// MetalRast.metal — optimized Metal port of CuRast (Schütz et al., 2026).
//
// Pipeline:
//   clear_framebuffer  → reset visibility buffer to UINT64_MAX (per-pixel)
//   stage1_rasterize   → persistent threadgroup-batched, threadgroup-shared mesh data
//   prepare_indirect   → write threadgroup count for stage 2 / stage 3 dispatches
//   stage2_rasterize   → 32 threads/triangle (one Apple SIMD group); warp-coop setup
//   stage3_rasterize   → 64 threads/tile, view-space Möller-Trumbore ray cast
//   resolve            → decode visibility buffer → find draw call → shade
//
// Visibility buffer encoding (paper §4.1):
//   high 28 bits = depth (sign + low-3 mantissa bits dropped from float32)
//   low  36 bits = global triangle index (so 64-bit atomic_min picks closest)
//
// vs. the literal CUDA reference, the perf optimizations applied here:
//   1. CPU pre-computes worldView = view * model per draw call → eliminates
//      3× modelMul + 3× viewMul per triangle in stages 1/2/3.
//   2. Stage 1 is a persistent batched kernel: each TG claims a 256-tri batch
//      via threadgroup-broadcast atomicAdd, walks its own mesh cursor in TG-
//      shared memory, and loads `DrawCall` + `worldView` once per mesh
//      transition. Eliminates per-triangle binary-search and per-triangle
//      DrawCall struct loads.
//   3. Stage 1 inner pixel loop uses incremental barycentric stepping
//      (s += ds_dx; t += dt_dx) and incremental pixelID instead of
//      multiplications + edge_fn() per pixel.
//   4. Sub-pixel cull: tris whose bbox doesn't contain a pixel sample are
//      dropped before the inner loop (huge win on highly-tessellated meshes).
//   5. Stage 2 setup (sutherland-hodgman near-clip, screen bbox, edges) runs
//      on lane 0 only and is broadcast via simd_broadcast — the other 31
//      lanes idle for ~10 instructions instead of redundantly doing the same
//      ~50 instructions of setup.
//   6. Stage 3 ray-casts in view space (origin = 0, no per-pixel matrix
//      transform). View-space vertices are pre-computed once per tile,
//      broadcast to all 64 threads.

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
#include "SharedTypes.h"
using namespace metal;

// ---------- function constants ---------------------------------------------
//
// Pipeline-specialization knobs. Set at pipeline creation time via
// MTLFunctionConstantValues; the Metal compiler bakes them in and dead-
// strips the unused branches/buffer params. This gives us CuRast's
// uncompressed×compressed × instanced×non-instanced kernel matrix without
// duplicating source.

// USE_COMPRESSED: false → kernel reads `device const float3* verticesU` at
// slot 0 (16 B/vertex). true → reads `device const ushort4* verticesC` at
// slot 0 (8 B/vertex), decoded with the per-mesh AABB stored on DrawCall.
// Instancing is handled by separate kernel entry points (`stage1_rasterize`
// vs `stage1_rasterize_instanced`) — easier to read than a combined kernel.
constant bool USE_COMPRESSED [[function_constant(0)]];
// USE_HIZ_MASK: false → stage1 elides the per-instance visibility-mask read
// at compile time, restoring the no-feature baseline cost when Hi-Z is OFF.
// true → stage1 reads the bit-packed mask before rasterizing each (DC,
// instance) pair. Built as a separate pipeline variant per scene; renderer
// picks based on the runtime toggle.
constant bool USE_HIZ_MASK   [[function_constant(1)]];
// USE_PREFIX_SUM: false → resolve does the plain log2(N) binary search over
// drawCalls[].cumulativeTriangleStart. true → resolve reads a precomputed
// 64K-entry bucket table to narrow the search to ~1-2 DCs per pixel. Two
// resolve pipelines are built so the no-feature variant pays no buffer-bind
// or memory-fetch cost.
constant bool USE_PREFIX_SUM [[function_constant(2)]];
// USE_TEXTURING gates the entire texture-sample path:
//   - the per-vertex UV pool / texture-set arg-buffer / sampler kernel
//     params (not declared in the untextured variant; host skips bind),
//   - the ~50-line bary-reconstruct + perspective-UV + gradient + sample
//     block in resolve (compile-time eliminated to lambert).
// Two PSOs (textured / untextured) × two for USE_PREFIX_SUM = 4 total.
constant bool USE_TEXTURING  [[function_constant(3)]];
// USE_COMPACTED_INSTANCES (Hijma §6.2.2 / runbook #12) — when ON, the
// instanced-stage1 kernel iterates a packed `visibleInstanceIdx[]`
// array produced by `compact_visible_instances` (one element per
// visible instance) instead of looping over the dense [firstInstance,
// firstInstance + instCount) range with a per-iteration Hi-Z bit test.
// Eliminates wasted iterations on Hi-Z-culled instances. The runtime
// only enables this PSO when Hi-Z is also active — the visibility mask
// is the input to the compaction.
constant bool USE_COMPACTED_INSTANCES [[function_constant(4)]];
// MSL's [[function_constant(...)]] attribute on buffer params can only
// reference a program-scope constant — not an expression. So we derive
// the inverse explicitly. (function-constant-derived constants get folded
// at pipeline-creation time too.)
constant bool USE_UNCOMPRESSED = !USE_COMPRESSED;
constant bool NO_PREFIX_SUM    = !USE_PREFIX_SUM;

// ---------- visibility-buffer encoding --------------------------------------

constant ulong PAYLOAD_MASK   = (1UL << 36) - 1UL;       // 0xF FFFF FFFF
constant uint  DEPTH_INVALID  = 0xFFFFFFFu;              // 28-bit "infinity"
constant uint  STAGE1_BATCH   = 256u;
constant int   STAGE2_TG      = 32;
constant int   STAGE3_TG      = 64;
constant int   STAGE3_TILE    = 64;
constant int   THRESHOLD_S    = 128;
constant int   THRESHOLD_L    = 4096;

inline ulong encode_fragment(float depth, ulong payload) {
    uint udepth = as_type<uint>(depth);
    uint d28    = (udepth & 0x7FFFFFF8u) >> 3;
    return (ulong(d28) << 36) | (payload & PAYLOAD_MASK);
}

inline ulong decode_payload(ulong encoded) {
    return encoded & PAYLOAD_MASK;
}

// ---------- vertex load -----------------------------------------------------
//
// Compressed layout: each `ushort4` slot holds (x, y, z, _pad). Decompression:
//   pos = float3(u.xyz) * factor + aabbMin
// Uncompressed layout: each `float3` slot holds (x, y, z) directly (16 B in
// MSL because float3 is 16-byte aligned).
//
// Kernels call LOAD_VPOS(idx, dc) — a macro because the function_constant
// attribute on the buffer params means only one of `verticesC` / `verticesU`
// is actually present at any one specialization. The dead branch gets
// stripped at pipeline-creation time.
inline float3 decompress_pos(ushort4 u, constant DrawCall &dc) {
    return float3(float(u.x) * dc.compressionFactorX + dc.aabbMinX,
                  float(u.y) * dc.compressionFactorY + dc.aabbMinY,
                  float(u.z) * dc.compressionFactorZ + dc.aabbMinZ);
}
inline float3 decompress_pos(ushort4 u, thread const DrawCall &dc) {
    return float3(float(u.x) * dc.compressionFactorX + dc.aabbMinX,
                  float(u.y) * dc.compressionFactorY + dc.aabbMinY,
                  float(u.z) * dc.compressionFactorZ + dc.aabbMinZ);
}
inline float3 decompress_pos(ushort4 u, threadgroup const DrawCall &dc) {
    return float3(float(u.x) * dc.compressionFactorX + dc.aabbMinX,
                  float(u.y) * dc.compressionFactorY + dc.aabbMinY,
                  float(u.z) * dc.compressionFactorZ + dc.aabbMinZ);
}

// Macro requires `verticesC` (ushort4*) and `verticesU` (float3*) to be in
// scope at the call site. Both are declared on every vertex-using kernel as
// conditional buffer parameters bound to the SAME slot — exactly one exists
// per specialization (the dead branch is stripped at pipeline creation).
#define LOAD_VPOS(idx, dc) \
    (USE_COMPRESSED ? decompress_pos(verticesC[(idx)], (dc)) : verticesU[(idx)])

// ---------- helpers ----------------------------------------------------------

// Project view-space point to screen space. Caller must guard against z<=0.
inline float3 project_to_screen(float3 view, constant CameraUniforms &cam) {
    float invZ = 1.0f / view.z;
    float2 ndc = view.xy * cam.projectionXY * invZ;          // NDC.x,y in [-1,1]
    float2 scr = (ndc * 0.5f + 0.5f) * float2(float(cam.screenWidth),
                                              float(cam.screenHeight));
    return float3(scr.x, scr.y, view.z);
}

// Linear-tail binary search for the draw call containing a global triangle ID.
// Used in resolve only. ulong globalTri because Zorah-scale scenes have > 4.3B
// post-instance triangles.
inline uint find_draw_call(device const DrawCall *drawCalls,
                           uint numDrawCalls,
                           ulong globalTri)
{
    uint lo = 0, hi = numDrawCalls;
    while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (drawCalls[mid].cumulativeTriangleStart <= globalTri) lo = mid;
        else                                                     hi = mid;
    }
    return lo;
}

// Prefix-sum-accelerated DC lookup. Bucket b of the precomputed `pfxBuckets`
// table holds `first DC i with cumulativeTriangleStart > b * (1<<pfxShift)`.
// For a query t, the answer DC i satisfies pfxBuckets[b] - 1 <= i < pfxBuckets[b+1]
// where b = t >> pfxShift. Search range size ≈ N / B (≪ 1 for our scenes), so
// the inner binary search exits in 0-2 iterations vs log2(N) for the plain
// search. Bucket count is fixed at compile-time on the host side; this kernel
// only needs the table pointer + shift.
inline uint find_draw_call_pfx(device const DrawCall *drawCalls,
                               device const uint     *pfxBuckets,
                               uint   pfxBucketCount,
                               uint   pfxShift,
                               ulong  globalTri)
{
    uint b   = uint(min(globalTri >> pfxShift, ulong(pfxBucketCount - 1)));
    uint lo  = pfxBuckets[b];
    uint hi  = pfxBuckets[b + 1];
    if (lo > 0) lo -= 1;          // pfx[b] = first i with cum > b*stride;
                                  // valid range starts at pfx[b]-1.
    if (hi <= lo + 1) return lo;  // single-DC bucket — common case
    while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (drawCalls[mid].cumulativeTriangleStart <= globalTri) lo = mid;
        else                                                     hi = mid;
    }
    return lo;
}

// ---------- BitEdit::readU32 in MSL (paper §4.6) ----------------------------
// Reads `bits` bits from a uint32 word stream starting at `bitPos`.
// Mirrors `BitEdit::readU32` from the CuRast source; handles cross-uint32
// straddling reads. Caller must ensure data has 1 trailing slack uint32.
//
// Unconditional 2-word load: the conditional `if (bitShift + bits > 32u)`
// caused intra-SIMD divergence — when one lane in a warp straddled while
// others didn't, the predicated tail-load serialized. Always loading both
// words removes the branch (and the divergence stall) at the cost of one
// guaranteed-cached neighbor load. The trailing-slack contract that the
// host already maintains makes the 2nd load safe regardless of `bits`.
// Stage1 -3.0% / total frame -3.0% on komainu (5M tri, --astc-block 4x4,
// --hi-z, 500/100, 3 runs); Zorah 100K saved-cam unchanged within noise.
inline uint readU32_chunk(device const uint* data, ulong bitPos, uint bits) {
    ulong  wordIdx  = bitPos / 32ul;
    uint   bitShift = uint(bitPos & 31ul);
    uint   w0 = data[wordIdx];
    uint   w1 = data[wordIdx + 1ul];
    ulong  combined = ulong(w0) | (ulong(w1) << 32);
    uint mask = (bits == 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return uint((combined >> bitShift) & mask);
}

// Multi-buffer index lookup. The host splits indices across up to 4 separate
// MTLBuffers (each ≤ ~18 GB on M2 Max) and tags each DrawCall with its chunk
// index. For scenes that fit in one buffer, all 4 slots get bound to chunk 0.
//
// Two paths:
//   bitsPerIndex == 0  → raw uint32 indices; offset is u32 element offset
//                        from the chunk start.
//   bitsPerIndex >  0  → packed; offset is BIT offset from the chunk start;
//                        we read `bitsPerIndex` bits starting there.
inline uint load_index_raw(device const uint *idx0,
                           device const uint *idx1,
                           device const uint *idx2,
                           device const uint *idx3,
                           uint chunkIdx, ulong offset)
{
    if (chunkIdx == 0u) return idx0[offset];
    if (chunkIdx == 1u) return idx1[offset];
    if (chunkIdx == 2u) return idx2[offset];
    return idx3[offset];
}

inline uint load_index_packed(device const uint *idx0,
                              device const uint *idx1,
                              device const uint *idx2,
                              device const uint *idx3,
                              uint chunkIdx, ulong bitPos, uint bits)
{
    if (chunkIdx == 0u) return readU32_chunk(idx0, bitPos, bits);
    if (chunkIdx == 1u) return readU32_chunk(idx1, bitPos, bits);
    if (chunkIdx == 2u) return readU32_chunk(idx2, bitPos, bits);
    return readU32_chunk(idx3, bitPos, bits);
}

// One-call API for the kernels: looks up the index for `indexInPrim`-th
// triangle vertex of a draw call (e.g. dc.indexOffset + 3*localTri + k).
// Branches on dc.bitsPerIndex (uniform across a SIMD group during stage1's
// mesh-by-mesh sweep, so cost ≈ 0).
//
// `dc.indexOffset` is:
//   bits = 0 → u32 element offset from chunk start
//   bits > 0 → bit offset from chunk start
// Returned value is the mesh-local vertex index, with `dc.indexMin` already
// added back (paper §4.6 — packed values are stored as actualIndex - indexMin
// for tighter bit width).
inline uint load_index(device const uint *idx0,
                       device const uint *idx1,
                       device const uint *idx2,
                       device const uint *idx3,
                       thread const DrawCall &dc,
                       ulong  indexInPrim)
{
    if (dc.bitsPerIndex == 0u) {
        // Raw uint32 path: indices are stored absolute (no indexMin bias).
        return load_index_raw(idx0, idx1, idx2, idx3, dc.indexBufferIdx,
                              dc.indexOffset + indexInPrim);
    }
    ulong bitPos = dc.indexOffset + indexInPrim * ulong(dc.bitsPerIndex);
    uint  packed = load_index_packed(idx0, idx1, idx2, idx3, dc.indexBufferIdx,
                                     bitPos, dc.bitsPerIndex);
    return packed + dc.indexMin;
}

// View-space backface cull (paper §4.1). Triangle is back-facing if the
// view-space face normal points the same way as v0 (which equals
// `v0 - origin`, since the camera is at the view-space origin).
//
// Accounts for negative-determinant worldView (mirrored instances) by
// flipping the sign of the test.
inline bool backface_cull_view(float3 v0, float3 v1, float3 v2,
                               float3x3 wvLinear)
{
    float3 N = cross(v1 - v0, v2 - v0);
    float  d = dot(v0, N);
    float  det = determinant(wvLinear);
    if (det < 0.0f) d = -d;
    return d >= 0.0f;
}

// =============================================================================
//   CLEAR
// =============================================================================
//
// Plain (non-atomic) writes — encoder serializes us before any rasterize
// dispatch. MSL doesn't expose 64-bit atomic_store anyway.
kernel void clear_framebuffer(device ulong *fb        [[buffer(0)]],
                              constant uint &count    [[buffer(1)]],
                              uint           gid      [[thread_position_in_grid]])
{
    // Vectorized 16B store per thread (one ulong2 = native AGX vector
    // width). Halves the dispatch grid count vs 1 thread/ulong while
    // keeping per-warp lane utilization identical to the scalar form.
    // Empirical: clear cost on 1920×1080 framebuffer drops from 28.5 µs
    // to 17.6 µs (-38%) on M2 Max. ulong4 (2× wider) was tried first
    // and clears even faster but caused a downstream Stage1 regression
    // (suspected: smaller dispatch finishes earlier → next stage starts
    // before the L2 settle, contended writes degrade atomic_min issue
    // throughput). ulong2 keeps the pipeline behavior intact.
    uint base = gid * 2u;
    if (base + 2u <= count) {
        device ulong2 *fb2 = reinterpret_cast<device ulong2*>(fb);
        fb2[gid] = ulong2(0xFFFFFFFFFFFFFFFFUL);
        return;
    }
    if (base < count) fb[base] = 0xFFFFFFFFFFFFFFFFUL;
}

// =============================================================================
//   PREPARE INDIRECT DISPATCH ARGS
// =============================================================================

kernel void prepare_indirect(
    device const atomic_uint     *itemCounter [[buffer(0)]],
    device IndirectDispatchArgs  *args        [[buffer(1)]],
    constant uint                 &maxItems   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint count = atomic_load_explicit(itemCounter, memory_order_relaxed);
    args->x = min(count, maxItems);
    args->y = 1;
    args->z = 1;
}

// =============================================================================
//   STAGE 1 — persistent batched kernel, threadgroup-shared mesh data
// =============================================================================
//
// 256 threads per TG, ~32 persistent TGs. Each TG keeps shared state
// (mesh + worldView) and walks meshes in lock-step, claiming one 256-tri batch
// per global atomic_add.

struct Stage1TG {
    DrawCall  mesh;              // current mesh metadata (instanceCount, firstInstance, etc.)
    uint      meshIdx;           // index into drawCalls[]
    uint      blockBatchIdx;     // last claimed global batch idx
    uint      blockLocalBatchIdx;// local batch idx within current mesh
    uint      _pad;
};

// Transform + cull + rasterize one (triangle, instance) pair. Object-space
// vertices are passed in by the caller (they're loaded once per triangle in
// the instanced kernel and reused across instances). Returns nothing.
//
// `worldView` is per-instance. `globalTri` includes the instance offset.
// 64-bit because Zorah-scale scenes have > 4.3B post-instance triangles, so
// the cumulative prefix-sum does not fit in 32 bits.
inline void stage1_one(
    float3 p0, float3 p1, float3 p2,
    float4x4 worldView,
    float wvDetSign,                      // +1 or -1; sign of det(worldView 3x3)
    uint dcID, uint instanceID, uint localTri,
    ulong globalTri,
    constant CameraUniforms &cam,
    device atomic_ulong *framebuffer,
    device atomic_uint *stage2Counter,
    device Stage2Item *stage2Queue,
    uint stage2Capacity,
    float W, float H, float zN, float zF)
{
    // World→view in a single 4x4 mul (worldView = view * model).
    float3 v0 = (worldView * float4(p0, 1.0f)).xyz;
    float3 v1 = (worldView * float4(p1, 1.0f)).xyz;
    float3 v2 = (worldView * float4(p2, 1.0f)).xyz;

    // Z-frustum cull.
    if (v0.z >= zF && v1.z >= zF && v2.z >= zF) return;
    if (v0.z <  zN && v1.z <  zN && v2.z <  zN) return;

    bool nearIntersect = (v0.z < zN) || (v1.z < zN) || (v2.z < zN);

    // View-space backface cull. The worldView determinant's sign is
    // hoisted to per-DC/per-instance scope by the caller (passed in
    // as `wvDetSign`); this kernel multiplies the per-tri `d` by it
    // to flip the sense for mirrored matrices. Saves a cross+dot
    // per tri (~10 fmul + 4 fadd) — pure repacking of arithmetic
    // that was previously redundant work.
    {
        float3 N = cross(v1 - v0, v2 - v0);
        float  d = dot(v0, N) * wvDetSign;
        if (d >= 0.0f) return;
    }

    // Defer near-crossing tris to stage 2 for sutherland-hodgman.
    if (nearIntersect) {
        uint slot = atomic_fetch_add_explicit(stage2Counter, 1u,
                                              memory_order_relaxed);
        if (slot < stage2Capacity)
            stage2Queue[slot] = Stage2Item{ dcID, localTri, instanceID, 0u };
        return;
    }

    // Project to screen.
    float3 s0 = project_to_screen(v0, cam);
    float3 s1 = project_to_screen(v1, cam);
    float3 s2 = project_to_screen(v2, cam);
    s0.xy -= 0.5f;  s1.xy -= 0.5f;  s2.xy -= 0.5f;

    float min_x = min3(s0.x, s1.x, s2.x);
    float max_x = max3(s0.x, s1.x, s2.x);
    float min_y = min3(s0.y, s1.y, s2.y);
    float max_y = max3(s0.y, s1.y, s2.y);

    if (max_x < 0.0f || min_x >= W || max_y < 0.0f || min_y >= H) return;

    min_x = max(min_x, 0.0f);
    max_x = min(max_x, W - 1.0f);
    min_y = max(min_y, 0.0f);
    max_y = min(max_y, H - 1.0f);

    // size_x / size_y count integer pixel samples in the bbox along each
    // axis. If either is 0 the bbox holds no sample → cull. (The old
    // floor()-based bbox-axis cull this replaced was mathematically
    // redundant with this check; proof: min_x > floor(min_x) AND
    // max_x < floor(min_x)+1 forces ceil(max_x) == ceil(min_x).)
    int size_x = int(ceil(max_x)) - int(ceil(min_x));
    int size_y = int(ceil(max_y)) - int(ceil(min_y));
    if (size_x <= 0 || size_y <= 0) return;

    if (size_x * size_y > THRESHOLD_S) {
        uint slot = atomic_fetch_add_explicit(stage2Counter, 1u,
                                              memory_order_relaxed);
        if (slot < stage2Capacity)
            stage2Queue[slot] = Stage2Item{ dcID, localTri, instanceID, 0u };
        return;
    }

    // Direct rasterization with incremental barycentric stepping.
    float2 v_ab = s1.xy - s0.xy;
    float2 v_ac = s2.xy - s0.xy;
    float  cr   = v_ab.x * v_ac.y - v_ab.y * v_ac.x;
    if (fabs(cr) < 1e-12f) return;
    float  factor = 1.0f / cr;

    float ds_dx =  v_ac.y * factor;
    float ds_dy = -v_ac.x * factor;
    float dt_dx = -v_ab.y * factor;
    float dt_dy =  v_ab.x * factor;

    float start_x   = ceil(min_x);
    float start_y   = ceil(min_y);
    float sample0_x = start_x - s0.x;
    float sample0_y = start_y - s0.y;
    float s_row = (sample0_x * v_ac.y - sample0_y * v_ac.x) * factor;
    float t_row = (v_ab.x * sample0_y - v_ab.y * sample0_x) * factor;

    float invZ0 = 1.0f / v0.z;
    float invZ1 = 1.0f / v1.z;
    float invZ2 = 1.0f / v2.z;

    uint pix_row = uint(start_y) * cam.screenWidth + uint(start_x);

    for (int dy = 0; dy < size_y; ++dy) {
        float s    = s_row;
        float t    = t_row;
        uint  pidx = pix_row;

        for (int dx = 0; dx < size_x; ++dx) {
            float v = 1.0f - (s + t);
            if (s >= 0.0f && t >= 0.0f && v >= 0.0f) {
                float invZ  = v * invZ0 + s * invZ1 + t * invZ2;
                float depth = 1.0f / invZ;
                if (depth >= zN && depth <= zF) {
                    ulong encoded = encode_fragment(depth, globalTri);
                    atomic_min_explicit(&framebuffer[pidx], encoded,
                                        memory_order_relaxed);
                }
            }
            s += ds_dx;
            t += dt_dx;
            pidx++;
        }
        s_row   += ds_dy;
        t_row   += dt_dy;
        pix_row += cam.screenWidth;
    }
}

// Single-instance kernel — the fast path when no draw call has > 1 instance.
// Each TG keeps the current mesh + worldView in TG-shared memory and walks
// meshes in lock-step.
kernel void stage1_rasterize(
    device const ushort4     *verticesC      [[buffer(0), function_constant(USE_COMPRESSED)]],
    device const float3      *verticesU      [[buffer(0), function_constant(USE_UNCOMPRESSED)]],
    device const uint        *indices0       [[buffer(1)]],
    device const uint        *indices1       [[buffer(15)]],
    device const uint        *indices2       [[buffer(16)]],
    device const uint        *indices3       [[buffer(17)]],
    device const DrawCall    *drawCalls      [[buffer(2)]],
    constant uint            &numDrawCalls   [[buffer(3)]],
    constant uint            &totalTriangles [[buffer(4)]],   // unused
    constant CameraUniforms  &cam            [[buffer(5)]],
    device atomic_ulong      *framebuffer    [[buffer(6)]],
    device atomic_uint       *batchCounter   [[buffer(7)]],
    device atomic_uint       *stage2Counter  [[buffer(8)]],
    device Stage2Item        *stage2Queue    [[buffer(9)]],
    constant uint            &stage2Capacity [[buffer(10)]],
    device const uint        *visibleBits    [[buffer(11)]],   // hiZ cull mask, bit-packed (per worldView slot)
    device const float4x4    *worldViewArr   [[buffer(14)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]])
{
    threadgroup Stage1TG sh;
    threadgroup float4x4 sh_worldView;
    // Sign of det(worldView 3x3), hoisted from per-tri to per-DC. The
    // backface cull in stage1_one needs it to handle mirrored matrices;
    // computing it once amortizes ~10 fmul + 4 fadd per DC across all
    // tris of that DC instead of paying it per-tri.
    threadgroup float    sh_wvDetSign;

    if (tid_in_tg == 0) {
        sh.mesh                = drawCalls[0];
        sh_worldView           = worldViewArr[drawCalls[0].firstInstance];
        // Det sign is precomputed by the host into wv[3][3] (an otherwise
        // unread slot in an affine matrix), saving a per-DC cross+dot and
        // — more importantly — a per-(tri, instance) cross+dot in the
        // instanced kernel below. Bottom row of affine wv is [0,0,0,1]
        // so this packing doesn't affect any other consumer.
        sh_wvDetSign           = sh_worldView[3].w;
        sh.meshIdx             = 0;
        sh.blockBatchIdx       = 0;
        sh.blockLocalBatchIdx  = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float W = float(cam.screenWidth);
    const float H = float(cam.screenHeight);
    const float zN = cam.nearPlane, zF = cam.farPlane;

    while (true) {
        if (tid_in_tg == 0) {
            uint next = atomic_fetch_add_explicit(batchCounter, 1u,
                                                  memory_order_relaxed);
            uint diff = next - sh.blockBatchIdx;
            sh.blockBatchIdx       = next;
            sh.blockLocalBatchIdx += diff;

            uint nbim = (sh.mesh.triangleCount + STAGE1_BATCH - 1u) / STAGE1_BATCH;
            while (sh.blockLocalBatchIdx >= nbim) {
                sh.meshIdx += 1u;
                if (sh.meshIdx >= numDrawCalls) break;
                sh.blockLocalBatchIdx -= nbim;
                sh.mesh       = drawCalls[sh.meshIdx];
                sh_worldView  = worldViewArr[sh.mesh.firstInstance];
                sh_wvDetSign  = sh_worldView[3].w;     // host pre-packed the sign
                nbim          = (sh.mesh.triangleCount + STAGE1_BATCH - 1u) / STAGE1_BATCH;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sh.meshIdx >= numDrawCalls) return;

        uint localTri = sh.blockLocalBatchIdx * STAGE1_BATCH + tid_in_tg;
        if (localTri >= sh.mesh.triangleCount) continue;

        DrawCall dc      = sh.mesh;
        float4x4 wv      = sh_worldView;

        // Hi-Z cull (per (DC, instance)): instanceCount==1 here so the slot
        // is simply firstInstance. Bit-packed: bit s of word s/32 is the
        // flag. USE_HIZ_MASK=false elides the entire load+branch at compile
        // time → no overhead when toggle is OFF.
        if (USE_HIZ_MASK) {
            uint sl = dc.firstInstance;
            if (((visibleBits[sl >> 5u] >> (sl & 31u)) & 1u) == 0u) continue;
        }

        uint i0 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 0ul);
        uint i1 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 1ul);
        uint i2 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 2ul);
        float3 p0 = LOAD_VPOS(dc.vertexOffset + i0, dc);
        float3 p1 = LOAD_VPOS(dc.vertexOffset + i1, dc);
        float3 p2 = LOAD_VPOS(dc.vertexOffset + i2, dc);

        // instanceCount == 1 here; instanceID = 0; globalTri = cumStart + localTri.
        ulong globalTri = dc.cumulativeTriangleStart + ulong(localTri);
        stage1_one(p0, p1, p2, wv, sh_wvDetSign,
                   sh.meshIdx, 0u, localTri, globalTri,
                   cam, framebuffer, stage2Counter, stage2Queue, stage2Capacity,
                   W, H, zN, zF);
    }
}

// Instanced kernel (paper §4.5). For each (loaded) triangle, loop over
// `mesh.instanceCount` instances and rasterize each with its own worldView.
// The per-triangle vertex/index loads are amortized across all instances.
kernel void stage1_rasterize_instanced(
    device const ushort4     *verticesC      [[buffer(0), function_constant(USE_COMPRESSED)]],
    device const float3      *verticesU      [[buffer(0), function_constant(USE_UNCOMPRESSED)]],
    device const uint        *indices0       [[buffer(1)]],
    device const uint        *indices1       [[buffer(15)]],
    device const uint        *indices2       [[buffer(16)]],
    device const uint        *indices3       [[buffer(17)]],
    device const DrawCall    *drawCalls      [[buffer(2)]],
    constant uint            &numDrawCalls   [[buffer(3)]],
    constant uint            &totalTriangles [[buffer(4)]],   // unused
    constant CameraUniforms  &cam            [[buffer(5)]],
    device atomic_ulong      *framebuffer    [[buffer(6)]],
    device atomic_uint       *batchCounter   [[buffer(7)]],
    device atomic_uint       *stage2Counter  [[buffer(8)]],
    device Stage2Item        *stage2Queue    [[buffer(9)]],
    constant uint            &stage2Capacity [[buffer(10)]],
    device const uint        *visibleBits    [[buffer(11)]],   // hiZ cull mask, bit-packed (per worldView slot)
    device const float4x4    *worldViewArr   [[buffer(14)]],
    device const uint        *visibleInstanceIdx    [[buffer(18), function_constant(USE_COMPACTED_INSTANCES)]],
    device const uint        *visibleInstancesPerDC [[buffer(19), function_constant(USE_COMPACTED_INSTANCES)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]])
{
    threadgroup Stage1TG sh;

    if (tid_in_tg == 0) {
        sh.mesh                = drawCalls[0];
        sh.meshIdx             = 0;
        sh.blockBatchIdx       = 0;
        sh.blockLocalBatchIdx  = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float W = float(cam.screenWidth);
    const float H = float(cam.screenHeight);
    const float zN = cam.nearPlane, zF = cam.farPlane;

    while (true) {
        if (tid_in_tg == 0) {
            uint next = atomic_fetch_add_explicit(batchCounter, 1u,
                                                  memory_order_relaxed);
            uint diff = next - sh.blockBatchIdx;
            sh.blockBatchIdx       = next;
            sh.blockLocalBatchIdx += diff;

            uint nbim = (sh.mesh.triangleCount + STAGE1_BATCH - 1u) / STAGE1_BATCH;
            while (sh.blockLocalBatchIdx >= nbim) {
                sh.meshIdx += 1u;
                if (sh.meshIdx >= numDrawCalls) break;
                sh.blockLocalBatchIdx -= nbim;
                sh.mesh   = drawCalls[sh.meshIdx];
                nbim      = (sh.mesh.triangleCount + STAGE1_BATCH - 1u) / STAGE1_BATCH;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sh.meshIdx >= numDrawCalls) return;

        uint localTri = sh.blockLocalBatchIdx * STAGE1_BATCH + tid_in_tg;
        if (localTri >= sh.mesh.triangleCount) continue;

        DrawCall dc = sh.mesh;

        // Load the triangle's vertices ONCE. They get reused across all instances.
        uint i0 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 0ul);
        uint i1 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 1ul);
        uint i2 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 2ul);
        float3 p0 = LOAD_VPOS(dc.vertexOffset + i0, dc);
        float3 p1 = LOAD_VPOS(dc.vertexOffset + i1, dc);
        float3 p2 = LOAD_VPOS(dc.vertexOffset + i2, dc);

        // Loop over instances of this triangle. Two paths:
        //   - USE_COMPACTED_INSTANCES off: original dense iteration with
        //     a per-instance Hi-Z bit test (compiled out if !USE_HIZ_MASK).
        //   - USE_COMPACTED_INSTANCES on: iterate the packed
        //     `visibleInstanceIdx[]` produced by `compact_visible_instances`.
        //     Hi-Z is implicit (only visible instances were packed).
        uint instCount = dc.instanceCount;
        uint firstInst = dc.firstInstance;
        uint loopCount = USE_COMPACTED_INSTANCES
                       ? visibleInstancesPerDC[sh.meshIdx]
                       : instCount;
        for (uint k = 0; k < loopCount; ++k) {
            uint inst = USE_COMPACTED_INSTANCES
                      ? visibleInstanceIdx[firstInst + k]
                      : k;
            if (!USE_COMPACTED_INSTANCES && USE_HIZ_MASK) {
                uint sl = firstInst + inst;
                if (((visibleBits[sl >> 5u] >> (sl & 31u)) & 1u) == 0u) continue;
            }
            float4x4 wv      = worldViewArr[firstInst + inst];
            // Per-instance det sign was pre-packed by the host into wv[3][3]
            // (otherwise-unread .w slot of the bottom row). One register
            // load instead of 3 fmul + 5 fadd + 1 fcmp per (tri, instance).
            float    wvDetSign = wv[3].w;
            ulong    globalTri = dc.cumulativeTriangleStart
                               + ulong(inst) * ulong(dc.triangleCount)
                               + ulong(localTri);
            stage1_one(p0, p1, p2, wv, wvDetSign,
                       sh.meshIdx, inst, localTri, globalTri,
                       cam, framebuffer, stage2Counter, stage2Queue, stage2Capacity,
                       W, H, zN, zF);
        }
    }
}

// =============================================================================
//   STAGE 2 — medium triangles, 32 threads / triangle, warp-cooperative setup
// =============================================================================

inline int clip_near(thread float3 in_pts[3], float zN,
                     thread float3 out_pts[4])
{
    int n = 0;
    for (int i = 0; i < 3; ++i) {
        float3 a = in_pts[i];
        float3 b = in_pts[(i+1) % 3];
        bool aIn = a.z >= zN;
        bool bIn = b.z >= zN;
        if (aIn) out_pts[n++] = a;
        if (aIn != bIn) {
            float t = (zN - a.z) / (b.z - a.z);
            out_pts[n++] = a + t * (b - a);
        }
    }
    return n;
}

kernel void stage2_rasterize(
    device const ushort4     *verticesC      [[buffer(0), function_constant(USE_COMPRESSED)]],
    device const float3      *verticesU      [[buffer(0), function_constant(USE_UNCOMPRESSED)]],
    device const uint        *indices0       [[buffer(1)]],
    device const uint        *indices1       [[buffer(15)]],
    device const uint        *indices2       [[buffer(16)]],
    device const uint        *indices3       [[buffer(17)]],
    device const DrawCall    *drawCalls      [[buffer(2)]],
    constant uint            &numDrawCalls   [[buffer(3)]],
    constant CameraUniforms  &cam            [[buffer(5)]],
    device atomic_ulong      *framebuffer    [[buffer(6)]],
    device const Stage2Item  *stage2Queue    [[buffer(9)]],
    device atomic_uint       *stage3Counter  [[buffer(11)]],
    device Stage3Item        *stage3Queue    [[buffer(12)]],
    constant uint            &stage3Capacity [[buffer(13)]],
    device const float4x4    *worldViewArr   [[buffer(14)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]],
    uint  tg_id     [[threadgroup_position_in_grid]])
{
    Stage2Item item     = stage2Queue[tg_id];
    DrawCall   dc       = drawCalls[item.drawCallID];
    // globalTri encodes (mesh, instance, localTri) per the formula in SharedTypes.h.
    ulong      globalTri= dc.cumulativeTriangleStart
                        + ulong(item.instanceID) * ulong(dc.triangleCount)
                        + ulong(item.localTriangleID);

    // ---- Warp-cooperative setup: lane 0 does the work, broadcasts results.
    float min_x = 0, min_y = 0, max_x = 0, max_y = 0;
    float3 v0_ = float3(0), v1_ = float3(0), v2_ = float3(0);
    float3 s0_ = float3(0), s1_ = float3(0), s2_ = float3(0);
    bool   forward3 = false;
    bool   skip     = false;

    if (tid_in_tg == 0) {
        // Per-instance worldView: dc.firstInstance + item.instanceID.
        float4x4 worldView = worldViewArr[dc.firstInstance + item.instanceID];

        uint i0 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 0ul);
        uint i1 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 1ul);
        uint i2 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 2ul);
        float3 p0 = LOAD_VPOS(dc.vertexOffset + i0, dc);
        float3 p1 = LOAD_VPOS(dc.vertexOffset + i1, dc);
        float3 p2 = LOAD_VPOS(dc.vertexOffset + i2, dc);

        v0_ = (worldView * float4(p0, 1.0f)).xyz;
        v1_ = (worldView * float4(p1, 1.0f)).xyz;
        v2_ = (worldView * float4(p2, 1.0f)).xyz;

        float3x3 wvL = float3x3(worldView[0].xyz,
                                worldView[1].xyz,
                                worldView[2].xyz);
        if (backface_cull_view(v0_, v1_, v2_, wvL)) {
            skip = true;
        } else {
            float zN = cam.nearPlane;

            float3 vIn[3]   = { v0_, v1_, v2_ };
            float3 vClip[4];
            int nClip = clip_near(vIn, zN, vClip);
            if (nClip < 3) {
                skip = true;
            } else {
                float W = float(cam.screenWidth), H = float(cam.screenHeight);
                min_x =  1e30f; min_y =  1e30f;
                max_x = -1e30f; max_y = -1e30f;
                for (int k = 0; k < nClip; ++k) {
                    float3 sk = project_to_screen(vClip[k], cam);
                    min_x = min(min_x, sk.x); max_x = max(max_x, sk.x);
                    min_y = min(min_y, sk.y); max_y = max(max_y, sk.y);
                }
                if (max_x < 0.0f || min_x >= W || max_y < 0.0f || min_y >= H) {
                    skip = true;
                } else {
                    int ix0 = max(0, int(floor(min_x)));
                    int iy0 = max(0, int(floor(min_y)));
                    int ix1 = min(int(W) - 1, int(floor(max_x)));
                    int iy1 = min(int(H) - 1, int(floor(max_y)));
                    int bw  = ix1 - ix0 + 1;
                    int bh  = iy1 - iy0 + 1;

                    bool nearIntersect = (v0_.z < zN) || (v1_.z < zN) || (v2_.z < zN);
                    if (bw * bh > THRESHOLD_L || nearIntersect) {
                        forward3 = true;
                        // Pass integer pixel bbox to broadcast block via min/max.
                        min_x = float(ix0); min_y = float(iy0);
                        max_x = float(ix1); max_y = float(iy1);
                    } else {
                        s0_ = project_to_screen(v0_, cam);
                        s1_ = project_to_screen(v1_, cam);
                        s2_ = project_to_screen(v2_, cam);
                        s0_.xy -= 0.5f; s1_.xy -= 0.5f; s2_.xy -= 0.5f;
                        // Pass clamped float bbox to broadcast block.
                        min_x = max(min(min(s0_.x, s1_.x), s2_.x), 0.0f);
                        min_y = max(min(min(s0_.y, s1_.y), s2_.y), 0.0f);
                        max_x = min(max(max(s0_.x, s1_.x), s2_.x), W - 1.0f);
                        max_y = min(max(max(s0_.y, s1_.y), s2_.y), H - 1.0f);
                    }
                }
            }
        }
    }

    // simd_broadcast doesn't accept bool — round-trip via uint.
    if (simd_broadcast(uint(skip), 0) != 0u) return;

    forward3 = (simd_broadcast(uint(forward3), 0) != 0u);
    min_x    = simd_broadcast(min_x, 0);
    min_y    = simd_broadcast(min_y, 0);
    max_x    = simd_broadcast(max_x, 0);
    max_y    = simd_broadcast(max_y, 0);

    if (forward3) {
        if (tid_in_tg == 0) {
            int tx0 = int(min_x) / STAGE3_TILE;
            int ty0 = int(min_y) / STAGE3_TILE;
            int tx1 = int(max_x) / STAGE3_TILE;
            int ty1 = int(max_y) / STAGE3_TILE;
            for (int ty = ty0; ty <= ty1; ++ty) {
                for (int tx = tx0; tx <= tx1; ++tx) {
                    uint slot = atomic_fetch_add_explicit(stage3Counter, 1u,
                                                          memory_order_relaxed);
                    if (slot < stage3Capacity) {
                        stage3Queue[slot] = Stage3Item{
                            item.drawCallID, item.localTriangleID,
                            item.instanceID,
                            uint(tx) * STAGE3_TILE, uint(ty) * STAGE3_TILE,
                            0u, 0u, 0u
                        };
                    }
                }
            }
        }
        return;
    }

    // Direct medium-triangle rasterization.
    v0_ = float3(simd_broadcast(v0_.x, 0), simd_broadcast(v0_.y, 0), simd_broadcast(v0_.z, 0));
    v1_ = float3(simd_broadcast(v1_.x, 0), simd_broadcast(v1_.y, 0), simd_broadcast(v1_.z, 0));
    v2_ = float3(simd_broadcast(v2_.x, 0), simd_broadcast(v2_.y, 0), simd_broadcast(v2_.z, 0));
    s0_ = float3(simd_broadcast(s0_.x, 0), simd_broadcast(s0_.y, 0), 0.0f);
    s1_ = float3(simd_broadcast(s1_.x, 0), simd_broadcast(s1_.y, 0), 0.0f);
    s2_ = float3(simd_broadcast(s2_.x, 0), simd_broadcast(s2_.y, 0), 0.0f);

    float zN = cam.nearPlane, zF = cam.farPlane;

    int ix0 = int(ceil(min_x));
    int iy0 = int(ceil(min_y));
    int ix1 = int(ceil(max_x));
    int iy1 = int(ceil(max_y));
    int bw  = ix1 - ix0;
    int bh  = iy1 - iy0;
    if (bw <= 0 || bh <= 0) return;

    float2 v_ab = s1_.xy - s0_.xy;
    float2 v_ac = s2_.xy - s0_.xy;
    float  cr   = v_ab.x * v_ac.y - v_ab.y * v_ac.x;
    if (fabs(cr) < 1e-12f) return;
    float  factor = 1.0f / cr;

    float invZ0 = 1.0f / v0_.z;
    float invZ1 = 1.0f / v1_.z;
    float invZ2 = 1.0f / v2_.z;

    int npix = bw * bh;
    // AGX has a multi-cycle integer divide; replace per-iteration `p / bw`
    // and `p % bw` with a float-reciprocal multiply (~1-2 cyc) and a
    // ±1 correction step. `(uint)(p * 1/bw)` can round down by 1 ULP at
    // exact multiples of bw (where the true quotient is N but the float
    // product is N - ε), and very rarely round up by 1 (N + ε with N*ε > 1).
    // The fix-up handles both cases via a residue check; the corrected
    // branch is taken at most once and only for ~1-2 of every (bw*bh)
    // iterations, so cost is essentially zero.
    float inv_bw = 1.0f / float(bw);
    for (int p = int(tid_in_tg); p < npix; p += STAGE2_TG) {
        int dy = int(float(p) * inv_bw);
        int dx = p - dy * bw;
        if (dx >= bw)        { dy += 1; dx -= bw; }      // recip under-shot
        else if (dx < 0)     { dy -= 1; dx += bw; }      // recip over-shot
        int x  = ix0 + dx;
        int y  = iy0 + dy;
        float sx = float(x) - s0_.x;
        float sy = float(y) - s0_.y;
        float s = (sx * v_ac.y - sy * v_ac.x) * factor;
        float t = (v_ab.x * sy - v_ab.y * sx) * factor;
        float v = 1.0f - (s + t);
        if (s < 0.0f || t < 0.0f || v < 0.0f) continue;

        float invZ  = v * invZ0 + s * invZ1 + t * invZ2;
        float depth = 1.0f / invZ;
        if (depth < zN || depth > zF) continue;

        ulong encoded = encode_fragment(depth, globalTri);
        uint  pidx    = uint(y) * cam.screenWidth + uint(x);
        atomic_min_explicit(&framebuffer[pidx], encoded,
                            memory_order_relaxed);
    }
}

// =============================================================================
//   STAGE 3 — large triangles, 64 threads / 64×64 tile, view-space MT
// =============================================================================
//
// View-space ray: origin = (0,0,0), dir = (u/(f/aspect), v/-f, 1). No
// per-pixel matrix transform. Triangle vertices transformed once by lane 0
// then made visible to all lanes via threadgroup memory.

kernel void stage3_rasterize(
    device const ushort4     *verticesC      [[buffer(0), function_constant(USE_COMPRESSED)]],
    device const float3      *verticesU      [[buffer(0), function_constant(USE_UNCOMPRESSED)]],
    device const uint        *indices0      [[buffer(1)]],
    device const uint        *indices1      [[buffer(15)]],
    device const uint        *indices2      [[buffer(16)]],
    device const uint        *indices3      [[buffer(17)]],
    device const DrawCall    *drawCalls     [[buffer(2)]],
    constant CameraUniforms  &cam           [[buffer(5)]],
    device atomic_ulong      *framebuffer   [[buffer(6)]],
    device const Stage3Item  *stage3Queue   [[buffer(12)]],
    device const float4x4    *worldViewArr  [[buffer(14)]],
    uint  tid_in_tg [[thread_position_in_threadgroup]],
    uint  tg_id     [[threadgroup_position_in_grid]])
{
    Stage3Item item     = stage3Queue[tg_id];
    DrawCall   dc       = drawCalls[item.drawCallID];
    ulong      globalTri= dc.cumulativeTriangleStart
                        + ulong(item.instanceID) * ulong(dc.triangleCount)
                        + ulong(item.localTriangleID);

    threadgroup float3 sh_v0;
    threadgroup float3 sh_v1;
    threadgroup float3 sh_v2;

    if (tid_in_tg == 0) {
        float4x4 worldView = worldViewArr[dc.firstInstance + item.instanceID];
        uint i0 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 0ul);
        uint i1 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 1ul);
        uint i2 = load_index(indices0, indices1, indices2, indices3, dc, ulong(item.localTriangleID)*3ul + 2ul);
        float3 p0 = LOAD_VPOS(dc.vertexOffset + i0, dc);
        float3 p1 = LOAD_VPOS(dc.vertexOffset + i1, dc);
        float3 p2 = LOAD_VPOS(dc.vertexOffset + i2, dc);
        sh_v0 = (worldView * float4(p0, 1.0f)).xyz;
        sh_v1 = (worldView * float4(p1, 1.0f)).xyz;
        sh_v2 = (worldView * float4(p2, 1.0f)).xyz;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float3 v0 = sh_v0, v1 = sh_v1, v2 = sh_v2;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    float invPx = 1.0f / cam.projectionXY.x;   // 1 / (f / aspect)
    float invPy = 1.0f / cam.projectionXY.y;   // 1 / -f
    float zN    = cam.nearPlane, zF = cam.farPlane;

    int W = int(cam.screenWidth), H = int(cam.screenHeight);
    int tileX0 = int(item.tileX);
    int tileY0 = int(item.tileY);

    for (int rowOffset = 0; rowOffset < STAGE3_TILE; ++rowOffset) {
        int y = tileY0 + rowOffset;
        int x = tileX0 + int(tid_in_tg);
        if (x < 0 || x >= W || y < 0 || y >= H) continue;

        float ndcX = (float(x) + 0.5f) / float(W) * 2.0f - 1.0f;
        float ndcY = (float(y) + 0.5f) / float(H) * 2.0f - 1.0f;
        float3 dir = float3(ndcX * invPx, ndcY * invPy, 1.0f);

        float3 P  = cross(dir, e2);
        float  det = dot(e1, P);
        if (fabs(det) < 1e-20f) continue;
        float invDet = 1.0f / det;

        float3 T  = -v0;
        float u   = dot(T, P) * invDet;
        if (u < 0.0f || u > 1.0f) continue;

        float3 Q  = cross(T, e1);
        float v   = dot(dir, Q) * invDet;
        if (v < 0.0f || u + v > 1.0f) continue;

        float tHit = dot(e2, Q) * invDet;
        if (tHit <= 0.0f) continue;

        float depth = tHit;       // dir.z = 1
        if (depth < zN || depth > zF) continue;

        ulong encoded = encode_fragment(depth, globalTri);
        uint  pidx    = uint(y) * cam.screenWidth + uint(x);
        atomic_min_explicit(&framebuffer[pidx], encoded,
                            memory_order_relaxed);
    }
}

// =============================================================================
//   RESOLVE — decode visibility buffer + tinted lambertian shade
// =============================================================================

// visMode passed via setBytes (slot 11) so flipping it doesn't trigger a
// pipeline rebuild. See ViewerState::visMode.
//   0 = lambert / normal-shaded (default)
//   1 = depth (greyscale, near→white far→black)
//   2 = mesh ID  (hash dcID)
//   3 = triangle ID (hash globalTri)
//   4 = stage (no encoded stage; we proxy via screen-space triangle area:
//             tiny→blue (s1), mid→green (s2), large→red (s3))
// Texture set — argument buffer with one texture handle per glTF image.
// The resolve kernel indexes into `textures[dc.textureHandle]` and samples
// with hardware bilinear / trilinear / aniso. Empty when scene has no
// textures; a function constant gates the entire path so untextured scenes
// don't pay the binding cost.
constant constexpr uint kMaxTextures = 256;
struct TextureSet {
    array<texture2d<half, access::sample>, kMaxTextures> tex [[id(0)]];
};

kernel void resolve(
    device const ulong         *framebuffer    [[buffer(0)]],
    device const ushort4     *verticesC      [[buffer(1), function_constant(USE_COMPRESSED)]],
    device const float3      *verticesU      [[buffer(1), function_constant(USE_UNCOMPRESSED)]],
    device const uint          *indices0       [[buffer(2)]],
    device const uint          *indices1       [[buffer(8)]],
    device const uint          *indices2       [[buffer(9)]],
    device const uint          *indices3       [[buffer(10)]],
    device const DrawCall      *drawCalls      [[buffer(3)]],
    constant uint              &numDrawCalls   [[buffer(4)]],
    constant CameraUniforms    &cam            [[buffer(5)]],
    device const float4x4      *modelMatrices  [[buffer(6)]],
    constant uint              &visMode        [[buffer(11)]],
    device const uint          *pfxBuckets     [[buffer(12), function_constant(USE_PREFIX_SUM)]],
    constant uint              &pfxBucketCount [[buffer(13), function_constant(USE_PREFIX_SUM)]],
    constant uint              &pfxShift       [[buffer(14), function_constant(USE_PREFIX_SUM)]],
    device const float2        *uvs            [[buffer(15), function_constant(USE_TEXTURING)]],
    constant TextureSet        &texSet         [[buffer(16), function_constant(USE_TEXTURING)]],
    device const float4x4      *worldView      [[buffer(17)]],
    sampler                     texSampler     [[sampler(0),  function_constant(USE_TEXTURING)]],
    texture2d<float, access::write> outTex     [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cam.screenWidth || gid.y >= cam.screenHeight) return;
    uint pidx = gid.y * cam.screenWidth + gid.x;

    ulong encoded = framebuffer[pidx];
    uint d28      = uint(encoded >> 36);

    if (d28 == DEPTH_INVALID) {
        float t = float(gid.y) / float(cam.screenHeight);
        float3 bg = mix(float3(0.05, 0.06, 0.10), float3(0.18, 0.22, 0.30), t);
        outTex.write(float4(bg, 1.0), gid);
        return;
    }

    if (visMode == 1u) {
        // d28 is a bit-packed view of the IEEE float depth (top 28 bits,
        // sign stripped) used for atomic_min ordering — NOT a normalized
        // [0,1]. Reverse the pack to get a float depth back, then log-map
        // it to [near, far] so near-plane detail isn't crushed.
        uint udepth = d28 << 3;
        float depth = as_type<float>(udepth);
        float zN = max(cam.nearPlane, 1e-4f);
        float zF = max(cam.farPlane,  zN * 1.001f);
        float t  = saturate(log(max(depth, zN) / zN) / log(zF / zN));
        float v  = 1.0f - t;       // near = white, far = black
        outTex.write(float4(v, v, v, 1.0f), gid);
        return;
    }

    ulong globalTri = decode_payload(encoded);
    uint  dcID      = USE_PREFIX_SUM
        ? find_draw_call_pfx(drawCalls, pfxBuckets, pfxBucketCount, pfxShift, globalTri)
        : find_draw_call    (drawCalls, numDrawCalls, globalTri);
    DrawCall dc     = drawCalls[dcID];
    // Decode (instance, localTri) from the global ID. Per-DC offset fits in
    // u32 (no single DC has > 4B triangles), so it's safe to narrow here.
    uint offsetInDC = uint(globalTri - dc.cumulativeTriangleStart);
    uint instanceID = (dc.instanceCount > 1u) ? (offsetInDC / dc.triangleCount) : 0u;
    uint localTri   = offsetInDC - instanceID * dc.triangleCount;

    if (visMode == 2u) {
        // Mesh ID — hash dcID alone (no instance), so all instances of the
        // same mesh share the same colour.
        uint  h     = dcID * 2654435761u;
        float3 col  = float3(((h >> 16) & 0xFFu) / 255.0f,
                             ((h >>  8) & 0xFFu) / 255.0f,
                             ((h >>  0) & 0xFFu) / 255.0f);
        outTex.write(float4(col, 1.0), gid);
        return;
    }
    if (visMode == 3u) {
        // Triangle ID — hash global triangle number. Fold the high 32 bits in
        // so triangles past 2^32 don't collide with the lower range.
        uint  h     = (uint(globalTri) ^ uint(globalTri >> 32u)) * 374761393u + 668265263u;
        h ^= (h >> 13); h *= 1274126177u; h ^= (h >> 16);
        float3 col  = float3(((h >> 16) & 0xFFu) / 255.0f,
                             ((h >>  8) & 0xFFu) / 255.0f,
                             ((h >>  0) & 0xFFu) / 255.0f);
        outTex.write(float4(col, 1.0), gid);
        return;
    }

    // Single matrix per vertex: worldView = view * model is precomputed on
    // the host (one mat-mul per instance per frame), so resolve does ONE
    // matvec per vertex to land in view space, not the previous two
    // (model * vert, then viewMatrix * world). Saves 3 mat4-vec4 muls per
    // textured pixel — ~48 muls/pixel × visible-tex-coverage at 1080p.
    //
    // Follow-up still on the table: a deferred per-triangle setup pass
    // that runs once per *visible* triangle (compacted via atomic-add into
    // a list) and writes screen-space s0/s1/s2 + edge equations + invZ
    // into a small buffer; resolve then reads them by triId. That would
    // eliminate the remaining per-pixel `project_to_screen × 3` plus the
    // gradient-reconstruction work below. ~2-4× speedup on resolve at the
    // cost of one new kernel + buffer. Out of scope for this commit.
    float4x4 wv = worldView[dc.firstInstance + instanceID];

    uint i0 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 0ul);
    uint i1 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 1ul);
    uint i2 = load_index(indices0, indices1, indices2, indices3, dc, ulong(localTri)*3ul + 2ul);
    float3 mp0 = LOAD_VPOS(dc.vertexOffset + i0, dc);
    float3 mp1 = LOAD_VPOS(dc.vertexOffset + i1, dc);
    float3 mp2 = LOAD_VPOS(dc.vertexOffset + i2, dc);
    // View-space positions. Camera is at the origin, +Z = forward.
    float3 v0 = (wv * float4(mp0, 1)).xyz;
    float3 v1 = (wv * float4(mp1, 1)).xyz;
    float3 v2 = (wv * float4(mp2, 1)).xyz;

    if (visMode == 4u) {
        // Stage: project verts to screen, classify by max screen-extent.
        // Skip behind-camera (shouldn't happen since the pixel is drawn,
        // but guard against numerical edge-cases).
        float zmin = min(v0.z, min(v1.z, v2.z));
        float3 col = float3(0.5);
        if (zmin > cam.nearPlane) {
            float3 s0 = project_to_screen(v0, cam);
            float3 s1 = project_to_screen(v1, cam);
            float3 s2 = project_to_screen(v2, cam);
            float2 lo = min(s0.xy, min(s1.xy, s2.xy));
            float2 hi = max(s0.xy, max(s1.xy, s2.xy));
            float  ext = max(hi.x - lo.x, hi.y - lo.y);
            col = (ext < 8.0f)  ? float3(0.30, 0.55, 0.95)    // tiny → stage1 (blue)
                : (ext < 64.0f) ? float3(0.35, 0.85, 0.40)    // mid → stage2 (green)
                                : float3(0.95, 0.45, 0.30);   // big → stage3 (red)
        }
        outTex.write(float4(col, 1.0), gid);
        return;
    }

    // Pure lambertian in *view space* (camera at origin). Many GLB assets
    // ship with inconsistent triangle winding — the same surface authored
    // CW in some primitives, CCW in others — which produces opposite
    // normals from `cross(v1-v0, v2-v0)` and then half-bright / half-dark
    // patches. Flip the normal toward the camera (-Z in view space) so
    // winding direction stops mattering.
    float3 N        = normalize(cross(v1 - v0, v2 - v0));
    float3 viewDir  = -normalize(v0);    // from surface toward camera-at-origin
    if (dot(N, viewDir) < 0.0f) N = -N;
    float3 base     = float3(0.78);
    // World-space light, transformed to view space (rotation only, w=0).
    // `cam.viewMatrix * float4(L, 0).xyz` is mat3(viewMatrix) * L.
    float3 lightDir = normalize((cam.viewMatrix * float4(0.4, 0.9, -0.3, 0.0)).xyz);
    float  diff     = max(0.0f, dot(N, lightDir));
    // View-space fresnel rim: bright when the surface is edge-on to the
    // camera (N perpendicular to view Z). Cheaper and more physically
    // meaningful than the previous world-Y-axis rim.
    float  rim      = pow(1.0f - max(0.0f, abs(N.z)), 2.0f) * 0.10f;
    float3 color    = base * (0.20f + 0.80f * diff) + rim;

    // ---- Texture sampling ----
    // Outer guard is the function constant: the untextured PSO compiles
    // out the entire block, so untextured scenes pay zero cost (no per-
    // pixel branch, no buffer dereferences). Inner guard is the runtime
    // textureHandle check for mixed scenes where some DCs are untextured.
    if (USE_TEXTURING
        && dc.textureHandle != INVALID_TEXTURE_HANDLE
        && dc.textureHandle < kMaxTextures)
    {
        // Project the triangle to screen space and compute the pixel-
        // center barycentrics via signed edge functions (same convention
        // as stage1's rasterizer).
        float3 s0 = project_to_screen(v0, cam);
        float3 s1 = project_to_screen(v1, cam);
        float3 s2 = project_to_screen(v2, cam);
        float2 px = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);
        float2 e0 = float2(s2.x - s1.x, s2.y - s1.y);
        float2 e1 = float2(s0.x - s2.x, s0.y - s2.y);
        float2 e2 = float2(s1.x - s0.x, s1.y - s0.y);
        float  area2 = e2.x * (-e1.y) - e2.y * (-e1.x);
        if (fabs(area2) > 1e-6f) {
            float w0 = ((px.x - s1.x) * e0.y - (px.y - s1.y) * e0.x) / area2;
            float w1 = ((px.x - s2.x) * e1.y - (px.y - s2.y) * e1.x) / area2;
            float w2 = 1.0f - w0 - w1;

            float2 uv0 = uvs[dc.uvOffset + i0];
            float2 uv1 = uvs[dc.uvOffset + i1];
            float2 uv2 = uvs[dc.uvOffset + i2];
            // Perspective-correct UV: weight each by 1/z, sum, renormalise.
            // Without this, UVs swim across the surface as the camera moves
            // (screen-space barycentrics are affine, not perspective).
            float invZ0 = 1.0f / max(s0.z, 1e-4f);
            float invZ1 = 1.0f / max(s1.z, 1e-4f);
            float invZ2 = 1.0f / max(s2.z, 1e-4f);
            float pw0   = w0 * invZ0;
            float pw1   = w1 * invZ1;
            float pw2   = w2 * invZ2;
            float pwSum = pw0 + pw1 + pw2;
            // Guard against a zero / near-zero denominator at very oblique
            // triangles. Sign-preserving epsilon keeps the texel sample
            // stable while never producing NaN.
            pwSum = (pwSum >= 0.0f) ? max(pwSum,  1e-6f) : min(pwSum, -1e-6f);
            float2 uv = (uv0 * pw0 + uv1 * pw1 + uv2 * pw2) / pwSum;

            // ---- Explicit gradient for the sampler ----
            // We're inside a compute kernel and the texture sample is
            // gated by the visibility check + textureHandle != INVALID,
            // so neighbouring threads in a SIMD-quad may have diverged
            // (background pixels return early). MSL's implicit-gradient
            // `sample(s, uv)` is undefined under divergent control flow.
            // Reconstruct the UV at (+1, 0) and (0, +1) screen neighbours
            // via the same perspective-correct barycentric and feed the
            // sampler explicit gradients — same math main's LOD path uses.
            float2 pxR = px + float2(1.0f, 0.0f);
            float2 pxD = px + float2(0.0f, 1.0f);
            float w0R = ((pxR.x - s1.x) * e0.y - (pxR.y - s1.y) * e0.x) / area2;
            float w1R = ((pxR.x - s2.x) * e1.y - (pxR.y - s2.y) * e1.x) / area2;
            float w2R = 1.0f - w0R - w1R;
            float w0D = ((pxD.x - s1.x) * e0.y - (pxD.y - s1.y) * e0.x) / area2;
            float w1D = ((pxD.x - s2.x) * e1.y - (pxD.y - s2.y) * e1.x) / area2;
            float w2D = 1.0f - w0D - w1D;
            float pw0R = w0R * invZ0, pw1R = w1R * invZ1, pw2R = w2R * invZ2;
            float pw0D = w0D * invZ0, pw1D = w1D * invZ1, pw2D = w2D * invZ2;
            float sumR = pw0R + pw1R + pw2R;
            float sumD = pw0D + pw1D + pw2D;
            sumR = (sumR >= 0.0f) ? max(sumR,  1e-6f) : min(sumR, -1e-6f);
            sumD = (sumD >= 0.0f) ? max(sumD,  1e-6f) : min(sumD, -1e-6f);
            float2 uvR = (uv0 * pw0R + uv1 * pw1R + uv2 * pw2R) / sumR;
            float2 uvD = (uv0 * pw0D + uv1 * pw1D + uv2 * pw2D) / sumD;
            float2 dudv_dx = uvR - uv;
            float2 dudv_dy = uvD - uv;

            // Hardware sampler does mip selection from the explicit
            // gradient, plus bilinear / trilinear / anisotropic. ASTC 4×4
            // blocks are decoded by the sampler itself.
            half4 sampled = texSet.tex[dc.textureHandle].sample(
                texSampler, uv, gradient2d(dudv_dx, dudv_dy));
            color *= float3(sampled.rgb);
        }
    }
    outTex.write(float4(saturate(color), 1.0f), gid);
}

// =============================================================================
//   HI-Z OCCLUSION CULLING (Frostbite 2015 / Aaltonen 2015 / Nanite 2021)
// =============================================================================
//
// Single-phase: at end of frame N we build a Hi-Z pyramid from the visibility
// buffer's encoded depth, then test each (DC, instance) AABB and write a 1-byte
// visibility mask consumed by frame N+1's stage1.
//
// Conservative everywhere it matters:
//   - Mip-0 depth uses the same 28-bit-truncated representation as the vis
//     buffer, which rounds DOWN. A smaller stored depth → smaller occluder
//     reach → less likely to cull → never wrongly cull visible geometry.
//   - AABBs straddling the near plane are marked visible (we don't try to
//     handle clip-space straddlers; cheap and correct).
//   - Mip is picked so AABB rect spans ≤ 2 texels at that level; we sample
//     up to 4 texels (clamped) and take the max as the occluder threshold.
//
// Rejection rule: if AABB.nearZ_view (= smallest depth among 8 transformed
// corners) is GREATER than the max depth in the covered Hi-Z tile, every
// visible pixel under the AABB is closer than the AABB itself → cull.

// Reconstruct the float depth that was stored in the vis buffer's top 28 bits.
// `(udepth & 0x7FFFFFF8) >> 3` truncates the 3 LSBs (rounds toward 0 for
// non-negative floats), so this returns a value ≤ the actual closest pixel
// depth — which makes the Hi-Z conservative for occlusion.
inline float hiz_unpack_depth(ulong encoded) {
    uint d28 = uint(encoded >> 36);
    if (d28 == DEPTH_INVALID) return INFINITY;   // no fragment → "infinitely far"
    return as_type<float>(d28 << 3);
}

// Mip-0 builder. One thread per output texel. Reads the encoded vis-buffer
// fragment and writes the unpacked depth (or +INF for empty pixels) to the
// hiZ texture mip 0. We don't bother with a sampler — integer reads only.
kernel void hiZ_extractDepth(
    device const ulong       *framebuffer [[buffer(0)]],
    constant uint            &fbW         [[buffer(1)]],
    constant uint            &fbH         [[buffer(2)]],
    texture2d<float, access::write> hiZ0  [[texture(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= fbW || gid.y >= fbH) return;
    uint  pidx = gid.y * fbW + gid.x;
    float d    = hiz_unpack_depth(framebuffer[pidx]);
    hiZ0.write(float4(d, 0, 0, 0), gid);
}

// Mip-(i+1) builder. One thread per output texel; reads up to 4 texels from
// mip i (clamped), writes their max. Source/dest are explicitly bound mip
// levels of the same texture, with read/write separation handled host-side.
// Used as a fallback when only ONE mip remains to build (odd remainder).
kernel void hiZ_downsample(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint dstW = dst.get_width();
    uint dstH = dst.get_height();
    if (gid.x >= dstW || gid.y >= dstH) return;

    uint srcW = src.get_width();
    uint srcH = src.get_height();
    uint x0   = min(gid.x * 2u,        srcW - 1u);
    uint y0   = min(gid.y * 2u,        srcH - 1u);
    uint x1   = min(gid.x * 2u + 1u,   srcW - 1u);
    uint y1   = min(gid.y * 2u + 1u,   srcH - 1u);
    float a = src.read(uint2(x0, y0)).r;
    float b = src.read(uint2(x1, y0)).r;
    float c = src.read(uint2(x0, y1)).r;
    float d = src.read(uint2(x1, y1)).r;
    dst.write(float4(max(max(a, b), max(c, d)), 0, 0, 0), gid);
}

// Two-mips-per-dispatch downsampler. TG = 8×8 = 64 threads, covers a 16×16
// region of `src` and produces both mip k+1 (8×8 dst0) and mip k+2 (4×4 dst1)
// in a single dispatch. Phase 1: each thread reads 2×2 source → mip k+1 +
// stash in TG memory. Barrier. Phase 2: top-left 4×4 of threads reads 2×2
// from TG memory → mip k+2. Halves the dispatch count vs the simple 2×2
// kernel above (and removes the barrier-between-dispatches cost on TBDR).
kernel void hiZ_downsample2(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::write> dst0 [[texture(1)]],   // mip k+1
    texture2d<float, access::write> dst1 [[texture(2)]],   // mip k+2
    uint2 gid    [[thread_position_in_grid]],
    uint2 lid    [[thread_position_in_threadgroup]],
    uint2 tg_id  [[threadgroup_position_in_grid]])
{
    threadgroup float shTile[8][8];

    uint dst0W = dst0.get_width();
    uint dst0H = dst0.get_height();
    uint srcW  = src.get_width();
    uint srcH  = src.get_height();

    // ---- Phase 1: 2×2 source → mip k+1 -----------------------------------
    float v = 0.0f;
    if (gid.x < dst0W && gid.y < dst0H) {
        uint x0 = min(gid.x * 2u,      srcW - 1u);
        uint y0 = min(gid.y * 2u,      srcH - 1u);
        uint x1 = min(gid.x * 2u + 1u, srcW - 1u);
        uint y1 = min(gid.y * 2u + 1u, srcH - 1u);
        float a = src.read(uint2(x0, y0)).r;
        float b = src.read(uint2(x1, y0)).r;
        float c = src.read(uint2(x0, y1)).r;
        float d = src.read(uint2(x1, y1)).r;
        v = max(max(a, b), max(c, d));
        dst0.write(float4(v, 0, 0, 0), gid);
    }
    shTile[lid.y][lid.x] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 2: 2×2 mip k+1 (from TG) → mip k+2 -------------------------
    // Top-left 4×4 of threads, each writing a UNIQUE dst1 cell. The TG owns
    // an 8×8 region of dst0 → 4×4 region of dst1, so the 16 phase-2 threads
    // map to dst1 coords (tg_id*4 + lid). The earlier `gid.x/2` formula
    // collapsed pairs of threads onto the same dst1 cell — silent write race
    // that manifested as triangle flicker / vanish. Don't reintroduce.
    if (lid.x < 4u && lid.y < 4u) {
        uint dst1W = dst1.get_width();
        uint dst1H = dst1.get_height();
        uint2 outCoord = uint2(tg_id.x * 4u + lid.x, tg_id.y * 4u + lid.y);
        if (outCoord.x < dst1W && outCoord.y < dst1H) {
            float a = shTile[lid.y * 2u    ][lid.x * 2u    ];
            float b = shTile[lid.y * 2u    ][lid.x * 2u + 1];
            float c = shTile[lid.y * 2u + 1][lid.x * 2u    ];
            float d = shTile[lid.y * 2u + 1][lid.x * 2u + 1];
            dst1.write(float4(max(max(a, b), max(c, d)), 0, 0, 0), outCoord);
        }
    }
}

// Per-(DC, instance) AABB cull. One thread per worldView slot. Each Apple-
// Silicon SIMD group is 32 lanes — we dispatch with TG=32 so each TG covers
// 32 contiguous slots which all share one packed-bit word. Lane 0 commits
// the word via simd_ballot (one device write per 32 instances, no atomics).
// `phaseMode`:
//   0 (single-phase / final two-phase): write the new mask absolutely.
//   1 (two-phase phase-1 → phase-2 hand-off): write (newly-visible AND NOT
//      already-rendered-this-frame). prevBits is the mask phase 1 just
//      consumed; we want only the disocclusion delta for phase 2.
kernel void cull_instances_aabb(
    device const DrawCall   *drawCalls           [[buffer(0)]],
    constant uint           &numDrawCalls        [[buffer(1)]],
    device const float4x4   *worldViewArr        [[buffer(2)]],
    constant uint           &numInstances        [[buffer(3)]],
    device const uint       *instanceToDrawCall  [[buffer(4)]],
    constant CameraUniforms &cam                 [[buffer(5)]],
    device uint             *visibleBits         [[buffer(6)]],
    constant uint           &hiZMipCount         [[buffer(7)]],
    constant uint           &phaseMode           [[buffer(8)]],
    device const uint       *prevBits            [[buffer(9)]],
    texture2d<float, access::read> hiZ           [[texture(0)]],
    uint                     slot                [[thread_position_in_grid]],
    uint                     lane                [[thread_index_in_simdgroup]])
{
    bool visibleHere = false;

    if (slot < numInstances) do {
        uint dcIdx = instanceToDrawCall[slot];
        if (dcIdx >= numDrawCalls) { visibleHere = true; break; }
        DrawCall dc = drawCalls[dcIdx];

        // Object-space AABB corners.
        float3 lo = float3(dc.aabbMinX, dc.aabbMinY, dc.aabbMinZ);
        // aabbMax = aabbMin + compressionFactor * 65535 (host quantization).
        float3 hi = lo + float3(dc.compressionFactorX,
                                dc.compressionFactorY,
                                dc.compressionFactorZ) * 65535.0f;

        float4x4 wv = worldViewArr[slot];

        // Project all 8 corners; track view-space nearZ + screen-space rect.
        // If any corner crosses the near plane the AABB straddles it: bail
        // visible (handling clip-space straddlers cleanly is more code than
        // it's worth and the cull rate hit is small).
        bool straddlesNear = false;
        float3 ssMin = float3( INFINITY,  INFINITY,  INFINITY);
        float3 ssMax = float3(-INFINITY, -INFINITY, -INFINITY);
        for (uint i = 0; i < 8; ++i) {
            float3 corner = float3((i & 1u) ? hi.x : lo.x,
                                   (i & 2u) ? hi.y : lo.y,
                                   (i & 4u) ? hi.z : lo.z);
            float4 v = wv * float4(corner, 1.0);
            if (v.z <= cam.nearPlane) { straddlesNear = true; break; }
            float2 ndc = v.xy * cam.projectionXY / v.z;
            float2 uv  = ndc * 0.5f + 0.5f;
            ssMin = min(ssMin, float3(uv.x, uv.y, v.z));
            ssMax = max(ssMax, float3(uv.x, uv.y, v.z));
        }
        if (straddlesNear) { visibleHere = true; break; }

        // Frustum reject (entirely outside the [0,1] screen rect).
        if (ssMax.x < 0.0f || ssMin.x > 1.0f ||
            ssMax.y < 0.0f || ssMin.y > 1.0f) {
            visibleHere = false;
            break;
        }

        float2 r0 = clamp(ssMin.xy, float2(0.0), float2(1.0));
        float2 r1 = clamp(ssMax.xy, float2(0.0), float2(1.0));

        // Pick mip such that the rect spans ≤ 2 texels at that mip.
        uint mip0W   = hiZ.get_width(0);
        uint mip0H   = hiZ.get_height(0);
        float rectW = (r1.x - r0.x) * float(mip0W);
        float rectH = (r1.y - r0.y) * float(mip0H);
        float maxDim = max(rectW, rectH);
        int mip = (maxDim <= 1.0f) ? 0
                                   : int(ceil(log2(maxDim))) - 1;
        mip = clamp(mip, 0, int(hiZMipCount) - 1);

        // Sample up to 4 texels at the chosen mip.
        uint mipW = max(1u, mip0W >> uint(mip));
        uint mipH = max(1u, mip0H >> uint(mip));
        uint x0   = min(uint(r0.x * float(mipW)), mipW - 1u);
        uint y0   = min(uint(r0.y * float(mipH)), mipH - 1u);
        uint x1   = min(uint(r1.x * float(mipW)), mipW - 1u);
        uint y1   = min(uint(r1.y * float(mipH)), mipH - 1u);
        float occ = 0.0f;
        for (uint y = y0; y <= y1; ++y) {
            for (uint x = x0; x <= x1; ++x) {
                occ = max(occ, hiZ.read(uint2(x, y), uint(mip)).r);
            }
        }

        // aabbNearZ = ssMin.z; cull if the AABB's nearest corner is FARTHER
        // than every visible pixel in the covered tile. Tiny bias avoids
        // self-occlusion at depth ties.
        float aabbNearZ = ssMin.z;
        visibleHere = !(aabbNearZ > occ * 1.0001f);
    } while (false);

    // Pack 32 lanes' visibility into a single word, write once per SIMD
    // group. simd_ballot returns a simd_vote whose low 32 bits map to
    // lanes 0..31 — exactly the slot range owned by this TG.
    simd_vote vote = simd_ballot(visibleHere);
    uint mask = uint((simd_vote::vote_t)(vote));
    if (phaseMode == 1u) {
        // Diff mode: emit only NEWLY-disoccluded instances. prevBits[w] is
        // what phase-1 already drew → mask out those bits.
        mask &= ~prevBits[slot >> 5u];
    }
    if (lane == 0) {
        visibleBits[slot >> 5u] = mask;
    }
}

// =============================================================================
//   INSTANCE COMPACTION — Hijma §6.2.2
// =============================================================================
//
// One TG per draw call. Threads cooperatively scan that DC's instance range
// (`[firstInstance, firstInstance + instanceCount)`), test the Hi-Z bit, and
// pack the LOCAL instance indices (0 .. instanceCount-1) of survivors into
// `visibleInstanceIdx[firstInstance .. firstInstance + visibleCount)`. The
// per-DC count is written to `visibleInstancesPerDC[meshIdx]`.
//
// Stage 1 (with USE_COMPACTED_INSTANCES on) reads visibleCount + iterates the
// packed indices, eliminating the per-iteration Hi-Z bit test. Pays back when
// (instances/DC) × (cull rate) is large enough to outweigh the extra
// dispatch + write traffic. Useless on small scenes — the host gates this
// kernel behind a config flag.
kernel void compact_visible_instances(
    device const DrawCall*   drawCalls            [[buffer(0)]],
    constant uint&           numDrawCalls         [[buffer(1)]],
    device const uint*       visibleBits          [[buffer(2)]],
    device uint*             visibleInstanceIdx   [[buffer(3)]],
    device uint*             visibleInstancesPerDC[[buffer(4)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]])
{
    if (tg_id >= numDrawCalls) return;
    DrawCall dc = drawCalls[tg_id];

    // Per-TG visible-count (atomic so multiple lanes can append in parallel).
    threadgroup atomic_uint sh_count;
    if (tid_in_tg == 0) {
        atomic_store_explicit(&sh_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kTG = 64u;   // matches host dispatch
    for (uint i = tid_in_tg; i < dc.instanceCount; i += kTG) {
        uint sl = dc.firstInstance + i;
        bool visible = ((visibleBits[sl >> 5u] >> (sl & 31u)) & 1u) != 0u;
        if (visible) {
            uint slot = atomic_fetch_add_explicit(&sh_count, 1u,
                                                  memory_order_relaxed);
            visibleInstanceIdx[dc.firstInstance + slot] = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0) {
        visibleInstancesPerDC[tg_id] =
            atomic_load_explicit(&sh_count, memory_order_relaxed);
    }
}

// =============================================================================
//   PRESENT — fullscreen blit of our render texture to the layer drawable
// =============================================================================
//
// The interactive viewer renders the scene to an offscreen RGBA8 texture (our
// existing output) and then runs this tiny vertex+fragment pair to copy it
// onto the CAMetalLayer drawable, with bilinear scaling if sizes differ.

struct PresentVOut {
    float4 pos [[position]];
    float2 uv;
};

vertex PresentVOut present_vs(uint vid [[vertex_id]]) {
    // Fullscreen triangle (oversized — the rasterizer clips). Beats a quad.
    //   v0 = (-1, -1)   uv (0, 1)
    //   v1 = ( 3, -1)   uv (2, 1)
    //   v2 = (-1,  3)   uv (0, -1)
    float2 p = float2((vid == 1) ?  3.0 : -1.0,
                      (vid == 2) ?  3.0 : -1.0);
    PresentVOut o;
    o.pos = float4(p, 0.0, 1.0);
    // Map clip-space [-1,1] → texture uv [0,1], flipping Y so screen-Y down
    // matches our framebuffer convention.
    o.uv  = float2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return o;
}

fragment float4 present_fs(PresentVOut in [[stage_in]],
                           texture2d<float, access::sample> src [[texture(0)]],
                           sampler smp [[sampler(0)]])
{
    return src.sample(smp, in.uv);
}
