#include "Mesh.h"

#include "Camera.h"      // makeIdentity / makeRotationY (used by Zorah generator)

#include <algorithm>
#include <bit>
#include <cmath>

namespace metalrast {

// ---------- BitEdit helpers (CPU side, mirror of MSL decode) ----------------

uint32_t bitsForIndexRange(uint32_t numVertices) {
    if (numVertices <= 1) return 0;
    // ceil(log2(numVertices)). std::bit_width returns ceil(log2(n+1)) for n>0,
    // i.e. bits needed to represent n. For an index range [0, n-1] we need
    // bit_width(n-1).  Edge: n=2 → bit_width(1)=1.
    return static_cast<uint32_t>(std::bit_width(numVertices - 1));
}

void bitstreamWriteU32(std::vector<uint32_t>& words,
                       uint64_t bitOffset, uint32_t bitCount, uint32_t value)
{
    if (bitCount == 0) return;
    if (bitCount > 32) bitCount = 32;

    const uint64_t wordIdx = bitOffset / 32;
    const uint32_t shift   = static_cast<uint32_t>(bitOffset % 32);
    const uint32_t mask    = (bitCount == 32) ? 0xFFFFFFFFu
                                              : ((1u << bitCount) - 1u);

    // Grow as needed; +1 trailing word so a straddling read at the end won't OOB.
    const uint64_t neededWords = wordIdx + 2 + ((shift + bitCount > 32) ? 0 : 0);
    if (words.size() < neededWords) words.resize(static_cast<size_t>(neededWords), 0u);

    value &= mask;
    words[static_cast<size_t>(wordIdx)] &= ~(mask << shift);
    words[static_cast<size_t>(wordIdx)] |=  (value << shift);
    if (shift + bitCount > 32) {
        const uint32_t bitsInFirst = 32 - shift;
        const uint32_t hiMask      = mask >> bitsInFirst;
        words[static_cast<size_t>(wordIdx + 1)] &= ~hiMask;
        words[static_cast<size_t>(wordIdx + 1)] |=  (value >> bitsInFirst);
    }
}

Mesh makeUVSphere(int nLat, int nLon, float radius) {
    Mesh m;
    m.positions.reserve(static_cast<size_t>(nLat + 1) * (nLon + 1));
    m.indices.reserve(static_cast<size_t>(nLat) * nLon * 6);

    constexpr float PI = 3.14159265358979323846f;

    // Vertex grid: (nLat+1) rings × (nLon+1) verts (duplicate seam for clean wrap).
    for (int i = 0; i <= nLat; ++i) {
        float v     = static_cast<float>(i) / static_cast<float>(nLat);
        float theta = v * PI;                          // [0, π]   pole→pole
        float sinT  = std::sin(theta);
        float cosT  = std::cos(theta);

        for (int j = 0; j <= nLon; ++j) {
            float u    = static_cast<float>(j) / static_cast<float>(nLon);
            float phi  = u * 2.0f * PI;                // [0, 2π]
            float sinP = std::sin(phi);
            float cosP = std::cos(phi);

            simd_float3 p = {
                radius * sinT * cosP,
                radius * cosT,
                radius * sinT * sinP,
            };
            m.positions.push_back(p);
        }
    }

    auto idx = [nLon](int i, int j) -> uint32_t {
        return static_cast<uint32_t>(i * (nLon + 1) + j);
    };

    // Wind triangles so cross(p1-p0, p2-p0) points outward (away from origin).
    // For an outward-pointing normal at the +Z face, with i increasing southward
    // and j increasing eastward, the order (a, c, b) gives outward winding.
    for (int i = 0; i < nLat; ++i) {
        for (int j = 0; j < nLon; ++j) {
            uint32_t a = idx(i,     j);
            uint32_t b = idx(i,     j + 1);
            uint32_t c = idx(i + 1, j);
            uint32_t d = idx(i + 1, j + 1);

            // tri 1: a, c, b  (outward)
            m.indices.push_back(a);
            m.indices.push_back(c);
            m.indices.push_back(b);
            // tri 2: b, c, d  (outward)
            m.indices.push_back(b);
            m.indices.push_back(c);
            m.indices.push_back(d);
        }
    }
    return m;
}

Mesh makeGroundPlane(float halfSize, float yLevel, int subdiv) {
    if (subdiv < 1) subdiv = 1;

    Mesh m;
    m.positions.reserve(static_cast<size_t>(subdiv + 1) * (subdiv + 1));
    m.indices.reserve(static_cast<size_t>(subdiv) * subdiv * 6);

    float step = (2.0f * halfSize) / static_cast<float>(subdiv);

    for (int j = 0; j <= subdiv; ++j) {
        float z = -halfSize + step * static_cast<float>(j);
        for (int i = 0; i <= subdiv; ++i) {
            float x = -halfSize + step * static_cast<float>(i);
            m.positions.push_back(simd_float3{x, yLevel, z});
        }
    }

    auto idx = [subdiv](int i, int j) -> uint32_t {
        return static_cast<uint32_t>(j * (subdiv + 1) + i);
    };

    // Wind so the normal points +Y (up). With +X = east, +Z = north, an
    // outward (+Y) normal needs CW winding when viewed from above:
    //   cross(b-a, c-a) = +Y  for  a=(0,0,0), b=(1,0,0), c=(0,0,-1)
    for (int j = 0; j < subdiv; ++j) {
        for (int i = 0; i < subdiv; ++i) {
            uint32_t a = idx(i,     j);
            uint32_t b = idx(i + 1, j);
            uint32_t c = idx(i,     j + 1);
            uint32_t d = idx(i + 1, j + 1);
            // tri 1: a, c, b
            m.indices.push_back(a);
            m.indices.push_back(c);
            m.indices.push_back(b);
            // tri 2: b, c, d
            m.indices.push_back(b);
            m.indices.push_back(c);
            m.indices.push_back(d);
        }
    }
    return m;
}

uint32_t addInstancedMeshToScene(Scene& scene, const Mesh& mesh,
                                 const std::vector<simd_float4x4>& models)
{
    if (models.empty() || mesh.positions.empty()) return UINT32_MAX;

    // 1. Compute mesh-space AABB. We always need this for camera framing,
    //    even in uncompressed mode (the kernel's view-space cull uses it
    //    indirectly through the DrawCall fields for the bbox cache).
    simd_float3 lo = mesh.positions[0], hi = mesh.positions[0];
    for (auto const& p : mesh.positions) {
        lo = simd_make_float3(std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z));
        hi = simd_make_float3(std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z));
    }

    // 2. Compression factor (only meaningful when scene.compressed).
    simd_float3 size = (simd_float3){ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    simd_float3 factor = (simd_float3){
        size.x > 0.0f ? size.x / 65535.0f : 0.0f,
        size.y > 0.0f ? size.y / 65535.0f : 0.0f,
        size.z > 0.0f ? size.z / 65535.0f : 0.0f,
    };
    simd_float3 inv = (simd_float3){
        size.x > 0.0f ? 65535.0f / size.x : 0.0f,
        size.y > 0.0f ? 65535.0f / size.y : 0.0f,
        size.z > 0.0f ? 65535.0f / size.z : 0.0f,
    };

    // Vertex offset: indexes into whichever positions buffer is active.
    size_t vertCount = scene.compressed ? scene.positions.size()
                                        : scene.positionsFloat.size();

    DrawCall dc{};
    dc.vertexOffset            = static_cast<uint32_t>(vertCount);
    dc.triangleCount           = mesh.triangleCount();
    dc.cumulativeTriangleStart = scene.totalInstancedTriangles();   // strict prefix sum (uint64)
    dc.instanceCount           = static_cast<uint32_t>(models.size());
    dc.firstInstance           = static_cast<uint32_t>(scene.modelMatrices.size());
    dc.indexBufferIdx          = 0;   // synthetic scenes: single buffer is enough
    dc.indexMin                = 0;   // generators reference [0, numVerts-1]
    // Synthetic scenes are untextured; resolve falls back to pure lambert
    // when textureHandle is INVALID. uvOffset stays 0 (harmless — never
    // read for untextured DCs).
    dc.textureHandle           = INVALID_TEXTURE_HANDLE;
    dc.uvOffset                = 0;
    dc.aabbMinX                = lo.x;
    dc.aabbMinY                = lo.y;
    dc.aabbMinZ                = lo.z;
    dc.compressionFactorX      = factor.x;
    dc.compressionFactorY      = factor.y;
    dc.compressionFactorZ      = factor.z;

    if (scene.indicesPacked) {
        // Mesh-local indices (0..numVerts-1) → bits/index = ceil(log2(numVerts)).
        const uint32_t numVerts = static_cast<uint32_t>(mesh.positions.size());
        const uint32_t bits     = bitsForIndexRange(numVerts);
        const uint64_t bitStart = uint64_t(scene.indicesBitstream.size()) * 32ull;
        dc.bitsPerIndex = bits;
        dc.indexOffset  = bitStart;        // bit offset (when bitsPerIndex>0)
        if (bits > 0) {
            const size_t baseWord = scene.indicesBitstream.size();
            const uint64_t totalBits = uint64_t(mesh.indices.size()) * uint64_t(bits);
            // Pre-grow + 1 trailing slack word so unaligned end-of-stream reads
            // never OOB the chunk.
            const uint64_t totalWords = (totalBits + 31) / 32 + 1;
            scene.indicesBitstream.resize(baseWord + size_t(totalWords), 0u);
            for (size_t i = 0; i < mesh.indices.size(); ++i) {
                bitstreamWriteU32(scene.indicesBitstream,
                                  bitStart + uint64_t(i) * uint64_t(bits),
                                  bits, mesh.indices[i]);
            }
        } else {
            // Degenerate (≤1 vertex): no indices to pack, but keep the slot.
        }
    } else {
        dc.bitsPerIndex = 0;
        dc.indexOffset  = static_cast<uint64_t>(scene.indices.size());   // u32 element offset
    }

    uint32_t id = static_cast<uint32_t>(scene.drawCalls.size());

    if (scene.compressed) {
        // Quantize and push 8-byte ushort4 positions.
        scene.positions.reserve(scene.positions.size() + mesh.positions.size());
        for (auto const& p : mesh.positions) {
            PackedVertex pv;
            float qx = std::round((p.x - lo.x) * inv.x);
            float qy = std::round((p.y - lo.y) * inv.y);
            float qz = std::round((p.z - lo.z) * inv.z);
            pv.x = static_cast<uint16_t>(std::clamp(qx, 0.0f, 65535.0f));
            pv.y = static_cast<uint16_t>(std::clamp(qy, 0.0f, 65535.0f));
            pv.z = static_cast<uint16_t>(std::clamp(qz, 0.0f, 65535.0f));
            pv._pad = 0;
            scene.positions.push_back(pv);
        }
    } else {
        // Push raw float3 positions (16 B each in MSL/simd_float3 layout).
        scene.positionsFloat.insert(scene.positionsFloat.end(),
                                    mesh.positions.begin(), mesh.positions.end());
    }
    if (!scene.indicesPacked) {
        scene.indices.insert(scene.indices.end(),
                             mesh.indices.begin(), mesh.indices.end());
    }
    scene.modelMatrices.insert(scene.modelMatrices.end(),
                               models.begin(), models.end());
    scene.drawCalls.push_back(dc);
    return id;
}

uint32_t addMeshToScene(Scene& scene, const Mesh& mesh, simd_float4x4 model) {
    return addInstancedMeshToScene(scene, mesh, {model});
}

// ---------- Zorah-style scene generator -------------------------------------

namespace {
// Cheap LCG so the scene is deterministic across runs.
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed * 6364136223846793005ull + 1) {}
    uint32_t next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(state >> 32); }
    float    nextF() { return float(next()) / float(0xFFFFFFFFu); }
    float    range(float lo, float hi) { return lo + (hi - lo) * nextF(); }
};

}  // namespace

simd_float4x4 makeRotationXYZ(float ax, float ay, float az) {
    float cx = std::cos(ax), sx = std::sin(ax);
    float cy = std::cos(ay), sy = std::sin(ay);
    float cz = std::cos(az), sz = std::sin(az);
    // Z * Y * X
    simd_float4x4 m = makeIdentity();
    m.columns[0] = (simd_float4){ cy*cz,             cy*sz,            -sy,    0 };
    m.columns[1] = (simd_float4){ sx*sy*cz - cx*sz,  sx*sy*sz + cx*cz, sx*cy,  0 };
    m.columns[2] = (simd_float4){ cx*sy*cz + sx*sz,  cx*sy*sz - sx*cz, cx*cy,  0 };
    return m;
}

simd_float4x4 transformTRS(simd_float3 t, simd_float3 rot, float scale) {
    simd_float4x4 r = makeRotationXYZ(rot.x, rot.y, rot.z);
    r.columns[0] *= scale;
    r.columns[1] *= scale;
    r.columns[2] *= scale;
    r.columns[3] = (simd_float4){ t.x, t.y, t.z, 1.0f };
    return r;
}

Scene makeZorahLikeScene(uint32_t targetInstances) {
    Scene scene;

    // A few base meshes with varying tessellation. Each ~ a few thousand tris.
    // (Zorah uses ~1M-tri base meshes; we keep ours small so the host-side
    //  scene build stays fast — total post-instance tri count is what matters.)
    std::vector<Mesh> bases;
    bases.push_back(makeUVSphere(20, 30, 1.0f));   // ~1.2k tris
    bases.push_back(makeUVSphere(30, 50, 1.0f));   // ~3k tris
    bases.push_back(makeUVSphere(40, 60, 0.7f));   // ~4.8k tris
    bases.push_back(makeUVSphere(15, 25, 1.2f));   // ~750 tris

    // Distribute instances across base meshes roughly evenly.
    uint32_t perBase = (targetInstances + bases.size() - 1) / bases.size();

    LCG rng(0x9E3779B97F4A7C15ull);

    // Lay out instances in a 3D grid sized to fit `targetInstances`.
    // Cube-rootish to keep the world cube-like; slight Y-bias so it spreads
    // across the camera framing.
    uint32_t gridSide = 1;
    while (gridSide * gridSide * gridSide < targetInstances) ++gridSide;
    float spacing = 2.5f;
    float halfExtent = (gridSide - 1) * 0.5f * spacing;

    uint32_t emitted = 0;
    for (size_t b = 0; b < bases.size() && emitted < targetInstances; ++b) {
        std::vector<simd_float4x4> transforms;
        transforms.reserve(perBase);

        uint32_t toEmit = std::min(perBase, targetInstances - emitted);
        for (uint32_t i = 0; i < toEmit; ++i) {
            uint32_t idx = emitted + i;
            uint32_t gx = idx % gridSide;
            uint32_t gy = (idx / gridSide) % gridSide;
            uint32_t gz = idx / (gridSide * gridSide);

            simd_float3 pos = {
                gx * spacing - halfExtent + rng.range(-0.4f, 0.4f),
                gy * spacing - halfExtent + rng.range(-0.4f, 0.4f),
                gz * spacing - halfExtent + rng.range(-0.4f, 0.4f),
            };
            simd_float3 rot = {
                rng.range(0.0f, 6.2832f),
                rng.range(0.0f, 6.2832f),
                rng.range(0.0f, 6.2832f),
            };
            float scl = rng.range(0.4f, 0.9f);
            transforms.push_back(transformTRS(pos, rot, scl));
        }

        addInstancedMeshToScene(scene, bases[b], transforms);
        emitted += toEmit;
    }
    return scene;
}

// ---------- Scene replication (cheap stress test) ---------------------------

Scene replicateScene(Scene&& src, int copies) {
    if (copies <= 1) return std::move(src);

    // Compute world-space AABB of the source scene to pick a sensible grid spacing.
    simd_float3 lo = {  1e30f,  1e30f,  1e30f };
    simd_float3 hi = { -1e30f, -1e30f, -1e30f };
    for (auto const& dc : src.drawCalls) {
        simd_float3 mLo = (simd_float3){ dc.aabbMinX, dc.aabbMinY, dc.aabbMinZ };
        simd_float3 mHi = (simd_float3){
            dc.aabbMinX + dc.compressionFactorX * 65535.0f,
            dc.aabbMinY + dc.compressionFactorY * 65535.0f,
            dc.aabbMinZ + dc.compressionFactorZ * 65535.0f,
        };
        for (uint32_t inst = 0; inst < dc.instanceCount; ++inst) {
            simd_float4x4 m = src.modelMatrices[dc.firstInstance + inst];
            simd_float3 corners[8] = {
                {mLo.x, mLo.y, mLo.z}, {mHi.x, mLo.y, mLo.z},
                {mLo.x, mHi.y, mLo.z}, {mHi.x, mHi.y, mLo.z},
                {mLo.x, mLo.y, mHi.z}, {mHi.x, mLo.y, mHi.z},
                {mLo.x, mHi.y, mHi.z}, {mHi.x, mHi.y, mHi.z},
            };
            for (auto const& c : corners) {
                simd_float4 w = simd_mul(m, (simd_float4){c.x, c.y, c.z, 1.0f});
                lo = simd_make_float3(std::min(lo.x, w.x), std::min(lo.y, w.y), std::min(lo.z, w.z));
                hi = simd_make_float3(std::max(hi.x, w.x), std::max(hi.y, w.y), std::max(hi.z, w.z));
            }
        }
    }
    simd_float3 size = (simd_float3){ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    float maxSide = std::max({size.x, size.y, size.z});
    float spacing = maxSide * 1.15f;   // ~15% gap between copies

    // Lay out as a cube of side ceil(N^(1/3)).
    int side = 1;
    while (side * side * side < copies) ++side;

    Scene out;
    out.compressed       = src.compressed;
    out.indicesPacked    = src.indicesPacked;
    // Move host vectors and Metal buffers across (the per-copy iteration
    // doesn't mutate them; all replicas share the same vertex/index data on
    // the GPU via per-copy DrawCalls referencing the same buffer).
    out.positions        = std::move(src.positions);
    out.positionsFloat   = std::move(src.positionsFloat);
    out.indices          = std::move(src.indices);
    out.indicesBitstream = std::move(src.indicesBitstream);
    out.metalVertices    = src.metalVertices;     src.metalVertices    = nullptr;
    out.metalIndices     = src.metalIndices;      src.metalIndices     = nullptr;
    out.metalVertCount   = src.metalVertCount;
    out.metalIndexBytes  = src.metalIndexBytes;
    // UVs + texture set are shared across all replicas (each replica's DCs
    // reference the same vertex / UV data on the GPU). Without these moves
    // a `--repeat N` of a textured scene would render with valid
    // textureHandles + uvOffsets pointing into an empty UV buffer,
    // OOB-reading on the GPU.
    out.uvs              = std::move(src.uvs);
    out.textures         = std::move(src.textures);
    out.drawCalls.reserve(src.drawCalls.size() * size_t(copies));
    out.modelMatrices.reserve(src.modelMatrices.size() * size_t(copies));

    uint64_t cumTri = 0;
    int emitted = 0;
    for (int gz = 0; gz < side && emitted < copies; ++gz)
    for (int gy = 0; gy < side && emitted < copies; ++gy)
    for (int gx = 0; gx < side && emitted < copies; ++gx, ++emitted) {
        if (emitted >= copies) break;
        simd_float3 offset = (simd_float3){
            (gx - (side - 1) * 0.5f) * spacing,
            (gy - (side - 1) * 0.5f) * spacing,
            (gz - (side - 1) * 0.5f) * spacing,
        };

        for (auto const& origDC : src.drawCalls) {
            DrawCall dc = origDC;
            dc.cumulativeTriangleStart = cumTri;
            dc.firstInstance = static_cast<uint32_t>(out.modelMatrices.size());

            for (uint32_t inst = 0; inst < origDC.instanceCount; ++inst) {
                simd_float4x4 m = src.modelMatrices[origDC.firstInstance + inst];
                m.columns[3].x += offset.x;
                m.columns[3].y += offset.y;
                m.columns[3].z += offset.z;
                out.modelMatrices.push_back(m);
            }

            out.drawCalls.push_back(dc);
            cumTri += uint64_t(dc.triangleCount) * uint64_t(dc.instanceCount);
        }
    }
    return out;
}

}  // namespace metalrast
