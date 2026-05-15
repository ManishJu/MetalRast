// glTF / GLB loader. Single-header cgltf parses the file; we walk the scene
// graph, accumulate world-space transforms by node hierarchy, and group nodes
// that share a mesh into instance lists.
//
// Memory: for huge scenes (Zorah's 9.3 GB .bin), we override cgltf's default
// file_read with an mmap-based one. The bin is paged in by the kernel as the
// loader walks accessors — peak working set stays manageable on a 32 GB Mac.

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <meshoptimizer.h>

#include "GLBLoader.h"
#include "Camera.h"      // makeIdentity / TRS helpers (only need identity here)
#include "MeshRegistry.h"
#include "ThreadPool.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace metalrast {

namespace {

// ---------- cgltf custom file_read: mmap the bin instead of loading -----
//
// cgltf's default `file_read` malloc's a buffer the size of the bin and reads
// it in. For Zorah's 9.3 GB bin that's most of our 32 GB RAM gone before we
// even start quantizing. mmap lets the kernel page bin data in/out as the
// accessor walk touches it.
//
// `file_release` is also overridden to munmap. We can't tell mmap'd vs
// malloc'd pointers apart from the void* alone, so we tag mmap'd ones in a
// thread_local set. This is fine since cgltf only loads one file's buffers at
// a time inside `cgltf_load_buffers`.

struct MMapTracking {
    std::unordered_map<void*, size_t> mmaps;   // ptr → length, for munmap
};
static MMapTracking& tracking() { static MMapTracking t; return t; }

cgltf_result mmap_file_read(const cgltf_memory_options*,
                            const cgltf_file_options*,
                            const char* path,
                            cgltf_size* outSize,
                            void** outData)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return cgltf_result_file_not_found;

    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); return cgltf_result_io_error; }
    size_t len = static_cast<size_t>(st.st_size);

    void* p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);   // mmap holds a ref; safe to close fd
    if (p == MAP_FAILED) return cgltf_result_out_of_memory;

    // Hint: we'll read sequentially-ish. Helps macOS prefetch.
    madvise(p, len, MADV_SEQUENTIAL);

    tracking().mmaps[p] = len;

    *outSize = len;
    *outData = p;
    return cgltf_result_success;
}

void mmap_file_release(const cgltf_memory_options*,
                       const cgltf_file_options*,
                       void* data,
                       cgltf_size /* size, unused */)
{
    auto& m = tracking().mmaps;
    auto it = m.find(data);
    if (it != m.end()) {
        munmap(data, it->second);
        m.erase(it);
    } else {
        std::free(data);   // cgltf default fallback
    }
}

simd_float4x4 mat_from_cgltf_node_local(const cgltf_node& n) {
    if (n.has_matrix) {
        simd_float4x4 m;
        std::memcpy(&m, n.matrix, 16 * sizeof(float));
        return m;
    }
    // Otherwise compose TRS.
    simd_float4x4 t = makeIdentity();
    if (n.has_translation)
        t.columns[3] = (simd_float4){ n.translation[0], n.translation[1],
                                      n.translation[2], 1.0f };
    simd_float4x4 r = makeIdentity();
    if (n.has_rotation) {
        // Quaternion (x, y, z, w) → rotation matrix.
        float x = n.rotation[0], y = n.rotation[1], z = n.rotation[2], w = n.rotation[3];
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, xz = x*z, yz = y*z;
        float wx = w*x, wy = w*y, wz = w*z;
        r.columns[0] = (simd_float4){ 1 - 2*(yy+zz),     2*(xy+wz),       2*(xz-wy), 0 };
        r.columns[1] = (simd_float4){     2*(xy-wz), 1 - 2*(xx+zz),       2*(yz+wx), 0 };
        r.columns[2] = (simd_float4){     2*(xz+wy),     2*(yz-wx),   1 - 2*(xx+yy), 0 };
    }
    simd_float4x4 s = makeIdentity();
    if (n.has_scale) {
        s.columns[0].x = n.scale[0];
        s.columns[1].y = n.scale[1];
        s.columns[2].z = n.scale[2];
    }
    return simd_mul(t, simd_mul(r, s));
}

// Recursively walk the node tree, accumulating world-space transforms.
void walk_node(const cgltf_node* node, simd_float4x4 parentMat,
               std::unordered_map<const cgltf_mesh*, std::vector<simd_float4x4>>& out,
               const cgltf_mesh* meshBase)
{
    simd_float4x4 local = mat_from_cgltf_node_local(*node);
    simd_float4x4 world = simd_mul(parentMat, local);

    // EXT_mesh_gpu_instancing: each node may carry inline instance TRS arrays.
    bool hasGpuInstancing = false;
    {
        for (cgltf_size i = 0; i < node->extensions_count; ++i) {
            // cgltf surfaces extension JSON as raw strings — we'd have to parse it
            // ourselves. For now, log a warning and fall back to single-instance
            // node behavior. (Test scenes that need this extension can re-export
            // with one node per instance instead.)
            const cgltf_extension& ext = node->extensions[i];
            if (ext.name && std::strcmp(ext.name, "EXT_mesh_gpu_instancing") == 0) {
                std::fprintf(stderr,
                    "GLB warning: node uses EXT_mesh_gpu_instancing — "
                    "this loader does not parse the inline instance array yet; "
                    "will treat as one instance.\n");
                hasGpuInstancing = true;
            }
        }
    }
    (void)hasGpuInstancing;  // silenced — single instance for now

    if (node->mesh) {
        // Key by the cgltf_mesh pointer; nodes that point at the same mesh
        // become instances of one DrawCall.
        out[node->mesh].push_back(world);
    }

    for (cgltf_size i = 0; i < node->children_count; ++i) {
        walk_node(node->children[i], world, out, meshBase);
    }
}

// Find the POSITION attribute accessor for a primitive, or null.
const cgltf_accessor* find_position_accessor(const cgltf_primitive& prim) {
    for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
        if (prim.attributes[ai].type == cgltf_attribute_type_position)
            return prim.attributes[ai].data;
    }
    return nullptr;
}

// Find TEXCOORD_0 attribute accessor for a primitive, or null. We only
// support a single UV channel for now — most glTF assets only carry
// TEXCOORD_0 anyway, and the resolve kernel reads a flat float2 per vertex.
const cgltf_accessor* find_texcoord0_accessor(const cgltf_primitive& prim) {
    for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
        if (prim.attributes[ai].type == cgltf_attribute_type_texcoord
            && prim.attributes[ai].index == 0)
            return prim.attributes[ai].data;
    }
    return nullptr;
}

// Resolve material → pbr_metallic_roughness → base_color_texture → texture →
// image. Returns null at any missing link (untextured material, alpha-only
// material, KHR_materials_pbrSpecularGlossiness not yet handled).
const cgltf_image* base_color_image(const cgltf_primitive& prim) {
    const cgltf_material* mat = prim.material;
    if (!mat) return nullptr;
    if (!mat->has_pbr_metallic_roughness) return nullptr;
    const cgltf_texture_view& tv = mat->pbr_metallic_roughness.base_color_texture;
    if (!tv.texture || !tv.texture->image) return nullptr;
    return tv.texture->image;
}

// Index of `img` inside data->images[], or 0xFFFFFFFFu if not in the array.
uint32_t image_index_in(const cgltf_data* data, const cgltf_image* img) {
    if (!data || !img) return 0xFFFFFFFFu;
    auto idx = cgltf_size(img - data->images);
    if (idx >= data->images_count) return 0xFFFFFFFFu;
    return uint32_t(idx);
}

// Copy TEXCOORD_0 into a flat float2 destination. glTF spec allows float,
// u8 (normalized), u16 (normalized) — fast-path the float case (common),
// fall back to cgltf's read_float for the rest.
void copy_uvs_block(const cgltf_accessor* uvAcc, simd_float2* dst) {
    const uint8_t* base = nullptr;
    size_t stride = 0;
    {
        const cgltf_buffer_view* bv = uvAcc ? uvAcc->buffer_view : nullptr;
        if (bv && bv->data) {
            base = reinterpret_cast<const uint8_t*>(bv->data) + uvAcc->offset;
        } else if (bv && bv->buffer && bv->buffer->data) {
            base = reinterpret_cast<const uint8_t*>(bv->buffer->data)
                 + bv->offset + uvAcc->offset;
        }
        stride = (bv && bv->stride) ? bv->stride : uvAcc->stride;
        if (stride == 0) stride = uvAcc->stride;
    }
    const size_t n = uvAcc ? uvAcc->count : 0;

    if (uvAcc && uvAcc->component_type == cgltf_component_type_r_32f
        && uvAcc->type == cgltf_type_vec2 && base) {
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* p = base + i * stride;
            float u, v;
            std::memcpy(&u, p + 0, 4);
            std::memcpy(&v, p + 4, 4);
            dst[i] = (simd_float2){ u, v };
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            float v[2] = { 0, 0 };
            cgltf_accessor_read_float(uvAcc, i, v, 2);
            dst[i] = (simd_float2){ v[0], v[1] };
        }
    }
}

// ---------- Direct typed buffer access ---------------------------------------
// cgltf_accessor_read_float / _index branch on component type per call. For
// large meshes (komainu's 30M positions, Zorah's billions) that overhead is
// the dominant CPU cost. The helpers below resolve component type once per
// accessor and then loop with a typed inner read (memcpy for unaligned-safety).

// Raw byte pointer to element 0 of an accessor inside its (possibly
// meshopt-decompressed) buffer view. Returns nullptr when there is no
// usable storage — caller falls back to cgltf_accessor_read_*.
//
// When `bv->data` is set (extension-decoded data, e.g. EXT_meshopt_compression),
// the decompressed buffer's first byte corresponds to bv->offset = 0 in the
// view's own coordinate system. So the per-element offset is just `a->offset`.
// When `bv->data` is null, the underlying `bv->buffer->data` is used and the
// real offset is `bv->offset + a->offset`.
inline const uint8_t* accessor_base_ptr(const cgltf_accessor* a) {
    if (!a) return nullptr;
    const cgltf_buffer_view* bv = a->buffer_view;
    if (!bv) return nullptr;
    if (bv->data) {
        return reinterpret_cast<const uint8_t*>(bv->data) + a->offset;
    }
    if (!bv->buffer || !bv->buffer->data) return nullptr;
    return reinterpret_cast<const uint8_t*>(bv->buffer->data)
         + bv->offset + a->offset;
}

// Effective byte stride for an accessor (buffer_view's stride if set, else the
// accessor's packed stride filled in by cgltf during parse).
inline size_t accessor_stride_bytes(const cgltf_accessor* a) {
    const cgltf_buffer_view* bv = a->buffer_view;
    if (bv->stride) return bv->stride;
    return a->stride;
}

template <typename T>
inline T read_unaligned(const void* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// Quantize one primitive's positions into a pre-reserved range of the
// destination buffer. AABB lo + inv (= 65535 / size) are the per-MESH AABB
// (shared across the mesh's primitives so all use the same compression
// factors).
void quantize_positions_block(const cgltf_accessor* posAcc,
                              simd_float3 lo, simd_float3 inv,
                              PackedVertex* dst /* dst[0..count-1] */)
{
    const uint8_t* base   = accessor_base_ptr(posAcc);
    const size_t   stride = accessor_stride_bytes(posAcc);
    const size_t   n      = posAcc->count;
    // Positions are float VEC3 by spec — komainu/Zorah and ~all modern .glbs
    // use this. Defensive: keep cgltf_accessor_read_float as a fallback when
    // not float-VEC3-3.
    if (posAcc->component_type == cgltf_component_type_r_32f &&
        posAcc->type == cgltf_type_vec3)
    {
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* p = base + i * stride;
            float x = read_unaligned<float>(p + 0);
            float y = read_unaligned<float>(p + 4);
            float z = read_unaligned<float>(p + 8);
            PackedVertex pv;
            pv.x = uint16_t(std::clamp(std::round((x - lo.x) * inv.x), 0.0f, 65535.0f));
            pv.y = uint16_t(std::clamp(std::round((y - lo.y) * inv.y), 0.0f, 65535.0f));
            pv.z = uint16_t(std::clamp(std::round((z - lo.z) * inv.z), 0.0f, 65535.0f));
            pv._pad = 0;
            dst[i] = pv;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            float v[3] = {0,0,0};
            cgltf_accessor_read_float(posAcc, i, v, 3);
            PackedVertex pv;
            pv.x = uint16_t(std::clamp(std::round((v[0] - lo.x) * inv.x), 0.0f, 65535.0f));
            pv.y = uint16_t(std::clamp(std::round((v[1] - lo.y) * inv.y), 0.0f, 65535.0f));
            pv.z = uint16_t(std::clamp(std::round((v[2] - lo.z) * inv.z), 0.0f, 65535.0f));
            pv._pad = 0;
            dst[i] = pv;
        }
    }
}

void copy_positions_float_block(const cgltf_accessor* posAcc,
                                simd_float3* dst /* dst[0..count-1] */)
{
    const uint8_t* base   = accessor_base_ptr(posAcc);
    const size_t   stride = accessor_stride_bytes(posAcc);
    const size_t   n      = posAcc->count;
    if (posAcc->component_type == cgltf_component_type_r_32f &&
        posAcc->type == cgltf_type_vec3)
    {
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* p = base + i * stride;
            float x = read_unaligned<float>(p + 0);
            float y = read_unaligned<float>(p + 4);
            float z = read_unaligned<float>(p + 8);
            dst[i] = simd_make_float3(x, y, z);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            float v[3] = {0,0,0};
            cgltf_accessor_read_float(posAcc, i, v, 3);
            dst[i] = simd_make_float3(v[0], v[1], v[2]);
        }
    }
}

// Copy raw indices to a pre-reserved uint32 slot, biasing each by primVertBase.
// Specializes the inner loop on component type.
void copy_indices_raw_block(const cgltf_accessor* idxAcc,
                            uint32_t primVertBase,
                            uint32_t* dst /* dst[0..count-1] */)
{
    const uint8_t* base   = accessor_base_ptr(idxAcc);
    const size_t   stride = accessor_stride_bytes(idxAcc);
    const size_t   n      = idxAcc->count;
    switch (idxAcc->component_type) {
        case cgltf_component_type_r_8u:
            for (size_t i = 0; i < n; ++i)
                dst[i] = primVertBase + uint32_t(base[i * stride]);
            break;
        case cgltf_component_type_r_16u:
            for (size_t i = 0; i < n; ++i)
                dst[i] = primVertBase + uint32_t(read_unaligned<uint16_t>(base + i * stride));
            break;
        case cgltf_component_type_r_32u:
            if (stride == sizeof(uint32_t) && primVertBase == 0) {
                std::memcpy(dst, base, n * sizeof(uint32_t));
            } else {
                for (size_t i = 0; i < n; ++i)
                    dst[i] = primVertBase + read_unaligned<uint32_t>(base + i * stride);
            }
            break;
        default:
            // Fall back to cgltf for esoteric component types (signed, etc).
            for (size_t i = 0; i < n; ++i)
                dst[i] = primVertBase + uint32_t(cgltf_accessor_read_index(idxAcc, i));
            break;
    }
}

// Pack indices into a pre-reserved bit range of the destination word stream.
// `dstWords` MUST be pre-zeroed for this mesh's slice (we rely on |= writes)
// and have +1 trailing slack uint32 so a straddling write at the end doesn't
// OOB. CuRast's BitEdit::writeU32 inlined into a per-component-type loop.
void pack_indices_block(const cgltf_accessor* idxAcc,
                        uint32_t primVertBase, uint32_t bitsPerIndex,
                        uint32_t* dstWords, uint64_t startBit)
{
    if (bitsPerIndex == 0) return;
    const uint8_t* base   = accessor_base_ptr(idxAcc);
    const size_t   stride = accessor_stride_bytes(idxAcc);
    const size_t   n      = idxAcc->count;
    const uint32_t mask   = (bitsPerIndex == 32) ? 0xFFFFFFFFu
                                                  : ((1u << bitsPerIndex) - 1u);
    const auto write_value = [&](size_t i, uint32_t value) {
        uint64_t bitPos  = startBit + uint64_t(i) * uint64_t(bitsPerIndex);
        uint64_t wordIdx = bitPos / 32;
        uint32_t shift   = uint32_t(bitPos & 31u);
        uint32_t v       = (value + primVertBase) & mask;
        // Pre-zeroed slice → |= is enough; no AND-clear needed.
        dstWords[wordIdx] |= (v << shift);
        if (shift + bitsPerIndex > 32u) {
            uint32_t bitsInFirst = 32u - shift;
            dstWords[wordIdx + 1] |= (v >> bitsInFirst);
        }
    };

    switch (idxAcc->component_type) {
        case cgltf_component_type_r_8u:
            for (size_t i = 0; i < n; ++i)
                write_value(i, uint32_t(base[i * stride]));
            break;
        case cgltf_component_type_r_16u:
            for (size_t i = 0; i < n; ++i)
                write_value(i, uint32_t(read_unaligned<uint16_t>(base + i * stride)));
            break;
        case cgltf_component_type_r_32u:
            for (size_t i = 0; i < n; ++i)
                write_value(i, read_unaligned<uint32_t>(base + i * stride));
            break;
        default:
            for (size_t i = 0; i < n; ++i)
                write_value(i, uint32_t(cgltf_accessor_read_index(idxAcc, i)));
            break;
    }
}

}  // namespace

Scene loadGLB(const std::string& path, uint64_t maxIndices, bool compressed,
              bool bitPackIndices, MTL::Device* directDevice,
              const TextureLoader::EncodeOpts& texOpts) {
    cgltf_options opts{};
    opts.file.read    = mmap_file_read;
    opts.file.release = mmap_file_release;
    cgltf_data*   data = nullptr;
    cgltf_result  res  = cgltf_parse_file(&opts, path.c_str(), &data);
    if (res != cgltf_result_success) {
        throw std::runtime_error("cgltf_parse_file failed for " + path);
    }
    res = cgltf_load_buffers(&opts, data, path.c_str());
    if (res != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("cgltf_load_buffers failed for " + path);
    }

    // EXT_meshopt_compression decompression happens lazily — only for the
    // bufferViews actually referenced by primitives we decide to keep (post-
    // maxIndices cap). At Zorah scale the decompressed total is ~31 GB, so
    // we MUST avoid decompressing everything just to throw most of it away.
    // The helper below decompresses one bufferView in place; we call it from
    // both the plan pass (when accessor.has_min/max isn't set) and after the
    // cap, before we read indices in pass 2a.
    // Decompressed bytes go into malloc'd buffers handed to cgltf via bv->data.
    // cgltf_free will call its own free() on every bv->data (whether it
    // allocated it or not), so we MUST use the same allocator as cgltf's
    // default — i.e. plain `malloc`, not `std::vector`. Otherwise cgltf_free
    // hands a non-malloc pointer to free() → UB / SIGABRT.
    std::vector<std::mutex> bvMutexes(data->buffer_views_count);
    auto ensureBufferViewDecoded = [&](cgltf_size bvIdx) -> bool {
        if (bvIdx >= data->buffer_views_count) return false;
        cgltf_buffer_view& bv = data->buffer_views[bvIdx];
        if (!bv.has_meshopt_compression) return true;       // already raw
        if (bv.data) return true;                            // already decoded
        std::lock_guard<std::mutex> lk(bvMutexes[bvIdx]);
        if (bv.data) return true;                            // raced; done
        const cgltf_meshopt_compression& mc = bv.meshopt_compression;
        if (!mc.buffer || !mc.buffer->data) return false;
        uint8_t* dst = static_cast<uint8_t*>(std::malloc(bv.size));
        if (!dst) return false;
        std::memset(dst, 0, bv.size);
        const uint8_t* src = (const uint8_t*)mc.buffer->data + mc.offset;
        int rc = -1;
        switch (mc.mode) {
            case cgltf_meshopt_compression_mode_attributes:
                rc = meshopt_decodeVertexBuffer(dst, mc.count, mc.stride, src, mc.size); break;
            case cgltf_meshopt_compression_mode_triangles:
                rc = meshopt_decodeIndexBuffer(dst, mc.count, mc.stride, src, mc.size);  break;
            case cgltf_meshopt_compression_mode_indices:
                rc = meshopt_decodeIndexSequence(dst, mc.count, mc.stride, src, mc.size);break;
            default: break;
        }
        if (rc != 0) {
            std::free(dst);
            return false;
        }
        switch (mc.filter) {
            case cgltf_meshopt_compression_filter_octahedral:
                meshopt_decodeFilterOct(dst, mc.count, mc.stride);   break;
            case cgltf_meshopt_compression_filter_quaternion:
                meshopt_decodeFilterQuat(dst, mc.count, mc.stride);  break;
            case cgltf_meshopt_compression_filter_exponential:
                meshopt_decodeFilterExp(dst, mc.count, mc.stride);   break;
            case cgltf_meshopt_compression_filter_color:
                meshopt_decodeFilterColor(dst, mc.count, mc.stride); break;
            case cgltf_meshopt_compression_filter_none:
            default: break;
        }
        bv.data = dst;
        return true;
    };
    // Helper: walk an accessor → its bufferView idx so we can decompress on demand.
    auto bvIndexOf = [&](const cgltf_accessor* a) -> cgltf_size {
        if (!a || !a->buffer_view) return data->buffer_views_count;
        return cgltf_size(a->buffer_view - data->buffer_views);
    };
    auto ensureAccessorDecoded = [&](const cgltf_accessor* a) -> bool {
        cgltf_size i = bvIndexOf(a);
        if (i >= data->buffer_views_count) return false;
        return ensureBufferViewDecoded(i);
    };

    // Walk every scene's roots and collect per-mesh instance world matrices.
    std::unordered_map<const cgltf_mesh*, std::vector<simd_float4x4>> instances;
    if (data->scenes_count == 0 || data->nodes_count == 0) {
        cgltf_free(data);
        throw std::runtime_error("GLB has no scene/nodes: " + path);
    }
    const cgltf_mesh* meshBase = data->meshes;
    simd_float4x4 root = makeIdentity();
    for (cgltf_size si = 0; si < data->scenes_count; ++si) {
        const cgltf_scene& sc = data->scenes[si];
        for (cgltf_size i = 0; i < sc.nodes_count; ++i) {
            walk_node(sc.nodes[i], root, instances, meshBase);
        }
    }
    if (instances.empty()) {
        cgltf_free(data);
        throw std::runtime_error("GLB has no mesh-bearing nodes: " + path);
    }

    using clock = std::chrono::steady_clock;
    auto t_plan_start = clock::now();

    // ---------- PASS 1 — per-PRIMITIVE plans (CuRast convention) ------------
    // One DrawCall = one glTF primitive (matches CuRast's per-accessor task
    // model). Multi-primitive glTF meshes produce multiple DrawCalls that
    // share their parent mesh's instance transforms. Per-primitive packing
    // gives strictly tighter `bitsPerIndex` than per-mesh would.
    struct PrimPlan {
        const cgltf_mesh*                  glmesh;       // for stable sort
        cgltf_size                         primIdx;      // for stable sort
        const cgltf_primitive*             prim;
        std::vector<simd_float4x4>         transforms;
        simd_float3                        lo, hi;
        simd_float3                        compressionFactor;
        simd_float3                        inv;          // 65535 / size

        uint32_t                           numVerts      = 0;
        uint64_t                           numIndices    = 0;
        uint32_t                           triangleCount = 0;

        // Filled by the index-scan pass (only when bitPackIndices). For raw
        // mode we leave indexMin = 0 and bitsPerIndex = 0.
        uint32_t                           indexMin     = 0;
        uint32_t                           indexMax     = 0;
        uint32_t                           bitsPerIndex = 0;

        // Texturing.
        // textureHandle = imageIdx into data->images[]; INVALID = untextured.
        // uvAcc resolved here (Pass 1), copied in Pass 3 alongside positions.
        uint32_t                           textureHandle = INVALID_TEXTURE_HANDLE;
        const cgltf_accessor*              uvAcc         = nullptr;

        // Output slots (filled in pass 2 after scan).
        uint32_t                           vertexOffset    = 0;
        uint32_t                           uvOffset        = 0;   // float2 elements into scene.uvs
        uint64_t                           indexOffsetU32  = 0;   // raw mode
        uint64_t                           indexOffsetBits = 0;   // packed mode
        uint32_t                           firstInstance   = 0;   // shared across siblings of same mesh
    };

    std::vector<PrimPlan> plans;
    plans.reserve(instances.size() * 2);   // most meshes are 1-prim; oversize a bit
    for (auto const& [glmesh, transforms] : instances) {
        for (cgltf_size pi = 0; pi < glmesh->primitives_count; ++pi) {
            const cgltf_primitive& prim = glmesh->primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* posAcc = find_position_accessor(prim);
            if (!posAcc || !prim.indices) continue;

            PrimPlan p;
            p.glmesh     = glmesh;
            p.primIdx    = pi;
            p.prim       = &prim;
            p.transforms = transforms;     // shared with siblings of same mesh
            // Phase 1 texturing — base-color image index (one MTLTexture per
            // glTF image), and TEXCOORD_0 accessor for the UV copy in Pass 3.
            p.uvAcc         = find_texcoord0_accessor(prim);
            // Without TEXCOORD_0 the UV slot stays zero-init; sampling at
            // origin would produce a constant colour over the surface
            // instead of the lambert grey we'd otherwise show. Treat the
            // primitive as untextured.
            p.textureHandle = (p.uvAcc != nullptr)
                ? image_index_in(data, base_color_image(prim))
                : INVALID_TEXTURE_HANDLE;
            p.lo = (simd_float3){  1e30f,  1e30f,  1e30f };
            p.hi = (simd_float3){ -1e30f, -1e30f, -1e30f };
            // Per-primitive AABB (not per-mesh): each glTF primitive carries
            // its own POSITION accessor with min/max stored. This matches
            // CuRast (whose accessors are the unit of compression).
            if (posAcc->has_min && posAcc->has_max) {
                p.lo = (simd_float3){ posAcc->min[0], posAcc->min[1], posAcc->min[2] };
                p.hi = (simd_float3){ posAcc->max[0], posAcc->max[1], posAcc->max[2] };
            } else {
                for (cgltf_size i = 0; i < posAcc->count; ++i) {
                    float v[3] = {0,0,0};
                    cgltf_accessor_read_float(posAcc, i, v, 3);
                    p.lo = simd_make_float3(std::min(p.lo.x, v[0]), std::min(p.lo.y, v[1]), std::min(p.lo.z, v[2]));
                    p.hi = simd_make_float3(std::max(p.hi.x, v[0]), std::max(p.hi.y, v[1]), std::max(p.hi.z, v[2]));
                }
            }
            p.numVerts      = static_cast<uint32_t>(posAcc->count);
            p.numIndices    = prim.indices->count;
            p.triangleCount = static_cast<uint32_t>(prim.indices->count / 3);
            simd_float3 size = (simd_float3){ p.hi.x - p.lo.x, p.hi.y - p.lo.y, p.hi.z - p.lo.z };
            p.compressionFactor = (simd_float3){
                size.x > 0 ? size.x / 65535.0f : 0.0f,
                size.y > 0 ? size.y / 65535.0f : 0.0f,
                size.z > 0 ? size.z / 65535.0f : 0.0f };
            p.inv = (simd_float3){
                size.x > 0 ? 65535.0f / size.x : 0.0f,
                size.y > 0 ? 65535.0f / size.y : 0.0f,
                size.z > 0 ? 65535.0f / size.z : 0.0f };
            plans.push_back(std::move(p));
        }
    }
    // Stable order: parent mesh pointer, then primitive index within mesh.
    std::sort(plans.begin(), plans.end(),
              [&](const PrimPlan& a, const PrimPlan& b){
                  if (a.glmesh != b.glmesh) return uintptr_t(a.glmesh) < uintptr_t(b.glmesh);
                  return a.primIdx < b.primIdx;
              });

    uint64_t totalVerts = 0, totalIdx = 0, totalInstances = 0;
    for (auto const& p : plans) {
        totalVerts += p.numVerts;
        totalIdx   += p.numIndices;
    }
    for (auto const& [glmesh, transforms] : instances) totalInstances += transforms.size();
    std::printf("GLB: %zu unique meshes, %zu primitives, %llu unique vertices, "
                "%llu indices (%llu tris), %llu instances.\n",
                instances.size(),
                plans.size(),
                (unsigned long long)totalVerts,
                (unsigned long long)totalIdx,
                (unsigned long long)(totalIdx / 3),
                (unsigned long long)totalInstances);
    std::fflush(stdout);

    // ---------- PASS 1.5 — apply maxIndices cap (single-threaded) -----------
    // Walk plans in stable order; drop later primitives once cap is hit.
    std::vector<PrimPlan> kept;
    kept.reserve(plans.size());
    uint64_t totalIndicesEmitted = 0;
    uint64_t skippedMeshes  = 0;       // (named for legacy log; counts primitives)
    uint64_t skippedIndices = 0;
    for (auto& p : plans) {
        if (maxIndices != 0 && totalIndicesEmitted >= maxIndices) {
            ++skippedMeshes;
            skippedIndices += p.numIndices;
            continue;
        }
        totalIndicesEmitted += p.numIndices;
        kept.push_back(std::move(p));
    }

    // ---------- PASS 1.7 — refcount bufferViews for streaming decompression
    // We decompress meshopt-compressed bvs LAZILY during the parallel fill
    // (pass 3) — i.e. each fill task ensures its own bv is decoded, and as
    // soon as the last primitive using a bv finishes, we free its
    // decompressed buffer. This keeps peak host memory bounded by
    // (in-flight tasks × per-bv decompressed size) rather than the sum over
    // all bvs (~31 GB on Zorah).
    std::vector<std::atomic<int>> bvUsers(data->buffer_views_count);
    for (cgltf_size i = 0; i < data->buffer_views_count; ++i) bvUsers[i].store(0);
    for (auto const& p : kept) {
        const cgltf_accessor* posAcc = find_position_accessor(*p.prim);
        if (posAcc) ++bvUsers[bvIndexOf(posAcc)];
        if (p.prim->indices) ++bvUsers[bvIndexOf(p.prim->indices)];   // fill
        // Index-min/max scan also consults the index buffer once.
        if (bitPackIndices && p.prim->indices)
            ++bvUsers[bvIndexOf(p.prim->indices)];
        // Phase 1 — UV bv has one fill-time user count.
        if (p.uvAcc) ++bvUsers[bvIndexOf(p.uvAcc)];
    }
    auto releaseBufferView = [&](cgltf_size bvIdx) {
        if (bvIdx >= data->buffer_views_count) return;
        if (--bvUsers[bvIdx] == 0) {
            cgltf_buffer_view& bv = data->buffer_views[bvIdx];
            if (bv.data && bv.has_meshopt_compression) {
                std::free(bv.data);
                bv.data = nullptr;
            }
        }
    };

    // ---------- PASS 2a — parallel index_min/max scan (packed mode only) ----
    // CuRast (paper §4.6) packs each primitive's indices against the actual
    // referenced range [indexMin, indexMax]. We compute that range here,
    // then pass 2b uses bitsPerIndex = ceil(log2(indexMax - indexMin + 1)).
    auto t_scan_start = clock::now();
    if (bitPackIndices) {
        size_t hw = std::max<size_t>(1u, std::thread::hardware_concurrency());
        size_t numThreads = std::min<size_t>(16, std::max<size_t>(2, hw));
        if (kept.size() <= 1) numThreads = 1;
        ThreadPool pool(numThreads);

        for (auto& p : kept) {
            pool.enqueue([&, pPtr = &p](int){
                auto& plan = *pPtr;
                const cgltf_accessor* idxAcc = plan.prim->indices;
                // Decompress the index bv on demand. Released at end of scan.
                cgltf_size idxBv = bvIndexOf(idxAcc);
                ensureBufferViewDecoded(idxBv);
                if (idxAcc->has_min && idxAcc->has_max) {
                    plan.indexMin = static_cast<uint32_t>(idxAcc->min[0]);
                    plan.indexMax = static_cast<uint32_t>(idxAcc->max[0]);
                } else {
                    const uint8_t* base   = accessor_base_ptr(idxAcc);
                    const size_t   stride = accessor_stride_bytes(idxAcc);
                    const size_t   n      = idxAcc->count;
                    uint32_t mn = 0xFFFFFFFFu, mx = 0;
                    if (!base) {
                        for (size_t i = 0; i < n; ++i) {
                            uint32_t v = static_cast<uint32_t>(cgltf_accessor_read_index(idxAcc, i));
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                        }
                    } else {
                        switch (idxAcc->component_type) {
                            case cgltf_component_type_r_8u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = base[i * stride];
                                    if (v < mn) mn = v;
                                    if (v > mx) mx = v;
                                }
                                break;
                            case cgltf_component_type_r_16u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = read_unaligned<uint16_t>(base + i * stride);
                                    if (v < mn) mn = v;
                                    if (v > mx) mx = v;
                                }
                                break;
                            case cgltf_component_type_r_32u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = read_unaligned<uint32_t>(base + i * stride);
                                    if (v < mn) mn = v;
                                    if (v > mx) mx = v;
                                }
                                break;
                            default:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = static_cast<uint32_t>(cgltf_accessor_read_index(idxAcc, i));
                                    if (v < mn) mn = v;
                                    if (v > mx) mx = v;
                                }
                                break;
                        }
                    }
                    if (n == 0) { mn = 0; mx = 0; }
                    plan.indexMin = mn;
                    plan.indexMax = mx;
                }
                uint64_t range = uint64_t(plan.indexMax) - uint64_t(plan.indexMin) + 1;
                plan.bitsPerIndex = (range > 1)
                    ? static_cast<uint32_t>(std::ceil(std::log2(double(range))))
                    : 0u;
                // Decrement scan's user count for this idxBv. If fill is
                // already done, this also frees the decoded buffer.
                releaseBufferView(idxBv);
            });
        }
        pool.wait();
    }
    auto t_scan_end = clock::now();
    double scanMs = std::chrono::duration<double, std::milli>(t_scan_end - t_scan_start).count();


    // ---------- PASS 2b — reserve output ranges + build DrawCalls -----------
    Scene scene;
    scene.compressed    = compressed;
    scene.indicesPacked = bitPackIndices;
    scene.drawCalls.reserve(kept.size());
    scene.modelMatrices.reserve(totalInstances);

    // Per-parent-mesh first-instance lookup so siblings share the instance
    // range (CuRast also has multiple primitives reference the same node
    // transform set).
    std::unordered_map<const cgltf_mesh*, uint32_t> meshFirstInstance;
    meshFirstInstance.reserve(instances.size());

    uint64_t globalTriCounter = 0;
    uint32_t nextVertOffset   = 0;
    uint64_t nextIndexU32     = 0;
    uint64_t nextIndexBitOff  = 0;

    for (auto& p : kept) {
        // Reserve position slot (one slot per primitive — CuRast convention).
        p.vertexOffset = nextVertOffset;
        // UVs are 1:1 with positions — same slot offset, into scene.uvs[].
        // Untextured prims don't have a uvAcc; their slot stays zero-init
        // and the resolve kernel never reads it (textureHandle == INVALID).
        p.uvOffset     = nextVertOffset;
        nextVertOffset += p.numVerts;

        // Reserve index slot.
        if (bitPackIndices) {
            p.indexOffsetBits = nextIndexBitOff;
            nextIndexBitOff   += p.numIndices * p.bitsPerIndex;
            // Round up so the next primitive's bit range starts on a uint32
            // word boundary. Guarantees parallel writers never share a word.
            nextIndexBitOff = (nextIndexBitOff + 31ull) & ~uint64_t(31);
        } else {
            p.indexOffsetU32 = nextIndexU32;
            nextIndexU32     += p.numIndices;
        }

        // Share `firstInstance` across siblings of the same parent mesh.
        auto it = meshFirstInstance.find(p.glmesh);
        if (it == meshFirstInstance.end()) {
            uint32_t fi = static_cast<uint32_t>(scene.modelMatrices.size());
            scene.modelMatrices.insert(scene.modelMatrices.end(),
                                       p.transforms.begin(), p.transforms.end());
            meshFirstInstance.emplace(p.glmesh, fi);
            p.firstInstance = fi;
        } else {
            p.firstInstance = it->second;
        }

        DrawCall dc{};
        dc.vertexOffset            = p.vertexOffset;
        dc.indexOffset             = bitPackIndices ? p.indexOffsetBits
                                                    : p.indexOffsetU32;
        dc.bitsPerIndex            = p.bitsPerIndex;
        dc.indexMin                = p.indexMin;
        dc.triangleCount           = p.triangleCount;
        dc.cumulativeTriangleStart = globalTriCounter;
        dc.instanceCount           = static_cast<uint32_t>(p.transforms.size());
        dc.firstInstance           = p.firstInstance;
        dc.indexBufferIdx          = 0;   // populated by renderer's chunk-split
        dc.aabbMinX = p.lo.x; dc.aabbMinY = p.lo.y; dc.aabbMinZ = p.lo.z;
        dc.compressionFactorX = p.compressionFactor.x;
        dc.compressionFactorY = p.compressionFactor.y;
        dc.compressionFactorZ = p.compressionFactor.z;
        // textureHandle == INVALID is treated as untextured by resolve.
        // uvOffset is the float2 element offset into scene.uvs[]; safe to
        // set even for untextured DCs since the kernel never reads it
        // for those.
        dc.textureHandle = p.textureHandle;
        dc.uvOffset      = p.uvOffset;
        scene.drawCalls.push_back(dc);
        globalTriCounter += uint64_t(p.triangleCount) * p.transforms.size();
    }

    // ---- Allocate destination storage --------------------------------------
    // Two paths:
    //   directDevice == null   → host std::vector storage (small scenes,
    //                            multi-chunk index split fall-back).
    //   directDevice != null   → Metal-allocated buffers; fill writes
    //                            straight into buffer->contents() with no
    //                            host-vector duplicate. Single index chunk
    //                            only — total must fit in Metal's per-buffer
    //                            cap (~18.7 GB on M2 Max).
    void*    vertDstRaw = nullptr;        // typed below
    uint32_t* idxDst    = nullptr;        // raw or packed both write to uint32 words
    {
        size_t vbStride = compressed ? sizeof(PackedVertex) : sizeof(simd_float3);
        size_t vbBytes  = size_t(nextVertOffset) * vbStride;
        size_t idxWords = bitPackIndices
            ? size_t((nextIndexBitOff + 31) / 32) + 1   // +1 slack
            : size_t(nextIndexU32);
        size_t idxBytes = idxWords * sizeof(uint32_t);

        if (directDevice) {
            // Per-buffer cap on M2 Max is ~18.7 GiB. If we'd exceed, fall
            // through to host path (caller should not have set directDevice).
            const size_t cap = directDevice->maxBufferLength();
            if (vbBytes > cap || idxBytes > cap) {
                throw std::runtime_error(
                    "loadGLB direct mode: scene exceeds Metal per-buffer cap "
                    "(call without directDevice to use multi-chunk fallback)");
            }
            scene.metalVertices = directDevice->newBuffer(vbBytes,
                                                          MTL::ResourceStorageModeShared);
            if (!scene.metalVertices) {
                throw std::runtime_error("Metal vertex buffer alloc failed");
            }
            scene.metalIndices  = directDevice->newBuffer(idxBytes,
                                                          MTL::ResourceStorageModeShared);
            if (!scene.metalIndices) {
                scene.metalVertices->release();
                scene.metalVertices = nullptr;
                throw std::runtime_error("Metal index buffer alloc failed");
            }
            scene.metalVertCount   = nextVertOffset;
            scene.metalIndexBytes  = idxBytes;

            vertDstRaw = scene.metalVertices->contents();
            idxDst     = static_cast<uint32_t*>(scene.metalIndices->contents());

            // Packed path uses |= writes; need zeroed destination for the
            // bits that aren't explicitly written. (Whole-element writes in
            // raw / position paths overwrite, so no zero needed there — but
            // it's cheap and safer.)
            if (bitPackIndices) std::memset(idxDst, 0, idxBytes);
        } else {
            if (compressed) {
                scene.positions.resize(nextVertOffset);
                vertDstRaw = scene.positions.data();
            } else {
                scene.positionsFloat.resize(nextVertOffset);
                vertDstRaw = scene.positionsFloat.data();
            }
            if (bitPackIndices) {
                scene.indicesBitstream.assign(idxWords, 0u);
                idxDst = scene.indicesBitstream.data();
            } else {
                scene.indices.resize(nextIndexU32);
                idxDst = scene.indices.data();
            }
        }
        // Phase 1 UVs — one slot per vertex, parallel to positions.
        // Untextured prims have a zero-init slice; resolve never reads it
        // (textureHandle == INVALID gates the texture path).
        bool sceneHasUVs = false;
        for (auto const& p : kept) if (p.uvAcc) { sceneHasUVs = true; break; }
        if (sceneHasUVs) {
            scene.uvs.assign(nextVertOffset, (simd_float2){ 0.0f, 0.0f });
        }
    }

    auto t_plan_end = clock::now();
    double planMs = std::chrono::duration<double, std::milli>(t_plan_end - t_plan_start).count();

    // ---------- PASS 3 — parallel per-primitive fill -----------------------
    // Each task writes into its primitive's pre-reserved position / index
    // slots. Since slots are disjoint and packed-index slots end at uint32
    // word boundaries, no two tasks share memory.
    auto t_fill_start = clock::now();
    {
        size_t hw = std::max<size_t>(1u, std::thread::hardware_concurrency());
        size_t numThreads = std::min<size_t>(16, std::max<size_t>(2, hw));
        if (kept.size() <= 1) numThreads = 1;
        ThreadPool pool(numThreads);

        for (auto const& p : kept) {
            pool.enqueue([&, &plan = p](int /*tid*/){
                const cgltf_primitive* prim = plan.prim;
                const cgltf_accessor*  posAcc = find_position_accessor(*prim);
                const cgltf_accessor*  idxAcc = prim->indices;
                cgltf_size posBv = bvIndexOf(posAcc);
                cgltf_size idxBv = bvIndexOf(idxAcc);

                // Decompress on demand. Each fill task owns one user count
                // for its position bv and one for its index bv.
                ensureBufferViewDecoded(posBv);
                ensureBufferViewDecoded(idxBv);

                // ---- Positions: write into pre-resolved destination
                // (either scene.positions[].data() or
                //  scene.metalVertices->contents()).
                if (compressed) {
                    PackedVertex* dst = static_cast<PackedVertex*>(vertDstRaw)
                                      + plan.vertexOffset;
                    quantize_positions_block(posAcc, plan.lo, plan.inv, dst);
                } else {
                    simd_float3* dst = static_cast<simd_float3*>(vertDstRaw)
                                     + plan.vertexOffset;
                    copy_positions_float_block(posAcc, dst);
                }

                // ---- UVs — only when this prim has
                // a TEXCOORD_0 accessor. ensureBufferViewDecoded handles
                // meshopt-compressed UV bvs; copy_uvs_block fast-paths
                // float2 LE storage and falls back to cgltf_accessor_read_float
                // for normalized u8/u16.
                if (plan.uvAcc) {
                    cgltf_size uvBv = bvIndexOf(plan.uvAcc);
                    ensureBufferViewDecoded(uvBv);
                    copy_uvs_block(plan.uvAcc,
                                   scene.uvs.data() + plan.uvOffset);
                    releaseBufferView(uvBv);
                }

                // ---- Indices
                if (bitPackIndices) {
                    uint32_t bias = uint32_t(0u) - plan.indexMin;
                    pack_indices_block(idxAcc, bias, plan.bitsPerIndex,
                                       idxDst, plan.indexOffsetBits);
                } else {
                    uint32_t* dst = idxDst + plan.indexOffsetU32;
                    copy_indices_raw_block(idxAcc, /*primVertBase*/ 0, dst);
                }

                // Release fill's user count. Frees the decompressed buffer
                // once both scan + fill have completed for it.
                releaseBufferView(posBv);
                releaseBufferView(idxBv);
            });
        }
        pool.wait();
    }
    auto t_fill_end = clock::now();
    double fillMs = std::chrono::duration<double, std::milli>(t_fill_end - t_fill_start).count();
    std::printf("GLB load: plan %.1f ms (idx-scan %.1f ms), parallel fill %.1f ms "
                "(%zu DCs / %zu primitives)\n",
                planMs, scanMs, fillMs, scene.drawCalls.size(), plans.size());

    // Decode every glTF image and encode it as ASTC LDR (4×4 or 6×6)
    // packaged in our MRTC container. The Metal renderer then parses MRTC
    // and uploads each mip's raw blocks straight into a texture — no
    // transcoder step. Run before cgltf_free; the encoder reads bytes
    // through cgltf's buffer_view accessors.
    if (data->images_count > 0) {
        auto t_tex_start = clock::now();
        scene.textures = TextureLoader::encodeAll(data, texOpts);
        size_t okCount = 0, n4x4 = 0, n6x6 = 0, blobBytes = 0;
        for (auto const& t : scene.textures) {
            if (!t.mrtcBytes.empty()) {
                ++okCount;
                blobBytes += t.mrtcBytes.size();
                if (t.block == AstcBlock::B4x4) ++n4x4; else ++n6x6;
            }
        }
        double texMs = std::chrono::duration<double, std::milli>(
                           clock::now() - t_tex_start).count();
        std::printf("GLB textures: %zu/%zu encoded mrtc "
                    "(%zu @ 4×4, %zu @ 6×6, %.2f MiB), %.0f ms\n",
                    okCount, scene.textures.size(),
                    n4x4, n6x6,
                    blobBytes / (1024.0 * 1024.0),
                    texMs);
    }

    cgltf_free(data);

    if (skippedMeshes > 0) {
        const size_t loadedIndexBytes = bitPackIndices
            ? scene.indicesBitstream.size() * 4
            : scene.indices.size() * 4;
        std::printf("Memory cap: skipped %llu meshes / %llu indices "
                    "(loaded %zu draw calls / %.2f GiB indices).\n",
                    (unsigned long long)skippedMeshes,
                    (unsigned long long)skippedIndices,
                    scene.drawCalls.size(),
                    loadedIndexBytes / 1073741824.0);
    }

    if (scene.drawCalls.empty()) {
        throw std::runtime_error("GLB had no triangle primitives: " + path);
    }
    return scene;
}

// ===========================================================================
//   loadGLBToRegistry — CuRast-style streaming load
// ===========================================================================
//
// Parses the .glb / .gltf, walks each primitive, registers a MeshRecord with
// the ResidencyManager. Stores source pointers (cgltf_primitive*) so the
// manager's worker thread can decompress on demand later.
//
// Computes per-mesh AABB, bitsPerIndex, and indexMin up-front (parallel scan
// of index accessors) — these are needed for camera framing + DrawCall
// emission and are cheap to compute relative to the actual decompression.
//
// Hands cgltfData to the manager via adoptCgltfData; the manager free()s it
// at destruction. The mmap'd .bin therefore stays alive for the scene's
// entire lifetime, ready for on-demand reads.
void loadGLBToRegistry(const std::string& path,
                       ResidencyManager& mgr,
                       uint64_t maxIndices)
{
    cgltf_options opts{};
    opts.file.read    = mmap_file_read;
    opts.file.release = mmap_file_release;

    cgltf_data*   data = nullptr;
    cgltf_result  res  = cgltf_parse_file(&opts, path.c_str(), &data);
    if (res != cgltf_result_success)
        throw std::runtime_error("cgltf_parse_file failed for " + path);
    res = cgltf_load_buffers(&opts, data, path.c_str());
    if (res != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("cgltf_load_buffers failed for " + path);
    }

    if (data->scenes_count == 0 || data->nodes_count == 0) {
        cgltf_free(data);
        throw std::runtime_error("GLB has no scene/nodes: " + path);
    }

    std::unordered_map<const cgltf_mesh*, std::vector<simd_float4x4>> instances;
    const cgltf_mesh* meshBase = data->meshes;
    simd_float4x4 root = makeIdentity();
    for (cgltf_size si = 0; si < data->scenes_count; ++si) {
        const cgltf_scene& sc = data->scenes[si];
        for (cgltf_size i = 0; i < sc.nodes_count; ++i)
            walk_node(sc.nodes[i], root, instances, meshBase);
    }
    if (instances.empty()) {
        cgltf_free(data);
        throw std::runtime_error("GLB has no mesh-bearing nodes: " + path);
    }

    // ---- Pass 1: build per-primitive plans (no buffer access) ------------
    struct PrimPlan {
        const cgltf_mesh*      glmesh;
        cgltf_size             primIdx;
        const cgltf_primitive* prim;
        std::vector<simd_float4x4> transforms;
        simd_float3            lo, hi;
        simd_float3            compressionFactor;
        simd_float3            inv;
        uint32_t               numVerts      = 0;
        uint32_t               numIndices    = 0;
        uint32_t               triangleCount = 0;
        uint32_t               indexMin      = 0;
        uint32_t               indexMax      = 0;
        uint32_t               bitsPerIndex  = 0;
    };
    std::vector<PrimPlan> plans;
    plans.reserve(instances.size());
    for (auto const& [glmesh, transforms] : instances) {
        for (cgltf_size pi = 0; pi < glmesh->primitives_count; ++pi) {
            const cgltf_primitive& prim = glmesh->primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* posAcc = find_position_accessor(prim);
            if (!posAcc || !prim.indices) continue;

            PrimPlan p;
            p.glmesh     = glmesh;
            p.primIdx    = pi;
            p.prim       = &prim;
            p.transforms = transforms;
            p.lo = (simd_float3){  1e30f,  1e30f,  1e30f };
            p.hi = (simd_float3){ -1e30f, -1e30f, -1e30f };
            if (posAcc->has_min && posAcc->has_max) {
                p.lo = (simd_float3){ posAcc->min[0], posAcc->min[1], posAcc->min[2] };
                p.hi = (simd_float3){ posAcc->max[0], posAcc->max[1], posAcc->max[2] };
            } else {
                // No min/max: leave as full range; the manager will refine
                // after first decode if needed. (For Zorah every accessor
                // has min/max set so this branch is rarely taken.)
                p.lo = (simd_float3){ -1, -1, -1 };
                p.hi = (simd_float3){  1,  1,  1 };
            }
            p.numVerts      = static_cast<uint32_t>(posAcc->count);
            p.numIndices    = static_cast<uint32_t>(prim.indices->count);
            p.triangleCount = static_cast<uint32_t>(prim.indices->count / 3);
            simd_float3 size = (simd_float3){ p.hi.x - p.lo.x,
                                              p.hi.y - p.lo.y,
                                              p.hi.z - p.lo.z };
            p.compressionFactor = (simd_float3){
                size.x > 0 ? size.x / 65535.0f : 0.0f,
                size.y > 0 ? size.y / 65535.0f : 0.0f,
                size.z > 0 ? size.z / 65535.0f : 0.0f };
            p.inv = (simd_float3){
                size.x > 0 ? 65535.0f / size.x : 0.0f,
                size.y > 0 ? 65535.0f / size.y : 0.0f,
                size.z > 0 ? 65535.0f / size.z : 0.0f };
            plans.push_back(std::move(p));
        }
    }
    std::sort(plans.begin(), plans.end(),
              [&](const PrimPlan& a, const PrimPlan& b){
                  if (a.glmesh != b.glmesh) return uintptr_t(a.glmesh) < uintptr_t(b.glmesh);
                  return a.primIdx < b.primIdx;
              });

    // Apply maxIndices cap (deterministic order).
    std::vector<PrimPlan> kept;
    kept.reserve(plans.size());
    uint64_t emitted = 0, skippedPrims = 0;
    for (auto& p : plans) {
        if (maxIndices != 0 && emitted >= maxIndices) { ++skippedPrims; continue; }
        emitted += p.numIndices;
        kept.push_back(std::move(p));
    }

    // ---- Pass 2: parallel index_min/max scan + bitsPerIndex --------------
    // We need bitsPerIndex BEFORE registering (the manager pre-allocates pool
    // slot bytes from numIndices × bitsPerIndex). One-time cost per primitive.
    // Decompresses meshopt source bvs on demand; releases as soon as scan
    // for that primitive completes.
    {
        std::vector<std::vector<uint8_t>> ownedDecoded(data->buffer_views_count);
        std::vector<std::mutex>           bvLocks(data->buffer_views_count);

        auto ensureDecoded = [&](cgltf_size bvIdx) -> bool {
            cgltf_buffer_view& bv = data->buffer_views[bvIdx];
            if (!bv.has_meshopt_compression) return true;
            if (bv.data) return true;
            std::lock_guard<std::mutex> lk(bvLocks[bvIdx]);
            if (bv.data) return true;
            const cgltf_meshopt_compression& mc = bv.meshopt_compression;
            if (!mc.buffer || !mc.buffer->data) return false;
            ownedDecoded[bvIdx].assign(bv.size, 0u);
            uint8_t* dst = ownedDecoded[bvIdx].data();
            const uint8_t* src = (const uint8_t*)mc.buffer->data + mc.offset;
            int rc = -1;
            switch (mc.mode) {
                case cgltf_meshopt_compression_mode_attributes:
                    rc = meshopt_decodeVertexBuffer(dst, mc.count, mc.stride, src, mc.size); break;
                case cgltf_meshopt_compression_mode_triangles:
                    rc = meshopt_decodeIndexBuffer(dst, mc.count, mc.stride, src, mc.size);  break;
                case cgltf_meshopt_compression_mode_indices:
                    rc = meshopt_decodeIndexSequence(dst, mc.count, mc.stride, src, mc.size);break;
                default: break;
            }
            if (rc != 0) { ownedDecoded[bvIdx].clear(); return false; }
            switch (mc.filter) {
                case cgltf_meshopt_compression_filter_octahedral:
                    meshopt_decodeFilterOct(dst, mc.count, mc.stride);   break;
                case cgltf_meshopt_compression_filter_quaternion:
                    meshopt_decodeFilterQuat(dst, mc.count, mc.stride);  break;
                case cgltf_meshopt_compression_filter_exponential:
                    meshopt_decodeFilterExp(dst, mc.count, mc.stride);   break;
                case cgltf_meshopt_compression_filter_color:
                    meshopt_decodeFilterColor(dst, mc.count, mc.stride); break;
                case cgltf_meshopt_compression_filter_none:
                default: break;
            }
            bv.data = dst;
            return true;
        };

        size_t hw = std::max<size_t>(1u, std::thread::hardware_concurrency());
        ThreadPool pool(std::min<size_t>(16, std::max<size_t>(2, hw)));
        for (auto& p : kept) {
            pool.enqueue([&, pPtr = &p](int){
                auto& plan = *pPtr;
                const cgltf_accessor* idxAcc = plan.prim->indices;
                if (idxAcc->has_min && idxAcc->has_max) {
                    plan.indexMin = static_cast<uint32_t>(idxAcc->min[0]);
                    plan.indexMax = static_cast<uint32_t>(idxAcc->max[0]);
                } else {
                    cgltf_size bvIdx = idxAcc->buffer_view
                        ? cgltf_size(idxAcc->buffer_view - data->buffer_views)
                        : data->buffer_views_count;
                    if (bvIdx < data->buffer_views_count) ensureDecoded(bvIdx);
                    const uint8_t* base   = accessor_base_ptr(idxAcc);
                    const size_t   stride = accessor_stride_bytes(idxAcc);
                    const size_t   n      = idxAcc->count;
                    uint32_t mn = 0xFFFFFFFFu, mx = 0;
                    if (!base) {
                        for (size_t i = 0; i < n; ++i) {
                            uint32_t v = static_cast<uint32_t>(cgltf_accessor_read_index(idxAcc, i));
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                        }
                    } else {
                        switch (idxAcc->component_type) {
                            case cgltf_component_type_r_8u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = base[i*stride];
                                    if (v < mn) mn = v; if (v > mx) mx = v;
                                } break;
                            case cgltf_component_type_r_16u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = read_unaligned<uint16_t>(base + i*stride);
                                    if (v < mn) mn = v; if (v > mx) mx = v;
                                } break;
                            case cgltf_component_type_r_32u:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = read_unaligned<uint32_t>(base + i*stride);
                                    if (v < mn) mn = v; if (v > mx) mx = v;
                                } break;
                            default:
                                for (size_t i = 0; i < n; ++i) {
                                    uint32_t v = static_cast<uint32_t>(cgltf_accessor_read_index(idxAcc, i));
                                    if (v < mn) mn = v; if (v > mx) mx = v;
                                } break;
                        }
                    }
                    if (n == 0) { mn = 0; mx = 0; }
                    plan.indexMin = mn;
                    plan.indexMax = mx;
                }
                uint64_t range = uint64_t(plan.indexMax) - uint64_t(plan.indexMin) + 1;
                plan.bitsPerIndex = (range > 1)
                    ? static_cast<uint32_t>(std::ceil(std::log2(double(range))))
                    : 0u;
            });
        }
        pool.wait();
        // The lazy ownedDecoded buffers go out of scope here; bv->data
        // pointers reference them and become dangling. We re-decompress
        // on demand later in the manager's worker (it owns its own
        // bvDecoded_ pool with malloc/free for cgltf_free compatibility).
        for (auto& bv : std::vector<cgltf_buffer_view*>{}) (void)bv; // silence unused
        for (cgltf_size i = 0; i < data->buffer_views_count; ++i) {
            cgltf_buffer_view& bv = data->buffer_views[i];
            if (bv.has_meshopt_compression && !ownedDecoded[i].empty()
                && bv.data == ownedDecoded[i].data())
            {
                bv.data = nullptr;   // drop dangling pointer; manager will redecode
            }
        }
    }

    // ---- Pass 3: hand cgltf_data to the manager + register meshes --------
    mgr.adoptCgltfData(data);
    for (auto& p : kept) {
        MeshRecord rec;
        rec.prim              = p.prim;
        rec.numVerts          = p.numVerts;
        rec.numIndices        = p.numIndices;
        rec.triangleCount     = p.triangleCount;
        rec.lo                = p.lo;
        rec.hi                = p.hi;
        rec.compressionFactor = p.compressionFactor;
        rec.inv               = p.inv;
        rec.bitsPerIndex      = p.bitsPerIndex;
        rec.indexMin          = p.indexMin;
        rec.transforms        = std::move(p.transforms);
        mgr.registerMesh(std::move(rec));
    }

    std::printf("loadGLBToRegistry: %zu primitives registered (%llu skipped by cap), "
                "source mmap'd, decompression deferred\n",
                kept.size(), (unsigned long long)skippedPrims);
    mgr.finalizeRegistration();
}

}  // namespace metalrast
