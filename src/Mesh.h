#pragma once

#include <Metal/Metal.hpp>

#include <simd/simd.h>
#include <cstdint>
#include <vector>

#include "SharedTypes.h"
#include "TextureLoader.h"

namespace metalrast {

// A single mesh — vertices in object space, triangles as flat uint32 triples.
// Caller composes Scene by concatenating multiple Mesh into one big buffer
// and emitting a DrawCall per mesh.
struct Mesh {
    std::vector<simd_float3> positions;
    std::vector<uint32_t>    indices;       // triangle list, length % 3 == 0

    uint32_t triangleCount() const { return static_cast<uint32_t>(indices.size() / 3); }
};

// Compressed vertex format: 3 × uint16 (X, Y, Z in mesh-space quantized
// coordinates) + 1 × uint16 padding for 8-byte alignment.
struct alignas(8) PackedVertex {
    uint16_t x, y, z, _pad;
};
static_assert(sizeof(PackedVertex) == 8, "PackedVertex must be 8 bytes");

// Tessellated UV sphere centered on origin. Outward-facing winding so the
// world-space backface cull (dot(N, p0 - cam) > 0 → cull) shows the front.
//
//   nLat = number of latitude *segments* (rings between poles).
//   nLon = number of longitude *segments* (slices around the equator).
//
// Total triangles ≈ 2 * nLat * nLon (the two pole rings degenerate to fans
// with nLon triangles each; rest are 2 per quad).
Mesh makeUVSphere(int nLat, int nLon, float radius);

// Generic TRS: translate × XYZ-rotate × uniform-scale. Used by Zorah-style
// instance generators in both the static (Scene) and residency paths.
simd_float4x4 transformTRS(simd_float3 t, simd_float3 rot, float scale);

// A ground plane: one large quad split into (subdiv × subdiv) cells. Sits at
// y = yLevel, spans [-halfSize, +halfSize] in x and z. Faces +Y (up).
// subdiv=1 → 2 triangles (good for stage-1/2 stress-testing the large path).
// subdiv=N → 2N² small triangles (mostly stage-1).
Mesh makeGroundPlane(float halfSize, float yLevel, int subdiv);

// Combines several meshes into a single big position+index buffer plus a
// matching DrawCall array (with cumulativeTriangleStart correctly set).
//
// Instancing: `modelMatrices[firstInstance .. firstInstance+instanceCount)`
// is the per-instance world transform for draw call `i`. Every draw call has
// at least one instance; non-instanced meshes simply have instanceCount = 1.
struct Scene {
    // When `compressed`, positions are 16-bit fixed-point per-mesh-AABB-
    // quantized in `positions` (8 B/vertex). Otherwise they're raw simd_float3
    // in `positionsFloat` (16 B/vertex). Exactly one of the two is populated
    // based on the flag — the other stays empty.
    bool                       compressed = true;
    std::vector<PackedVertex>  positions;          // when compressed
    std::vector<simd_float3>   positionsFloat;     // when uncompressed

    // When `indicesPacked`, `indicesBitstream` carries variable-bit-width
    // mesh-local indices in a contiguous uint32 word stream (paper §4.6),
    // and each DrawCall's `bitsPerIndex` and `indexOffset` (= bit offset)
    // describe how to read its slice. Otherwise `indices` carries raw
    // uint32 indices and DrawCall::bitsPerIndex == 0.
    bool                       indicesPacked = false;
    std::vector<uint32_t>      indices;            // raw uint32 path
    std::vector<uint32_t>      indicesBitstream;   // packed path (uint32 words holding bit-packed indices)

    std::vector<DrawCall>      drawCalls;
    std::vector<simd_float4x4> modelMatrices;     // length = total instance count

    // Textures + per-vertex UVs for the textured-resolve path.
    // `uvs[dc.uvOffset + localIdx]` is the float2 UV for the dc's local
    // vertex `localIdx`. Untextured DCs keep dc.textureHandle == ~0u and
    // resolve never reads `uvs`. `textures` is one EncodedTexture per
    // glTF image (ASTC LDR blocks packaged in an MRTC blob); the renderer
    // parses and uploads each mip via uploadTextures().
    std::vector<simd_float2>      uvs;
    std::vector<EncodedTexture>   textures;

    // ---- Pre-populated GPU buffers (CuRast-style streaming load) ----------
    // When the GLB loader is given an MTL::Device, it allocates Metal
    // vertex + index buffers and writes geometry DIRECTLY into their
    // contents() pointer (no host-vector intermediate). Halves peak host
    // RAM at Zorah scale. uploadScene then adopts these buffers without
    // copying.
    //
    // Ownership: the loader retains the freshly-created buffers. When a
    // renderer takes them via `take*()`, the slot becomes null; this Scene
    // no longer owns them. If a Scene is destroyed without anyone taking
    // them, the destructor releases them (no leak).
    //
    // `metalIndices` is single-chunk only; multi-chunk fallback stays on
    // the host-vector path.
    void releaseMetalBuffers() {
        if (metalVertices) { metalVertices->release(); metalVertices = nullptr; }
        if (metalIndices)  { metalIndices->release();  metalIndices  = nullptr; }
    }

    MTL::Buffer* takeMetalVertices() {
        MTL::Buffer* b = metalVertices; metalVertices = nullptr; return b;
    }
    MTL::Buffer* takeMetalIndices() {
        MTL::Buffer* b = metalIndices;  metalIndices  = nullptr; return b;
    }

    MTL::Buffer*               metalVertices    = nullptr;
    MTL::Buffer*               metalIndices     = nullptr;
    uint64_t                   metalVertCount   = 0;
    uint64_t                   metalIndexBytes  = 0;

    // Triangles in the visibility buffer's global ID space — i.e. counting
    // each instance's triangles separately. Returns uint64 because Zorah-
    // scale scenes (~13B post-instancing tris) overflow uint32.
    uint64_t totalInstancedTriangles() const {
        uint64_t n = 0;
        for (auto const& d : drawCalls)
            n += uint64_t(d.triangleCount) * uint64_t(d.instanceCount);
        return n;
    }

    uint64_t totalTriangles() const { return totalInstancedTriangles(); }

    uint32_t totalInstances() const {
        return static_cast<uint32_t>(modelMatrices.size());
    }

    // Move-only because of the owned MTL::Buffer*. Copy of the scene would
    // duplicate the pointer and double-release on destruction.
    Scene() = default;
    Scene(const Scene&)            = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&& o) noexcept { *this = std::move(o); }
    Scene& operator=(Scene&& o) noexcept {
        if (this != &o) {
            releaseMetalBuffers();
            compressed       = o.compressed;
            positions        = std::move(o.positions);
            positionsFloat   = std::move(o.positionsFloat);
            indicesPacked    = o.indicesPacked;
            indices          = std::move(o.indices);
            indicesBitstream = std::move(o.indicesBitstream);
            drawCalls        = std::move(o.drawCalls);
            modelMatrices    = std::move(o.modelMatrices);
            uvs              = std::move(o.uvs);
            textures         = std::move(o.textures);
            metalVertices    = o.metalVertices;    o.metalVertices    = nullptr;
            metalIndices     = o.metalIndices;     o.metalIndices     = nullptr;
            metalVertCount   = o.metalVertCount;   o.metalVertCount   = 0;
            metalIndexBytes  = o.metalIndexBytes;  o.metalIndexBytes  = 0;
        }
        return *this;
    }
    ~Scene() { releaseMetalBuffers(); }
};

// Bit-pack helpers (host side, mirror of CuRast's BitEdit + minimum-bits calc).
// For a mesh with `numVertices` vertices, the smallest count of bits per index
// that can address every vertex is ceil(log2(numVertices)). Returns 0 for
// numVertices ≤ 1 (degenerate; callers fall back to raw 32-bit).
uint32_t bitsForIndexRange(uint32_t numVertices);

// Append `value` to a uint32 word stream at bit position `bitOffset` using
// `bitCount` bits (1..32). Resizes `words` if needed. value's high bits
// outside `bitCount` are masked off.
void bitstreamWriteU32(std::vector<uint32_t>& words,
                       uint64_t bitOffset, uint32_t bitCount, uint32_t value);

// Add a single non-instanced mesh. Returns the draw-call ID.
uint32_t addMeshToScene(Scene& scene, const Mesh& mesh, simd_float4x4 model);

// Add a mesh that will be drawn `models.size()` times, each with a different
// world transform. Returns the draw-call ID. Big win on Zorah-style scenes:
// vertex/index data is loaded once per triangle and re-rasterized per
// instance via the instanced stage-1 kernel.
uint32_t addInstancedMeshToScene(Scene& scene, const Mesh& mesh,
                                 const std::vector<simd_float4x4>& models);

// Build a Zorah-style scene: a few small base meshes (UV spheres of varying
// tessellation) instanced many times across a grid with random rotations.
// `numInstances` is approximate (rounded to the grid). Triangle count
// post-instancing ≈ numInstances × ~4k tris-per-base.
//
// The intent is to exercise the instancing path with significant overdraw
// at 4K — closer to the workload CuRast benchmarks against than a single
// high-poly sphere is.
Scene makeZorahLikeScene(uint32_t targetInstances);

// Replicate an existing scene `N` times across a 3D grid. Vertex / index
// data is REUSED (no duplication on the GPU): only the DrawCall + model-
// matrix arrays grow. Each new copy stays on the non-instanced path
// (instanceCount stays 1 per draw call). Useful for stress-testing a real
// asset like komainu without needing the instancing kernel or Zorah-scale
// memory.
//
// Takes Scene by rvalue (move-only) and returns a new Scene. When copies==1
// the input is moved through unchanged.
Scene replicateScene(Scene&& src, int copies);

}  // namespace metalrast
