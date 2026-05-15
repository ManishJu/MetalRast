#pragma once

#include <Metal/Metal.hpp>

#include <string>

#include "Mesh.h"
#include "TextureLoader.h"

namespace metalrast {

// Load a .glb (or .gltf) file into a Scene.
//
// Each glTF mesh becomes a metalrast Mesh; nodes that reference the same mesh
// are collected as instances of that mesh (so a glTF scene that repeats the
// same mesh via N node transforms produces N entries in modelMatrices and one
// DrawCall with instanceCount = N — matching the paper §4.5 layout).
//
// Also handles the `EXT_mesh_gpu_instancing` extension when present (per-node
// inline instance arrays of TRS).
//
// Throws std::runtime_error on parse / file-IO failure.
//
// `maxIndices` caps the total index count (across all meshes/primitives) so
// huge scenes like Zorah (~4.9B uint32 indices ≈ 18 GiB) don't exceed Metal's
// per-buffer + working-set limits on M2 Max. Loader walks meshes in glTF
// order and stops adding new ones once the cap is reached. 0 means no cap.
//
// `compressed`: true → quantize positions to 16-bit per-mesh-AABB ushort4
// (8 B/vertex). false → keep raw simd_float3 (16 B/vertex). Should match
// Scene::compressed.
//
// `bitPackIndices`: true → emit per-mesh variable-bit-width packed indices
// into Scene::indicesBitstream (paper §4.6). false → raw uint32 in
// Scene::indices.
//
// `directDevice`: optional. When non-null, the loader allocates the Metal
// vertex + index buffers directly and fills them via their `contents()`
// pointer (CuRast-style: no host-vector intermediate). The returned Scene
// has `metalVertices` / `metalIndices` populated and the `positions` /
// `positionsFloat` / `indices` / `indicesBitstream` host vectors empty.
// `MetalRastRenderer::uploadScene` then adopts those buffers without a copy.
// This roughly halves peak host RAM at Zorah scale.
//
// Single-chunk only when `directDevice` is set: the renderer's multi-buffer
// index split is bypassed. Total indices must fit in Metal's per-buffer cap
// (~18.7 GB on M2 Max). If the scene would exceed this, the loader throws
// — call without `directDevice` to fall back to the host-vector path that
// supports up to 4 chunks.
Scene loadGLB(const std::string& path, uint64_t maxIndices = 0,
              bool compressed = true,
              bool bitPackIndices = true,
              MTL::Device* directDevice = nullptr,
              const TextureLoader::EncodeOpts& texOpts = {});

class ResidencyManager;

// CuRast-style streaming load: parse the .glb, register every primitive's
// metadata + source pointers with a ResidencyManager, but do NOT
// decompress or upload geometry. The first per-frame visibility pass
// queues page-ins for whichever meshes the camera sees; the manager
// services them asynchronously and reuses pool slots LRU-style.
//
// `maxIndices` still caps how many indices we'll register total (rarely
// useful in residency mode — point of residency is unbounded scenes).
//
// Throws std::runtime_error on parse / file-IO failure.
void loadGLBToRegistry(const std::string& path,
                       ResidencyManager& mgr,
                       uint64_t maxIndices = 0);

}  // namespace metalrast
