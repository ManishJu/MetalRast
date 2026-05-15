// SharedTypes.h — layout-compatible struct definitions for both MSL and host C++.
//
// Included by CuRast.metal (via #include) and by the C++ host code. Apple's
// `simd/simd.h` provides matching types on both sides; we pick the matrix /
// vector type in each language so MSL gets `metal::float4x4` (which has
// operator*), while C++ gets the layout-compatible `simd_float4x4`.

#pragma once

#ifdef __METAL_VERSION__
    #include <metal_stdlib>
    using cu_float4x4 = metal::float4x4;
    using cu_float4   = metal::float4;
    using cu_float2   = metal::float2;
    using cu_uint     = metal::uint;
    using cu_ulong    = metal::ulong;
#else
    #include <simd/simd.h>
    #include <stdint.h>
    typedef simd_float4x4 cu_float4x4;
    typedef simd_float4   cu_float4;
    typedef simd_float2   cu_float2;
    typedef uint32_t      cu_uint;
    typedef uint64_t      cu_ulong;
#endif

// Camera + screen + projection state.
typedef struct {
    cu_float4x4 viewMatrix;       // world → view (camera at origin, looking +Z)
    cu_float4   cameraWorldPos;   // .xyz = world-space camera position
    cu_float2   projectionXY;     // (f / aspect, -f); negative Y flips Y axis
    float       nearPlane;
    float       farPlane;
    cu_uint     screenWidth;
    cu_uint     screenHeight;
    float       _pad0;
    float       _pad1;
} CameraUniforms;

// Per-mesh draw call. Static across frames. The per-frame `worldView` matrix
// (= viewMatrix * modelMatrix_per_instance) lives in a separate parallel
// array `worldView[firstInstance .. firstInstance+instanceCount-1]`.
//
// Triangle ID encoding (paper §4.5):
//   global =  cumulativeTriangleStart
//          + instanceID * triangleCount
//          + localTriangleID
// The 36-bit visibility-buffer payload is enough for ~68B such IDs (Zorah-
// scale comfortably fits).
// Position compression (paper §4.6):
// Each vertex is stored as 3 × uint16 (in a `ushort4` slot, .w unused for
// alignment). Decompression in the shader is:
//   pos = float3(uShort.xyz) * compressionFactor + aabbMin
// `compressionFactor = (aabbMax - aabbMin) / 65535`. Per-mesh AABB and
// factor are stored on the DrawCall; quantization happens once on the host.
// Index storage (paper §4.6):
//   bitsPerIndex == 0  → raw uint32 indices; `indexOffset` is in u32 elements
//                         (existing behaviour — back-compat with uncompressed mode).
//   bitsPerIndex >  0  → bit-packed indices (CuRast IX_PU16 family); `indexOffset`
//                         is in BITS into the chosen index chunk; each index is
//                         the next `bitsPerIndex` bits, decoded via `BitEdit::readU32`.
// `indexMin`            stores the smallest mesh-local index actually referenced
//                       by this draw call; packed values are stored relative to
//                       it so `bitsPerIndex = ceil(log2(indexMax - indexMin + 1))`
//                       (matches CuRast). Decoded index = packedValue + indexMin.
//                       For raw uint32 indices, indexMin == 0.
//
// One DrawCall corresponds to ONE glTF primitive (CuRast convention). Multi-
// primitive glTF meshes produce multiple DrawCalls that share a parent set of
// instance transforms.
typedef struct {
    cu_uint   vertexOffset;             // first vertex of this DC in vertex buffer (offset 0)
    cu_uint   triangleCount;            //                                              (offset 4)
    cu_ulong  indexOffset;              // raw: u32-element offset; packed: bit offset (offset 8, u64)
    cu_uint   instanceCount;            // number of instances of this mesh (>= 1)      (offset 16)
    cu_uint   firstInstance;            // into modelMatrices[] / worldView[]            (offset 20)
    cu_uint   indexBufferIdx;           // 0..N-1 — which indices buffer to use          (offset 24)
    cu_uint   bitsPerIndex;             // 0 = raw uint32; 1..32 = packed bits/index    (offset 28)
    float     aabbMinX;                 //                                                (offset 32)
    float     aabbMinY;                 //                                                (offset 36)
    float     aabbMinZ;                 //                                                (offset 40)
    float     compressionFactorX;       // = (aabbMax - aabbMin) / 65535                  (offset 44)
    float     compressionFactorY;       //                                                (offset 48)
    float     compressionFactorZ;       //                                                (offset 52)
    cu_ulong  cumulativeTriangleStart;  // u64; Zorah needs > 4.3B                       (offset 56)
    cu_uint   indexMin;                 // bias added back at decode (paper §4.6)        (offset 64)
    // Texturing. textureHandle is the slot index into the renderer's
    // texture array (= the glTF image-index); 0xFFFFFFFF means
    // "untextured, lambert only". uvOffset is the float2 element offset
    // into the global UV buffer for this DC's local vertex 0.
    cu_uint   textureHandle;            //                                                (offset 68)
    cu_uint   uvOffset;                 // float2 element offset into uvs[]               (offset 72)
    cu_uint   _pad2;                    //                                                (offset 76)
} DrawCall;                             // 80 bytes total

#ifndef __METAL_VERSION__
#  define INVALID_TEXTURE_HANDLE 0xFFFFFFFFu
#else
constant uint INVALID_TEXTURE_HANDLE = 0xFFFFFFFFu;
#endif

// Stage1 → Stage2 work item.
typedef struct {
    cu_uint drawCallID;
    cu_uint localTriangleID;
    cu_uint instanceID;
    cu_uint _pad;
} Stage2Item;

// Stage1/Stage2 → Stage3 work item (one entry per 64×64 tile of a large triangle).
typedef struct {
    cu_uint drawCallID;
    cu_uint localTriangleID;
    cu_uint instanceID;
    cu_uint tileX;                   // pixel coord, multiple of 64
    cu_uint tileY;
    cu_uint _pad0;
    cu_uint _pad1;
    cu_uint _pad2;
} Stage3Item;

// MTLDispatchThreadgroupsIndirectArguments — three uint32 thread-group counts.
typedef struct {
    cu_uint x;
    cu_uint y;
    cu_uint z;
} IndirectDispatchArgs;
