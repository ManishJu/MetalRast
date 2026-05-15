#pragma once

// Residency / streaming for huge scenes (CuRast §4.7 + §4.8 in spirit).
//
// MeshRegistry / VRAMPool / ResidencyManager keep ALL of a scene's metadata
// in a fixed-size CPU-side registry, but pin only the *currently visible*
// geometry into two large GPU "pool" buffers (vertex + index). Per frame:
//
//   1. Frustum-cull MeshRecords against the camera.
//   2. For visible-but-not-resident meshes, enqueue page-in.
//   3. A worker pool services up to N page-ins per frame: it decompresses
//      meshopt source bytes from the mmap'd .bin, quantises positions,
//      bit-packs indices, and writes the result directly into a pool slot
//      via the buffer's `contents()` pointer.
//   4. When the pool fills, evict the LRU resident slots.
//   5. Build a per-frame DrawCall + worldView + modelMatrices triplet
//      containing only resident-and-visible meshes; hand it to the
//      renderer via FrameView.
//
// Why on Apple Silicon: GPU working set is the binding constraint past ~1B
// source-tris; full Zorah's ~10-15 GiB of geometry can't all be resident
// at once. CuRast's residency manager solves the same problem on CUDA via
// `MemoryManager::allocVirtualCuda` + per-frame paging; we do the
// equivalent with two big `MTLBuffer`s and a free-list allocator.

#include <Metal/Metal.hpp>

#include <simd/simd.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "SharedTypes.h"

struct cgltf_data;
struct cgltf_primitive;

namespace metalrast {

class Camera;
struct Mesh;

// Procedurally-staged mesh bytes — produced once at registration for
// non-GLB sources (e.g. synthetic Zorah). The page-in worker memcpies
// these straight into pool slots, bypassing meshopt entirely.
struct ProcStaging {
    std::vector<uint8_t> vertBytes;   // already quantised to PackedVertex / simd_float3
    std::vector<uint8_t> idxBytes;    // already bit-packed (or u32 if !bitPackIndices)
};

// ---- Per-mesh persistent record -------------------------------------------
//
// Created once during loadGLBToRegistry. Stays resident on the CPU for the
// scene's entire lifetime — its size is small (~few hundred bytes) so 26K
// records (Zorah scale) fit comfortably in a few MB. The actual vertex/
// index data lives on disk (mmap'd .bin) until the manager pages it in.
struct MeshRecord {
    // Stable identity ------------------------------------------------------
    uint32_t                   meshId           = 0;     // index into registry
    // Source: exactly one of {prim, procStaging} is non-null. prim points
    // into the manager's cgltf_data (mmap'd .bin); procStaging holds CPU
    // bytes already laid out as the GPU expects.
    const cgltf_primitive*     prim             = nullptr;
    std::shared_ptr<ProcStaging> procStaging;

    // Decompressed counts (computed at registration) ----------------------
    uint32_t                   numVerts         = 0;
    uint32_t                   numIndices       = 0;
    uint32_t                   triangleCount    = 0;

    // Mesh-local AABB + quantisation params (precomputed once) ------------
    simd_float3                lo                = (simd_float3){ 0, 0, 0 };
    simd_float3                hi                = (simd_float3){ 0, 0, 0 };
    simd_float3                compressionFactor = (simd_float3){ 0, 0, 0 };
    simd_float3                inv               = (simd_float3){ 0, 0, 0 };

    // Per-mesh packed-index parameters (computed once via parallel scan) --
    uint32_t                   bitsPerIndex      = 0;
    uint32_t                   indexMin          = 0;

    // Instance transforms (one row per instance node referencing this prim)
    std::vector<simd_float4x4> transforms;

    // ---- Residency state -------------------------------------------------
    // The state machine: NotResident → Loading → Resident → (evicted) NotResident.
    enum class State : uint8_t { NotResident, Loading, Resident };
    std::atomic<State>         state            { State::NotResident };

    // Current pool slot extents — only valid when state == Resident.
    uint32_t                   poolVertexOffset = 0;     // in PackedVertex / simd_float3 stride
    uint64_t                   poolIndexBitOffset = 0;   // bit offset for packed indices
    uint64_t                   poolIndexU32Offset = 0;   // u32-element offset for raw indices
    // Slot identity returned by VRAMPool — used for free()/touch().
    int32_t                    vertSlotId       = -1;
    int32_t                    idxSlotId        = -1;
    // Index pool is split into ≤kMaxIndexBuffers chunks; this records which
    // chunk this mesh's index slot lives in. Mirrored into DrawCall.indexBufferIdx
    // so the renderer reads from the right MTL::Buffer slot.
    int8_t                     idxChunkIdx      = 0;

    // LRU bookkeeping — frame index when this mesh last appeared in the
    // visible-and-resident set. Updated by ResidencyManager::prepareFrame.
    std::atomic<uint64_t>      lastVisibleFrame { 0 };

    MeshRecord() = default;
    // Move-only because of std::atomic members.
    MeshRecord(const MeshRecord&) = delete;
    MeshRecord& operator=(const MeshRecord&) = delete;
    MeshRecord(MeshRecord&& o) noexcept { *this = std::move(o); }
    MeshRecord& operator=(MeshRecord&& o) noexcept {
        if (this != &o) {
            meshId             = o.meshId;
            prim               = o.prim;
            procStaging        = std::move(o.procStaging);
            numVerts           = o.numVerts;
            numIndices         = o.numIndices;
            triangleCount      = o.triangleCount;
            lo                 = o.lo;
            hi                 = o.hi;
            compressionFactor  = o.compressionFactor;
            inv                = o.inv;
            bitsPerIndex       = o.bitsPerIndex;
            indexMin           = o.indexMin;
            transforms         = std::move(o.transforms);
            state              .store(o.state.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
            poolVertexOffset   = o.poolVertexOffset;
            poolIndexBitOffset = o.poolIndexBitOffset;
            poolIndexU32Offset = o.poolIndexU32Offset;
            vertSlotId         = o.vertSlotId;
            idxSlotId          = o.idxSlotId;
            idxChunkIdx        = o.idxChunkIdx;
            lastVisibleFrame   .store(o.lastVisibleFrame.load(std::memory_order_relaxed),
                                       std::memory_order_relaxed);
        }
        return *this;
    }
};

// ---- VRAMPool — first-fit free-list allocator over one Metal buffer ------
//
// One pool for vertices, one for indices. Each allocation reserves a
// contiguous byte range. Allocations are tracked as `Slot` records with
// LRU stamps. When alloc fails, the LRU eviction routine frees the oldest
// resident slots until enough space is contiguous.
//
// Single-threaded API; the ResidencyManager serialises pool operations on
// its own mutex (allocations are rare relative to workloads).
class VRAMPool {
public:
    struct Slot {
        uint64_t offset       = 0;     // bytes from buffer start
        uint64_t size         = 0;     // bytes
        int32_t  meshId       = -1;    // -1 == free
        uint64_t lastUsedFrame = 0;
    };

    VRAMPool(MTL::Device* device, size_t totalBytes, const char* label);
    ~VRAMPool();

    VRAMPool(const VRAMPool&) = delete;
    VRAMPool& operator=(const VRAMPool&) = delete;

    // Returns slot ID on success, -1 on out-of-space (caller may evict + retry).
    int32_t alloc(uint64_t bytes, uint32_t meshId);
    void    free(int32_t slotId);
    void    touch(int32_t slotId, uint64_t frameIdx);

    const Slot& slot(int32_t id) const { return slots_[size_t(id)]; }

    // Evict the LRU `count` allocated slots whose `lastUsedFrame < cutoffFrame`.
    // Returns number actually freed. Caller's job to retry the allocation.
    uint32_t evictLRU(uint32_t count, uint64_t cutoffFrame,
                      std::vector<int32_t>* outFreedMeshIds);

    MTL::Buffer* buffer()    const { return buf_; }
    void*        contents()  const { return buf_->contents(); }
    uint64_t     capacity()  const { return capacity_; }
    uint64_t     used()      const { return used_; }

private:
    // Internal: insert a free range and merge adjacent runs.
    void insertFreeRange(uint64_t offset, uint64_t size);
    int32_t  newSlotId();

    MTL::Buffer*        buf_       = nullptr;
    uint64_t            capacity_  = 0;
    uint64_t            used_      = 0;
    // Slot record store. Slot IDs are indices; freed-slot indices are
    // recycled via freeSlotIds_.
    std::vector<Slot>   slots_;
    std::vector<int32_t> freeSlotIds_;
    // Free byte ranges, sorted by offset.
    struct FreeRange { uint64_t offset; uint64_t size; };
    std::vector<FreeRange> freeList_;
};

// ---- ResidencyManager ------------------------------------------------------
class ResidencyManager {
public:
    struct Config {
        // Pool sizes: 1 GiB each is the sweet spot on Apple Silicon. Empirical:
        // GPU performance falls off a cliff above ~4 GiB Shared MTLBuffers on
        // Apple Silicon (likely TLB / coherency-tracking overhead). On small
        // scenes the cliff is brutal: 1 GiB komainu (700 MiB working set) runs
        // at 6.4 ms, 6 GiB komainu at 949 ms — 150× slower. On Zorah-scale
        // workloads, however, an 8 GiB vert pool is the smallest size that
        // holds the visible working set without thrashing, and the per-frame
        // rasterizer cost dominates the cliff penalty. We default to 8 GiB so
        // big scenes work out of the box; override smaller via
        // --residency-vert-mib for komainu-class workloads.
        size_t   vertPoolBytes      = size_t(8) << 30;   // 8 GiB
        size_t   idxPoolBytes       = size_t(12) << 30;  // 12 GiB (= max: 4 chunks × 3 GiB)
        // Per-frame load budget + worker count. With 3163-primitive Zorah and
        // the prior 64+4 defaults, page-in took ~12s to settle on a still
        // camera and never settled on a moving one. 256+8 lets the workers
        // catch up reasonably fast.
        uint32_t maxLoadsPerFrame   = 256;
        uint32_t numWorkerThreads   = 8;
        bool     compressed         = true;
        bool     bitPackIndices     = true;
    };

    ResidencyManager(MTL::Device* device, Config cfg);
    ~ResidencyManager();

    ResidencyManager(const ResidencyManager&) = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;

    // ---- Registration phase (called by loader during scene load) ---------
    // Takes ownership of cgltfData (kept alive for our lifetime — primitives'
    // mmap'd source bytes must remain valid).
    void adoptCgltfData(cgltf_data* data);

    // Register a mesh. Returns its meshId. Body of MeshRecord is moved in.
    uint32_t registerMesh(MeshRecord&& rec);

    // Convenience: register a procedurally-generated mesh (no GLB source).
    // Quantises positions + bit-packs indices once on the calling thread,
    // stores the result on the MeshRecord; the page-in worker just memcpies.
    // Used by `--zorah N --residency` and synthetic stress tests.
    uint32_t registerProceduralMesh(const Mesh& mesh,
                                    const std::vector<simd_float4x4>& transforms);

    // After all meshes are registered: build the global instance count, the
    // initial scene AABB (for camera framing), etc.
    void finalizeRegistration();

    // ---- Per-frame view ---------------------------------------------------
    // What the renderer reads from this frame.
    static constexpr int kMaxIdxChunks = 4;
    struct FrameView {
        MTL::Buffer* vertexPool       = nullptr;
        // Multiple index-pool chunks (≤ kMaxIdxChunks); chunk 0 is always
        // populated, higher chunks are nullptr if unused. Renderer's existing
        // multi-buffer index path consumes these directly.
        MTL::Buffer* indexPools[kMaxIdxChunks] = { nullptr, nullptr, nullptr, nullptr };
        int          numIndexChunks   = 1;
        MTL::Buffer* drawCalls        = nullptr;
        MTL::Buffer* worldView        = nullptr;
        MTL::Buffer* modelMatrices    = nullptr;
        uint32_t     numDrawCalls     = 0;
        uint32_t     numInstances     = 0;
        uint64_t     totalTriangles   = 0;
        bool         compressed       = true;
        bool         indicesPacked    = true;
    };

    // Cull, page-in, build per-frame buffers. Returns the FrameView.
    FrameView prepareFrame(const Camera& cam, uint64_t frameIdx);

    // ---- Stats for UI panels ---------------------------------------------
    struct Stats {
        uint32_t totalMeshes      = 0;
        uint32_t residentMeshes   = 0;
        uint32_t visibleMeshes    = 0;
        uint32_t loadingMeshes    = 0;
        uint64_t vertPoolUsed     = 0;
        uint64_t vertPoolCapacity = 0;
        uint64_t idxPoolUsed      = 0;
        uint64_t idxPoolCapacity  = 0;
        uint32_t loadsThisFrame   = 0;
        uint64_t totalLoadsLifetime = 0;
    };
    Stats stats() const;

    // World-space scene AABB across all instance transforms (filled by
    // finalizeRegistration). Used for the initial camera framing.
    void sceneAABB(simd_float3& outLo, simd_float3& outHi) const {
        outLo = sceneLo_; outHi = sceneHi_;
    }

private:
    // ---- Worker thread ----------------------------------------------------
    // Pops mesh IDs off loadQueue_, decompresses meshopt source from the
    // mmap'd cgltf_buffer, quantises positions, packs indices, and writes
    // straight into the pool slots' contents() pointer.
    void workerLoop();

    // Synchronous load body — called from worker. Returns true on success.
    bool loadMeshIntoPool(uint32_t meshId);

    // Allocate vert + idx pool slots for a mesh. May evict LRU slots if
    // pools are full. Returns false if even after eviction we can't fit.
    bool ensurePoolSlots(uint32_t meshId);

    // Frustum cull a mesh against the camera. Tests the mesh's world-space
    // AABB (per-instance × mesh-local AABB) against the 6 frustum planes.
    bool isMeshVisible(const MeshRecord& rec,
                       const simd_float4* frustumPlanes /*[6]*/) const;

    // ---- State ------------------------------------------------------------
    MTL::Device*               device_     = nullptr;
    Config                     cfg_;

    cgltf_data*                cgltfData_  = nullptr;     // owned (we cgltf_free in dtor)

    std::vector<MeshRecord>    meshes_;
    simd_float3                sceneLo_    = (simd_float3){ 0, 0, 0 };
    simd_float3                sceneHi_    = (simd_float3){ 0, 0, 0 };
    uint32_t                   totalInstanceCount_ = 0;

    std::unique_ptr<VRAMPool>                 vertPool_;
    // Index pool split across N chunks of ≤2 GiB each. Sized in the ctor
    // from cfg.idxPoolBytes; we ceil-divide by the per-chunk cap (2 GiB)
    // so the user sees a single logical pool size.
    std::vector<std::unique_ptr<VRAMPool>>    idxPools_;

    // Per-frame buffers (Shared, rewritten in place each frame).
    MTL::Buffer*               drawCallsBuf_     = nullptr;
    MTL::Buffer*               worldViewBuf_     = nullptr;
    MTL::Buffer*               modelMatricesBuf_ = nullptr;
    uint32_t                   drawCallsCap_     = 0;     // sized to meshes_.size()
    uint32_t                   instancesCap_     = 0;     // sized to totalInstances

    // Async loader.
    std::vector<std::thread>   workers_;
    std::queue<uint32_t>       loadQueue_;
    std::mutex                 queueMutex_;
    std::condition_variable    queueCv_;
    std::atomic<bool>          stopWorkers_   { false };
    std::atomic<uint32_t>      meshesLoading_ { 0 };
    std::atomic<uint64_t>      totalLoads_    { 0 };

    // Pool ops mutex (alloc / free / evict are not internally thread-safe).
    std::mutex                 poolMutex_;

    // Frame counter / per-frame stats.
    uint64_t                   currentFrame_  = 0;
    uint32_t                   loadsThisFrame_ = 0;
    uint32_t                   visibleMeshes_  = 0;
    uint32_t                   residentMeshes_ = 0;

    // Per-meshopt-bv decompression state. The worker decompresses on demand
    // and refcounts so siblings (rare) don't double-decode.
    std::vector<std::atomic<int>> bvUsers_;
    std::vector<void*>            bvDecoded_;
    std::vector<std::mutex>       bvMutexes_;
    bool                          ensureBufferViewDecoded(uint32_t bvIdx);
    void                          releaseBufferView(uint32_t bvIdx);
};

// Build a Zorah-like synthetic scene directly into a ResidencyManager.
// Mirrors makeZorahLikeScene's geometry / instance distribution; calls
// registerProceduralMesh + finalizeRegistration. After this returns the
// manager is ready for prepareFrame.
void buildZorahLikeRegistry(ResidencyManager& mgr, uint32_t targetInstances);

}  // namespace metalrast
