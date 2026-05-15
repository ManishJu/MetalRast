#include "MeshRegistry.h"

#include <Foundation/Foundation.hpp>

#include <cgltf.h>
#include <meshoptimizer.h>

#include "Camera.h"
#include "Mesh.h"           // bitsForIndexRange + PackedVertex layout

namespace {
struct LCG_R {
    uint64_t s;
    uint64_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return s; }
    float    range(float lo, float hi) { return lo + (hi - lo) * float((next() >> 8) & 0xFFFFFF) / float(0xFFFFFF); }
};
}

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace metalrast {

// ===========================================================================
//  VRAMPool
// ===========================================================================

VRAMPool::VRAMPool(MTL::Device* device, size_t totalBytes, const char* label)
    : capacity_(totalBytes)
{
    buf_ = device->newBuffer(totalBytes, MTL::ResourceStorageModeShared);
    if (!buf_) {
        std::fprintf(stderr,
            "VRAMPool '%s': newBuffer(%.2f GiB) failed (Metal max %.2f GiB)\n",
            label, totalBytes / 1073741824.0,
            device->maxBufferLength() / 1073741824.0);
        throw std::runtime_error("VRAMPool alloc failed");
    }
    if (label) {
        NS::String* s = NS::String::string(label, NS::UTF8StringEncoding);
        buf_->setLabel(s);
    }
    freeList_.push_back({0, totalBytes});
    slots_.reserve(8192);
    freeSlotIds_.reserve(1024);
}

VRAMPool::~VRAMPool() {
    if (buf_) buf_->release();
}

int32_t VRAMPool::newSlotId() {
    if (!freeSlotIds_.empty()) {
        int32_t id = freeSlotIds_.back();
        freeSlotIds_.pop_back();
        return id;
    }
    slots_.push_back({});
    return static_cast<int32_t>(slots_.size() - 1);
}

int32_t VRAMPool::alloc(uint64_t bytes, uint32_t meshId) {
    // Round up to 16-byte alignment (covers PackedVertex 8B and uint32 4B).
    bytes = (bytes + 15ull) & ~uint64_t(15);
    // First-fit. Walk the free list in offset order.
    for (size_t i = 0; i < freeList_.size(); ++i) {
        if (freeList_[i].size >= bytes) {
            uint64_t off = freeList_[i].offset;
            freeList_[i].offset += bytes;
            freeList_[i].size   -= bytes;
            if (freeList_[i].size == 0) {
                freeList_.erase(freeList_.begin() + i);
            }
            int32_t id = newSlotId();
            slots_[id] = Slot{ off, bytes, int32_t(meshId), 0 };
            used_ += bytes;
            return id;
        }
    }
    return -1;
}

void VRAMPool::free(int32_t slotId) {
    if (slotId < 0 || slotId >= int32_t(slots_.size())) return;
    Slot& s = slots_[slotId];
    if (s.meshId < 0) return;            // already free
    insertFreeRange(s.offset, s.size);
    used_ -= s.size;
    s.meshId        = -1;
    s.size          = 0;
    s.lastUsedFrame = 0;
    freeSlotIds_.push_back(slotId);
}

void VRAMPool::touch(int32_t slotId, uint64_t frameIdx) {
    if (slotId < 0 || slotId >= int32_t(slots_.size())) return;
    Slot& s = slots_[slotId];
    if (s.lastUsedFrame < frameIdx) s.lastUsedFrame = frameIdx;
}

void VRAMPool::insertFreeRange(uint64_t offset, uint64_t size) {
    // Find insertion point (sorted by offset) and merge with neighbours.
    auto it = std::lower_bound(freeList_.begin(), freeList_.end(), offset,
        [](const FreeRange& r, uint64_t o){ return r.offset < o; });
    auto inserted = freeList_.insert(it, FreeRange{ offset, size });
    // Try merge with NEXT neighbour first (so iterator math stays sane).
    auto next = inserted + 1;
    if (next != freeList_.end() && inserted->offset + inserted->size == next->offset) {
        inserted->size += next->size;
        freeList_.erase(next);
    }
    if (inserted != freeList_.begin()) {
        auto prev = inserted - 1;
        if (prev->offset + prev->size == inserted->offset) {
            prev->size += inserted->size;
            freeList_.erase(inserted);
        }
    }
}

uint32_t VRAMPool::evictLRU(uint32_t count, uint64_t cutoffFrame,
                            std::vector<int32_t>* outFreedMeshIds)
{
    // Build a list of currently-allocated slots (skip frees) sorted by
    // lastUsedFrame ascending. Evict the oldest `count` whose frame is
    // strictly before `cutoffFrame` (don't evict slots used this frame).
    struct Entry { int32_t id; uint64_t frame; };
    std::vector<Entry> live;
    live.reserve(slots_.size() - freeSlotIds_.size());
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].meshId < 0) continue;
        if (slots_[i].lastUsedFrame >= cutoffFrame) continue;
        live.push_back({ int32_t(i), slots_[i].lastUsedFrame });
    }
    std::sort(live.begin(), live.end(),
        [](const Entry& a, const Entry& b){ return a.frame < b.frame; });
    uint32_t evicted = 0;
    for (auto const& e : live) {
        if (evicted >= count) break;
        if (outFreedMeshIds) outFreedMeshIds->push_back(slots_[e.id].meshId);
        free(e.id);
        ++evicted;
    }
    return evicted;
}

// ===========================================================================
//  ResidencyManager
// ===========================================================================

ResidencyManager::ResidencyManager(MTL::Device* device, Config cfg)
    : device_(device), cfg_(cfg)
{
    // Cap each pool to maxBufferLength; warn if user requested more.
    size_t cap = device_->maxBufferLength();
    if (cfg_.vertPoolBytes > cap) {
        std::fprintf(stderr, "ResidencyManager: vertPool %.2f GiB > Metal cap %.2f GiB; clamping\n",
                     cfg_.vertPoolBytes / 1073741824.0, cap / 1073741824.0);
        cfg_.vertPoolBytes = cap;
    }
    if (cfg_.idxPoolBytes > cap * size_t(kMaxIdxChunks)) {
        std::fprintf(stderr, "ResidencyManager: idxPool %.2f GiB > Metal cap × %d chunks; clamping\n",
                     cfg_.idxPoolBytes / 1073741824.0, kMaxIdxChunks);
        cfg_.idxPoolBytes = cap * size_t(kMaxIdxChunks);
    }
    vertPool_ = std::make_unique<VRAMPool>(device_, cfg_.vertPoolBytes, "vertex pool");

    // Split the idx pool into ≤kMaxIdxChunks chunks, each ≤3 GiB. The
    // per-chunk cap keeps individual buffers well under the Apple-Silicon
    // perf cliff (catastrophic at ~6 GiB, mild ~14% degradation at 4 GiB
    // on a single Shared MTLBuffer). Splitting lets us keep total idx
    // capacity at e.g. 8-12 GiB for Zorah-class scenes without slideshow
    // performance.
    constexpr size_t kPerChunkCap = size_t(3) << 30;   // 3 GiB
    size_t remain = cfg_.idxPoolBytes;
    int chunks = 0;
    while (remain > 0 && chunks < kMaxIdxChunks) {
        size_t sz = std::min(remain, kPerChunkCap);
        char label[32];
        std::snprintf(label, sizeof(label), "index pool [%d]", chunks);
        idxPools_.emplace_back(std::make_unique<VRAMPool>(device_, sz, label));
        remain -= sz;
        ++chunks;
    }

    std::printf("ResidencyManager: vert pool %.2f GiB + idx pool %.2f GiB "
                "across %zu chunk%s (%u workers, max %u loads/frame)\n",
                cfg_.vertPoolBytes / 1073741824.0,
                cfg_.idxPoolBytes  / 1073741824.0,
                idxPools_.size(),
                idxPools_.size() == 1 ? "" : "s",
                cfg_.numWorkerThreads, cfg_.maxLoadsPerFrame);
}

ResidencyManager::~ResidencyManager() {
    stopWorkers_.store(true, std::memory_order_release);
    queueCv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    auto rel = [](auto*& p){ if (p) { p->release(); p = nullptr; } };
    rel(drawCallsBuf_); rel(worldViewBuf_); rel(modelMatricesBuf_);
    // Free any decompressed bv buffers we own.
    for (size_t i = 0; i < bvDecoded_.size(); ++i) {
        if (bvDecoded_[i]) { std::free(bvDecoded_[i]); bvDecoded_[i] = nullptr; }
    }
    if (cgltfData_) { cgltf_free(cgltfData_); cgltfData_ = nullptr; }
}

void ResidencyManager::adoptCgltfData(cgltf_data* data) {
    cgltfData_ = data;
    // Pre-size the bv decompression refcount + lock arrays.
    if (data) {
        bvUsers_   = std::vector<std::atomic<int>>(data->buffer_views_count);
        bvDecoded_ .assign(data->buffer_views_count, nullptr);
        bvMutexes_ = std::vector<std::mutex>(data->buffer_views_count);
        for (auto& a : bvUsers_) a.store(0, std::memory_order_relaxed);
    }
}

uint32_t ResidencyManager::registerMesh(MeshRecord&& rec) {
    rec.meshId = static_cast<uint32_t>(meshes_.size());
    meshes_.push_back(std::move(rec));
    return meshes_.back().meshId;
}

uint32_t ResidencyManager::registerProceduralMesh(
    const Mesh& mesh,
    const std::vector<simd_float4x4>& transforms)
{
    if (mesh.positions.empty() || mesh.indices.empty() || transforms.empty()) {
        throw std::runtime_error("registerProceduralMesh: empty mesh");
    }

    MeshRecord rec;
    rec.prim          = nullptr;
    rec.numVerts      = static_cast<uint32_t>(mesh.positions.size());
    rec.numIndices    = static_cast<uint32_t>(mesh.indices.size());
    rec.triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    rec.transforms    = transforms;

    // Mesh-local AABB.
    simd_float3 lo = mesh.positions[0], hi = mesh.positions[0];
    for (auto const& p : mesh.positions) {
        lo = simd_make_float3(std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z));
        hi = simd_make_float3(std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z));
    }
    rec.lo = lo; rec.hi = hi;
    simd_float3 size = (simd_float3){ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    rec.compressionFactor = (simd_float3){
        size.x > 0 ? size.x / 65535.0f : 0.0f,
        size.y > 0 ? size.y / 65535.0f : 0.0f,
        size.z > 0 ? size.z / 65535.0f : 0.0f };
    rec.inv = (simd_float3){
        size.x > 0 ? 65535.0f / size.x : 0.0f,
        size.y > 0 ? 65535.0f / size.y : 0.0f,
        size.z > 0 ? 65535.0f / size.z : 0.0f };
    rec.indexMin     = 0;   // generators use 0..numVerts-1
    rec.bitsPerIndex = bitsForIndexRange(rec.numVerts);

    // ---- Stage GPU-layout bytes once -----------------------------------
    auto staging = std::make_shared<ProcStaging>();

    // Vertex bytes.
    if (cfg_.compressed) {
        staging->vertBytes.resize(rec.numVerts * sizeof(PackedVertex));
        auto* dst = reinterpret_cast<PackedVertex*>(staging->vertBytes.data());
        for (uint32_t i = 0; i < rec.numVerts; ++i) {
            const simd_float3& p = mesh.positions[i];
            PackedVertex pv;
            pv.x = uint16_t(std::clamp(std::round((p.x - lo.x) * rec.inv.x), 0.0f, 65535.0f));
            pv.y = uint16_t(std::clamp(std::round((p.y - lo.y) * rec.inv.y), 0.0f, 65535.0f));
            pv.z = uint16_t(std::clamp(std::round((p.z - lo.z) * rec.inv.z), 0.0f, 65535.0f));
            pv._pad = 0;
            dst[i] = pv;
        }
    } else {
        staging->vertBytes.resize(rec.numVerts * sizeof(simd_float3));
        auto* dst = reinterpret_cast<simd_float3*>(staging->vertBytes.data());
        for (uint32_t i = 0; i < rec.numVerts; ++i) dst[i] = mesh.positions[i];
    }

    // Index bytes.
    if (cfg_.bitPackIndices) {
        const uint32_t bits = rec.bitsPerIndex;
        const uint64_t totalBits = uint64_t(rec.numIndices) * uint64_t(bits);
        // +1 trailing slack uint32 so unaligned end reads can't OOB.
        const uint64_t totalU32 = (totalBits + 31) / 32 + 1;
        staging->idxBytes.assign(size_t(totalU32) * 4, 0u);
        auto* dst = reinterpret_cast<uint32_t*>(staging->idxBytes.data());
        if (bits > 0) {
            const uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
            uint64_t cursor = 0;
            for (uint32_t i = 0; i < rec.numIndices; ++i) {
                uint32_t v = mesh.indices[i] & mask;
                uint64_t w = cursor / 32;
                uint32_t s = uint32_t(cursor & 31u);
                dst[w] |= (v << s);
                if (s + bits > 32u) dst[w + 1] |= (v >> (32u - s));
                cursor += bits;
            }
        }
    } else {
        staging->idxBytes.resize(size_t(rec.numIndices) * 4);
        std::memcpy(staging->idxBytes.data(),
                    mesh.indices.data(),
                    size_t(rec.numIndices) * 4);
    }

    rec.procStaging = std::move(staging);
    return registerMesh(std::move(rec));
}

void ResidencyManager::finalizeRegistration() {
    // Compute scene-wide world-space AABB and total instance count.
    sceneLo_ = (simd_float3){  1e30f,  1e30f,  1e30f };
    sceneHi_ = (simd_float3){ -1e30f, -1e30f, -1e30f };
    uint32_t totalInstances = 0;
    for (auto const& m : meshes_) {
        totalInstances += static_cast<uint32_t>(m.transforms.size());
        simd_float3 mLo = m.lo, mHi = m.hi;
        simd_float3 corners[8] = {
            {mLo.x, mLo.y, mLo.z}, {mHi.x, mLo.y, mLo.z},
            {mLo.x, mHi.y, mLo.z}, {mHi.x, mHi.y, mLo.z},
            {mLo.x, mLo.y, mHi.z}, {mHi.x, mLo.y, mHi.z},
            {mLo.x, mHi.y, mHi.z}, {mHi.x, mHi.y, mHi.z},
        };
        for (auto const& T : m.transforms) {
            for (auto const& c : corners) {
                simd_float4 w = simd_mul(T, (simd_float4){c.x, c.y, c.z, 1.0f});
                sceneLo_ = simd_make_float3(std::min(sceneLo_.x, w.x),
                                             std::min(sceneLo_.y, w.y),
                                             std::min(sceneLo_.z, w.z));
                sceneHi_ = simd_make_float3(std::max(sceneHi_.x, w.x),
                                             std::max(sceneHi_.y, w.y),
                                             std::max(sceneHi_.z, w.z));
            }
        }
    }
    if (sceneLo_.x > sceneHi_.x) {
        sceneLo_ = (simd_float3){ -1, -1, -1 };
        sceneHi_ = (simd_float3){  1,  1,  1 };
    }
    totalInstanceCount_ = totalInstances;

    // Pre-allocate per-frame buffers sized to worst case (all meshes / all
    // instances visible at once). They're Shared so writes go straight to
    // GPU-visible memory; we rewrite in place each frame.
    drawCallsCap_ = static_cast<uint32_t>(meshes_.size());
    instancesCap_ = totalInstances;
    drawCallsBuf_     = device_->newBuffer(size_t(drawCallsCap_) * sizeof(DrawCall),
                                            MTL::ResourceStorageModeShared);
    worldViewBuf_     = device_->newBuffer(size_t(instancesCap_) * sizeof(simd_float4x4),
                                            MTL::ResourceStorageModeShared);
    modelMatricesBuf_ = device_->newBuffer(size_t(instancesCap_) * sizeof(simd_float4x4),
                                            MTL::ResourceStorageModeShared);
    if (!drawCallsBuf_ || !worldViewBuf_ || !modelMatricesBuf_) {
        throw std::runtime_error("ResidencyManager: per-frame buffer alloc failed");
    }

    // Spin up worker threads.
    workers_.reserve(cfg_.numWorkerThreads);
    for (uint32_t i = 0; i < cfg_.numWorkerThreads; ++i) {
        workers_.emplace_back([this]{ workerLoop(); });
    }

    std::printf("ResidencyManager: %zu meshes / %u instances registered\n",
                meshes_.size(), totalInstanceCount_);
}

// ---- Frustum cull ---------------------------------------------------------
// Plane is `dot(plane.xyz, p) + plane.w >= 0` for points "inside".
bool ResidencyManager::isMeshVisible(const MeshRecord& rec,
                                     const simd_float4* planes) const
{
    simd_float3 mLo = rec.lo, mHi = rec.hi;
    simd_float3 corners[8] = {
        {mLo.x, mLo.y, mLo.z}, {mHi.x, mLo.y, mLo.z},
        {mLo.x, mHi.y, mLo.z}, {mHi.x, mHi.y, mLo.z},
        {mLo.x, mLo.y, mHi.z}, {mHi.x, mLo.y, mHi.z},
        {mLo.x, mHi.y, mHi.z}, {mHi.x, mHi.y, mHi.z},
    };
    // Any instance in-frustum → mesh is visible.
    for (auto const& T : rec.transforms) {
        // Transform the 8 corners; cheap enough for our 3-2K instances/mesh
        // worst case at ~5M corner-tests/frame even on Zorah.
        simd_float3 wMin = (simd_float3){  1e30f,  1e30f,  1e30f };
        simd_float3 wMax = (simd_float3){ -1e30f, -1e30f, -1e30f };
        for (auto const& c : corners) {
            simd_float4 w = simd_mul(T, (simd_float4){c.x, c.y, c.z, 1.0f});
            wMin = simd_make_float3(std::min(wMin.x, w.x),
                                     std::min(wMin.y, w.y),
                                     std::min(wMin.z, w.z));
            wMax = simd_make_float3(std::max(wMax.x, w.x),
                                     std::max(wMax.y, w.y),
                                     std::max(wMax.z, w.z));
        }
        // Test world AABB against each frustum plane (n-vertex / p-vertex test).
        bool inAll = true;
        for (int p = 0; p < 6; ++p) {
            simd_float3 n = (simd_float3){ planes[p].x, planes[p].y, planes[p].z };
            // p-vertex: corner of the AABB furthest along +n.
            simd_float3 pv = (simd_float3){
                n.x >= 0 ? wMax.x : wMin.x,
                n.y >= 0 ? wMax.y : wMin.y,
                n.z >= 0 ? wMax.z : wMin.z };
            float d = simd_dot(n, pv) + planes[p].w;
            if (d < 0.0f) { inAll = false; break; }
        }
        if (inAll) return true;
    }
    return false;
}

// ---- Pool slot allocation + eviction -------------------------------------
bool ResidencyManager::ensurePoolSlots(uint32_t meshId) {
    MeshRecord& m = meshes_[meshId];

    const size_t vbStride  = cfg_.compressed ? sizeof(PackedVertex) : sizeof(simd_float3);
    const size_t vbBytes   = size_t(m.numVerts) * vbStride;
    const uint64_t idxBits = uint64_t(m.numIndices) * uint64_t(m.bitsPerIndex);
    const size_t idxBytes  = cfg_.bitPackIndices
        ? size_t((idxBits + 31) / 32 + 1) * sizeof(uint32_t)   // +1 slack word
        : size_t(m.numIndices) * sizeof(uint32_t);

    std::lock_guard<std::mutex> lk(poolMutex_);

    auto evictAndRetry = [&](VRAMPool& pool, size_t need, int32_t& outSlot) {
        std::vector<int32_t> freedMeshIds;
        // Don't evict slots touched this frame.
        uint64_t cutoff = currentFrame_;
        // Evict in batches until we fit or run dry.
        for (int attempt = 0; attempt < 8 && outSlot < 0; ++attempt) {
            uint32_t freed = pool.evictLRU(/*count*/ 64, cutoff, &freedMeshIds);
            if (freed == 0) break;
            outSlot = pool.alloc(need, meshId);
        }
        // Mark freed meshes NotResident.
        for (int32_t fid : freedMeshIds) {
            MeshRecord& fm = meshes_[fid];
            fm.state.store(MeshRecord::State::NotResident, std::memory_order_release);
            fm.vertSlotId = -1;
            fm.idxSlotId  = -1;
        }
    };

    // Vertex pool: single chunk.
    int32_t vSlot = vertPool_->alloc(vbBytes, meshId);
    if (vSlot < 0) evictAndRetry(*vertPool_, vbBytes, vSlot);

    // Index pool: try each chunk in order. First-fit across chunks
    // distributes load roughly evenly without needing a smarter policy.
    int32_t iSlot     = -1;
    int8_t  iChunkIdx = -1;
    for (size_t c = 0; c < idxPools_.size() && iSlot < 0; ++c) {
        iSlot = idxPools_[c]->alloc(idxBytes, meshId);
        if (iSlot >= 0) iChunkIdx = int8_t(c);
    }
    // No raw alloc fit anywhere → evict from each chunk in order.
    if (iSlot < 0) {
        for (size_t c = 0; c < idxPools_.size() && iSlot < 0; ++c) {
            evictAndRetry(*idxPools_[c], idxBytes, iSlot);
            if (iSlot >= 0) iChunkIdx = int8_t(c);
        }
    }

    if (vSlot < 0 || iSlot < 0) {
        if (vSlot >= 0) vertPool_->free(vSlot);
        if (iSlot >= 0 && iChunkIdx >= 0) idxPools_[size_t(iChunkIdx)]->free(iSlot);
        return false;
    }

    m.vertSlotId         = vSlot;
    m.idxSlotId          = iSlot;
    m.idxChunkIdx        = iChunkIdx;
    m.poolVertexOffset   = static_cast<uint32_t>(vertPool_->slot(vSlot).offset / vbStride);
    VRAMPool& iPool      = *idxPools_[size_t(iChunkIdx)];
    if (cfg_.bitPackIndices) {
        m.poolIndexBitOffset = iPool.slot(iSlot).offset * 8ull;   // bytes → bits
        m.poolIndexU32Offset = 0;
    } else {
        m.poolIndexBitOffset = 0;
        m.poolIndexU32Offset = iPool.slot(iSlot).offset / sizeof(uint32_t);
    }
    return true;
}

ResidencyManager::Stats ResidencyManager::stats() const {
    Stats s;
    s.totalMeshes        = static_cast<uint32_t>(meshes_.size());
    s.residentMeshes     = residentMeshes_;
    s.visibleMeshes      = visibleMeshes_;
    s.loadingMeshes      = meshesLoading_.load(std::memory_order_acquire);
    s.vertPoolUsed       = vertPool_ ? vertPool_->used() : 0;
    s.vertPoolCapacity   = vertPool_ ? vertPool_->capacity() : 0;
    s.idxPoolUsed        = 0;
    s.idxPoolCapacity    = 0;
    for (auto const& p : idxPools_) {
        s.idxPoolUsed     += p->used();
        s.idxPoolCapacity += p->capacity();
    }
    s.loadsThisFrame     = loadsThisFrame_;
    s.totalLoadsLifetime = totalLoads_.load(std::memory_order_acquire);
    return s;
}

// ---- Per-frame: cull → enqueue page-ins → build DC + worldView ------------
//
// Steps:
//   1. Compute camera viewProj + frustum planes from CameraUniforms.
//   2. Per mesh: visibility test (any instance in-frustum). If visible:
//      ensure-resident (touch slots) or enqueue page-in.
//   3. Build DC list from visible-and-resident meshes:
//      - DC's vertexOffset / indexOffset come from the MeshRecord's current
//        pool slot (poolVertexOffset, poolIndexBitOffset/U32Offset).
//      - cumulativeTriangleStart is recomputed to be a strict prefix sum
//        across THIS frame's DC list.
//      - per-instance worldView and modelMatrices arrays are built in
//        lock-step.
//
ResidencyManager::FrameView
ResidencyManager::prepareFrame(const Camera& cam, uint64_t frameIdx) {
    currentFrame_   = frameIdx;
    loadsThisFrame_ = 0;
    visibleMeshes_  = 0;
    residentMeshes_ = 0;

    // Build viewProj + extract 6 frustum planes (Gribb/Hartmann).
    CameraUniforms cu = cam.uniforms(/*w*/1920u, /*h*/1080u);   // size cancels in plane test
    // Build a proper perspective matrix here for plane extraction. CameraUniforms
    // stores (viewMatrix, projectionXY, near, far). We reconstruct a clip-space
    // matrix M = P * V from those:
    simd_float4x4 V = cu.viewMatrix;
    float fy = -cu.projectionXY.y;          // f
    float fx = cu.projectionXY.x;           // f / aspect
    float zn = cu.nearPlane, zf = cu.farPlane;
    // Standard zero-to-one D3D-style perspective; matches our project_to_screen
    // (camera at origin, +Z forward). We just need the planes — sign is fine.
    simd_float4x4 P = (simd_float4x4){{
        (simd_float4){ fx, 0,            0,                              0 },
        (simd_float4){ 0,  fy,           0,                              0 },
        (simd_float4){ 0,  0,            zf / (zf - zn),                 1 },
        (simd_float4){ 0,  0,           -zn * zf / (zf - zn),            0 },
    }};
    simd_float4x4 M = simd_mul(P, V);
    // Rows of M^T:
    auto row = [&](int i) -> simd_float4 {
        return (simd_float4){ M.columns[0][i], M.columns[1][i],
                              M.columns[2][i], M.columns[3][i] };
    };
    simd_float4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    simd_float4 planes[6] = {
        r3 + r0,        // left
        r3 - r0,        // right
        r3 + r1,        // bottom
        r3 - r1,        // top
        r2,             // near (z >= 0)
        r3 - r2,        // far
    };
    // Normalise so that .xyz is a unit vector — keeps the AABB test cheap.
    for (int p = 0; p < 6; ++p) {
        float len = std::sqrt(planes[p].x * planes[p].x +
                              planes[p].y * planes[p].y +
                              planes[p].z * planes[p].z);
        if (len > 1e-6f) planes[p] = planes[p] * (1.0f / len);
    }

    // ---- Visibility + page-in queueing ------------------------------------
    DrawCall*       dcDst   = static_cast<DrawCall*>(drawCallsBuf_->contents());
    simd_float4x4*  wvDst   = static_cast<simd_float4x4*>(worldViewBuf_->contents());
    simd_float4x4*  mmDst   = static_cast<simd_float4x4*>(modelMatricesBuf_->contents());

    uint32_t numDCs       = 0;
    uint32_t numInstances = 0;
    uint64_t cumTri       = 0;

    // Local queue of meshes to request loads for this frame; we drain after
    // visibility so we hold the queueMutex_ once.
    std::vector<uint32_t> wantLoad;
    wantLoad.reserve(64);

    for (uint32_t mid = 0; mid < meshes_.size(); ++mid) {
        MeshRecord& rec = meshes_[mid];
        if (!isMeshVisible(rec, planes)) continue;
        ++visibleMeshes_;
        rec.lastVisibleFrame.store(frameIdx, std::memory_order_relaxed);

        MeshRecord::State st = rec.state.load(std::memory_order_acquire);
        if (st != MeshRecord::State::Resident) {
            if (st == MeshRecord::State::NotResident) {
                // Use compare-exchange so multiple frames don't double-enqueue.
                MeshRecord::State expected = MeshRecord::State::NotResident;
                if (rec.state.compare_exchange_strong(expected, MeshRecord::State::Loading,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire))
                {
                    wantLoad.push_back(mid);
                }
            }
            continue;   // skip emission this frame
        }

        // Resident + visible → emit one DC for this mesh, all of its instances.
        const uint32_t inst = static_cast<uint32_t>(rec.transforms.size());
        if (numDCs >= drawCallsCap_ || numInstances + inst > instancesCap_) {
            // Per-frame buffer would overflow — should not happen with our
            // sizing (worst-case pre-allocated). Stop emission gracefully.
            break;
        }

        DrawCall dc{};
        dc.vertexOffset            = rec.poolVertexOffset;
        dc.indexOffset             = cfg_.bitPackIndices ? rec.poolIndexBitOffset
                                                         : rec.poolIndexU32Offset;
        dc.bitsPerIndex            = cfg_.bitPackIndices ? rec.bitsPerIndex : 0u;
        dc.indexMin                = rec.indexMin;
        dc.triangleCount           = rec.triangleCount;
        dc.cumulativeTriangleStart = cumTri;
        dc.instanceCount           = inst;
        dc.firstInstance           = numInstances;
        dc.indexBufferIdx          = uint32_t(rec.idxChunkIdx);
        dc.aabbMinX = rec.lo.x; dc.aabbMinY = rec.lo.y; dc.aabbMinZ = rec.lo.z;
        dc.compressionFactorX = rec.compressionFactor.x;
        dc.compressionFactorY = rec.compressionFactor.y;
        dc.compressionFactorZ = rec.compressionFactor.z;
        // Residency mode is untextured today (texture upload lives in the
        // static path only). Resolve falls back to pure lambert when
        // textureHandle is INVALID.
        dc.textureHandle = INVALID_TEXTURE_HANDLE;
        dc.uvOffset      = 0;
        dcDst[numDCs++] = dc;
        cumTri += uint64_t(inst) * uint64_t(rec.triangleCount);

        // Per-instance worldView (= V * M) and modelMatrices.
        // Pack det(wv 3x3) sign into wv[3][3]; stage1 reads it from there
        // to skip the per-(tri, instance) cross+dot in the hot path.
        // Bottom row of an affine wv is [0,0,0,1] so wv[3][3] is otherwise
        // unread.
        for (uint32_t k = 0; k < inst; ++k) {
            const simd_float4x4& mm = rec.transforms[k];
            mmDst[numInstances]     = mm;
            simd_float4x4 wv        = simd_mul(V, mm);
            simd_float3 c0          = simd_make_float3(wv.columns[0]);
            simd_float3 c1          = simd_make_float3(wv.columns[1]);
            simd_float3 c2          = simd_make_float3(wv.columns[2]);
            float det               = simd_dot(simd_cross(c0, c1), c2);
            wv.columns[3].w         = (det < 0.0f) ? -1.0f : 1.0f;
            wvDst[numInstances]     = wv;
            ++numInstances;
        }

        // Touch slots for LRU.
        vertPool_->touch(rec.vertSlotId, frameIdx);
        idxPools_[size_t(rec.idxChunkIdx)]->touch(rec.idxSlotId, frameIdx);
        ++residentMeshes_;
    }

    // ---- Drain wantLoad queue under queueMutex_ ---------------------------
    if (!wantLoad.empty()) {
        const uint32_t budget = cfg_.maxLoadsPerFrame;
        std::lock_guard<std::mutex> lk(queueMutex_);
        // Cap total in-flight loads at budget — gates GPU pool churn.
        uint32_t toQueue = std::min<uint32_t>(uint32_t(wantLoad.size()), budget);
        for (uint32_t i = 0; i < toQueue; ++i) loadQueue_.push(wantLoad[i]);
        // For meshes we couldn't queue this frame, revert their state to
        // NotResident so a later frame can retry.
        for (uint32_t i = toQueue; i < wantLoad.size(); ++i) {
            meshes_[wantLoad[i]].state.store(MeshRecord::State::NotResident,
                                              std::memory_order_release);
        }
        loadsThisFrame_ = toQueue;
        if (toQueue > 0) queueCv_.notify_all();
    }

    FrameView v;
    v.vertexPool     = vertPool_->buffer();
    v.numIndexChunks = int(idxPools_.size());
    for (int c = 0; c < v.numIndexChunks && c < kMaxIdxChunks; ++c) {
        v.indexPools[c] = idxPools_[size_t(c)]->buffer();
    }
    v.drawCalls      = drawCallsBuf_;
    v.worldView      = worldViewBuf_;
    v.modelMatrices  = modelMatricesBuf_;
    v.numDrawCalls   = numDCs;
    v.numInstances   = numInstances;
    v.totalTriangles = cumTri;
    v.compressed     = cfg_.compressed;
    v.indicesPacked  = cfg_.bitPackIndices;
    return v;
}

void ResidencyManager::workerLoop() {
    while (!stopWorkers_.load(std::memory_order_acquire)) {
        uint32_t meshId = UINT32_MAX;
        {
            std::unique_lock<std::mutex> lk(queueMutex_);
            queueCv_.wait(lk, [this]{
                return stopWorkers_.load(std::memory_order_acquire) || !loadQueue_.empty();
            });
            if (stopWorkers_.load(std::memory_order_acquire)) return;
            if (loadQueue_.empty()) continue;
            meshId = loadQueue_.front();
            loadQueue_.pop();
        }
        meshesLoading_.fetch_add(1, std::memory_order_acq_rel);
        bool ok = loadMeshIntoPool(meshId);
        meshesLoading_.fetch_sub(1, std::memory_order_acq_rel);
        if (ok) {
            totalLoads_.fetch_add(1, std::memory_order_acq_rel);
        } else {
            // Failed: revert to NotResident so a future attempt can retry.
            meshes_[meshId].state.store(MeshRecord::State::NotResident,
                                        std::memory_order_release);
        }
    }
}

// ---- meshopt-on-demand decompression --------------------------------------
// Lazy decompress + refcount: each load task increments user count, frees
// when count returns to 0.
bool ResidencyManager::ensureBufferViewDecoded(uint32_t bvIdx) {
    if (bvIdx >= cgltfData_->buffer_views_count) return false;
    cgltf_buffer_view& bv = cgltfData_->buffer_views[bvIdx];
    if (!bv.has_meshopt_compression) return true;
    if (bv.data) return true;

    std::lock_guard<std::mutex> lk(bvMutexes_[bvIdx]);
    if (bv.data) return true;
    const cgltf_meshopt_compression& mc = bv.meshopt_compression;
    if (!mc.buffer || !mc.buffer->data) return false;

    void* dst = std::malloc(bv.size);
    if (!dst) return false;
    std::memset(dst, 0, bv.size);
    const uint8_t* src = static_cast<const uint8_t*>(mc.buffer->data) + mc.offset;
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
    if (rc != 0) { std::free(dst); return false; }
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
    bv.data           = dst;
    bvDecoded_[bvIdx] = dst;
    return true;
}

void ResidencyManager::releaseBufferView(uint32_t bvIdx) {
    if (bvIdx >= cgltfData_->buffer_views_count) return;
    if (--bvUsers_[bvIdx] != 0) return;
    cgltf_buffer_view& bv = cgltfData_->buffer_views[bvIdx];
    if (bv.data && bv.has_meshopt_compression && bvDecoded_[bvIdx] == bv.data) {
        std::free(bv.data);
        bv.data           = nullptr;
        bvDecoded_[bvIdx] = nullptr;
    }
}

namespace {

inline const uint8_t* accessor_base_ptr(const cgltf_accessor* a) {
    if (!a) return nullptr;
    const cgltf_buffer_view* bv = a->buffer_view;
    if (!bv) return nullptr;
    if (bv->data) return static_cast<const uint8_t*>(bv->data) + a->offset;
    if (!bv->buffer || !bv->buffer->data) return nullptr;
    return static_cast<const uint8_t*>(bv->buffer->data) + bv->offset + a->offset;
}

inline size_t accessor_stride_bytes(const cgltf_accessor* a) {
    const cgltf_buffer_view* bv = a->buffer_view;
    if (bv->stride) return bv->stride;
    return a->stride;
}

template <typename T>
inline T read_unaligned(const void* p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }

const cgltf_accessor* find_position_accessor(const cgltf_primitive& prim) {
    for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
        if (prim.attributes[ai].type == cgltf_attribute_type_position)
            return prim.attributes[ai].data;
    }
    return nullptr;
}

}  // namespace

bool ResidencyManager::loadMeshIntoPool(uint32_t meshId) {
    MeshRecord& m = meshes_[meshId];

    // Procedural source: bytes already in GPU layout, just memcpy.
    if (m.procStaging) {
        if (!ensurePoolSlots(meshId)) return false;
        VRAMPool& iPool = *idxPools_[size_t(m.idxChunkIdx)];
        void* vDst = static_cast<uint8_t*>(vertPool_->contents())
                   + vertPool_->slot(m.vertSlotId).offset;
        void* iDst = static_cast<uint8_t*>(iPool.contents())
                   + iPool.slot(m.idxSlotId).offset;
        std::memcpy(vDst, m.procStaging->vertBytes.data(), m.procStaging->vertBytes.size());
        std::memcpy(iDst, m.procStaging->idxBytes.data(),  m.procStaging->idxBytes.size());
        m.state.store(MeshRecord::State::Resident, std::memory_order_release);
        return true;
    }

    const cgltf_primitive* prim = m.prim;
    if (!prim) return false;
    const cgltf_accessor* posAcc = find_position_accessor(*prim);
    const cgltf_accessor* idxAcc = prim->indices;
    if (!posAcc || !idxAcc) return false;

    if (!ensurePoolSlots(meshId)) return false;

    cgltf_size posBv = posAcc->buffer_view
        ? cgltf_size(posAcc->buffer_view - cgltfData_->buffer_views)
        : cgltfData_->buffer_views_count;
    cgltf_size idxBv = idxAcc->buffer_view
        ? cgltf_size(idxAcc->buffer_view - cgltfData_->buffer_views)
        : cgltfData_->buffer_views_count;
    bvUsers_[posBv].fetch_add(1, std::memory_order_acq_rel);
    bvUsers_[idxBv].fetch_add(1, std::memory_order_acq_rel);

    if (!ensureBufferViewDecoded(uint32_t(posBv)) ||
        !ensureBufferViewDecoded(uint32_t(idxBv)))
    {
        std::lock_guard<std::mutex> lk(poolMutex_);
        vertPool_->free(m.vertSlotId); m.vertSlotId = -1;
        idxPools_[size_t(m.idxChunkIdx)]->free(m.idxSlotId);
        m.idxSlotId  = -1;
        releaseBufferView(uint32_t(posBv));
        releaseBufferView(uint32_t(idxBv));
        return false;
    }

    VRAMPool& iPool = *idxPools_[size_t(m.idxChunkIdx)];
    void*    vertBase = static_cast<uint8_t*>(vertPool_->contents())
                      + vertPool_->slot(m.vertSlotId).offset;
    uint32_t* idxBase = reinterpret_cast<uint32_t*>(
                          static_cast<uint8_t*>(iPool.contents())
                          + iPool.slot(m.idxSlotId).offset);

    // ---- Positions ---------------------------------------------------------
    {
        const uint8_t* base   = accessor_base_ptr(posAcc);
        const size_t   stride = accessor_stride_bytes(posAcc);
        const size_t   n      = posAcc->count;
        if (cfg_.compressed) {
            PackedVertex* dst = static_cast<PackedVertex*>(vertBase);
            const simd_float3 lo = m.lo, inv = m.inv;
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
            simd_float3* dst = static_cast<simd_float3*>(vertBase);
            for (size_t i = 0; i < n; ++i) {
                const uint8_t* p = base + i * stride;
                dst[i] = simd_make_float3(read_unaligned<float>(p + 0),
                                           read_unaligned<float>(p + 4),
                                           read_unaligned<float>(p + 8));
            }
        }
    }

    // ---- Indices -----------------------------------------------------------
    {
        const uint8_t* base   = accessor_base_ptr(idxAcc);
        const size_t   stride = accessor_stride_bytes(idxAcc);
        const size_t   n      = idxAcc->count;
        if (cfg_.bitPackIndices) {
            uint64_t bytes = (uint64_t(n) * uint64_t(m.bitsPerIndex) + 7) / 8 + 4;
            std::memset(idxBase, 0, bytes);
            const uint32_t bits = m.bitsPerIndex;
            const uint32_t bias = uint32_t(0u) - m.indexMin;
            const uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
            uint64_t bitCursor = 0;
            auto write_value = [&](uint32_t value) {
                uint64_t wordIdx = bitCursor / 32;
                uint32_t shift   = uint32_t(bitCursor & 31u);
                uint32_t v       = (value + bias) & mask;
                idxBase[wordIdx] |= (v << shift);
                if (shift + bits > 32u) {
                    uint32_t bitsInFirst = 32u - shift;
                    idxBase[wordIdx + 1] |= (v >> bitsInFirst);
                }
                bitCursor += bits;
            };
            switch (idxAcc->component_type) {
                case cgltf_component_type_r_8u:
                    for (size_t i = 0; i < n; ++i) write_value(uint32_t(base[i * stride]));
                    break;
                case cgltf_component_type_r_16u:
                    for (size_t i = 0; i < n; ++i)
                        write_value(uint32_t(read_unaligned<uint16_t>(base + i * stride)));
                    break;
                case cgltf_component_type_r_32u:
                    for (size_t i = 0; i < n; ++i)
                        write_value(read_unaligned<uint32_t>(base + i * stride));
                    break;
                default:
                    for (size_t i = 0; i < n; ++i)
                        write_value(uint32_t(cgltf_accessor_read_index(idxAcc, i)));
                    break;
            }
        } else {
            uint32_t* dst = idxBase;
            switch (idxAcc->component_type) {
                case cgltf_component_type_r_8u:
                    for (size_t i = 0; i < n; ++i) dst[i] = uint32_t(base[i * stride]);
                    break;
                case cgltf_component_type_r_16u:
                    for (size_t i = 0; i < n; ++i)
                        dst[i] = uint32_t(read_unaligned<uint16_t>(base + i * stride));
                    break;
                case cgltf_component_type_r_32u:
                    if (stride == 4) std::memcpy(dst, base, n * 4);
                    else for (size_t i = 0; i < n; ++i)
                             dst[i] = read_unaligned<uint32_t>(base + i * stride);
                    break;
                default:
                    for (size_t i = 0; i < n; ++i)
                        dst[i] = uint32_t(cgltf_accessor_read_index(idxAcc, i));
                    break;
            }
        }
    }

    releaseBufferView(uint32_t(posBv));
    releaseBufferView(uint32_t(idxBv));

    m.state.store(MeshRecord::State::Resident, std::memory_order_release);
    return true;
}

// ===========================================================================
//   buildZorahLikeRegistry
// ===========================================================================
//
// Mirrors makeZorahLikeScene in Mesh.cpp (same base meshes, same RNG, same
// instance grid) but registers each unique mesh with a ResidencyManager
// instead of building a Scene. After this returns, mgr.finalizeRegistration
// has run and the manager is ready for prepareFrame.
void buildZorahLikeRegistry(ResidencyManager& mgr, uint32_t targetInstances) {
    std::vector<Mesh> bases;
    bases.push_back(makeUVSphere(20, 30, 1.0f));
    bases.push_back(makeUVSphere(30, 50, 1.0f));
    bases.push_back(makeUVSphere(40, 60, 0.7f));
    bases.push_back(makeUVSphere(15, 25, 1.2f));

    uint32_t perBase = (targetInstances + uint32_t(bases.size()) - 1) / uint32_t(bases.size());
    LCG_R rng{0x9E3779B97F4A7C15ull};

    uint32_t gridSide = 1;
    while (gridSide * gridSide * gridSide < targetInstances) ++gridSide;
    float spacing    = 2.5f;
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
        mgr.registerProceduralMesh(bases[b], transforms);
        emitted += toEmit;
    }
    mgr.finalizeRegistration();
    std::printf("buildZorahLikeRegistry: %zu unique meshes, %u instances\n",
                bases.size(), emitted);
}

}  // namespace metalrast
