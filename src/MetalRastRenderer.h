#pragma once

#include <Metal/Metal.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "Mesh.h"
#include "MeshRegistry.h"
#include "SharedTypes.h"
#include "TextureLoader.h"

namespace metalrast {

// Per-frame GPU timing info, populated after waitUntilCompleted().
// `stage*Ms` are populated only when a per-stage breakdown is requested
// (renderAndWaitProfiled); for the default render path they remain zero.
struct FrameStats {
    double gpuMilliseconds = 0.0;   // GPUEndTime - GPUStartTime (full frame)
    double clearMs   = 0.0;
    double stage1Ms  = 0.0;
    double stage23Ms = 0.0;         // stage2 + indirect-prep + stage3 combined
    double resolveMs = 0.0;
    uint32_t stage2Items   = 0;
    uint32_t stage3Items   = 0;
};

// 1:1 with CuRast's exposed CUDA rasterizer toggles
// (`RASTERIZER_VISBUFFER_INDEXED` and `RASTERIZER_VISBUFFER_INSTANCED` in
// `HostDeviceInterface.h`), plus an `Auto` setting that picks `Instanced`
// only when the scene actually has multi-instance draw calls.
enum class RasterMode : uint32_t {
    Auto               = 0,
    VisbufferIndexed   = 8,    // matches CuRast enum value
    VisbufferInstanced = 10,   // matches CuRast enum value
};

struct RendererConfig {
    uint32_t width  = 1920;
    uint32_t height = 1080;

    // Hard caps for the inter-stage queues. Items past the cap are dropped.
    // Per the handoff: stage2 ≈ 0.5×nTris, stage3 ≈ 64k for paper-style scenes.
    uint32_t stage2Capacity = 1u << 23;   // 8M items × 16 B = 128 MB
    uint32_t stage3Capacity = 1u << 23;   // 8M items × 32 B = 256 MB.
                                          // Earlier 64K/2M caps overflowed in
                                          // dense detail (balcony balusters)
                                          // → silent drops → holes.

    // Persistent stage-1 grid: number of threadgroups to launch (each batches
    // 256 triangles via atomicAdd). The CuRast paper uses ~32 on NVIDIA
    // (each SM hides latency well via warp-level oversubscription). On Apple
    // Silicon the GPU clearly needs a much larger TG count — ~128 lands at
    // peak across all scene sizes we tested on M2 Max (3.4× the 32-TG number
    // for 4K / 72M tris). Likely cause: smaller per-core register file → much
    // lower per-TG occupancy → need more TGs to saturate.
    uint32_t stage1Threadgroups = 256;

    // Hi-Z occlusion culling (Frostbite 2015 / Aaltonen 2015 single-phase
    // pattern, last-frame HiZ as occluder). When ON: at end of every frame
    // we extract a max-mip Hi-Z pyramid from the visibility buffer and run a
    // per-(DC, instance) AABB cull whose result feeds the NEXT frame's stage1.
    // Default OFF — toggle in the Render panel.
    bool     hiZOcclusion       = false;
};

class MetalRastRenderer {
public:
    MetalRastRenderer(MTL::Device* device, const RendererConfig& cfg,
                   const std::string& metallibPath);
    ~MetalRastRenderer();

    // Upload an immutable scene. Call once after construction (or whenever the
    // scene changes — buffers get reallocated).
    // Non-const: at Zorah scale (~10 GB indices) keeping both the Scene's
    // host vector and the Metal buffer alive simultaneously OOMs. We free
    // each host vector as soon as we've copied it into a Metal buffer.
    void uploadScene(Scene& scene);

    // Parse each EncodedTexture's MRTC blob (raw ASTC blocks per mip),
    // allocate one MTL::Texture per image (shared storage, all mips), and
    // build an argument buffer of texture handles bound to the resolve
    // kernel. Hardware sampler does mip / bilinear / trilinear / aniso
    // for free. Empty `encoded` is a no-op (resolve falls back to lambert
    // via INVALID_TEXTURE_HANDLE on every DC).
    //
    // Mutates `encoded` in-place: each EncodedTexture's mrtcBytes vector
    // is freed after a successful upload (the GPU now owns the texture,
    // the host bytes are dead weight). Failed entries are left intact.
    void uploadTextures(std::vector<EncodedTexture>& encoded);

    // Update the per-frame camera uniform.
    void setCamera(const CameraUniforms& cam);

    // Visualisation mode passed to the resolve kernel each frame.
    //   0 = lambert  1 = depth  2 = mesh-id  3 = tri-id  4 = stage
    void setVisMode(uint32_t mode) { visMode_ = mode; }

    // Stage-1 dispatch grid: number of threadgroups for the small-tri pass.
    // Pure dispatch knob — no buffer or pipeline state depends on it — so the
    // slider can poke it every frame without rebuilding the renderer.
    void setStage1Threadgroups(uint32_t n) { cfg_.stage1Threadgroups = n; }

    // Toggle Hi-Z occlusion culling at runtime. When transitioning OFF the
    // visibility mask is reset to all-1 so the next frame renders everything.
    void setHiZOcclusion(bool on);

    // Two-phase Hi-Z (Frostbite/Nanite). Only takes effect when Hi-Z is on.
    // Phase 1: render with last-frame's mask. Phase 2: re-test current Hi-Z
    // and render any disoccluded instances. Eliminates the 1-frame pop on
    // fast camera movement; costs an extra cull dispatch + counter-reset.
    void setHiZTwoPhase(bool on) { hiZTwoPhase_ = on; }
    // Toggle Hijma §6.2.2 instance compaction. No-op unless Hi-Z is also on.
    void setCompactInstances(bool on) { compactInstances_ = on; }

    // Override which stage-1 kernel is dispatched. `Auto` (default) picks
    // the instanced kernel iff any draw call has > 1 instance.
    void       setRasterMode(RasterMode m) { rasterMode_ = m; }
    RasterMode rasterMode() const { return rasterMode_; }

    // Submit one render. Blocks on completion (so getOutputRGBA + frameStats
    // are immediately valid). Returns the per-frame stats.
    FrameStats renderAndWait();

    // Same as renderAndWait but splits the render into 4 command buffers
    // (clear / stage1 / stage2+stage3 / resolve) so each can be timed via
    // GPUStartTime/GPUEndTime. Adds ~50–200 µs of CPU overhead per frame —
    // use only for profiling.
    FrameStats renderAndWaitProfiled();

    // ----- MTLCounterSampleBuffer-based per-stage instrumentation ---------
    // Per-stage GPU timing using hardware timestamp counters sampled at
    // dispatch boundaries inside a single command buffer. Unlike
    // `renderAndWaitProfiled`, this does NOT split the frame into multiple
    // command buffers (so it does not perturb scheduling) and adds only
    // a few-cycle counter-sample call between dispatches. Use this whenever
    // possible — it's the truthful per-stage breakdown.
    //
    // Caller pre-allocates an MTLCounterSampleBuffer (timestamp counter set,
    // sample count >= 16) and passes it in. The renderer fills `report` with
    // one label per stage (Clear, Stage1, ...) and the µs each took.
    struct CounterStage {
        std::string label;
        double      microseconds = 0.0;
    };
    struct CounterReport {
        std::vector<CounterStage> stages;
        double   gpuTotalMs   = 0.0;
        uint32_t stage2Items  = 0;
        uint32_t stage3Items  = 0;
    };
    // Returns true on success; false if the device or driver refused to
    // sample (then `report` is left empty and gpuTotalMs is still valid).
    bool renderAndWaitWithCounters(MTL::CounterSampleBuffer* tsBuf,
                                   CounterReport& report);

    // After renderAndWait(), copy the output texture into an RGBA8 buffer.
    std::vector<uint8_t> readbackRGBA() const;

    // Encode the scene-render dispatches onto an existing command buffer
    // (no commit, no wait). For interactive mode the host drives the queue
    // and adds a present pass after this. Returns nothing immediately —
    // stage2/stage3 counters can be sampled after waitUntilCompleted via
    // the bufStage2Counter / bufStage3Counter pointers exposed below.
    void encodeRender(MTL::CommandBuffer* cb);

    // Residency path: bind transient buffers from a per-frame FrameView
    // (ResidencyManager owns them; we hold weak refs for the duration of
    // this frame). Subsequent renderAndWait / encodeRender uses them.
    // Caller must invoke this BEFORE every frame in residency mode.
    void bindFrameView(const ResidencyManager::FrameView& view);

    // Encode a fullscreen-triangle present pass onto cb that samples our
    // output texture into `dst`. dst's pixel format must match the
    // present-pipeline's color attachment (RGBA8Unorm by default).
    void encodePresent(MTL::CommandBuffer* cb, MTL::Texture* dst);

    // For ImGui overlays / dumb readbacks: the texture the renderer wrote.
    MTL::Texture* outputTexture() const { return outputTex_; }
    MTL::Buffer*  stage2CounterBuf() const { return bufStage2Counter_; }
    MTL::Buffer*  stage3CounterBuf() const { return bufStage3Counter_; }
    uint64_t      totalTriangles()  const { return totalTriangles_;    }

    uint32_t width()  const { return cfg_.width;  }
    uint32_t height() const { return cfg_.height; }
    uint32_t stage2Capacity() const { return cfg_.stage2Capacity; }
    uint32_t stage3Capacity() const { return cfg_.stage3Capacity; }

private:
    void buildPipelinesForCompression(bool useCompressed,
                                      const std::string& metallibPath);
    void allocatePersistentBuffers();
    void encodeFrame(MTL::CommandBuffer* cb);

    // ----- Counter-sampling state (set by renderAndWaitWithCounters) -----
    // When tsSampleBuf_ is non-null, encodeFrame's group helper opens a
    // fresh compute encoder per labeled stage with a ComputePassDescriptor
    // configured to sample timestamps at encoder start and end (the only
    // counter-sampling boundary Apple Silicon supports). The (start, end)
    // sample-buffer indices for each stage are recorded in tsStages_ for
    // later resolution into per-stage µs.
    MTL::CounterSampleBuffer* tsSampleBuf_ = nullptr;   // weak; caller owns
    uint32_t                  tsNextIdx_   = 0;
    struct TsStageRange { std::string label; uint32_t startIdx; uint32_t endIdx; };
    std::vector<TsStageRange> tsStages_;

    // Per-stage encoders for the profiled path.
    void encodeReset    (MTL::CommandBuffer* cb);
    void encodeClear    (MTL::ComputeCommandEncoder* enc);
    void encodeStage1   (MTL::ComputeCommandEncoder* enc, MTL::Buffer* maskBits = nullptr);
    void encodeStage23  (MTL::ComputeCommandEncoder* enc);
    void encodeResolve  (MTL::ComputeCommandEncoder* enc);

    // Bind the (up to 4) index-buffer chunks on the stage1/2/3 slot scheme
    // (1 / 15 / 16 / 17). Unused slots get chunk 0 as a placeholder so the
    // shader never reads from a null binding.
    void bindIndexBuffers(MTL::ComputeCommandEncoder* enc);

    // Helpers
    MTL::ComputePipelineState* makePipeline(const char* name);
    MTL::ComputePipelineState* makePipelineFCV(const char* name,
                                               MTL::FunctionConstantValues* fcv);

    // Scope-bound autorelease pool wrapper (RAII).
    struct PoolScope {
        NS::AutoreleasePool* pool;
        PoolScope() : pool(NS::AutoreleasePool::alloc()->init()) {}
        ~PoolScope() { pool->release(); }
    };

private:
    MTL::Device*        device_       = nullptr;   // not owned
    MTL::CommandQueue*  queue_        = nullptr;
    MTL::Library*       library_      = nullptr;

    MTL::ComputePipelineState* psoClear_              = nullptr;
    // 4 stage1 variants: (compressed × USE_HIZ_MASK). The non-Hiz pair has
    // the mask read elided at compile time — true zero overhead when toggle
    // OFF. Renderer picks pipeline at dispatch time based on hiZEnabled_.
    MTL::ComputePipelineState* psoStage1_              = nullptr;
    MTL::ComputePipelineState* psoStage1Instanced_     = nullptr;
    MTL::ComputePipelineState* psoStage1Hiz_           = nullptr;
    MTL::ComputePipelineState* psoStage1InstancedHiz_  = nullptr;
    // USE_COMPACTED_INSTANCES variant — only for the instanced-Hi-Z path.
    // Hijma §6.2.2: iterate a packed visibleInstanceIdx[] from
    // compact_visible_instances instead of looping over the dense instance
    // range with a per-iter Hi-Z bit test. Built only if compactInstances_
    // is enabled in the renderer config.
    MTL::ComputePipelineState* psoStage1InstancedHizCompacted_ = nullptr;
    MTL::ComputePipelineState* psoCompactInstances_   = nullptr;
    // Output of compact_visible_instances. Shapes:
    //   bufVisibleInstanceIdx_     : numInstances_   × uint
    //   bufVisibleInstancesPerDC_  : numDrawCalls_   × uint
    // Allocated lazily in ensureCompactionBuffers_().
    MTL::Buffer*               bufVisibleInstanceIdx_      = nullptr;
    MTL::Buffer*               bufVisibleInstancesPerDC_   = nullptr;
    MTL::ComputePipelineState* psoPrepareIndirect_    = nullptr;
    MTL::ComputePipelineState* psoStage2_             = nullptr;
    MTL::ComputePipelineState* psoStage3_             = nullptr;
    // 4 resolve PSOs — cross product of (USE_PREFIX_SUM, USE_TEXTURING).
    // The textured variants compile in the UV/texSet/sampler binds + the
    // ~50-line bary-reconstruct + sample block; untextured variants
    // compile them out entirely. encodeResolve picks one per frame based
    // on (usePrefixSum_ && bufPfxBuckets_, !textures_.empty()).
    MTL::ComputePipelineState* psoResolve_            = nullptr;   // !pfx, !tex
    MTL::ComputePipelineState* psoResolvePfx_         = nullptr;   //  pfx, !tex
    MTL::ComputePipelineState* psoResolveTex_         = nullptr;   // !pfx,  tex
    MTL::ComputePipelineState* psoResolvePfxTex_      = nullptr;   //  pfx,  tex

    // Present pass: vertex + fragment that copies outputTex_ to a drawable.
    MTL::RenderPipelineState*  psoPresent_            = nullptr;
    MTL::SamplerState*         presentSampler_       = nullptr;

    // Persistent buffers (reset/used every frame).
    MTL::Buffer* bufFramebuffer_  = nullptr;   // W*H × 8B  (private)
    MTL::Buffer* bufBatchCounter_ = nullptr;   // 4B        (shared)
    MTL::Buffer* bufStage2Counter_= nullptr;   // 4B        (shared)
    MTL::Buffer* bufStage3Counter_= nullptr;   // 4B        (shared)
    MTL::Buffer* bufStage2Queue_  = nullptr;   // cap × 8B  (private)
    MTL::Buffer* bufStage3Queue_  = nullptr;   // cap × 16B (private)
    MTL::Buffer* bufIndirectS2_   = nullptr;   // 12B       (private)
    MTL::Buffer* bufIndirectS3_   = nullptr;   // 12B       (private)
    MTL::Buffer* bufCameraUni_      = nullptr;   // 112B           (shared)
    MTL::Buffer* bufWorldView_      = nullptr;   // numInstances×64B (shared, per-frame)
    MTL::Buffer* bufModelMatrices_  = nullptr;   // numInstances×64B (shared, immutable; used by resolve)

    MTL::Texture* outputTex_ = nullptr;          // W×H RGBA8 (shared)

    // Scene buffers.
    MTL::Buffer* bufVertices_   = nullptr;
    // Up to 4 index buffers (each ≤ Metal's per-buffer limit, ~18 GB on M2 Max).
    // Each DrawCall picks one via its `indexBufferIdx` field.
    static constexpr int kMaxIndexBuffers = 4;
    MTL::Buffer* bufIndices_[kMaxIndexBuffers] = { nullptr, nullptr, nullptr, nullptr };
    int          numIndexBuffers_ = 1;
    MTL::Buffer* bufDrawCalls_  = nullptr;

    // ----- Textures -----
    // One MTL::Texture per glTF image (ASTC LDR 4×4 or 6×6 sRGB, all mips,
    // shared storage). Indexed by `dc.textureHandle`. Owned by the
    // renderer; released in dtor. `bufUVs_` carries a flat float2 per
    // vertex (parallel to positions); resolve reads
    // `uvs[dc.uvOffset + localIdx]`.
    static constexpr uint32_t kMaxTextures = 256;
    std::vector<MTL::Texture*>  textures_;
    // Compacted (no nulls) MTL::Resource* view of `textures_`, rebuilt
    // whenever uploadTextures changes the texture set. Lets encodeResolve
    // call `useResources(ptr, count, usage)` once per frame instead of
    // looping `useResource()` per texture — saves N-1 ObjC dispatches per
    // frame on textured scenes (Hijma §6.4.1).
    std::vector<const MTL::Resource*> textureResourceList_;
    MTL::Buffer*                bufUVs_         = nullptr;     // float2 per vertex (Shared)
    MTL::Buffer*                texArgBuffer_   = nullptr;     // bindless arg buffer
    MTL::ArgumentEncoder*       texArgEncoder_  = nullptr;     // built from psoResolve_
    MTL::SamplerState*          samplerTrilinear_ = nullptr;   // mip + linear + aniso
    void ensureTextureBindings_();

    uint32_t numDrawCalls_   = 0;
    uint64_t totalTriangles_ = 0;
    uint32_t numInstances_   = 0;
    bool     hasInstancing_  = false;
    bool     compressed_     = true;             // mirrors Scene::compressed; drives kernel pick
    std::string metallibPath_;                   // stashed for pipeline rebuild on compression flip
    std::vector<simd_float4x4> modelMatrices_;   // CPU copy (per instance), used to build worldView per frame

    RendererConfig cfg_;
    uint32_t       visMode_ = 0;            // resolve kernel uses this each frame
    RasterMode     rasterMode_ = RasterMode::Auto;
    bool           residencyMode_ = false;  // true → don't release bound buffers in dtor
    std::vector<uint8_t> readbackBuffer_;   // sized once, reused

    // ----- Hi-Z occlusion culling ----------------------------------------
    // Runtime flag (mirrors UI checkbox). Diverges from cfg_.hiZOcclusion only
    // because the toggle persists across renderer rebuilds.
    bool         hiZEnabled_     = false;
    bool         hiZTwoPhase_    = false;   // takes effect only when hiZEnabled_
    // Instance-compaction toggle (Hijma §6.2.2 / runbook #12). Only takes
    // effect when both hiZEnabled_ and the compacted PSO is built.
    bool         compactInstances_ = false;

    // ----- Prefix-sum DC lookup -----------------------------------------
    // 64K-entry uint32 table mapping `(globalTri >> pfxShift_)` to the first
    // DC index whose cumulativeTriangleStart exceeds `bucket * (1<<shift)`.
    // Rebuilt on the CPU whenever DCs change (uploadScene + bindFrameView).
    // Lookup in resolve becomes O(1) bucket fetch + O(1-2) refinement.
    static constexpr uint32_t kPfxBucketCount = 1u << 16;
    MTL::Buffer* bufPfxBuckets_ = nullptr;   // (kPfxBucketCount + 1) × uint32, Shared
    uint32_t     pfxShift_      = 0;
    bool         usePrefixSum_  = true;
public:
    void setUsePrefixSum(bool on) {
        bool wasOff = !usePrefixSum_;
        usePrefixSum_ = on;
        // OFF→ON transition: bucket table was skipped on the most recent
        // bindFrameView, so rebuild it now before the next resolve dispatch.
        if (on && wasOff) rebuildPrefixBuckets();
    }
    bool usePrefixSum() const     { return usePrefixSum_; }
private:
    void rebuildPrefixBuckets();
    // Single MTLTexture with a mip chain. Mip 0 is render-resolution; each
    // higher mip is the max-of-2×2 downsample. Re-built every frame.
    MTL::Texture*  hiZTex_       = nullptr;
    uint32_t       hiZW_         = 0;
    uint32_t       hiZH_         = 0;
    uint32_t       hiZMips_      = 0;
    // 1 byte per instance slot (firstInstance + inst). 0 = occluded last frame
    // → skip in stage1. 1 = visible (default). Reset to all-1 every time the
    // toggle is flipped OFF or the bound scene changes.
    MTL::Buffer*   bufVisibleInstances_ = nullptr;
    // Two-phase: mask consumed by phase 2 (= newly-disoccluded instances,
    // computed by the cull kernel as `now-visible AND NOT phase-1-mask`).
    // Allocated only when two-phase is enabled.
    MTL::Buffer*   bufVisibleInstancesPhase2_ = nullptr;
    // Per-instance → drawCallIdx lookup, rebuilt host-side whenever DCs change.
    // Lets the cull kernel run as a flat 1D dispatch (1 thread per worldView
    // slot) without doing a binary search per thread.
    MTL::Buffer*   bufInstanceToDrawCall_ = nullptr;
    uint32_t       instanceMaskCap_ = 0;
    // Per-mip texture views; views[i] is mip i bound as a 1-level texture so
    // we can pass src=views[i], dst=views[i+1] to the downsample kernel.
    std::vector<MTL::Texture*> hiZMipViews_;
    // Pipelines for the three Hi-Z kernels.
    MTL::ComputePipelineState* psoHiZExtract_     = nullptr;
    MTL::ComputePipelineState* psoHiZDownsample_  = nullptr;   // 1 mip / dispatch (fallback)
    MTL::ComputePipelineState* psoHiZDownsample2_ = nullptr;   // 2 mips / dispatch (primary)
    MTL::ComputePipelineState* psoCullInstances_  = nullptr;

    void ensureVisibilityMask_();    // mask + slot→DC lookup; cheap, always called
    void ensureHiZTexture_();        // Hi-Z mip pyramid; only when toggle ON
    void ensureCompactionBuffers_(); // Hijma §6.2.2 — alloc on first compacted dispatch
    void rebuildInstanceMapping_();
    void resetVisibilityMask_();
    // Dispatch compact_visible_instances. Reads bufVisibleInstances_, writes
    // bufVisibleInstanceIdx_ + bufVisibleInstancesPerDC_. Only called when
    // hiZEnabled_ && compactInstances_ && psoCompactInstances_ != nullptr.
    void encodeCompactInstances_(MTL::ComputeCommandEncoder* enc,
                                 MTL::Buffer* visBits);
    void encodeHiZAndCull_(MTL::ComputeCommandEncoder* enc);
    // Build Hi-Z and run cull writing to `outBits`. If `phaseMode==1`, the
    // cull diffs against `prevBits` (only emit newly-disoccluded instances).
    void encodeBuildHiZ_(MTL::ComputeCommandEncoder* enc);
    void encodeCull_    (MTL::ComputeCommandEncoder* enc,
                         MTL::Buffer* outBits, MTL::Buffer* prevBits,
                         uint32_t phaseMode);
};

}  // namespace metalrast
