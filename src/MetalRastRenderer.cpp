#include "MetalRastRenderer.h"

#include "TextureLoader.h"

#include <Foundation/Foundation.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace metalrast {

namespace {
    inline NS::String* nsstr(const char* s) {
        return NS::String::string(s, NS::UTF8StringEncoding);
    }
}

MetalRastRenderer::MetalRastRenderer(MTL::Device* device, const RendererConfig& cfg,
                               const std::string& metallibPath)
    : device_(device), cfg_(cfg), metallibPath_(metallibPath)
{
    queue_ = device_->newCommandQueue();
    if (!queue_) throw std::runtime_error("Failed to create command queue");

    // Default to compressed pipelines (matches default Scene::compressed = true).
    buildPipelinesForCompression(/*useCompressed=*/true, metallibPath);
    allocatePersistentBuffers();

    // ---- Present pipeline (interactive viewer) ----------------------------
    // Tiny vertex+fragment that samples outputTex_ to whatever colour target
    // the host gives us. Pixel format is fixed to RGBA8Unorm to match both
    // our output texture AND the layer's drawable.
    {
        MTL::Function* vfn = library_->newFunction(nsstr("present_vs"));
        MTL::Function* ffn = library_->newFunction(nsstr("present_fs"));
        if (!vfn || !ffn) {
            std::fprintf(stderr, "Present shaders not found in metallib\n");
            throw std::runtime_error("Present shaders missing");
        }
        MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vfn);
        desc->setFragmentFunction(ffn);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        NS::Error* err = nullptr;
        psoPresent_ = device_->newRenderPipelineState(desc, &err);
        desc->release();
        vfn->release(); ffn->release();
        if (!psoPresent_) {
            std::fprintf(stderr, "present pipeline failed: %s\n",
                         err ? err->localizedDescription()->utf8String() : "(none)");
            throw std::runtime_error("present pso");
        }
        MTL::SamplerDescriptor* sd = MTL::SamplerDescriptor::alloc()->init();
        sd->setMinFilter(MTL::SamplerMinMagFilterLinear);
        sd->setMagFilter(MTL::SamplerMinMagFilterLinear);
        presentSampler_ = device_->newSamplerState(sd);
        sd->release();
    }
}

MetalRastRenderer::~MetalRastRenderer() {
    auto rel = [](auto*& p){ if (p) { p->release(); p = nullptr; } };
    if (residencyMode_) {
        // Borrowed buffers (vertex pool, index pool, drawCalls, worldView,
        // modelMatrices) are owned by the ResidencyManager — null them out
        // so the rel() lambda doesn't release.
        bufVertices_      = nullptr;
        for (int b = 0; b < kMaxIndexBuffers; ++b) bufIndices_[b] = nullptr;
        bufDrawCalls_     = nullptr;
        bufWorldView_     = nullptr;
        bufModelMatrices_ = nullptr;
    }
    rel(bufVertices_);  rel(bufDrawCalls_); rel(bufUVs_);
    for (int b = 0; b < kMaxIndexBuffers; ++b) rel(bufIndices_[b]);
    // Textures + arg buffer + sampler.
    for (auto* t : textures_) if (t) t->release();
    textures_.clear();
    textureResourceList_.clear();
    rel(texArgBuffer_); rel(texArgEncoder_); rel(samplerTrilinear_);
    rel(bufFramebuffer_);
    rel(bufBatchCounter_); rel(bufStage2Counter_); rel(bufStage3Counter_);
    rel(bufStage2Queue_);  rel(bufStage3Queue_);
    rel(bufIndirectS2_);   rel(bufIndirectS3_);
    rel(bufCameraUni_);
    rel(bufWorldView_);
    rel(bufModelMatrices_);
    rel(outputTex_);
    rel(psoClear_); rel(psoStage1_); rel(psoStage1Instanced_);
    rel(psoStage1Hiz_); rel(psoStage1InstancedHiz_);
    rel(psoPrepareIndirect_);
    rel(psoStage2_); rel(psoStage3_);
    rel(psoResolve_); rel(psoResolvePfx_);
    rel(psoResolveTex_); rel(psoResolvePfxTex_);
    rel(psoPresent_); rel(presentSampler_);
    rel(psoHiZExtract_); rel(psoHiZDownsample_); rel(psoHiZDownsample2_);
    rel(psoCullInstances_);
    rel(psoStage1InstancedHizCompacted_); rel(psoCompactInstances_);
    rel(bufVisibleInstances_); rel(bufVisibleInstancesPhase2_);
    rel(bufInstanceToDrawCall_);
    rel(bufVisibleInstanceIdx_); rel(bufVisibleInstancesPerDC_);
    for (auto* v : hiZMipViews_) if (v) v->release();
    hiZMipViews_.clear();
    rel(hiZTex_);
    rel(library_);
    rel(queue_);
}

void MetalRastRenderer::bindFrameView(const ResidencyManager::FrameView& view) {
    // Switch into residency mode. Buffers are now owned by the manager.
    if (!residencyMode_) {
        // First entry into residency mode: free anything we may have
        // allocated via uploadScene() so we don't leak.
        auto rel = [](auto*& p){ if (p) { p->release(); p = nullptr; } };
        rel(bufVertices_);
        for (int b = 0; b < kMaxIndexBuffers; ++b) rel(bufIndices_[b]);
        rel(bufDrawCalls_); rel(bufWorldView_); rel(bufModelMatrices_);
        residencyMode_ = true;
    }
    if (view.compressed != compressed_) {
        std::printf("Rebuilding pipelines for compression=%s\n",
                    view.compressed ? "on" : "off");
        buildPipelinesForCompression(view.compressed, metallibPath_);
    }
    bufVertices_      = view.vertexPool;
    for (int b = 0; b < kMaxIndexBuffers; ++b) bufIndices_[b] = nullptr;
    int nChunks = std::min(view.numIndexChunks, kMaxIndexBuffers);
    for (int b = 0; b < nChunks; ++b) bufIndices_[b] = view.indexPools[b];
    numIndexBuffers_  = nChunks;
    bufDrawCalls_     = view.drawCalls;
    bufWorldView_     = view.worldView;
    bufModelMatrices_ = view.modelMatrices;
    numDrawCalls_     = view.numDrawCalls;
    numInstances_     = view.numInstances;
    totalTriangles_   = view.totalTriangles;
    hasInstancing_    = (view.numInstances > view.numDrawCalls);
    if (usePrefixSum_) rebuildPrefixBuckets();
}

// Build the bucket → first-DC table that resolve uses to skip most of the
// binary search. Single sweep over DCs (already sorted by cumulativeStart).
//
//   bucket[b] = smallest DC index i such that cum[i] > b * (1<<shift),
//               or numDrawCalls_ if no such i exists.
//
// `shift` is chosen so the largest cumulativeStart maps to bucket
// (kPfxBucketCount - 1) — i.e. T_total fits in [0, kPfxBucketCount<<shift).
// CPU cost: O(N + B). For N=3163, B=64K that's ~70K integer ops (~30 µs).
void MetalRastRenderer::rebuildPrefixBuckets() {
    if (!bufPfxBuckets_ || !bufDrawCalls_ || numDrawCalls_ == 0) return;
    auto* pfx = static_cast<uint32_t*>(bufPfxBuckets_->contents());
    const auto* dcs = static_cast<const DrawCall*>(bufDrawCalls_->contents());

    // Total post-instance triangle count = top of the prefix sum.
    const DrawCall& last = dcs[numDrawCalls_ - 1];
    uint64_t totalTris = last.cumulativeTriangleStart
                       + uint64_t(last.triangleCount) * last.instanceCount;

    // shift such that (totalTris >> shift) < kPfxBucketCount.
    uint32_t shift = 0;
    while ((totalTris >> shift) >= kPfxBucketCount) ++shift;
    pfxShift_ = shift;

    // Single linear sweep; cumulativeStart is monotonic so DC i's bucket is
    // monotonic too. Set pfx[b] = first i with cum > b*stride.
    uint32_t i = 0;
    for (uint32_t b = 0; b <= kPfxBucketCount; ++b) {
        uint64_t threshold = uint64_t(b) << shift;
        while (i < numDrawCalls_ && dcs[i].cumulativeTriangleStart <= threshold) ++i;
        pfx[b] = i;
    }
}

// Build a pipeline for `name`, optionally with a function-constants block
// applied at compile time. Pass nullptr `fcv` for kernels that don't use
// function constants (clear/prepare_indirect).
MTL::ComputePipelineState* MetalRastRenderer::makePipeline(const char* name) {
    return makePipelineFCV(name, nullptr);
}

MTL::ComputePipelineState* MetalRastRenderer::makePipelineFCV(
    const char* name, MTL::FunctionConstantValues* fcv)
{
    NS::Error* err = nullptr;
    MTL::Function* fn = nullptr;
    if (fcv) {
        fn = library_->newFunction(nsstr(name), fcv, &err);
    } else {
        fn = library_->newFunction(nsstr(name));
    }
    if (!fn) {
        const char* msg = err ? err->localizedDescription()->utf8String()
                              : "(no description)";
        std::fprintf(stderr, "Kernel '%s' not found in metallib: %s\n", name, msg);
        throw std::runtime_error("Missing kernel");
    }
    MTL::ComputePipelineState* pso = device_->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        const char* msg = err ? err->localizedDescription()->utf8String() : "unknown";
        std::fprintf(stderr, "Failed to build pipeline '%s': %s\n", name, msg);
        throw std::runtime_error("Pipeline creation failed");
    }
    return pso;
}

void MetalRastRenderer::buildPipelinesForCompression(bool useCompressed,
                                                    const std::string& metallibPath)
{
    // Lazy-load the library once; rebuild only the pipeline states on flip.
    if (!library_) {
        NS::Error* err = nullptr;
        NS::URL* url = NS::URL::fileURLWithPath(nsstr(metallibPath.c_str()));
        library_ = device_->newLibrary(url, &err);
        if (!library_) {
            const char* msg = err ? err->localizedDescription()->utf8String() : "unknown";
            std::fprintf(stderr, "Failed to load metallib '%s': %s\n",
                         metallibPath.c_str(), msg);
            throw std::runtime_error("metallib load failed");
        }
    }

    // Release any prior pipelines (in case of compression flip).
    auto rel = [](auto*& p){ if (p) { p->release(); p = nullptr; } };
    rel(psoClear_); rel(psoStage1_); rel(psoStage1Instanced_);
    rel(psoStage1Hiz_); rel(psoStage1InstancedHiz_);
    rel(psoStage1InstancedHizCompacted_); rel(psoCompactInstances_);
    rel(psoPrepareIndirect_); rel(psoStage2_); rel(psoStage3_);
    rel(psoResolve_); rel(psoResolvePfx_);
    rel(psoResolveTex_); rel(psoResolvePfxTex_);

    // FunctionConstantValues for the vertex-using kernels. Set both
    // USE_COMPRESSED (index 0) and USE_HIZ_MASK (index 1). For stage1 we
    // build BOTH USE_HIZ_MASK variants so the runtime toggle picks the
    // right pipeline; for stages 2/3/resolve the mask isn't read so the
    // value of USE_HIZ_MASK doesn't matter (we set it false).
    bool useC = useCompressed;
    bool hizFalse = false, hizTrue = true;
    bool pfxFalse = false, pfxTrue = true;
    // All three function constants must be supplied to every pipeline build,
    // even if a given kernel doesn't reference one — Metal requires every
    // declared `[[function_constant(N)]]` to have a value at PSO creation.
    bool texFalse = false, texTrue = true;
    // All FOUR function constants must be supplied to every pipeline build,
    // even if a given kernel doesn't reference one — Metal requires every
    // declared `[[function_constant(N)]]` to have a value at PSO creation.
    bool compFalse = false, compTrue = true;
    auto makeFcv = [&](bool hiz, bool pfx, bool tex, bool comp = false) {
        auto* fcv = MTL::FunctionConstantValues::alloc()->init();
        fcv->setConstantValue(&useC,                MTL::DataTypeBool, NS::UInteger(0));
        fcv->setConstantValue(hiz ? &hizTrue : &hizFalse,
                              MTL::DataTypeBool, NS::UInteger(1));
        fcv->setConstantValue(pfx ? &pfxTrue : &pfxFalse,
                              MTL::DataTypeBool, NS::UInteger(2));
        fcv->setConstantValue(tex ? &texTrue : &texFalse,
                              MTL::DataTypeBool, NS::UInteger(3));
        fcv->setConstantValue(comp ? &compTrue : &compFalse,
                              MTL::DataTypeBool, NS::UInteger(4));
        return fcv;
    };
    auto* fcvNoHiz          = makeFcv(false, false, false);
    auto* fcvWithHiz        = makeFcv(true,  false, false);
    auto* fcvNoHizPfx       = makeFcv(false, true,  false);
    auto* fcvNoHizTex       = makeFcv(false, false, true);
    auto* fcvNoHizPfxTex    = makeFcv(false, true,  true);
    auto* fcvWithHizComp    = makeFcv(true,  false, false, /*comp*/true);

    psoClear_              = makePipelineFCV("clear_framebuffer",          nullptr);
    psoPrepareIndirect_    = makePipelineFCV("prepare_indirect",           nullptr);
    psoStage1_             = makePipelineFCV("stage1_rasterize",           fcvNoHiz);
    psoStage1Instanced_    = makePipelineFCV("stage1_rasterize_instanced", fcvNoHiz);
    psoStage1Hiz_          = makePipelineFCV("stage1_rasterize",           fcvWithHiz);
    psoStage1InstancedHiz_ = makePipelineFCV("stage1_rasterize_instanced", fcvWithHiz);
    psoStage1InstancedHizCompacted_ =
        makePipelineFCV("stage1_rasterize_instanced", fcvWithHizComp);
    psoStage2_             = makePipelineFCV("stage2_rasterize",           fcvNoHiz);
    psoStage3_             = makePipelineFCV("stage3_rasterize",           fcvNoHiz);
    psoResolve_            = makePipelineFCV("resolve",                    fcvNoHiz);
    psoResolvePfx_         = makePipelineFCV("resolve",                    fcvNoHizPfx);
    psoResolveTex_         = makePipelineFCV("resolve",                    fcvNoHizTex);
    psoResolvePfxTex_      = makePipelineFCV("resolve",                    fcvNoHizPfxTex);
    // Hi-Z occlusion-cull kernels (no compression specialization needed).
    psoHiZExtract_         = makePipelineFCV("hiZ_extractDepth",           nullptr);
    psoHiZDownsample_      = makePipelineFCV("hiZ_downsample",             nullptr);
    psoHiZDownsample2_     = makePipelineFCV("hiZ_downsample2",            nullptr);
    psoCullInstances_      = makePipelineFCV("cull_instances_aabb",        nullptr);
    psoCompactInstances_   = makePipelineFCV("compact_visible_instances",  nullptr);

    fcvNoHiz->release();
    fcvWithHiz->release();
    fcvNoHizPfx->release();
    fcvNoHizTex->release();
    fcvNoHizPfxTex->release();
    fcvWithHizComp->release();
    compressed_ = useCompressed;

    // PSO stats — one-shot diagnostic. Apple doesn't expose registers/thread
    // directly, but `maxTotalThreadsPerThreadgroup` is implicitly throttled
    // by register pressure: high register usage → low max-TG. Comparing it
    // against 1024 (M2 Max's hard ceiling) tells us roughly how tight the
    // register budget is for each kernel.
    auto reportPSO = [](const char* name, MTL::ComputePipelineState* pso) {
        if (!pso) return;
        std::printf("  %-28s maxTG=%4u  simdWidth=%2u  tgMem=%5zu B\n",
                    name,
                    uint32_t(pso->maxTotalThreadsPerThreadgroup()),
                    uint32_t(pso->threadExecutionWidth()),
                    size_t(pso->staticThreadgroupMemoryLength()));
    };
    std::printf("PSO stats (compressed=%s) — maxTG headroom indicates register pressure (1024 = unconstrained):\n",
                useCompressed ? "yes" : "no");
    reportPSO("clear_framebuffer",          psoClear_);
    reportPSO("prepare_indirect",           psoPrepareIndirect_);
    reportPSO("stage1_rasterize",           psoStage1_);
    reportPSO("stage1_rasterize_instanced", psoStage1Instanced_);
    reportPSO("stage1_rasterize +HiZ",      psoStage1Hiz_);
    reportPSO("stage1_inst +HiZ",           psoStage1InstancedHiz_);
    reportPSO("stage2_rasterize",           psoStage2_);
    reportPSO("stage3_rasterize",           psoStage3_);
    reportPSO("resolve",                    psoResolve_);
    reportPSO("resolve +pfx",               psoResolvePfx_);
    reportPSO("resolve +tex",               psoResolveTex_);
    reportPSO("resolve +pfx +tex",          psoResolvePfxTex_);
    reportPSO("hiZ_extractDepth",           psoHiZExtract_);
    reportPSO("hiZ_downsample",             psoHiZDownsample_);
    reportPSO("hiZ_downsample2",            psoHiZDownsample2_);
    reportPSO("cull_instances_aabb",        psoCullInstances_);
    reportPSO("compact_visible_instances",  psoCompactInstances_);
    reportPSO("stage1_inst +HiZ +compact",  psoStage1InstancedHizCompacted_);
}

void MetalRastRenderer::allocatePersistentBuffers() {
    const uint32_t W = cfg_.width;
    const uint32_t H = cfg_.height;
    const size_t fbBytes = size_t(W) * H * sizeof(uint64_t);

    // Framebuffer: huge, GPU-only.
    bufFramebuffer_ = device_->newBuffer(fbBytes, MTL::ResourceStorageModePrivate);

    // Counters: 4B each, Shared so we can read back stage2/stage3 sizes.
    auto makeCounter = [&](){
        return device_->newBuffer(sizeof(uint32_t),
                                  MTL::ResourceStorageModeShared);
    };
    bufBatchCounter_  = makeCounter();
    bufStage2Counter_ = makeCounter();
    bufStage3Counter_ = makeCounter();

    // Stage2/3 queues: bulk, GPU-only.
    bufStage2Queue_ = device_->newBuffer(size_t(cfg_.stage2Capacity) * sizeof(Stage2Item),
                                         MTL::ResourceStorageModePrivate);
    bufStage3Queue_ = device_->newBuffer(size_t(cfg_.stage3Capacity) * sizeof(Stage3Item),
                                         MTL::ResourceStorageModePrivate);

    // Indirect-args: 3 uints each.
    bufIndirectS2_ = device_->newBuffer(sizeof(IndirectDispatchArgs),
                                        MTL::ResourceStorageModePrivate);
    bufIndirectS3_ = device_->newBuffer(sizeof(IndirectDispatchArgs),
                                        MTL::ResourceStorageModePrivate);

    // Camera uniform: small, written from CPU each frame.
    bufCameraUni_ = device_->newBuffer(sizeof(CameraUniforms),
                                       MTL::ResourceStorageModeShared);

    // Prefix-sum bucket table: (kPfxBucketCount + 1) × uint32 = 256 KB+.
    // Shared so the CPU can rewrite it whenever DCs change without going
    // through a blit. Initial contents = zeros (sentinel = no DCs yet).
    bufPfxBuckets_ = device_->newBuffer(
        size_t(kPfxBucketCount + 1) * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    std::memset(bufPfxBuckets_->contents(), 0, bufPfxBuckets_->length());

    // Output texture: RGBA8 sRGB, Shared so we can getBytes after the frame.
    // sRGB pixel format means the resolve kernel writes its (linear-light)
    // shaded color and the hardware encodes linear → sRGB on write. Without
    // this, linear bytes get stored raw and viewers/PNG decoders that expect
    // sRGB-encoded data interpret them as gamma-encoded, dividing by ~2.2 →
    // the whole frame looks too dark. The input ASTC textures are sampled
    // through ASTC_4x4_sRGB / ASTC_6x6_sRGB so the hardware decode brings
    // them into linear; lambert math is in linear; we must encode back to
    // sRGB at the output for the chain to be correct end-to-end.
    {
        MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatRGBA8Unorm_sRGB, W, H, /*mipmapped*/ false);
        desc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
        desc->setStorageMode(MTL::StorageModeShared);
        outputTex_ = device_->newTexture(desc);
        // texture2DDescriptor returns autoreleased; do not release it.
    }

    readbackBuffer_.assign(size_t(W) * H * 4, 0);
}

// Parse each MRTC blob and allocate one MTL::Texture per image with all
// mip levels populated. Plumbs the textures into the resolve kernel via
// an argument buffer (Metal's standard bindless pattern for N-texture
// arrays). Empty input is a no-op.
//
// MRTC payload is already raw ASTC LDR blocks per mip — no transcoder.
// On Apple Silicon the unified memory model means we can replaceRegion
// straight out of the host-side blob into the Metal texture; the GPU
// will fault the pages in on first sample.
void MetalRastRenderer::uploadTextures(std::vector<EncodedTexture>& encoded) {
    if (encoded.empty()) return;

    metalrast::TextureLoader::initOnce();

    // Sampler / arg encoder / arg buffer — created on first call and
    // reused. uploadTextures only registers per-image MTL::Textures into
    // the encoder's slots.
    ensureTextureBindings_();

    // Clear the cached Resource* list FIRST so the (very brief) window
    // between the release loop below and the rebuild at end-of-function
    // can never observe dangling pointers. Defensive: encodeFrame is
    // single-threaded with respect to uploadTextures so this can't be
    // reached today, but it's free insurance.
    textureResourceList_.clear();

    // Free any previously-uploaded textures.
    for (auto* t : textures_) if (t) t->release();
    textures_.assign(encoded.size(), nullptr);
    // Zero the argument buffer before re-encoding. ensureTextureBindings_
    // does this once on first call; on a SECOND uploadTextures (scene
    // swap) any slot index we don't write to below would otherwise retain
    // a Metal-texture ID from the just-released previous scene — a
    // dangling reference that could crash if a DC ever sampled it.
    if (texArgBuffer_) {
        std::memset(texArgBuffer_->contents(), 0,
                    texArgEncoder_->encodedLength());
    }

    size_t okCount = 0, ok4x4 = 0, ok6x6 = 0;
    size_t totalAstcBytes = 0;
    size_t hostBytesFreed = 0;
    for (size_t i = 0; i < encoded.size() && i < kMaxTextures; ++i) {
        auto& e = encoded[i];
        if (e.mrtcBytes.empty()) continue;

        metalrast::TextureLoader::MrtcView view;
        if (!metalrast::TextureLoader::readMrtc(e.mrtcBytes, view)) {
            std::fprintf(stderr,
                "uploadTextures[%zu]: MRTC parse failed (%zu bytes)\n",
                i, e.mrtcBytes.size());
            continue;
        }

        // Block-format guard. We only ship 4×4 and 6×6 sRGB pixel formats
        // today — anything else is a future-format leak.
        const bool is6x6 = (view.blockX == 6u);
        const uint32_t bd = is6x6 ? 6u : 4u;
        if (view.blockX != bd || view.blockY != bd) {
            std::fprintf(stderr,
                "uploadTextures[%zu]: unsupported block %ux%u\n",
                i, view.blockX, view.blockY);
            continue;
        }
        const auto pixfmt = is6x6 ? MTL::PixelFormatASTC_6x6_sRGB
                                  : MTL::PixelFormatASTC_4x4_sRGB;
        if (view.width < bd || view.height < bd || view.mipCount == 0) {
            std::fprintf(stderr,
                "uploadTextures[%zu]: dimensions too small (%ux%u, block=%u)\n",
                i, view.width, view.height, bd);
            continue;
        }

        // Autoreleased one-liner factory. Defaults: StorageModePrivate +
        // UsageShaderRead; we override storage mode to Shared so we can
        // replaceRegion from the host blob (Apple Silicon unified memory).
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(
            pixfmt, view.width, view.height, /*mipmapped*/ true);
        td->setMipmapLevelCount(view.mipCount);
        td->setStorageMode(MTL::StorageModeShared);
        MTL::Texture* tex = device_->newTexture(td);
        if (!tex) {
            std::fprintf(stderr,
                "uploadTextures[%zu]: newTexture failed (%ux%u, %u mips)\n",
                i, view.width, view.height, view.mipCount);
            continue;
        }

        // Upload each mip — raw ASTC bytes straight from the MRTC blob.
        for (uint32_t mip = 0; mip < view.mipCount; ++mip) {
            const auto& m = view.mips[mip];
            const uint32_t blocksX = (m.width  + bd - 1u) / bd;
            tex->replaceRegion(
                MTL::Region::Make2D(0, 0, m.width, m.height),
                mip, 0,
                m.data,
                /*bytesPerRow=*/ size_t(blocksX) * 16u,
                /*bytesPerImage=*/ 0);
            totalAstcBytes += m.size;
        }

        textures_[i] = tex;
        texArgEncoder_->setTexture(tex, uint32_t(i));
        ++okCount;
        if (is6x6) ++ok6x6; else ++ok4x4;

        // GPU now owns the texture; host-side blob is dead weight. Swap-
        // with-empty actually returns the capacity to the allocator
        // (clear() leaves it).
        hostBytesFreed += e.mrtcBytes.capacity();
        std::vector<uint8_t>().swap(e.mrtcBytes);
    }

    // Rebuild the compacted-no-nulls Resource view used by encodeResolve's
    // per-frame `useResources(...)` array call (Hijma §6.4.1 follow-up).
    textureResourceList_.clear();
    textureResourceList_.reserve(textures_.size());
    for (auto* t : textures_) {
        if (t) textureResourceList_.push_back(t);
    }

    std::printf("uploadTextures: %zu/%zu images uploaded "
                "(%zu @ ASTC 4×4 sRGB, %zu @ ASTC 6×6 sRGB, %.2f MiB GPU; "
                "sampler trilinear + 8× aniso; %.2f MiB host mrtc freed)\n",
                okCount, encoded.size(),
                ok4x4, ok6x6,
                totalAstcBytes / (1024.0 * 1024.0),
                hostBytesFreed / (1024.0 * 1024.0));
}

void MetalRastRenderer::uploadScene(Scene& scene) {
    const bool directVerts = (scene.metalVertices != nullptr);
    const bool directIdx   = (scene.metalIndices  != nullptr);
    bool noPositions = directVerts
        ? false
        : (scene.compressed ? scene.positions.empty()
                            : scene.positionsFloat.empty());
    bool noIndices   = directIdx
        ? false
        : (scene.indicesPacked ? scene.indicesBitstream.empty()
                               : scene.indices.empty());
    if (noPositions || noIndices || scene.drawCalls.empty()) {
        throw std::runtime_error("Scene is empty");
    }

    auto rel = [](auto*& p){ if (p) { p->release(); p = nullptr; } };
    // If we entered residency mode previously, the per-frame buffers
    // (vertex/index/draw-call/worldView/modelMatrices/UV pools) are *borrowed*
    // from ResidencyManager — calling release() on them is a double-free.
    // Drop the borrows before the cleanup pass and clear residencyMode_ so
    // subsequent paths (~MetalRastRenderer, the compression rebuild below)
    // treat us as a fresh static-upload renderer again.
    if (residencyMode_) {
        bufVertices_      = nullptr;
        for (int b = 0; b < kMaxIndexBuffers; ++b) bufIndices_[b] = nullptr;
        bufDrawCalls_     = nullptr;
        bufWorldView_     = nullptr;
        bufModelMatrices_ = nullptr;
        bufUVs_           = nullptr;
        residencyMode_    = false;
    }
    rel(bufVertices_); rel(bufDrawCalls_); rel(bufUVs_);
    for (int b = 0; b < kMaxIndexBuffers; ++b) rel(bufIndices_[b]);
    rel(bufWorldView_); rel(bufModelMatrices_);

    // Release any textures from a prior scene. uploadTextures() does this
    // internally for the textured→textured swap, but it early-returns when
    // the new scene has zero textures (line 393: `if (encoded.empty()) return;`)
    // — without this defensive release, a textured→untextured swap (drag-drop
    // in interactive mode) leaves stale MTL::Texture* in `textures_` and
    // dangling refs in `textureResourceList_`. encodeResolve gates on
    // `!textures_.empty()` so the textured PSO would still be picked, then
    // bind a freshly-released bufUVs_ (NULL) — kernel deref of NULL UV pool.
    for (auto* t : textures_) if (t) t->release();
    textures_.clear();
    textureResourceList_.clear();
    if (texArgBuffer_ && texArgEncoder_) {
        // Zero the arg buffer too, so the next textured scene starts from a
        // clean slate (matches uploadTextures' own scene-swap zero pass).
        std::memset(texArgBuffer_->contents(), 0,
                    texArgEncoder_->encodedLength());
    }

    // If the scene's compression mode flipped from what the pipelines were
    // built for, rebuild them. This recompiles ~5 specialized pipelines via
    // MTLFunctionConstantValues — typically <10 ms, only on change.
    if (scene.compressed != compressed_) {
        std::printf("Rebuilding pipelines for compression=%s\n",
                    scene.compressed ? "on" : "off");
        buildPipelinesForCompression(scene.compressed, metallibPath_);
    }

    if (directVerts) {
        // Loader pre-allocated and filled; just adopt.
        bufVertices_ = scene.takeMetalVertices();
        size_t stride = scene.compressed ? sizeof(PackedVertex) : sizeof(simd_float3);
        std::printf("Allocated vertex buffer: %.2f GiB (%s, %zu B/vertex, direct)\n",
                    bufVertices_->length() / 1073741824.0,
                    scene.compressed ? "compressed" : "uncompressed",
                    stride);
    } else {
        // Compressed: 8 B/vertex (PackedVertex). Uncompressed: 16 B/vertex
        // (simd_float3 padded to 16-byte alignment, matches MSL's float3 stride).
        const void*  src = scene.compressed ? (const void*)scene.positions.data()
                                            : (const void*)scene.positionsFloat.data();
        size_t       vc  = scene.compressed ? scene.positions.size()
                                            : scene.positionsFloat.size();
        size_t stride    = scene.compressed ? sizeof(PackedVertex) : sizeof(simd_float3);
        size_t vbBytes   = vc * stride;
        bufVertices_  = device_->newBuffer(src, vbBytes,
                                           MTL::ResourceStorageModeShared);
        if (!bufVertices_) {
            std::fprintf(stderr,
                "ERROR: vertex buffer allocation failed (%.2f GiB requested, "
                "Metal max %.2f GiB).\n",
                vbBytes / 1073741824.0,
                device_->maxBufferLength() / 1073741824.0);
            throw std::runtime_error("vertex buffer alloc failed");
        }
        std::printf("Allocated vertex buffer: %.2f GiB (%s, %zu B/vertex)\n",
                    vbBytes / 1073741824.0,
                    scene.compressed ? "compressed" : "uncompressed",
                    stride);
        // Vertex data is now in Metal — drop the host copy.
        std::vector<PackedVertex>{}.swap(scene.positions);
        std::vector<simd_float3>{}.swap(scene.positionsFloat);
    }

    std::vector<DrawCall> dcs = scene.drawCalls;

    if (directIdx) {
        // Loader pre-allocated a single Metal index buffer and filled it
        // directly. dcs already carry scene-wide indexOffset values; with a
        // single chunk those ARE the chunk-local offsets. Nothing to do.
        bufIndices_[0]    = scene.takeMetalIndices();
        // bufIndices_[1..3] stay null; bindIndexBuffers() falls back to
        // bufIndices_[0] for unused slots.
        numIndexBuffers_  = 1;
        std::printf("Index buffer split: 1 chunk (%s, direct), size (GiB): %.2f\n",
                    scene.indicesPacked ? "bit-packed" : "uint32",
                    bufIndices_[0]->length() / 1073741824.0);
        for (auto& dc : dcs) dc.indexBufferIdx = 0;
        goto index_upload_done;
    }

    // ---------- Multi-buffer index upload (host-vector path) ---------------
    // Metal caps individual buffers at maxBufferLength (~18.7 GB on M2 Max).
    // We split on draw-call boundaries; each DC's indices live entirely in
    // one chunk. The split criterion is byte-budget; the source data is
    // either:
    //   raw mode    — uint32 indices: chunk byte = (idxCount * 4)
    //   packed mode — bit stream:    chunk byte = (bits / 8 + 1)
    //
    // Each chunk also reserves +1 trailing uint32 (4B) of slack so an
    // unaligned packed read at the end of stream never OOBs.
    {
    const size_t kIdxBytesCap = device_->maxBufferLength();

    std::vector<uint32_t> order(dcs.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b){
        return dcs[a].indexOffset < dcs[b].indexOffset;
    });

    // Per-chunk source word ranges (uint32 word indices into the source array).
    struct Chunk { uint64_t srcWordStart; uint64_t srcWordEnd; };
    std::vector<Chunk> chunks;
    chunks.push_back({0, 0});

    if (!scene.indicesPacked) {
        // ----- raw uint32 indices -----
        // dc.indexOffset is in u32 elements, dc footprint = triCount*3 elements.
        for (uint32_t k : order) {
            DrawCall& dc = dcs[k];
            uint64_t dcStart = dc.indexOffset;
            uint64_t dcEnd   = dcStart + uint64_t(dc.triangleCount) * 3u;
            uint64_t chunkLimitElems = chunks.back().srcWordStart
                                     + (kIdxBytesCap / sizeof(uint32_t));
            if (dcEnd > chunkLimitElems) {
                // Close current chunk (srcWordEnd already set by prior DC) and
                // start a fresh chunk at this DC's source word.
                chunks.push_back({dcStart, dcStart});
            }
            dc.indexBufferIdx = static_cast<uint32_t>(chunks.size() - 1);
            dc.indexOffset    = dcStart - chunks.back().srcWordStart;  // local elem offset
            chunks.back().srcWordEnd = dcEnd;
        }
    } else {
        // ----- bit-packed indices -----
        // dc.indexOffset is in BITS, dc footprint in bits = triCount*3*bitsPerIndex.
        // Chunks must start on a uint32-word boundary so we can copy whole
        // uint32 words from the source bitstream.
        for (uint32_t k : order) {
            DrawCall& dc = dcs[k];
            uint64_t dcBitsLen = uint64_t(dc.triangleCount) * 3u * uint64_t(dc.bitsPerIndex);
            uint64_t dcBitStart = dc.indexOffset;
            uint64_t dcBitEnd   = dcBitStart + dcBitsLen;
            uint64_t chunkSrcWordStart = chunks.back().srcWordStart;
            uint64_t chunkBitStart = chunkSrcWordStart * 32ull;
            uint64_t newChunkBits  = dcBitEnd - chunkBitStart;
            uint64_t newChunkBytes = (newChunkBits + 7) / 8 + 4;   // +slack
            if (newChunkBytes > kIdxBytesCap) {
                // Close current chunk (its srcWordEnd was already updated by
                // the last DC that fit). Start a fresh chunk aligned DOWN to
                // the source-word that contains this DC's first bit.
                uint64_t newStart = dcBitStart / 32ull;
                chunks.push_back({newStart, newStart});
                chunkSrcWordStart = newStart;
                chunkBitStart     = newStart * 32ull;
            }
            dc.indexBufferIdx = static_cast<uint32_t>(chunks.size() - 1);
            dc.indexOffset    = dcBitStart - chunkBitStart;        // local bit offset
            chunks.back().srcWordEnd = (dcBitEnd + 31ull) / 32ull; // ceil-div to whole words
        }
    }

    numIndexBuffers_ = static_cast<int>(chunks.size());
    if (numIndexBuffers_ > kMaxIndexBuffers) {
        std::fprintf(stderr, "Scene needs %d index buffers; max is %d. "
                             "Bump kMaxIndexBuffers.\n",
                     numIndexBuffers_, kMaxIndexBuffers);
        throw std::runtime_error("too many index chunks");
    }

    std::printf("Index buffer split: %d chunk%s (%s), sizes (GiB): ",
                numIndexBuffers_, numIndexBuffers_ == 1 ? "" : "s",
                scene.indicesPacked ? "bit-packed" : "uint32");
    const uint32_t* srcPtr = scene.indicesPacked ? scene.indicesBitstream.data()
                                                 : scene.indices.data();
    const size_t    srcWordCount = scene.indicesPacked ? scene.indicesBitstream.size()
                                                       : scene.indices.size();
    for (int c = 0; c < numIndexBuffers_; ++c) {
        uint64_t startWord = chunks[c].srcWordStart;
        uint64_t endWord   = std::min<uint64_t>(chunks[c].srcWordEnd, srcWordCount);
        // +1 slack word so cross-uint32 packed reads at chunk-end don't OOB.
        size_t bytes = size_t(endWord - startWord + 1) * sizeof(uint32_t);
        std::printf("%.2f%s", bytes / 1073741824.0,
                    c + 1 < numIndexBuffers_ ? ", " : "");

        // For the last chunk, the slack word is already in the source buffer
        // (we always assign(words+1, 0)). For middle chunks (multi-chunk
        // splits, very rare on M2 Max) we need a temp copy with a zero tail.
        const uint32_t* uploadPtr = srcPtr + startWord;
        std::vector<uint32_t> chunkData;
        const bool isLastChunk = (c + 1 == numIndexBuffers_);
        const bool srcHasSlack = (endWord + 1 <= srcWordCount);
        if (!(isLastChunk && srcHasSlack)) {
            chunkData.assign(size_t(endWord - startWord) + 1, 0u);
            std::memcpy(chunkData.data(), srcPtr + startWord,
                        size_t(endWord - startWord) * sizeof(uint32_t));
            uploadPtr = chunkData.data();
        }
        bufIndices_[c] = device_->newBuffer(uploadPtr, bytes,
                                            MTL::ResourceStorageModeShared);
        if (!bufIndices_[c]) {
            std::fprintf(stderr,
                "\nERROR: index chunk %d allocation failed (%.2f GiB).\n",
                c, bytes / 1073741824.0);
            throw std::runtime_error("index chunk alloc failed");
        }
    }
    std::printf("\n");
    // Index data is now in Metal — drop the host copy.
    std::vector<uint32_t>{}.swap(scene.indices);
    std::vector<uint32_t>{}.swap(scene.indicesBitstream);
    }   // end of host-vector index-upload block

index_upload_done:
    bufDrawCalls_ = device_->newBuffer(dcs.data(),
                                       dcs.size() * sizeof(DrawCall),
                                       MTL::ResourceStorageModeShared);

    // Debug: dump the first few draw calls to verify struct layout + scene sanity.
    int dump = std::min<int>(3, int(dcs.size()));
    std::printf("Sizeof(DrawCall) = %zu bytes (expected 80)\n", sizeof(DrawCall));
    for (int i = 0; i < dump; ++i) {
        const DrawCall& dc = dcs[i];
        std::printf("  DC %d: vertOff=%u triCount=%u idxOff=%llu (%s) bits/idx=%u "
                    "indexMin=%u inst=%u firstInst=%u idxBuf=%u "
                    "aabb=(%.2f,%.2f,%.2f)–(%.2f,%.2f,%.2f) cumStart=%llu\n",
                    i, dc.vertexOffset, dc.triangleCount,
                    (unsigned long long)dc.indexOffset,
                    dc.bitsPerIndex == 0 ? "u32 elems" : "bits",
                    dc.bitsPerIndex,
                    dc.indexMin,
                    dc.instanceCount, dc.firstInstance, dc.indexBufferIdx,
                    dc.aabbMinX, dc.aabbMinY, dc.aabbMinZ,
                    dc.aabbMinX + dc.compressionFactorX * 65535.0f,
                    dc.aabbMinY + dc.compressionFactorY * 65535.0f,
                    dc.aabbMinZ + dc.compressionFactorZ * 65535.0f,
                    (unsigned long long)dc.cumulativeTriangleStart);
        if (i == 0 && dc.firstInstance < scene.modelMatrices.size()) {
            simd_float4 t = scene.modelMatrices[dc.firstInstance].columns[3];
            std::printf("    -> first inst world translation: (%.2f, %.2f, %.2f)\n",
                        t.x, t.y, t.z);
        }
    }

    numDrawCalls_   = static_cast<uint32_t>(scene.drawCalls.size());
    numInstances_   = scene.totalInstances();
    totalTriangles_ = scene.totalTriangles();
    hasInstancing_  = false;
    for (auto const& dc : scene.drawCalls) {
        if (dc.instanceCount > 1) { hasInstancing_ = true; break; }
    }

    // Stash per-instance model matrices on host (used to rebuild worldView per frame).
    modelMatrices_ = scene.modelMatrices;

    // Per-instance worldView buffer: 64 B × numInstances, written by CPU each frame.
    bufWorldView_ = device_->newBuffer(size_t(numInstances_) * sizeof(simd_float4x4),
                                       MTL::ResourceStorageModeShared);

    // Per-instance modelMatrices buffer: immutable across frames; resolve uses it
    // to compute world-space face normals for shading.
    bufModelMatrices_ = device_->newBuffer(scene.modelMatrices.data(),
                                           size_t(numInstances_) * sizeof(simd_float4x4),
                                           MTL::ResourceStorageModeShared);

    // ---- UVs + textures ----
    // The UV pool sits parallel to the position pool (one float2 per
    // vertex). Textures upload via uploadTextures() which parses each
    // EncodedTexture's MRTC blob, uploads the raw ASTC blocks per mip,
    // and registers them in the resolve kernel's argument buffer.
    //
    // The resolve kernel ALWAYS declares these inputs (no function
    // constant gating), so we must always have non-null bindings even
    // for untextured scenes — otherwise the GPU dereferences null
    // pointers in the kernel signature and faults. ensureTextureBindings_
    // creates 1-byte placeholders + an empty arg buffer when the scene
    // has none.
    ensureTextureBindings_();
    if (!scene.uvs.empty()) {
        if (bufUVs_) bufUVs_->release();
        bufUVs_ = device_->newBuffer(scene.uvs.data(),
                                     scene.uvs.size() * sizeof(simd_float2),
                                     MTL::ResourceStorageModeShared);
    }
    if (!scene.textures.empty()) {
        uploadTextures(scene.textures);
    }

    rebuildPrefixBuckets();
}

// Build the texture-path resources (sampler, arg encoder, arg buffer)
// once per renderer. Idempotent — uploadTextures() registers per-image
// MTL::Textures into the existing arg buffer; ensureTextureBindings_ is
// just the lazy first-time-allocate path. The placeholder UV buffer is
// gone — the untextured PSO doesn't declare the uv parameter at all
// (USE_TEXTURING=false function-constant elides it from the kernel
// signature), so there's nothing to bind for untextured scenes.
//
// The textured PSO is built whether or not the scene has textures, because
// FunctionConstantValues are baked at PSO-creation. uploadTextures, when
// called, populates the arg encoder's slots. encodeResolve picks the
// textured PSO variant only when textures_ is non-empty.
void MetalRastRenderer::ensureTextureBindings_() {
    if (!samplerTrilinear_) {
        MTL::SamplerDescriptor* sd = MTL::SamplerDescriptor::alloc()->init();
        sd->setMinFilter(MTL::SamplerMinMagFilterLinear);
        sd->setMagFilter(MTL::SamplerMinMagFilterLinear);
        sd->setMipFilter(MTL::SamplerMipFilterLinear);
        sd->setMaxAnisotropy(8);
        sd->setSAddressMode(MTL::SamplerAddressModeRepeat);
        sd->setTAddressMode(MTL::SamplerAddressModeRepeat);
        samplerTrilinear_ = device_->newSamplerState(sd);
        sd->release();
    }
    if (!texArgEncoder_) {
        // Build from the textured PSO variant — the only one that declares
        // [[buffer(16)]] = TextureSet. Untextured PSO has it elided.
        MTL::Function* fn = nullptr;
        {
            bool useC = compressed_;
            bool hizFalse = false;
            bool pfxFalse = false;
            bool texTrue  = true;
            auto* fcv = MTL::FunctionConstantValues::alloc()->init();
            fcv->setConstantValue(&useC,     MTL::DataTypeBool, NS::UInteger(0));
            fcv->setConstantValue(&hizFalse, MTL::DataTypeBool, NS::UInteger(1));
            fcv->setConstantValue(&pfxFalse, MTL::DataTypeBool, NS::UInteger(2));
            fcv->setConstantValue(&texTrue,  MTL::DataTypeBool, NS::UInteger(3));
            NS::Error* err = nullptr;
            fn = library_->newFunction(nsstr("resolve"), fcv, &err);
            fcv->release();
            if (!fn) {
                throw std::runtime_error("ensureTextureBindings_: resolve(textured) kernel build failed");
            }
        }
        texArgEncoder_ = fn->newArgumentEncoder(16);
        fn->release();
        if (!texArgEncoder_) {
            throw std::runtime_error("ensureTextureBindings_: arg encoder build failed");
        }
        texArgBuffer_ = device_->newBuffer(texArgEncoder_->encodedLength(),
                                           MTL::ResourceStorageModeShared);
        if (!texArgBuffer_) {
            throw std::runtime_error("ensureTextureBindings_: arg buffer alloc failed");
        }
        // Zero the arg buffer so any unset slots resolve to "null texture".
        std::memset(texArgBuffer_->contents(), 0, texArgEncoder_->encodedLength());
        texArgEncoder_->setArgumentBuffer(texArgBuffer_, 0);
    }
}

void MetalRastRenderer::setCamera(const CameraUniforms& cam) {
    std::memcpy(bufCameraUni_->contents(), &cam, sizeof(CameraUniforms));

    // Recompute worldView = view * model PER INSTANCE (not per draw call).
    // For a Zorah-style scene with millions of instances this is a real CPU
    // cost — but matrix mul is SIMD-friendly and the loop is trivially
    // parallelizable if it ever becomes the bottleneck.
    //
    // In residency mode the ResidencyManager owns bufWorldView_ and rebuilds
    // it in prepareFrame (only for the instances of currently-resident-and-
    // visible meshes — typically far fewer than numInstances_). Skip the
    // write here; modelMatrices_ is empty in that mode and the borrowed
    // buffer's size doesn't match numInstances_.
    if (bufWorldView_ && !residencyMode_) {
        auto* dst = static_cast<simd_float4x4*>(bufWorldView_->contents());
        for (uint32_t i = 0; i < numInstances_; ++i) {
            simd_float4x4 wv = simd_mul(cam.viewMatrix, modelMatrices_[i]);
            // Pack the sign of det(wv 3x3) into wv[3][3]. The bottom row of
            // an affine worldView is [0,0,0,1] so wv[3][3] is always 1
            // post-affine; nothing in the shader reads it (all uses do
            // `(wv * float4(p, 1)).xyz` which discards .w). Stage1 picks
            // it up via `wv[3].w` to skip the per-(tri, instance) cross+
            // dot. Saves ~9 ops × 9.6M instanced-tris on Zorah.
            simd_float3 c0 = simd_make_float3(wv.columns[0]);
            simd_float3 c1 = simd_make_float3(wv.columns[1]);
            simd_float3 c2 = simd_make_float3(wv.columns[2]);
            float det      = simd_dot(simd_cross(c0, c1), c2);
            wv.columns[3].w = (det < 0.0f) ? -1.0f : 1.0f;
            dst[i] = wv;
        }
    }
}

// ----------- Per-stage encoders (factored so profiled path can reuse) -------

void MetalRastRenderer::encodeReset(MTL::CommandBuffer* cb) {
    MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
    blit->fillBuffer(bufBatchCounter_,  NS::Range::Make(0, sizeof(uint32_t)), 0);
    blit->fillBuffer(bufStage2Counter_, NS::Range::Make(0, sizeof(uint32_t)), 0);
    blit->fillBuffer(bufStage3Counter_, NS::Range::Make(0, sizeof(uint32_t)), 0);
    blit->endEncoding();
}

void MetalRastRenderer::encodeClear(MTL::ComputeCommandEncoder* enc) {
    const uint32_t pixelCount = cfg_.width * cfg_.height;
    enc->setComputePipelineState(psoClear_);
    enc->setBuffer(bufFramebuffer_, 0, 0);
    enc->setBytes(&pixelCount, sizeof(uint32_t), 1);
    const uint32_t TG = 256;
    // Kernel writes 2 ulongs (one ulong2 = 16B = AGX-native vector) per
    // thread → grid divides by 2.
    const uint32_t threads = (pixelCount + 1u) / 2u;
    MTL::Size grid = MTL::Size::Make((threads + TG - 1) / TG, 1, 1);
    MTL::Size tg   = MTL::Size::Make(TG, 1, 1);
    enc->dispatchThreadgroups(grid, tg);
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

// Bind index-buffer chunks on slots 1, 15, 16, 17. Slot 1 always carries
// chunk 0; the others are populated only if the scene needed splitting. Bind
// chunk 0 to all unused slots so the shader can read any slot without a
// nullptr (the dispatcher branches on dc.indexBufferIdx).
void MetalRastRenderer::bindIndexBuffers(MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(bufIndices_[0], 0, 1);
    enc->setBuffer(numIndexBuffers_ > 1 ? bufIndices_[1] : bufIndices_[0], 0, 15);
    enc->setBuffer(numIndexBuffers_ > 2 ? bufIndices_[2] : bufIndices_[0], 0, 16);
    enc->setBuffer(numIndexBuffers_ > 3 ? bufIndices_[3] : bufIndices_[0], 0, 17);
}

void MetalRastRenderer::encodeStage1(MTL::ComputeCommandEncoder* enc, MTL::Buffer* maskBits) {
    // Pick the kernel variant. Auto (default) matches CuRast's normal
    // behaviour: instanced kernel iff any draw call has > 1 instance.
    // The explicit modes mirror CuRast's "Visbuffer[indexed]" /
    // "Visbuffer[instanced]" radios.
    bool useInstanced = false;
    switch (rasterMode_) {
        case RasterMode::Auto:               useInstanced = hasInstancing_; break;
        case RasterMode::VisbufferIndexed:   useInstanced = false;          break;
        case RasterMode::VisbufferInstanced: useInstanced = true;           break;
    }
    // Pick stage1 pipeline. Hi-Z toggle decides which pair (mask vs no-mask)
    // and `useInstanced` picks within the pair. The non-Hiz pipelines have
    // the mask read elided at compile time → zero overhead when OFF.
    // When compactInstances_ is also on, swap to the compacted-instances
    // variant for the Hi-Z + instanced case (the only case where the
    // Hijma §6.2.2 packed array applies).
    const bool useCompacted = hiZEnabled_ && compactInstances_ &&
                              useInstanced && psoStage1InstancedHizCompacted_;
    MTL::ComputePipelineState* pso =
          hiZEnabled_
        ? (useInstanced
              ? (useCompacted ? psoStage1InstancedHizCompacted_
                              : psoStage1InstancedHiz_)
              : psoStage1Hiz_)
        : (useInstanced ? psoStage1Instanced_    : psoStage1_);
    enc->setComputePipelineState(pso);
    enc->setBuffer(bufVertices_,    0, 0);
    bindIndexBuffers(enc);
    enc->setBuffer(bufDrawCalls_,   0, 2);
    enc->setBytes (&numDrawCalls_,   sizeof(uint32_t), 3);
    {
        uint32_t totalTris32 = static_cast<uint32_t>(totalTriangles_);   // unused in kernel
        enc->setBytes(&totalTris32, sizeof(uint32_t), 4);
    }
    enc->setBuffer(bufCameraUni_,   0, 5);
    enc->setBuffer(bufFramebuffer_, 0, 6);
    enc->setBuffer(bufBatchCounter_,0, 7);
    enc->setBuffer(bufStage2Counter_,0, 8);
    enc->setBuffer(bufStage2Queue_, 0, 9);
    enc->setBytes (&cfg_.stage2Capacity, sizeof(uint32_t), 10);
    // Hi-Z visibility mask (per worldView slot). Caller can override which
    // mask to bind (two-phase pass 2 reads a different one). Default = the
    // primary visibleInstances. Init = all-1 so OFF path is a no-op.
    ensureVisibilityMask_();
    enc->setBuffer(maskBits ? maskBits : bufVisibleInstances_, 0, 11);
    enc->setBuffer(bufWorldView_,   0, 14);
    if (useCompacted) {
        // Compacted-instances variant reads slots 18 + 19. The compaction
        // dispatch in encodeFrame must have already populated these.
        enc->setBuffer(bufVisibleInstanceIdx_,    0, 18);
        enc->setBuffer(bufVisibleInstancesPerDC_, 0, 19);
    }
    MTL::Size grid = MTL::Size::Make(cfg_.stage1Threadgroups, 1, 1);
    MTL::Size tg   = MTL::Size::Make(256, 1, 1);
    enc->dispatchThreadgroups(grid, tg);
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

// Hijma §6.2.2 — pack visible instances per DC into bufVisibleInstanceIdx_.
// One TG per DC, 64 threads; reads `visBits` (the per-instance Hi-Z mask)
// and writes counts into bufVisibleInstancesPerDC_.
void MetalRastRenderer::encodeCompactInstances_(MTL::ComputeCommandEncoder* enc,
                                                MTL::Buffer* visBits)
{
    if (!psoCompactInstances_ || numDrawCalls_ == 0) return;
    ensureCompactionBuffers_();
    enc->setComputePipelineState(psoCompactInstances_);
    enc->setBuffer(bufDrawCalls_,             0, 0);
    enc->setBytes (&numDrawCalls_, sizeof(uint32_t), 1);
    enc->setBuffer(visBits ? visBits : bufVisibleInstances_, 0, 2);
    enc->setBuffer(bufVisibleInstanceIdx_,    0, 3);
    enc->setBuffer(bufVisibleInstancesPerDC_, 0, 4);
    MTL::Size grid = MTL::Size::Make(numDrawCalls_, 1, 1);
    MTL::Size tg   = MTL::Size::Make(64, 1, 1);
    enc->dispatchThreadgroups(grid, tg);
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalRastRenderer::encodeStage23(MTL::ComputeCommandEncoder* enc) {
    // prep_indirect for stage 2
    enc->setComputePipelineState(psoPrepareIndirect_);
    enc->setBuffer(bufStage2Counter_, 0, 0);
    enc->setBuffer(bufIndirectS2_,    0, 1);
    enc->setBytes (&cfg_.stage2Capacity, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    // stage 2
    enc->setComputePipelineState(psoStage2_);
    enc->setBuffer(bufVertices_,    0, 0);
    bindIndexBuffers(enc);
    enc->setBuffer(bufDrawCalls_,   0, 2);
    enc->setBytes (&numDrawCalls_,   sizeof(uint32_t), 3);
    enc->setBuffer(bufCameraUni_,   0, 5);
    enc->setBuffer(bufFramebuffer_, 0, 6);
    enc->setBuffer(bufStage2Queue_, 0, 9);
    enc->setBuffer(bufStage3Counter_,0, 11);
    enc->setBuffer(bufStage3Queue_, 0, 12);
    enc->setBytes (&cfg_.stage3Capacity, sizeof(uint32_t), 13);
    enc->setBuffer(bufWorldView_,   0, 14);
    enc->dispatchThreadgroups(bufIndirectS2_, 0, MTL::Size::Make(32,1,1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    // prep_indirect for stage 3
    enc->setComputePipelineState(psoPrepareIndirect_);
    enc->setBuffer(bufStage3Counter_, 0, 0);
    enc->setBuffer(bufIndirectS3_,    0, 1);
    enc->setBytes (&cfg_.stage3Capacity, sizeof(uint32_t), 2);
    enc->dispatchThreadgroups(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    // stage 3
    enc->setComputePipelineState(psoStage3_);
    enc->setBuffer(bufVertices_,    0, 0);
    bindIndexBuffers(enc);
    enc->setBuffer(bufDrawCalls_,   0, 2);
    enc->setBuffer(bufCameraUni_,   0, 5);
    enc->setBuffer(bufFramebuffer_, 0, 6);
    enc->setBuffer(bufStage3Queue_, 0, 12);
    enc->setBuffer(bufWorldView_,   0, 14);
    enc->dispatchThreadgroups(bufIndirectS3_, 0, MTL::Size::Make(64,1,1));
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalRastRenderer::encodeResolve(MTL::ComputeCommandEncoder* enc) {
    const bool usePfx = usePrefixSum_ && bufPfxBuckets_ && numDrawCalls_ > 0;
    // useTex picks the textured PSO variant (compiles in the bary-recompute
    // + perspective-UV + sample block) when the scene has at least one
    // valid texture. Untextured scenes (synthetic sphere, Zorah, residency-
    // mode geometry on this branch) get the plain lambert variant — no
    // UV/sampler/arg-buffer binds, no per-pixel branch overhead.
    const bool useTex = !textures_.empty();
    MTL::ComputePipelineState* pso =
        useTex ? (usePfx ? psoResolvePfxTex_ : psoResolveTex_)
               : (usePfx ? psoResolvePfx_    : psoResolve_);
    enc->setComputePipelineState(pso);
    enc->setBuffer(bufFramebuffer_,    0, 0);
    enc->setBuffer(bufVertices_,       0, 1);
    enc->setBuffer(bufIndices_[0],     0, 2);
    enc->setBuffer(bufDrawCalls_,      0, 3);
    enc->setBytes (&numDrawCalls_,      sizeof(uint32_t), 4);
    enc->setBuffer(bufCameraUni_,      0, 5);
    enc->setBuffer(bufModelMatrices_,  0, 6);
    enc->setBuffer(bufWorldView_,      0, 17);
    // Index buffer chunks 1/2/3 for resolve on slots 8/9/10 (chunk 0 is on slot 2).
    enc->setBuffer(numIndexBuffers_ > 1 ? bufIndices_[1] : bufIndices_[0], 0, 8);
    enc->setBuffer(numIndexBuffers_ > 2 ? bufIndices_[2] : bufIndices_[0], 0, 9);
    enc->setBuffer(numIndexBuffers_ > 3 ? bufIndices_[3] : bufIndices_[0], 0, 10);
    enc->setBytes (&visMode_, sizeof(uint32_t), 11);
    if (usePfx) {
        enc->setBuffer(bufPfxBuckets_,  0, 12);
        const uint32_t bucketCount = kPfxBucketCount;
        enc->setBytes (&bucketCount,     sizeof(uint32_t), 13);
        enc->setBytes (&pfxShift_,       sizeof(uint32_t), 14);
    }
    // UV pool + texture argument buffer + sampler. Only the textured PSO
    // variant declares these kernel parameters — for the untextured PSO
    // we skip the binds and the useResource loop entirely.
    if (useTex) {
        enc->setBuffer(bufUVs_,        0, 15);
        enc->setBuffer(texArgBuffer_,  0, 16);
        enc->setSamplerState(samplerTrilinear_, 0);
        // Single ObjC dispatch instead of N. The cached
        // `textureResourceList_` was built once at uploadTextures time,
        // skipping any null slots (failed encodes). Hijma §6.4.1 — the
        // CPU-side encode-time savings scale with texture count.
        if (!textureResourceList_.empty()) {
            enc->useResources(textureResourceList_.data(),
                              textureResourceList_.size(),
                              MTL::ResourceUsageRead);
        }
    }
    enc->setTexture(outputTex_, 0);
    const uint32_t TGX = 16, TGY = 16;
    MTL::Size grid = MTL::Size::Make((cfg_.width  + TGX - 1) / TGX,
                                     (cfg_.height + TGY - 1) / TGY, 1);
    MTL::Size tg   = MTL::Size::Make(TGX, TGY, 1);
    enc->dispatchThreadgroups(grid, tg);
}

void MetalRastRenderer::encodeFrame(MTL::CommandBuffer* cb) {
    const bool useTwoPhase = hiZEnabled_ && hiZTwoPhase_
                          && numInstances_ > 0 && numDrawCalls_ > 0;

    encodeReset(cb);

    // ---- Stage-grouping helper -----------------------------------------
    // When `tsSampleBuf_` is non-null (renderAndWaitWithCounters path), each
    // labeled stage gets ITS OWN compute encoder configured with a
    // ComputePassDescriptor.sampleBufferAttachments → the GPU samples a
    // timestamp at encoder start AND end (this is the only counter-sampling
    // point Apple Silicon supports — dispatch boundary is x86-only).
    //
    // When tsSampleBuf_ is null we use one shared encoder for all stages,
    // wrapping each in pushDebugGroup/popDebugGroup so GPU traces still show
    // a labeled stage tree.
    //
    // The helper hides this so the call sites (single-phase, two-phase) read
    // identically across the two modes.
    MTL::ComputeCommandEncoder* curEnc = nullptr;
    auto closeEnc = [&]{
        if (curEnc) { curEnc->endEncoding(); curEnc = nullptr; }
    };
    auto openSharedEnc = [&](const char* label) {
        closeEnc();
        curEnc = cb->computeCommandEncoder();
        curEnc->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    };
    auto group = [&](const char* name, auto fn) {
        if (tsSampleBuf_) {
            // Per-stage encoder — close any prior one, open a new pass with
            // start/end sample slots set.
            closeEnc();
            const uint32_t startIdx = tsNextIdx_++;
            const uint32_t endIdx   = tsNextIdx_++;
            tsStages_.push_back({name, startIdx, endIdx});
            MTL::ComputePassDescriptor* d = MTL::ComputePassDescriptor::computePassDescriptor();
            auto* attArr = d->sampleBufferAttachments();
            auto* att    = attArr->object(0);
            att->setSampleBuffer(tsSampleBuf_);
            att->setStartOfEncoderSampleIndex(startIdx);
            att->setEndOfEncoderSampleIndex(endIdx);
            curEnc = cb->computeCommandEncoder(d);
            curEnc->setLabel(NS::String::string(name, NS::UTF8StringEncoding));
            curEnc->pushDebugGroup(NS::String::string(name, NS::UTF8StringEncoding));
            fn(curEnc);
            curEnc->popDebugGroup();
            // Don't close yet — let next group()/closeEnc() do it.
        } else {
            curEnc->pushDebugGroup(NS::String::string(name, NS::UTF8StringEncoding));
            fn(curEnc);
            curEnc->popDebugGroup();
        }
    };

    const bool useCompacted = hiZEnabled_ && compactInstances_
                              && hasInstancing_ && psoCompactInstances_
                              && psoStage1InstancedHizCompacted_;

    if (!useTwoPhase) {
        // Single-phase (today's path).
        if (!tsSampleBuf_) openSharedEnc("MetalRast frame");
        group("Clear",       [&](auto* enc){ encodeClear (enc); });
        if (useCompacted) {
            group("Compact instances",
                  [&](auto* enc){ encodeCompactInstances_(enc, bufVisibleInstances_); });
        }
        group("Stage1",      [&](auto* enc){ encodeStage1(enc); });
        group("Stage2+3",    [&](auto* enc){ encodeStage23(enc); });
        group("Resolve",     [&](auto* enc){ encodeResolve(enc); });
        if (hiZEnabled_) {
            group("HiZ+Cull", [&](auto* enc){ encodeHiZAndCull_(enc); });
        }
        closeEnc();
        return;
    }

    // ---- Two-phase Hi-Z (Frostbite 2015 / Nanite 2021) -------------------
    // Phase 1: render with last-frame's Hi-Z mask (bufVisibleInstances_).
    //          Build current Hi-Z. Run cull in DIFF mode → write
    //          bufVisibleInstancesPhase2_ = (now-visible) AND NOT (mask-1).
    // Counter reset (blit).
    // Phase 2: render with phase2 mask (only newly-disoccluded instances).
    //          Build Hi-Z again (more geometry now). Run cull in ABSOLUTE
    //          mode → bufVisibleInstances_ = next frame's phase-1 mask.
    // Resolve.

    ensureVisibilityMask_();
    ensureHiZTexture_();
    rebuildInstanceMapping_();

    if (!tsSampleBuf_) openSharedEnc("MetalRast Phase 1");
    group("Clear",        [&](auto* enc){ encodeClear (enc); });
    if (useCompacted) {
        group("Compact instances (P1)",
              [&](auto* enc){ encodeCompactInstances_(enc, bufVisibleInstances_); });
    }
    group("Stage1 (P1)",  [&](auto* enc){ encodeStage1(enc, bufVisibleInstances_); });
    group("Stage2+3 (P1)",[&](auto* enc){ encodeStage23(enc); });
    if (hiZTex_ && bufVisibleInstancesPhase2_) {
        group("HiZ build (P1)", [&](auto* enc){ encodeBuildHiZ_(enc); });
        group("Cull DIFF (P1)",
              [&](auto* enc){ encodeCull_(enc, bufVisibleInstancesPhase2_,
                                          bufVisibleInstances_, /*phaseMode*/ 1u); });
    }
    closeEnc();

    // Reset counters (batchCounter, stage2Counter, stage3Counter) BUT NOT
    // the framebuffer — phase 2 atomic-mins onto what phase 1 wrote.
    encodeReset(cb);

    if (!tsSampleBuf_) openSharedEnc("MetalRast Phase 2");
    if (useCompacted) {
        group("Compact instances (P2)",
              [&](auto* enc){ encodeCompactInstances_(enc, bufVisibleInstancesPhase2_); });
    }
    group("Stage1 (P2)",   [&](auto* enc){ encodeStage1(enc, bufVisibleInstancesPhase2_); });
    group("Stage2+3 (P2)", [&](auto* enc){ encodeStage23(enc); });
    if (hiZTex_ && bufVisibleInstances_) {
        group("HiZ build (P2)", [&](auto* enc){ encodeBuildHiZ_(enc); });
        group("Cull ABS (P2)",
              [&](auto* enc){ encodeCull_(enc, bufVisibleInstances_,
                                          /*prev*/ nullptr, /*phaseMode*/ 0u); });
    }
    group("Resolve",       [&](auto* enc){ encodeResolve(enc); });
    closeEnc();
}

// =========================================================================
//   HI-Z OCCLUSION CULLING — host orchestration
// =========================================================================
//
// Pattern: at end of frame N we extract depth → build mip chain →
// AABB-cull each (DC, instance) pair → write 1-byte mask.
// Frame N+1's stage1 reads that mask and skips occluded instances.

void MetalRastRenderer::setHiZOcclusion(bool on) {
    if (on == hiZEnabled_) return;
    hiZEnabled_ = on;
    if (!on) {
        // Reset so stage1 next frame renders everything (last-frame mask is
        // stale and would orphan disoccluded geometry).
        if (bufVisibleInstances_) resetVisibilityMask_();
    }
}

// Visibility mask + slot→DC lookup. Bit-packed: one uint32 per 32 instances.
// Init = all-FF so stage1 with toggle OFF behaves identically to before.
// Called every frame from encodeStage1 — cheap when already allocated.
void MetalRastRenderer::ensureVisibilityMask_() {
    if (numInstances_ == 0) return;
    if (numInstances_ > instanceMaskCap_) {
        if (bufVisibleInstances_)       bufVisibleInstances_->release();
        if (bufVisibleInstancesPhase2_) bufVisibleInstancesPhase2_->release();
        if (bufInstanceToDrawCall_)     bufInstanceToDrawCall_->release();
        const size_t maskWords = (size_t(numInstances_) + 31u) / 32u;
        bufVisibleInstances_       = device_->newBuffer(maskWords * sizeof(uint32_t),
                                                        MTL::ResourceStorageModeShared);
        bufVisibleInstancesPhase2_ = device_->newBuffer(maskWords * sizeof(uint32_t),
                                                        MTL::ResourceStorageModeShared);
        bufInstanceToDrawCall_     = device_->newBuffer(numInstances_ * sizeof(uint32_t),
                                                        MTL::ResourceStorageModeShared);
        instanceMaskCap_ = numInstances_;
        // Phase-1 mask init = all-1 (no cull on the very first frame). Phase-2
        // mask init = all-0 so phase-2 stage1 is a no-op until cull populates
        // it.
        std::memset(bufVisibleInstances_->contents(),       0xFF, maskWords * sizeof(uint32_t));
        std::memset(bufVisibleInstancesPhase2_->contents(), 0x00, maskWords * sizeof(uint32_t));
    }
    // The slot→DC map only needs to be current when the cull kernel runs. We
    // rebuild it in encodeHiZAndCull_; here we only care about the mask.
}

// Compaction buffers (Hijma §6.2.2): one per-instance slot for the packed
// `visibleInstanceIdx[]`, plus one per-DC slot for `visibleInstancesPerDC[]`.
// Resized when numInstances_ / numDrawCalls_ grow. Private storage — only
// read by the GPU.
void MetalRastRenderer::ensureCompactionBuffers_() {
    if (numInstances_ == 0 || numDrawCalls_ == 0) return;
    const size_t idxBytes = size_t(numInstances_) * sizeof(uint32_t);
    const size_t pdcBytes = size_t(numDrawCalls_) * sizeof(uint32_t);
    if (!bufVisibleInstanceIdx_ ||
        bufVisibleInstanceIdx_->length() < idxBytes) {
        if (bufVisibleInstanceIdx_) bufVisibleInstanceIdx_->release();
        bufVisibleInstanceIdx_ = device_->newBuffer(idxBytes,
                                                    MTL::ResourceStorageModePrivate);
    }
    if (!bufVisibleInstancesPerDC_ ||
        bufVisibleInstancesPerDC_->length() < pdcBytes) {
        if (bufVisibleInstancesPerDC_) bufVisibleInstancesPerDC_->release();
        bufVisibleInstancesPerDC_ = device_->newBuffer(pdcBytes,
                                                       MTL::ResourceStorageModePrivate);
    }
}

// Hi-Z mip pyramid + per-mip views. Allocated lazily on first toggle-ON; sticks
// around so toggle-flips are cheap. ~33 MB at 4K — only paid when used.
void MetalRastRenderer::ensureHiZTexture_() {
    const uint32_t W = cfg_.width;
    const uint32_t H = cfg_.height;

    // Number of mips = floor(log2(max(W, H))) + 1.
    uint32_t maxDim = std::max(W, H);
    uint32_t mips   = 1;
    while ((maxDim >> mips) > 0) ++mips;

    if (hiZTex_ && hiZW_ == W && hiZH_ == H && hiZMips_ == mips) return;

    if (hiZTex_) { hiZTex_->release(); hiZTex_ = nullptr; }
    for (auto* v : hiZMipViews_) if (v) v->release();
    hiZMipViews_.clear();

    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR32Float, W, H, /*mipped*/ true);
    td->setMipmapLevelCount(mips);
    td->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    td->setStorageMode(MTL::StorageModePrivate);
    hiZTex_ = device_->newTexture(td);

    hiZMipViews_.resize(mips, nullptr);
    for (uint32_t m = 0; m < mips; ++m) {
        hiZMipViews_[m] = hiZTex_->newTextureView(
            MTL::PixelFormatR32Float,
            MTL::TextureType2D,
            NS::Range(m, 1),    // levels
            NS::Range(0, 1));   // slices
    }

    hiZW_ = W; hiZH_ = H; hiZMips_ = mips;
}

// Build the slot → drawCallIdx lookup. drawCalls is in shared memory both in
// uploadScene mode and residency mode (MeshRegistry uses StorageModeShared),
// so we read directly from .contents().
void MetalRastRenderer::rebuildInstanceMapping_() {
    if (!bufDrawCalls_ || !bufInstanceToDrawCall_) return;
    const auto* dcs = static_cast<const DrawCall*>(bufDrawCalls_->contents());
    auto*       map = static_cast<uint32_t*>(bufInstanceToDrawCall_->contents());
    // Initialize to UINT32_MAX so any unbound slot reads as "no DC" → marked
    // visible by the kernel (defensive; in practice every slot is covered).
    std::memset(map, 0xFF, numInstances_ * sizeof(uint32_t));
    for (uint32_t i = 0; i < numDrawCalls_; ++i) {
        const DrawCall& dc = dcs[i];
        uint32_t end = dc.firstInstance + dc.instanceCount;
        end = std::min(end, numInstances_);
        for (uint32_t s = dc.firstInstance; s < end; ++s) map[s] = i;
    }
}

void MetalRastRenderer::resetVisibilityMask_() {
    if (!bufVisibleInstances_ || !instanceMaskCap_) return;
    const size_t bytes = ((size_t(instanceMaskCap_) + 31u) / 32u) * sizeof(uint32_t);
    std::memset(bufVisibleInstances_->contents(), 0xFF, bytes);
}

// Single-phase end-of-frame Hi-Z build + cull (legacy entry; thin wrapper).
void MetalRastRenderer::encodeHiZAndCull_(MTL::ComputeCommandEncoder* enc) {
    if (numInstances_ == 0 || numDrawCalls_ == 0) return;
    ensureVisibilityMask_();
    ensureHiZTexture_();
    if (!hiZTex_ || !bufVisibleInstances_ || !bufInstanceToDrawCall_) return;
    rebuildInstanceMapping_();
    encodeBuildHiZ_(enc);
    encodeCull_(enc, bufVisibleInstances_, /*prev*/ nullptr, /*phaseMode*/ 0u);
}

// Build the Hi-Z mip pyramid (extract mip 0 from vis buffer + downsample).
// Caller is responsible for ensuring resources allocated.
void MetalRastRenderer::encodeBuildHiZ_(MTL::ComputeCommandEncoder* enc) {
    const uint32_t W = cfg_.width;
    const uint32_t H = cfg_.height;

    // ---- 1. Extract mip 0 from the visibility buffer.
    enc->setComputePipelineState(psoHiZExtract_);
    enc->setBuffer (bufFramebuffer_, 0, 0);
    enc->setBytes  (&W, sizeof(uint32_t), 1);
    enc->setBytes  (&H, sizeof(uint32_t), 2);
    enc->setTexture(hiZMipViews_[0], 0);
    {
        const uint32_t TG = 16;
        MTL::Size grid = MTL::Size::Make((W + TG - 1) / TG, (H + TG - 1) / TG, 1);
        MTL::Size tg   = MTL::Size::Make(TG, TG, 1);
        enc->dispatchThreadgroups(grid, tg);
    }

    // ---- 2. Build mip chain. Walk pairs of mips per dispatch (2-mip kernel)
    // and fall back to the 1-mip kernel if there's an odd remainder. Halves
    // the dispatch count on Apple Silicon TBDR, where each compute dispatch
    // pays a non-trivial fence/encoder cost.
    uint32_t m = 1;
    while (m + 1 < hiZMips_) {
        enc->setComputePipelineState(psoHiZDownsample2_);
        enc->setTexture(hiZMipViews_[m - 1], 0);     // src   (mip m-1)
        enc->setTexture(hiZMipViews_[m],     1);     // dst0  (mip m)
        enc->setTexture(hiZMipViews_[m + 1], 2);     // dst1  (mip m+1)
        uint32_t mw = std::max(1u, W >> m);          // dst0 size
        uint32_t mh = std::max(1u, H >> m);
        const uint32_t TG = 8;
        MTL::Size grid = MTL::Size::Make((mw + TG - 1) / TG, (mh + TG - 1) / TG, 1);
        MTL::Size tg   = MTL::Size::Make(TG, TG, 1);
        enc->dispatchThreadgroups(grid, tg);
        m += 2;
    }
    if (m < hiZMips_) {
        enc->setComputePipelineState(psoHiZDownsample_);
        enc->setTexture(hiZMipViews_[m - 1], 0);
        enc->setTexture(hiZMipViews_[m],     1);
        uint32_t mw = std::max(1u, W >> m);
        uint32_t mh = std::max(1u, H >> m);
        const uint32_t TG = 16;
        MTL::Size grid = MTL::Size::Make((mw + TG - 1) / TG, (mh + TG - 1) / TG, 1);
        MTL::Size tg   = MTL::Size::Make(TG, TG, 1);
        enc->dispatchThreadgroups(grid, tg);
    }

}

// AABB cull → per-instance bit-packed mask. Single-phase callers pass
// phaseMode=0 and prevBits=nullptr. Two-phase phase-1→phase-2 hand-off
// passes phaseMode=1 and prevBits=mask phase 1 just consumed; the kernel
// emits ONLY the newly-disoccluded delta.
//
// TG=32 = one Apple-Silicon SIMD group, so simd_ballot inside the kernel
// packs 32 contiguous slots into one uint32 word (lane 0 commits it).
void MetalRastRenderer::encodeCull_(MTL::ComputeCommandEncoder* enc,
                                    MTL::Buffer* outBits,
                                    MTL::Buffer* prevBits,
                                    uint32_t     phaseMode)
{
    enc->setComputePipelineState(psoCullInstances_);
    enc->setBuffer (bufDrawCalls_,           0, 0);
    enc->setBytes  (&numDrawCalls_, sizeof(uint32_t), 1);
    enc->setBuffer (bufWorldView_,           0, 2);
    enc->setBytes  (&numInstances_, sizeof(uint32_t), 3);
    enc->setBuffer (bufInstanceToDrawCall_,  0, 4);
    enc->setBuffer (bufCameraUni_,           0, 5);
    enc->setBuffer (outBits,                 0, 6);
    enc->setBytes  (&hiZMips_, sizeof(uint32_t), 7);
    enc->setBytes  (&phaseMode, sizeof(uint32_t), 8);
    // prevBits is read only when phaseMode==1; bind whatever's safe otherwise
    // so Metal validation doesn't complain about a missing binding.
    enc->setBuffer (prevBits ? prevBits : bufVisibleInstances_, 0, 9);
    enc->setTexture(hiZTex_, 0);
    const uint32_t TG = 32;
    MTL::Size grid = MTL::Size::Make((numInstances_ + TG - 1) / TG, 1, 1);
    MTL::Size tg   = MTL::Size::Make(TG, 1, 1);
    enc->dispatchThreadgroups(grid, tg);
}

FrameStats MetalRastRenderer::renderAndWait() {
    if (!bufVertices_) throw std::runtime_error("renderAndWait: no scene uploaded");

    PoolScope pool;

    MTL::CommandBuffer* cb = queue_->commandBuffer();
    cb->retain();   // we'll inspect timing after waitUntilCompleted

    encodeFrame(cb);
    cb->commit();
    cb->waitUntilCompleted();

    if (cb->status() == MTL::CommandBufferStatusError) {
        NS::Error* err = cb->error();
        std::fprintf(stderr,
            "ERROR: command buffer failed: %s\n",
            err ? err->localizedDescription()->utf8String() : "(no description)");
    }

    FrameStats st{};
    st.gpuMilliseconds = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    st.stage2Items = *static_cast<const uint32_t*>(bufStage2Counter_->contents());
    st.stage3Items = *static_cast<const uint32_t*>(bufStage3Counter_->contents());

    cb->release();
    return st;
}

FrameStats MetalRastRenderer::renderAndWaitProfiled() {
    if (!bufVertices_) throw std::runtime_error("renderAndWait: no scene uploaded");

    PoolScope pool;

    auto runPass = [&](auto encodeFn) -> double {
        MTL::CommandBuffer* cb = queue_->commandBuffer();
        cb->retain();
        encodeFn(cb);
        cb->commit();
        cb->waitUntilCompleted();
        double ms = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        cb->release();
        return ms;
    };

    FrameStats st{};

    // Counter reset is included in the clear measurement.
    st.clearMs = runPass([&](MTL::CommandBuffer* cb){
        encodeReset(cb);
        auto* enc = cb->computeCommandEncoder();
        encodeClear(enc);
        enc->endEncoding();
    });

    st.stage1Ms = runPass([&](MTL::CommandBuffer* cb){
        auto* enc = cb->computeCommandEncoder();
        encodeStage1(enc);
        enc->endEncoding();
    });

    st.stage23Ms = runPass([&](MTL::CommandBuffer* cb){
        auto* enc = cb->computeCommandEncoder();
        encodeStage23(enc);
        enc->endEncoding();
    });

    st.resolveMs = runPass([&](MTL::CommandBuffer* cb){
        auto* enc = cb->computeCommandEncoder();
        encodeResolve(enc);
        if (hiZEnabled_) encodeHiZAndCull_(enc);
        enc->endEncoding();
    });

    st.gpuMilliseconds = st.clearMs + st.stage1Ms + st.stage23Ms + st.resolveMs;
    st.stage2Items = *static_cast<const uint32_t*>(bufStage2Counter_->contents());
    st.stage3Items = *static_cast<const uint32_t*>(bufStage3Counter_->contents());
    return st;
}

// =========================================================================
//   MTLCounterSampleBuffer-based per-stage timing
// =========================================================================
//
// Pattern: encodeFrame's `group` helper opens a fresh compute encoder per
// labeled stage, attaches a ComputePassDescriptor.sampleBufferAttachments
// with start/end sample slots, and the GPU writes a timestamp at encoder
// start AND end. After waitUntilCompleted we resolve the buffer, walk the
// (start, end) pairs, and convert raw GPU ticks → microseconds via a fresh
// CPU/GPU timestamp pair. Apple Silicon supports stage-boundary sampling
// only (dispatch-boundary is x86-only); per-encoder split is the cost of
// truthful per-stage timing on M-series.

bool MetalRastRenderer::renderAndWaitWithCounters(
    MTL::CounterSampleBuffer* tsBuf,
    CounterReport& report)
{
    report.stages.clear();
    if (!bufVertices_) throw std::runtime_error("renderAndWaitWithCounters: no scene uploaded");
    if (!tsBuf) {
        // Caller asked for counter sampling but didn't supply a buffer.
        FrameStats st = renderAndWait();
        report.gpuTotalMs  = st.gpuMilliseconds;
        report.stage2Items = st.stage2Items;
        report.stage3Items = st.stage3Items;
        return false;
    }

    PoolScope pool;

    // Take a CPU/GPU timestamp pair around the dispatch so we can convert
    // raw GPU-clock ticks into nanoseconds. Apple recommends sampling close
    // to the work for best accuracy.
    MTL::Timestamp cpuT0 = 0, gpuT0 = 0;
    device_->sampleTimestamps(&cpuT0, &gpuT0);

    tsSampleBuf_ = tsBuf;
    tsNextIdx_   = 0;
    tsStages_.clear();

    MTL::CommandBuffer* cb = queue_->commandBuffer();
    cb->retain();
    encodeFrame(cb);
    cb->commit();
    cb->waitUntilCompleted();

    // Disable sampling for any subsequent encodeFrame calls.
    MTL::CounterSampleBuffer* usedBuf = tsSampleBuf_;
    tsSampleBuf_ = nullptr;

    if (cb->status() == MTL::CommandBufferStatusError) {
        NS::Error* err = cb->error();
        std::fprintf(stderr,
            "ERROR: command buffer failed (counter pass): %s\n",
            err ? err->localizedDescription()->utf8String() : "(no description)");
    }

    report.gpuTotalMs  = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    report.stage2Items = *static_cast<const uint32_t*>(bufStage2Counter_->contents());
    report.stage3Items = *static_cast<const uint32_t*>(bufStage3Counter_->contents());
    cb->release();

    // Take a second pair to compute the GPU→ns ratio.
    MTL::Timestamp cpuT1 = 0, gpuT1 = 0;
    device_->sampleTimestamps(&cpuT1, &gpuT1);
    const double cpuDt = double(cpuT1 - cpuT0);             // CPU is in ns
    const double gpuDt = double(gpuT1 - gpuT0);             // GPU is opaque
    const double gpuTickToNs = (gpuDt > 0.0) ? (cpuDt / gpuDt) : 1.0;

    if (tsNextIdx_ == 0) return false;

    NS::Range range = NS::Range::Make(0, tsNextIdx_);
    NS::Data* data  = usedBuf->resolveCounterRange(range);
    if (!data || data->length() < tsNextIdx_ * sizeof(MTL::CounterResultTimestamp)) {
        // Driver couldn't resolve (possibly because boundaries weren't
        // supported on this device). Caller falls back to gpuTotalMs only.
        return false;
    }
    const auto* samples =
        static_cast<const MTL::CounterResultTimestamp*>(data->mutableBytes());

    report.stages.reserve(tsStages_.size());
    for (auto& s : tsStages_) {
        if (s.startIdx >= tsNextIdx_ || s.endIdx >= tsNextIdx_) continue;
        const uint64_t t0 = samples[s.startIdx].timestamp;
        const uint64_t t1 = samples[s.endIdx  ].timestamp;
        // CounterErrorValue (~0) marks an unresolvable slot — skip those.
        if (t0 == ~0ull || t1 == ~0ull) continue;
        const double ns = (t1 > t0) ? double(t1 - t0) * gpuTickToNs : 0.0;
        report.stages.push_back({s.label, ns / 1000.0});
    }
    return true;
}

std::vector<uint8_t> MetalRastRenderer::readbackRGBA() const {
    const uint32_t W = cfg_.width;
    const uint32_t H = cfg_.height;
    std::vector<uint8_t> out(size_t(W) * H * 4);
    outputTex_->getBytes(out.data(),
                         /*bytesPerRow*/ size_t(W) * 4,
                         MTL::Region::Make2D(0, 0, W, H),
                         /*level*/ 0);
    return out;
}

// ---- Interactive (compose with caller's command buffer) -----------------------

void MetalRastRenderer::encodeRender(MTL::CommandBuffer* cb) {
    if (!bufVertices_) throw std::runtime_error("encodeRender: no scene uploaded");
    encodeFrame(cb);
}

void MetalRastRenderer::encodePresent(MTL::CommandBuffer* cb, MTL::Texture* dst) {
    MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::alloc()->init();
    auto* att = rpd->colorAttachments()->object(0);
    att->setTexture(dst);
    att->setLoadAction(MTL::LoadActionClear);
    att->setStoreAction(MTL::StoreActionStore);
    att->setClearColor(MTL::ClearColor::Make(0, 0, 0, 1));

    MTL::RenderCommandEncoder* enc = cb->renderCommandEncoder(rpd);
    enc->setRenderPipelineState(psoPresent_);
    enc->setFragmentTexture(outputTex_, 0);
    enc->setFragmentSamplerState(presentSampler_, 0);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    enc->endEncoding();

    rpd->release();
}

}  // namespace metalrast
