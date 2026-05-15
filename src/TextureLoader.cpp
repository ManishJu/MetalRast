#include "TextureLoader.h"

#include "ThreadPool.h"

#include <condition_variable>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <thread>
#include <vector>
#include <unistd.h>   // isatty

// MRTC layout numbers are little-endian on disk. Apple Silicon is also
// little-endian, so the codec is a single memcpy. Big-endian targets would
// need byte-swapping here — caught at compile time.
static_assert(std::endian::native == std::endian::little,
              "MRTC codec assumes a little-endian host (Apple Silicon, x86)");

// stb_image — single-header JPEG/PNG decoder. We're the only consumer on
// this branch (main.cpp owns stb_image_write), so define the implementation
// here.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// stb_image_resize2 — single-header resampler. Used to build the mip
// pyramid before each ASTC encode. We need its sRGB-aware path so that
// downsamples are gamma-correct (downsample in linear, re-encode to sRGB).
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include "cgltf.h"

#include "astcenc.h"

namespace metalrast {

namespace {

// Return a non-owning view into a cgltf_image's encoded bytes. The cgltf
// buffer is mmapped/owned by cgltf_data and outlives all texture-encode
// work, so the span is valid for the whole encodeAll. Empty span on any
// failure → caller skips the image (lambert fallback).
//
// Avoids copying ~64 MiB per 8K-PNG-embedded image — komainu's 9 textures
// previously paid ~580 MiB of memcpy per scene load, even on cache hits
// (we read the bytes to compute the FNV cache key and the stbi_info dims).
std::span<const uint8_t> readImageBytes(const cgltf_data* data,
                                        const cgltf_image* img) {
    if (!img) return {};

    if (img->buffer_view && img->buffer_view->buffer
        && img->buffer_view->buffer->data) {
        const auto* bv = img->buffer_view;
        const auto* base = static_cast<const uint8_t*>(bv->buffer->data);
        // Defensive bounds check — cgltf already validates this in its parser
        // (third_party/cgltf.h `cgltf_validate`), but a future loader change
        // or a custom buffer override could land here without it.
        if (bv->buffer->size < bv->offset + bv->size) return {};
        return { base + bv->offset, bv->size };
    }

    (void)data;
    return {};
}

// ---------- MRTC container -------------------------------------------------
//
// Tiny custom format for our texture cache. Self-contained: a single
// .mrtc file holds a full ASTC mip chain plus enough metadata to re-
// allocate a Metal texture and replaceRegion() each level. Layout:
//
//   uint32  magic         'MRTC' (0x4354524D little-endian)
//   uint32  version       1
//   uint32  blockX        4 or 6
//   uint32  blockY        4 or 6
//   uint8   srgb          0/1
//   uint8   hasAlpha      0/1 (informational)
//   uint8   reserved[2]
//   uint32  width         level-0 width  (texels)
//   uint32  height        level-0 height (texels)
//   uint32  mipCount
//   uint32  mipSize[mipCount]   bytes per mip, in order largest→smallest
//   uint8   data[Σ mipSize]     raw ASTC blocks, concatenated
//
// Header is 32 bytes + 4·mipCount. ASTC LDR blocks are always 16 B regardless
// of block dimensions (4×4 / 6×6 / etc), so per-mip sizes are derivable, but
// we store them explicitly for safety against future block-format additions.

constexpr uint32_t kMrtcMagic   = 0x4354524Du;   // 'MRTC' (LE: M,R,T,C)
constexpr uint32_t kMrtcVersion = 1u;
constexpr size_t   kMrtcMinHeader = 32u;

inline void put32_le(std::vector<uint8_t>& out, uint32_t v) {
    const size_t off = out.size();
    out.resize(off + 4);
    std::memcpy(out.data() + off, &v, 4);
}
inline uint32_t rd32_le(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

std::vector<uint8_t> packMrtc(uint32_t blockX, uint32_t blockY,
                              bool srgb, bool hasAlpha,
                              uint32_t width, uint32_t height,
                              const std::vector<std::vector<uint8_t>>& mips)
{
    const uint32_t mipCount = uint32_t(mips.size());
    size_t totalMipBytes = 0;
    for (auto const& m : mips) totalMipBytes += m.size();

    std::vector<uint8_t> out;
    out.reserve(kMrtcMinHeader + 4 * size_t(mipCount) + totalMipBytes);

    put32_le(out, kMrtcMagic);
    put32_le(out, kMrtcVersion);
    put32_le(out, blockX);
    put32_le(out, blockY);
    out.push_back(srgb ? 1 : 0);
    out.push_back(hasAlpha ? 1 : 0);
    out.push_back(0);
    out.push_back(0);
    put32_le(out, width);
    put32_le(out, height);
    put32_le(out, mipCount);

    for (auto const& m : mips) put32_le(out, uint32_t(m.size()));
    for (auto const& m : mips) out.insert(out.end(), m.begin(), m.end());
    return out;
}

bool readMrtcDims(const std::vector<uint8_t>& bytes,
                  uint32_t& outW, uint32_t& outH) {
    if (bytes.size() < kMrtcMinHeader) return false;
    if (rd32_le(bytes.data()) != kMrtcMagic) return false;
    if (rd32_le(bytes.data() + 4) != kMrtcVersion) return false;
    outW = rd32_le(bytes.data() + 20);
    outH = rd32_le(bytes.data() + 24);
    return outW > 0 && outH > 0;
}

// ---------- Disk cache for encoded MRTC blobs ------------------------------

// Bumping this string invalidates every cached entry. Do this whenever the
// encoder library is updated, the encode parameter set changes, or the
// MRTC layout itself shifts. The hash key embeds this verbatim, so old
// `<hash>.mrtc` files become orphans (different filename, ignored).
constexpr const char* kCacheVersion = "MR_TC_v3__astcenc_4_8_fast_unorm8";

inline uint64_t fnv1a_64(const uint8_t* data, size_t n,
                         uint64_t h = 0xcbf29ce484222325ull) {
    constexpr uint64_t prime = 0x100000001b3ull;
    for (size_t i = 0; i < n; ++i) { h ^= data[i]; h *= prime; }
    return h;
}

uint64_t makeCacheKey(std::span<const uint8_t> src,
                      AstcBlock block, int quality, bool srgb) {
    uint64_t h = fnv1a_64(src.data(), src.size());
    uint8_t meta[3] = { uint8_t(block),
                        uint8_t(quality),
                        uint8_t(srgb ? 1 : 0) };
    h = fnv1a_64(meta, sizeof(meta), h);
    h = fnv1a_64(reinterpret_cast<const uint8_t*>(kCacheVersion),
                 std::strlen(kCacheVersion), h);
    return h;
}

std::filesystem::path cachePath(const std::string& dir, uint64_t key) {
    char name[24];
    std::snprintf(name, sizeof(name), "%016llx.mrtc",
                  (unsigned long long)key);
    return std::filesystem::path(dir) / name;
}

std::vector<uint8_t> readCacheFile(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    auto size = std::streamoff(f.tellg());
    if (size <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size_t{size_t(size)});
    f.read(reinterpret_cast<char*>(buf.data()), size);
    if (f.gcount() != size) return {};
    return buf;
}

// Atomic write: stage to <path>.tmp.<pid>.<ctr>, then rename. Crash-safe on POSIX.
// The .tmp suffix MUST be unique per writer — earlier versions used a fixed
// `.tmp`, which two concurrent metalrast processes encoding the same image
// would race on, with the loser's partial file overwriting the winner's
// completed entry. PID + per-process atomic counter disambiguates safely.
bool writeCacheFile(const std::filesystem::path& path,
                    const std::vector<uint8_t>& bytes) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return false;

    static std::atomic<uint64_t> tmpCtr{0};
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), ".tmp.%d.%llu",
                  int(::getpid()),
                  (unsigned long long)tmpCtr.fetch_add(1, std::memory_order_relaxed));
    auto tmp = path;
    tmp += suffix;
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(bytes.data()),
                std::streamsize(bytes.size()));
        if (!f.good()) { std::filesystem::remove(tmp); return false; }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(tmp); return false; }
    return true;
}

// ---------- ASTC encode ----------------------------------------------------

// Map our coarse 0..2 quality knob to one of astcenc's preset effort levels.
// 0 = FAST  (default — ~1-2s per 8K mip, ~30 dB PSNR on natural images)
// 1 = MEDIUM
// 2 = THOROUGH (slow, only worth it for hero textures with no cache)
inline float qualityPreset(int q) {
    switch (q) {
    case 0: default: return ASTCENC_PRE_FAST;
    case 1:          return ASTCENC_PRE_MEDIUM;
    case 2:          return ASTCENC_PRE_THOROUGH;
    }
}

// Generate mips 1..N-1 from level-0 RGBA8 pixels. Mip 0 is NOT copied — the
// caller already owns the source buffer and passes a pointer in to encodeOne.
// ASTC's block grid handles sub-block-size mips internally so we go all the
// way down to 1×1. sRGB-aware downsample for color textures.
//
// Allocation strategy: one contiguous arena holds all sub-mips back-to-back
// (vs. one std::vector per mip). For an 8K source the geometric series of
// sub-mip sizes sums to ~W·H·4 / 3 ≈ 85 MiB; we malloc that once instead of
// ~13 separate allocations. The downsample loop reads from mip N-1 and
// writes to mip N — both inside the same arena, contiguous, so the L2
// prefetcher stays warm.
//
// Returned spans point into `arena`; the arena MUST outlive them.
struct SubMipChain {
    std::vector<uint8_t>      arena;       // owns all sub-mip pixels
    struct Mip {
        const uint8_t* data;
        uint32_t       w, h;
        size_t         byteSize;
    };
    std::vector<Mip>          mips;        // size = mipCount - 1; mip index = m-1
};

SubMipChain buildSubMipChain(const uint8_t* px0,
                             uint32_t W, uint32_t H,
                             bool srgb)
{
    uint32_t mipCount = 1;
    {
        uint32_t w = W, h = H;
        while (w > 1u || h > 1u) {
            w = std::max(1u, w >> 1);
            h = std::max(1u, h >> 1);
            ++mipCount;
        }
    }
    if (mipCount <= 1) return {};

    // Closed-form total size for sub-mips. Compute exactly (ceil-of-half on
    // both axes) rather than the asymptotic W·H·4/3, since odd dims at
    // various levels affect the sum.
    size_t totalBytes = 0;
    for (uint32_t m = 1; m < mipCount; ++m) {
        const uint32_t mw = std::max(1u, W >> m);
        const uint32_t mh = std::max(1u, H >> m);
        totalBytes += size_t(mw) * size_t(mh) * 4u;
    }

    SubMipChain out;
    out.arena.resize(totalBytes);
    out.mips.reserve(mipCount - 1);

    size_t cursor = 0;
    for (uint32_t m = 1; m < mipCount; ++m) {
        const uint32_t prevW = std::max(1u, W >> (m - 1));
        const uint32_t prevH = std::max(1u, H >> (m - 1));
        const uint32_t curW  = std::max(1u, prevW >> 1);
        const uint32_t curH  = std::max(1u, prevH >> 1);
        const size_t curBytes = size_t(curW) * size_t(curH) * 4u;
        uint8_t* curDst = out.arena.data() + cursor;
        const uint8_t* prevPx = (m == 1) ? px0 : (out.mips.back().data);

        if (srgb) {
            stbir_resize_uint8_srgb(
                prevPx, int(prevW), int(prevH), 0,
                curDst, int(curW),  int(curH),  0,
                STBIR_RGBA);
        } else {
            stbir_resize_uint8_linear(
                prevPx, int(prevW), int(prevH), 0,
                curDst, int(curW),  int(curH),  0,
                STBIR_RGBA);
        }

        out.mips.push_back({ curDst, curW, curH, curBytes });
        cursor += curBytes;
    }
    return out;
}

// Barrier-based worker pool for astcenc's cooperative multi-thread encode.
//
// The general-purpose ThreadPool in ThreadPool.h can't be used here: astcenc
// has internal barriers that require *exactly N* concurrent participants
// each with a unique `thread_index` in [0, N) per call. The general pool's
// enqueue-based design lets one worker grab multiple tasks before another
// worker grabs its first, which would either (a) call astcenc with a
// duplicate thread_index (corruption) or (b) leave the encoder's barriers
// waiting for a missing participant (deadlock).
//
// AstcWorkerPool pre-spawns N-1 worker threads once per encodeAll and
// signals them per-mip via a condition variable. Each worker has a fixed
// `thread_index` for its entire lifetime, so astcenc's barriers always see
// the same N participants. Avoids ~1400 std::thread::thread spawns over a
// typical scene encode (9 textures × ~13 mips × ~11 workers).
class AstcWorkerPool {
public:
    using Fn = std::function<int(int ti)>;     // returns astcenc_error code

    explicit AstcWorkerPool(unsigned threadCount)
        : threadCount_(std::max(1u, threadCount))
    {
        if (threadCount_ <= 1) return;
        threads_.reserve(threadCount_ - 1);
        for (unsigned ti = 1; ti < threadCount_; ++ti) {
            threads_.emplace_back([this, ti] { workerLoop(int(ti)); });
        }
    }

    ~AstcWorkerPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++epoch_;
        }
        startCv_.notify_all();
        for (auto& t : threads_) t.join();
    }

    // Run `fn(ti)` on each ti in [0, threadCount_). The calling thread runs
    // ti=0; pool workers run ti=1..N-1. Returns the first non-zero error
    // seen (or 0 on success). Blocks until every participant has finished.
    int run(Fn fn) {
        if (threadCount_ == 1u) return fn(0);

        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_         = std::move(fn);
            ++epoch_;
            doneCount_  = 0;
            failure_    = 0;
        }
        startCv_.notify_all();

        // Calling thread participates with ti=0.
        int e0 = fn_(0);

        std::unique_lock<std::mutex> lk(mu_);
        if (e0 != 0 && failure_ == 0) failure_ = e0;
        ++doneCount_;
        if (doneCount_ == threadCount_) doneCv_.notify_one();
        doneCv_.wait(lk, [this]{ return doneCount_ == threadCount_; });
        return failure_;
    }

    unsigned threadCount() const { return threadCount_; }

private:
    void workerLoop(int ti) {
        uint64_t localEpoch = 0;
        for (;;) {
            Fn fn;
            {
                std::unique_lock<std::mutex> lk(mu_);
                startCv_.wait(lk, [&]{ return stop_ || epoch_ > localEpoch; });
                if (stop_) return;
                localEpoch = epoch_;
                fn = fn_;
            }
            int e = fn(ti);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (e != 0 && failure_ == 0) failure_ = e;
                ++doneCount_;
                if (doneCount_ == threadCount_) doneCv_.notify_all();
            }
        }
    }

    unsigned                       threadCount_;
    std::vector<std::thread>       threads_;
    std::mutex                     mu_;
    std::condition_variable        startCv_;
    std::condition_variable        doneCv_;
    Fn                             fn_;
    uint64_t                       epoch_     = 0;     // bumped per dispatch
    unsigned                       doneCount_ = 0;
    int                            failure_   = 0;
    bool                           stop_      = false;
};

// Pre-built astcenc context — one per (block size, srgb, quality) combo.
// Owned by encodeAll, lent to encodeOne by const reference. Created lazily
// when a matching image first needs it.
struct AstcCtx {
    astcenc_context* ctx = nullptr;
    uint32_t         threadCount = 0;
    uint32_t         block = 0;     // 4 or 6
    bool             srgb = true;
    int              quality = 0;
};

// Cache of contexts keyed by (block, srgb, quality). Up to 4 entries in
// practice (2 blocks × 1 quality × srgb-only), so a flat vector is fine.
struct AstcCtxCache {
    std::vector<AstcCtx> entries;
    unsigned threadCount = 1u;

    AstcCtx* get(uint32_t block, bool srgb, int quality) {
        for (auto& e : entries) {
            if (e.block == block && e.srgb == srgb && e.quality == quality)
                return &e;
        }
        // Build a new one. astcenc rebuilds internal LUTs here (~5 ms);
        // amortized across all images that share these parameters.
        astcenc_config cfg{};
        auto cfgErr = astcenc_config_init(
            srgb ? ASTCENC_PRF_LDR_SRGB : ASTCENC_PRF_LDR,
            block, block, /*block_z=*/1,
            qualityPreset(quality),
            ASTCENC_FLG_USE_DECODE_UNORM8,
            &cfg);
        if (cfgErr != ASTCENC_SUCCESS) {
            std::fprintf(stderr, "astcenc_config_init failed (%d)\n", int(cfgErr));
            return nullptr;
        }
        astcenc_context* ctx = nullptr;
        auto ctxErr = astcenc_context_alloc(&cfg, threadCount, &ctx, nullptr);
        if (ctxErr != ASTCENC_SUCCESS || !ctx) {
            std::fprintf(stderr,
                "astcenc_context_alloc failed (%d)\n", int(ctxErr));
            return nullptr;
        }
        entries.push_back({ ctx, threadCount, block, srgb, quality });
        return &entries.back();
    }

    ~AstcCtxCache() {
        for (auto& e : entries) {
            if (e.ctx) astcenc_context_free(e.ctx);
        }
    }
};

// Encode a single image's mip chain as ASTC blocks and pack it into MRTC.
// Returns empty vector on failure. `ctx` is borrowed from AstcCtxCache and
// must match the (block, srgb, quality) tuple. We `astcenc_compress_reset`
// between mips (and at the end, leaving the context ready for the next
// image).
//
// Threading: astcenc supports cooperative multi-threaded encode where N
// threads each call astcenc_compress_image() with a unique thread_index in
// [0, N). We spawn N-1 std::threads + run on the calling thread = N total
// participants. The std::thread spawn cost (~50 µs each) is negligible
// against any sane mip's encode time.
std::vector<uint8_t> encodeOne(const uint8_t* rgba, uint32_t W, uint32_t H,
                               bool srgb, AstcBlock block, bool hasAlpha,
                               const AstcCtx& cached,
                               AstcWorkerPool& pool)
{
    if (!rgba || W == 0 || H == 0) return {};

    const uint32_t bx = (block == AstcBlock::B6x6) ? 6u : 4u;
    const uint32_t by = bx;

    // Mip pyramid (CPU side, sRGB-aware downsampler). Mip 0 stays as a
    // non-owning view into the caller's RGBA buffer; mips 1..N-1 live in a
    // single contiguous arena.
    auto subMips = buildSubMipChain(rgba, W, H, srgb);
    const uint32_t mipCount = uint32_t(subMips.mips.size()) + 1u;

    astcenc_context* ctx = cached.ctx;
    // threadCount now lives in the pool; nothing in encodeOne needs to
    // know it directly.

    const astcenc_swizzle swz = { ASTCENC_SWZ_R, ASTCENC_SWZ_G,
                                  ASTCENC_SWZ_B, ASTCENC_SWZ_A };

    std::vector<std::vector<uint8_t>> mipAstc(mipCount);
    bool ok = true;
    for (uint32_t m = 0; m < mipCount; ++m) {
        const uint32_t mw = std::max(1u, W >> m);
        const uint32_t mh = std::max(1u, H >> m);

        astcenc_image image{};
        image.dim_x     = mw;
        image.dim_y     = mh;
        image.dim_z     = 1;
        image.data_type = ASTCENC_TYPE_U8;
        // astcenc_image::data is "array of 2D slice pointers" with dim_z
        // entries; for a 2D image we have one slice. astcenc takes a
        // non-const void**, but per the docs it only reads the pixels in
        // the U8 path — const_cast is safe.
        const uint8_t* mipPx = (m == 0u) ? rgba : subMips.mips[m - 1u].data;
        void* slicePtrs[1] = { const_cast<uint8_t*>(mipPx) };
        image.data = slicePtrs;

        const size_t blocksX = (mw + bx - 1u) / bx;
        const size_t blocksY = (mh + by - 1u) / by;
        const size_t outBytes = blocksX * blocksY * 16u;     // ASTC block = 16 B
        mipAstc[m].resize(outBytes);

        // Pre-spawned worker pool: all N participants (calling thread + N-1
        // workers) call astcenc_compress_image with their fixed ti in
        // [0, N). No per-mip std::thread spawns.
        int err = pool.run([&](int ti) -> int {
            return int(astcenc_compress_image(ctx, &image, &swz,
                                              mipAstc[m].data(),
                                              outBytes, unsigned(ti)));
        });
        if (err != ASTCENC_SUCCESS) {
            std::fprintf(stderr,
                "astcenc_compress_image mip %u failed (err=%d)\n",
                m, err);
            ok = false;
            break;
        }

        // Reset between images (and between mips — mips share the context's
        // working state which the API requires we reset before reuse).
        auto rstErr = astcenc_compress_reset(ctx);
        if (rstErr != ASTCENC_SUCCESS) {
            std::fprintf(stderr, "astcenc_compress_reset failed (%d)\n",
                         int(rstErr));
            ok = false;
            break;
        }
    }

    // Do NOT free ctx — it's borrowed from AstcCtxCache and will be reused
    // by the next image with the same (block, srgb, quality).
    if (!ok) return {};

    return packMrtc(bx, by, srgb, hasAlpha, W, H, mipAstc);
}

}  // namespace

namespace TextureLoader {

void initOnce() {
    // No-op for now. astcenc has no global init step; per-context tables
    // are built inside astcenc_context_alloc(). Hook reserved for future
    // ahead-of-time work (precomputing partition tables once, etc).
}

bool readMrtc(const std::vector<uint8_t>& bytes, MrtcView& out) {
    if (bytes.size() < kMrtcMinHeader) return false;
    const uint8_t* p = bytes.data();
    if (rd32_le(p + 0) != kMrtcMagic)        return false;
    if (rd32_le(p + 4) != kMrtcVersion)      return false;
    out.blockX = rd32_le(p + 8);
    out.blockY = rd32_le(p + 12);
    if (out.blockX != 4 && out.blockX != 6)  return false;
    if (out.blockY != 4 && out.blockY != 6)  return false;
    out.srgb     = (p[16] != 0);
    out.width    = rd32_le(p + 20);
    out.height   = rd32_le(p + 24);
    out.mipCount = rd32_le(p + 28);
    if (out.mipCount == 0 || out.mipCount > 32) return false;
    if (bytes.size() < kMrtcMinHeader + size_t(4) * out.mipCount) return false;

    std::vector<uint32_t> mipSizes(out.mipCount);
    for (uint32_t m = 0; m < out.mipCount; ++m)
        mipSizes[m] = rd32_le(p + kMrtcMinHeader + 4u * m);

    size_t cursor = kMrtcMinHeader + size_t(4) * out.mipCount;
    out.mips.clear();
    out.mips.reserve(out.mipCount);
    for (uint32_t m = 0; m < out.mipCount; ++m) {
        if (cursor + size_t(mipSizes[m]) > bytes.size()) return false;
        MrtcView::Mip mp;
        mp.data   = p + cursor;
        mp.size   = mipSizes[m];
        mp.width  = std::max(1u, out.width  >> m);
        mp.height = std::max(1u, out.height >> m);
        out.mips.push_back(mp);
        cursor += size_t(mipSizes[m]);
    }
    return true;
}

std::vector<EncodedTexture> encodeAll(const cgltf_data* data,
                                      const EncodeOpts& opts) {
    if (!data || data->images_count == 0) return {};

    initOnce();

    std::vector<EncodedTexture> out(data->images_count);

    const size_t total = data->images_count;
    const bool   useCache = !opts.cacheDir.empty();
    const bool   tty = isatty(fileno(stderr));
    // On TTY: \r overwrites the "starting" line with the "done" line so
    // each image gets exactly one line of output. On non-TTY (logs / CI):
    // print both as separate lines for grep-ability.
    const char*  startEol = tty ? "\r" : "\n";

    if (useCache) {
        std::error_code ec;
        std::filesystem::create_directories(opts.cacheDir, ec);
        // Failure here is non-fatal: writes will retry per-file and fall
        // back to "encode but don't cache".
    }

    std::fprintf(stderr,
        "GLB textures: %zu image%s to process%s%s%s\n",
        total, total == 1 ? "" : "s",
        useCache ? " (cache: " : " (no cache)",
        useCache ? opts.cacheDir.c_str() : "",
        useCache ? "/)" : "");

    using clock = std::chrono::steady_clock;
    const auto t_all0 = clock::now();
    size_t hits = 0, misses = 0;
    size_t totalBlobBytes = 0;
    double totalEncodeSec = 0.0;

    // Cache of astcenc contexts shared across all images. The encoder's
    // internal LUTs cost ~5 ms to build per context; with this cache we
    // pay that cost once per (block, srgb, quality) tuple instead of once
    // per image. Reset between mips/images is handled inside encodeOne.
    AstcCtxCache ctxCache;
    ctxCache.threadCount = std::max(1u, std::thread::hardware_concurrency());

    // Pre-spawn the astcenc worker pool once. Replaces ~1400 per-mip
    // std::thread::thread spawns over a typical scene encode.
    AstcWorkerPool astcPool(ctxCache.threadCount);

    std::atomic<size_t> failures{0};
    for (cgltf_size i = 0; i < data->images_count; ++i) {
        auto raw = readImageBytes(data, &data->images[i]);
        if (raw.empty()) {
            std::fprintf(stderr,
                "  [%zu/%zu] image_%zu — readImageBytes empty\n",
                size_t(i)+1, total, size_t(i));
            ++failures; continue;
        }

        // Decode source dims early so we can pick block size (the cache
        // key depends on block, and hero-min-dim depends on dims).
        // stbi_info is much cheaper than stbi_load — header-only parse.
        int infoW = 0, infoH = 0, infoComp = 0;
        stbi_info_from_memory(raw.data(), int(raw.size()),
                              &infoW, &infoH, &infoComp);
        if (infoW <= 0 || infoH <= 0) {
            std::fprintf(stderr,
                "  [%zu/%zu] image_%zu — stbi_info failed (raw=%zuB)\n",
                size_t(i)+1, total, size_t(i), raw.size());
            ++failures; continue;
        }

        const uint32_t maxDim = uint32_t(std::max(infoW, infoH));
        const AstcBlock block = (maxDim >= opts.heroMinDim)
                              ? AstcBlock::B4x4
                              : opts.defaultBlock;
        const char* blockStr = (block == AstcBlock::B4x4) ? "4×4" : "6×6";

        // Cache lookup.
        std::filesystem::path cpath;
        if (useCache) {
            uint64_t key = makeCacheKey(raw, block, opts.quality,
                                         /*srgb=*/true);
            cpath = cachePath(opts.cacheDir, key);
            if (std::filesystem::exists(cpath)) {
                auto bytes = readCacheFile(cpath);
                uint32_t kw = 0, kh = 0;
                // Hash-collision guard: 64-bit FNV-1a is fast but not
                // collision-proof for adversarial inputs. Verify the
                // cached MRTC's dims match the source's stbi_info dims;
                // a mismatch means a different image hashes to the same
                // key. Treat as miss → re-encode.
                bool dimsOK = !bytes.empty()
                           && readMrtcDims(bytes, kw, kh)
                           && kw == uint32_t(infoW)
                           && kh == uint32_t(infoH);
                if (dimsOK) {
                    const double mib = double(bytes.size())
                                     / (1024.0 * 1024.0);
                    std::fprintf(stderr,
                        "  [%zu/%zu] image_%zu  %u×%u %s  hit  %.2f MiB\n",
                        size_t(i)+1, total, size_t(i),
                        kw, kh, blockStr, mib);
                    EncodedTexture t;
                    t.mrtcBytes = std::move(bytes);
                    t.width     = kw;
                    t.height    = kh;
                    t.hasAlpha  = true;
                    t.srgb      = true;
                    t.block     = block;
                    ++hits;
                    totalBlobBytes += t.mrtcBytes.size();
                    out[i] = std::move(t);
                    continue;
                }
                if (!bytes.empty() && (kw != 0 || kh != 0) &&
                    (kw != uint32_t(infoW) || kh != uint32_t(infoH))) {
                    std::fprintf(stderr,
                        "  [%zu/%zu] image_%zu  cache hash collision: "
                        "key matches but dims differ "
                        "(cached %u×%u, source %d×%d) — re-encoding\n",
                        size_t(i)+1, total, size_t(i),
                        kw, kh, infoW, infoH);
                }
            }
        }

        // Cache miss → decode source pixels and run astcenc.
        std::fprintf(stderr,
            "  [%zu/%zu] image_%zu  %d×%d %s  encoding…%s",
            size_t(i)+1, total, size_t(i),
            infoW, infoH, blockStr, startEol);
        std::fflush(stderr);

        int w = 0, h = 0, comp = 0;
        unsigned char* px = stbi_load_from_memory(
            raw.data(), int(raw.size()), &w, &h, &comp, /*req_comp*/ 4);
        if (!px || w <= 0 || h <= 0) {
            std::fprintf(stderr,
                "%s  [%zu/%zu] image_%zu — stbi_load_from_memory failed: %s\n",
                tty ? "\r" : "",
                size_t(i)+1, total, size_t(i),
                stbi_failure_reason() ? stbi_failure_reason() : "(no reason)");
            if (px) stbi_image_free(px);
            ++failures; continue;
        }

        const uint32_t bsize = (block == AstcBlock::B6x6) ? 6u : 4u;
        AstcCtx* cached = ctxCache.get(bsize, /*srgb=*/true, opts.quality);
        if (!cached) {
            std::fprintf(stderr,
                "%s  [%zu/%zu] image_%zu — astcenc context unavailable\n",
                tty ? "\r" : "",
                size_t(i)+1, total, size_t(i));
            stbi_image_free(px);
            ++failures; continue;
        }
        const auto t0 = clock::now();
        std::vector<uint8_t> blob = encodeOne(px, uint32_t(w), uint32_t(h),
                                              /*srgb=*/true,
                                              block,
                                              /*hasAlpha=*/comp == 4,
                                              *cached,
                                              astcPool);
        const double secs = std::chrono::duration<double>(
            clock::now() - t0).count();
        stbi_image_free(px);

        if (blob.empty()) {
            std::fprintf(stderr,
                "%s  [%zu/%zu] image_%zu — astcenc encode returned empty\n",
                tty ? "\r" : "",
                size_t(i)+1, total, size_t(i));
            ++failures; continue;
        }

        const double mib = double(blob.size()) / (1024.0 * 1024.0);
        const char* tag = useCache ? "  miss → cached" : "  miss";
        std::fprintf(stderr,
            "%s  [%zu/%zu] image_%zu  %d×%d %s  encoded in %5.2fs  %.2f MiB%s\n",
            tty ? "\r" : "",
            size_t(i)+1, total, size_t(i),
            w, h, blockStr, secs, mib, tag);

        if (useCache && !cpath.empty()) {
            if (!writeCacheFile(cpath, blob)) {
                std::fprintf(stderr,
                    "  [%zu/%zu] image_%zu — cache write failed (path=%s)\n",
                    size_t(i)+1, total, size_t(i),
                    cpath.string().c_str());
            }
        }

        ++misses;
        totalEncodeSec += secs;
        totalBlobBytes += blob.size();

        EncodedTexture t;
        t.mrtcBytes = std::move(blob);
        t.width     = uint32_t(w);
        t.height    = uint32_t(h);
        t.hasAlpha  = (comp == 4);
        t.srgb      = true;
        t.block     = block;
        out[i]      = std::move(t);
    }

    const double totalSec = std::chrono::duration<double>(
        clock::now() - t_all0).count();
    const double totMiB   = double(totalBlobBytes) / (1024.0 * 1024.0);
    std::fprintf(stderr,
        "GLB textures: %zu/%zu cached%s, %zu/%zu encoded in %.1fs"
        "  |  %.1f MiB total  |  %.1fs wall\n",
        hits, total, useCache ? "" : " (cache disabled)",
        misses, total, totalEncodeSec, totMiB, totalSec);

    if (failures.load() > 0) {
        std::fprintf(stderr,
            "TextureLoader: %zu/%zu images failed to encode "
            "(unsupported format or zero-size source); "
            "those primitives will fall back to lambert.\n",
            failures.load(), size_t(data->images_count));
    }
    return out;
}

}  // namespace TextureLoader
}  // namespace metalrast
