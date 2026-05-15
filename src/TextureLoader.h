#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

struct cgltf_data;

namespace metalrast {

// ASTC block-size choice per texture. 4×4 = 8 bpp (high quality), 6×6 =
// 3.56 bpp (2.25× smaller, ~6-8 dB lower PSNR — visually transparent on
// natural surfaces, visible blocking on text/fine detail). Apple Silicon's
// hardware sampler handles both at full speed.
enum class AstcBlock : uint8_t { B4x4 = 0, B6x6 = 1 };

// One entry per glTF image. `mrtcBytes` is a self-contained MRTC blob
// (custom container — see TextureLoader.cpp for the layout) holding the
// pre-encoded ASTC LDR mip chain. The renderer parses the header and
// uploads each mip's raw blocks straight into a Metal texture; no
// transcode step.
//
// `width` / `height` are the level-0 dimensions for convenience (the same
// values are inside the MRTC header). Empty `mrtcBytes` means encoding
// failed for that image; the renderer treats it as untextured (lambert).
struct EncodedTexture {
    std::vector<uint8_t> mrtcBytes;
    uint32_t             width    = 0;
    uint32_t             height   = 0;
    bool                 hasAlpha = false;
    bool                 srgb     = true;   // base-color textures are sRGB per glTF spec
    AstcBlock            block    = AstcBlock::B4x4;
};

namespace TextureLoader {

// Encode policy for a whole scene. Defaults: 4×4 for everything (high
// quality). Bump `heroMinDim` to keep large textures at 4×4 — e.g. 4096
// means anything ≥ 4K stays at 4×4 (assumed close-up "hero" assets), the
// rest go to 6×6. UINT32_MAX (default) disables the override → blanket 4×4.
//
// Cache: `cacheDir` non-empty → look up encoded MRTC by content hash before
// encoding. Misses are written to disk after encode; subsequent runs hit the
// cache and skip the (slow) astcenc encode entirely. `cacheDir` empty → no
// cache (always re-encode). Hash inputs include the source bytes + block /
// quality / srgb / version-tag so changing any of them invalidates entries.
struct EncodeOpts {
    int         quality       = 0;        // mapped to ASTCENC_PRE_* range — see TextureLoader.cpp
    AstcBlock   defaultBlock  = AstcBlock::B4x4;
    uint32_t    heroMinDim    = std::numeric_limits<uint32_t>::max();
    std::string cacheDir      = "model_cache";   // empty = no cache
};

// One-shot init. astcenc has no global state to bootstrap; this is here as
// a hook for future ahead-of-time work and to keep the call site stable
// across encoder swaps. Idempotent.
void initOnce();

// Decode every image referenced by `data->images[]`, encode each as an
// MRTC blob in memory (full mip chain, ASTC LDR 4×4 or 6×6). Output is
// one EncodedTexture per image, in glTF image order (so a primitive's
// material->base_color->texture->image-index is also the index here).
//
// PNG / JPEG sources are decoded by stb_image to RGBA8, then a sRGB-aware
// box-filter mip pyramid is generated via stb_image_resize2, and each
// mip is encoded by astcenc.
std::vector<EncodedTexture> encodeAll(const cgltf_data* data,
                                      const EncodeOpts& opts = {});

// Read-only view into an MRTC blob. Pointers are borrowed from the
// underlying byte vector; do not store past its lifetime.
struct MrtcView {
    uint32_t width   = 0;
    uint32_t height  = 0;
    uint32_t blockX  = 0;
    uint32_t blockY  = 0;
    bool     srgb    = true;
    uint32_t mipCount = 0;
    struct Mip { const uint8_t* data; size_t size; uint32_t width; uint32_t height; };
    std::vector<Mip> mips;     // ordered largest → smallest, matching mip 0..N-1
};

// Parse an MRTC blob written by encodeAll(). Returns false on bad magic,
// version mismatch, truncation, or block-size out of {4,6}.
bool readMrtc(const std::vector<uint8_t>& bytes, MrtcView& out);

}  // namespace TextureLoader
}  // namespace metalrast
