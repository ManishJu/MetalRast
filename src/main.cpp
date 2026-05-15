// MetalRast — Metal port of CuRast — entry point.
//
// metal-cpp is header-only; the *_PRIVATE_IMPLEMENTATION defines below must
// appear in exactly one translation unit before the framework headers.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "Camera.h"
#include "GLBLoader.h"
#include "MeshRegistry.h"
#include "MetalRastRenderer.h"
#include "Mesh.h"
#include "SharedTypes.h"
#include "ui/InteractiveApp.h"

namespace {

struct Args {
    uint32_t width        = 1920;
    uint32_t height       = 1080;
    int      sphereLat    = 700;
    int      sphereLon    = 1000;
    int      groundSubdiv = 1;
    int      frames       = 1;       // total frames rendered
    int      warmup       = 4;       // warm-up frames excluded from stats
    std::string output    = "metalrast.png";
    bool benchmark        = false;
    bool noPng            = false;
    bool spinCamera       = false;
    bool profile          = false;   // per-stage GPU timing breakdown (split CB)
    bool profileCounters  = false;   // per-stage GPU timing via MTLCounterSampleBuffer
    std::string countersOutput = ""; // > 0 → write JSON dump per measured frame
    int  stage1TG         = 256;     // stage1 persistent threadgroup count (matches RendererConfig default)
    int  zorahInstances   = 0;       // > 0 → use Zorah-style scene with N instances
    std::string glbPath   = "";      // non-empty → load this .glb instead
    std::string camSide   = "+z";    // for --glb framing: -z, +z, +x, -x, +y, -y
    float       camPullback = 2.5f;  // distance multiplier (radius * this)
    uint64_t    maxTris     = 0;     // 0 = no cap; else GLB loader stops at N tris
    int         repeat      = 1;     // replicate the loaded scene N times (3D grid)
    bool        compressed  = true;  // 16-bit fixed-point positions (paper §4.6)
    bool        bitPacked   = true;  // variable-bit-width packed indices (paper §4.6)
    bool        directGpuLoad = true;// direct-Metal-write loader (CuRast-style, no host vec)
    bool        interactive = false; // open a GLFW window with ImGui debug UI
    int         residency   = -1;    // -1 auto, 0 force-off, 1 force-on
    int         residencyHeadlessFrames = 0;  // > 0 → register + N synthetic frames
    uint32_t    residencyPoolMiB        = 0;  // > 0 → set BOTH pools (legacy shorthand)
    uint32_t    residencyVertMiB        = 0;  // > 0 → override vertex pool only
    uint32_t    residencyIdxMiB         = 0;  // > 0 → override index pool only
    std::string loadCamera = "";              // non-empty → ignore auto-frame, replay saved camera
    bool        minimalUi  = false;           // interactive: hide all panels, show only Save Camera
    bool        hiZ        = false;           // enable Hi-Z occlusion culling (Frostbite-style)
    bool        hiZTwo     = false;           // two-phase Hi-Z (requires --hi-z)
    bool        compactInstances = false;     // Hijma §6.2.2 — pack visible instances before stage1 (requires --hi-z)
    bool        prefixSum  = true;            // resolve uses 64K-bucket DC lookup; --no-prefix-sum disables
    std::string astcBlock  = "4x4";           // 4x4 (high quality) or 6x6 (2.25× smaller); see TextureLoader.h
    uint32_t    astcHeroMin = 0;              // textures with max(W,H) >= N stay at 4×4 even when default is 6×6
    bool        textureCache = true;          // cache encoded MRTC in model_cache/; --no-texture-cache disables
    int         captureFrame = -1;            // > 0: write a .gputrace at this frame number (1-indexed)
    std::string captureOutput = "tmp/metalrast.gputrace";
};

void printUsage(const char* exe) {
    std::printf(
        "Usage: %s [options]\n"
        "  --width N           framebuffer width  (default 1920)\n"
        "  --height N          framebuffer height (default 1080)\n"
        "  --sphere LAT,LON    sphere tessellation (default 700,1000)\n"
        "  --ground N          ground-plane subdivisions (default 1, gives 2 large tris)\n"
        "  --frames N          total frames to render (default 1)\n"
        "  --warmup N          warmup frames to exclude from stats (default 4)\n"
        "  --benchmark         repeat render, report GPU timing stats\n"
        "  --spin              rotate the scene 360° across frames (visual sanity)\n"
        "  --output PATH       PNG output path (default metalrast.png)\n"
        "  --no-png            skip PNG write (benchmark-only mode)\n"
        "  --profile           per-stage GPU timing breakdown (CB-split, perturbative)\n"
        "  --profile-counters  per-stage GPU timing via MTLCounterSampleBuffer\n"
        "                          (single-CB, sub-µs counter cost, prefer this)\n"
        "  --counters-output P JSON dump path for --profile-counters per-frame samples\n"
        "  --tg1 N             stage1 persistent threadgroup count (default 256)\n"
        "  --zorah N           Zorah-style instanced scene with ~N instances\n"
        "                      (overrides --sphere/--ground)\n"
        "  --glb PATH          load a .glb / .gltf file as the scene\n"
        "                      (overrides --zorah/--sphere)\n"
        "  --repeat N          replicate the loaded GLB across an N-cell 3D grid\n"
        "                      (vertex/index data shared on GPU; only N×draw-calls)\n"
        "  --max-tris N        cap GLB load at N total instanced tris (memory budget)\n"
        "  --cam {-z,+z,...}   GLB camera angle (default +z)\n"
        "  --pullback F        camera distance multiplier (default 2.5)\n"
        "  --no-compress       disable 16-bit position compression (16 B/vertex float3)\n"
        "  --compress          force compression on (default; 8 B/vertex ushort4)\n"
        "  --no-bit-pack       disable variable-bit-width index packing (raw uint32)\n"
        "  --no-prefix-sum     disable 64K-bucket DC lookup in resolve (use plain binary search)\n"
        "  --bit-pack          force bit-packed indices on (default; paper §4.6)\n"
        "  --no-direct-gpu-load   disable CuRast-style direct Metal-buffer loader (use host vec)\n"
        "  --direct-gpu-load      force direct loader on (default; halves peak host RAM)\n"
        "  --residency            CuRast-style streaming: parse + register, page geometry in/out\n"
        "                          per-frame from a fixed-size pool (auto-on for large scenes)\n"
        "  --no-residency         force the static-upload path even for huge scenes\n"
        "  --interactive,-i    open a GLFW window + ImGui debug UI; drag-drop GLBs\n"
        "  --compact-instances pack visible instances before stage1 (Hijma §6.2.2; requires --hi-z;\n"
        "                          only useful on instance-heavy scenes with high cull rates)\n"
        "  --astc-block {4x4|6x6} ASTC block size for textures (default 4x4: 8 bpp high quality.\n"
        "                          6x6 is 2.25× smaller, ~6-8 dB lower PSNR — visually fine on natural surfaces)\n"
        "  --astc-hero-min N   any texture with max(W,H) >= N stays at 4×4 even when default is 6×6\n"
        "                          (e.g. --astc-block 6x6 --astc-hero-min 4096 keeps 4K+ textures sharp)\n"
        "  --no-texture-cache  disable model_cache/<hash>.mrtc lookup; force re-encode every time\n"
        "                          (cache is on by default — first run encodes + saves, later runs read)\n"
        "  --capture-frame N   write a Metal GPU trace at frame N to --capture-output\n"
        "                          (requires MTL_CAPTURE_ENABLED=1 in the environment)\n"
        "  --capture-output P  output path for --capture-frame (default tmp/metalrast.gputrace)\n"
        "  --help              this message\n",
        exe);
}

bool parseArgs(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else if (a == "--width")     out.width  = std::atoi(need("--width"));
        else if (a == "--height")    out.height = std::atoi(need("--height"));
        else if (a == "--sphere") {
            const char* v = need("--sphere");
            int lat = 0, lon = 0;
            if (std::sscanf(v, "%d,%d", &lat, &lon) != 2 || lat < 2 || lon < 3) {
                std::fprintf(stderr, "--sphere expects LAT,LON (e.g. 700,1000)\n");
                std::exit(2);
            }
            out.sphereLat = lat; out.sphereLon = lon;
        }
        else if (a == "--ground")    out.groundSubdiv = std::max(1, std::atoi(need("--ground")));
        else if (a == "--frames")    out.frames = std::max(1, std::atoi(need("--frames")));
        else if (a == "--warmup")    out.warmup = std::max(0, std::atoi(need("--warmup")));
        else if (a == "--benchmark") out.benchmark = true;
        else if (a == "--spin")      out.spinCamera = true;
        else if (a == "--output")    out.output = need("--output");
        else if (a == "--no-png")    out.noPng = true;
        else if (a == "--profile")   out.profile = true;
        else if (a == "--profile-counters") out.profileCounters = true;
        else if (a == "--counters-output") out.countersOutput = need("--counters-output");
        else if (a == "--tg1")       out.stage1TG = std::max(1, std::atoi(need("--tg1")));
        else if (a == "--zorah")     out.zorahInstances = std::max(1, std::atoi(need("--zorah")));
        else if (a == "--glb")       out.glbPath = need("--glb");
        else if (a == "--cam")       out.camSide = need("--cam");
        else if (a == "--pullback")  out.camPullback = std::max(0.5f, float(std::atof(need("--pullback"))));
        else if (a == "--max-tris")  out.maxTris    = static_cast<uint64_t>(std::atoll(need("--max-tris")));
        else if (a == "--repeat")    out.repeat     = std::max(1, std::atoi(need("--repeat")));
        else if (a == "--no-compress") out.compressed = false;
        else if (a == "--compress")    out.compressed = true;
        else if (a == "--no-bit-pack") out.bitPacked = false;
        else if (a == "--no-prefix-sum") out.prefixSum = false;
        else if (a == "--bit-pack")    out.bitPacked = true;
        else if (a == "--no-direct-gpu-load") out.directGpuLoad = false;
        else if (a == "--direct-gpu-load")    out.directGpuLoad = true;
        else if (a == "--residency")          out.residency = 1;
        else if (a == "--no-residency")       out.residency = 0;
        else if (a == "--residency-headless") {
            out.residencyHeadlessFrames = std::max(1, std::atoi(need("--residency-headless")));
            out.residency = 1;
        }
        else if (a == "--residency-pool-mib") {
            out.residencyPoolMiB = uint32_t(std::max(1, std::atoi(need("--residency-pool-mib"))));
        }
        else if (a == "--residency-vert-mib") {
            out.residencyVertMiB = uint32_t(std::max(1, std::atoi(need("--residency-vert-mib"))));
        }
        else if (a == "--residency-idx-mib") {
            out.residencyIdxMiB = uint32_t(std::max(1, std::atoi(need("--residency-idx-mib"))));
        }
        else if (a == "--load-camera") {
            out.loadCamera = need("--load-camera");
        }
        else if (a == "--minimal-ui") {
            out.minimalUi = true;
        }
        else if (a == "--hi-z" || a == "--hiz") {
            out.hiZ = true;
        }
        else if (a == "--two-phase" || a == "--hiz-two") {
            out.hiZTwo = true; out.hiZ = true;
        }
        else if (a == "--compact-instances") {
            out.compactInstances = true;
        }
        else if (a == "--interactive" || a == "-i") out.interactive = true;
        else if (a == "--astc-block") {
            std::string v = need("--astc-block");
            if (v != "4x4" && v != "6x6") {
                std::fprintf(stderr, "--astc-block must be '4x4' or '6x6' (got '%s')\n", v.c_str());
                return false;
            }
            out.astcBlock = v;
        }
        else if (a == "--astc-hero-min") {
            out.astcHeroMin = uint32_t(std::max(0, std::atoi(need("--astc-hero-min"))));
        }
        else if (a == "--no-texture-cache") {
            out.textureCache = false;
        }
        else if (a == "--capture-frame") {
            out.captureFrame = std::max(1, std::atoi(need("--capture-frame")));
        }
        else if (a == "--capture-output") {
            out.captureOutput = need("--capture-output");
        }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            printUsage(argv[0]);
            std::exit(2);
        }
    }
    return true;
}

metalrast::Scene buildDemoScene(const Args& a, MTL::Device* device) {
    if (!a.glbPath.empty()) {
        // maxTris is in triangles; GLB cap is in indices = 3 × tris.
        uint64_t maxIdx = a.maxTris ? a.maxTris * 3ull : 0ull;
        metalrast::TextureLoader::EncodeOpts texOpts;
        texOpts.defaultBlock = (a.astcBlock == "4x4")
                             ? metalrast::AstcBlock::B4x4
                             : metalrast::AstcBlock::B6x6;
        texOpts.heroMinDim   = a.astcHeroMin
                             ? a.astcHeroMin
                             : std::numeric_limits<uint32_t>::max();
        if (!a.textureCache) texOpts.cacheDir.clear();
        metalrast::Scene s = metalrast::loadGLB(a.glbPath, maxIdx,
                                                a.compressed, a.bitPacked,
                                                /*directDevice*/ a.directGpuLoad ? device : nullptr,
                                                texOpts);
        if (a.repeat > 1) {
            std::printf("Replicating scene %d×: %zu DC, %u inst, %llu tris → ",
                        a.repeat, s.drawCalls.size(), s.totalInstances(),
                        (unsigned long long)s.totalTriangles());
            s = metalrast::replicateScene(std::move(s), a.repeat);
            std::printf("%zu DC, %u inst, %llu tris\n",
                        s.drawCalls.size(), s.totalInstances(),
                        (unsigned long long)s.totalTriangles());
        }
        return s;
    }
    if (a.zorahInstances > 0) {
        // Build with the right compression mode from the start (otherwise
        // addInstancedMeshToScene quantizes into the wrong storage).
        if (!a.compressed) {
            std::fprintf(stderr,
                "Note: --zorah uses the compressed default; --no-compress "
                "is honored for --glb / --sphere only.\n");
        }
        return metalrast::makeZorahLikeScene(uint32_t(a.zorahInstances));
    }

    metalrast::Scene s;
    s.compressed    = a.compressed;
    s.indicesPacked = a.bitPacked;

    // Sphere at origin, radius 1.
    auto sphere = metalrast::makeUVSphere(a.sphereLat, a.sphereLon, 1.0f);
    metalrast::addMeshToScene(s, sphere, metalrast::makeTranslation((simd_float3){0, 0, 0}));

    // Large ground plane far below.
    auto plane = metalrast::makeGroundPlane(/*halfSize*/ 50.0f,
                                         /*yLevel*/   -1.0f,
                                         /*subdiv*/   a.groundSubdiv);
    metalrast::addMeshToScene(s, plane, metalrast::makeIdentity());

    return s;
}

void requireFeatures(MTL::Device* d) {
    bool ok = d->supportsFamily(MTL::GPUFamilyApple7);
    if (!ok) {
        std::fprintf(stderr,
            "ERROR: This GPU does not support GPUFamilyApple7 (M1+).\n"
            "       64-bit atomics required by MetalRast are unavailable.\n");
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout so log lines appear immediately even when redirected.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Args args;
    if (!parseArgs(argc, argv, args)) return 0;

    // Auto-resolve a per-model launch camera. If the user did not pass
    // --load-camera, look for "Launch Camera Angles/<glb-stem>.cam" relative
    // to the current working directory and use it. Lets us land on a curated
    // angle for each model instead of the default auto-frame.
    if (args.loadCamera.empty() && !args.glbPath.empty()) {
        std::filesystem::path p(args.glbPath);
        std::filesystem::path cam = std::filesystem::path("Launch Camera Angles")
            / (p.stem().string() + ".cam");
        std::error_code ec;
        if (std::filesystem::exists(cam, ec)) {
            args.loadCamera = cam.string();
            std::printf("Auto-loading launch camera: %s\n", args.loadCamera.c_str());
        }
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "No Metal device available\n"); return 1; }
    requireFeatures(device);

    std::printf("Device: %s\n", device->name()->utf8String());
    std::printf("Resolution: %u × %u\n", args.width, args.height);

    // -------- Decide residency mode --------------------------------------
    // Auto: --glb + interactive + file ≥ 512 MiB triggers streaming. The
    // user can override with --residency / --no-residency.
    bool useResidency = false;
    if (args.residency == 1) {
        useResidency = true;
    } else if (args.residency == 0) {
        useResidency = false;
    } else if (args.interactive && !args.glbPath.empty()) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(args.glbPath, ec);
        useResidency = (!ec && sz >= (size_t(512) << 20));
    }
    if (useResidency && args.glbPath.empty() && args.zorahInstances <= 0) {
        std::fprintf(stderr, "Note: --residency needs --glb PATH or --zorah N; ignoring.\n");
        useResidency = false;
    }

    // -------- Scene -------------------------------------------------------
    auto buildStart = std::chrono::steady_clock::now();
    metalrast::Scene scene;
    if (!useResidency) scene = buildDemoScene(args, device);
    auto buildEnd   = std::chrono::steady_clock::now();

    if (useResidency) {
        std::printf("Residency mode: streaming %s (geometry will be paged in per-frame)\n",
                    args.glbPath.c_str());
    } else {
        size_t vertCount = scene.compressed ? scene.positions.size()
                                            : scene.positionsFloat.size();
        std::printf("Scene: %zu vertices (%s), %zu indices, %llu triangles "
                    "(%zu draw calls, %u instances) — built in %.2f ms\n",
                    vertCount,
                    scene.compressed ? "compressed 8B" : "uncompressed 16B",
                    scene.indices.size(),
                    (unsigned long long)scene.totalTriangles(),
                    scene.drawCalls.size(),
                    scene.totalInstances(),
                    std::chrono::duration<double, std::milli>(buildEnd - buildStart).count());
    }

    // -------- Headless residency smoke -----------------------------------
    // Validation path: register a GLB with a ResidencyManager and run N
    // synthetic frames (camera framing the world AABB), printing stats.
    // No rendering — exercises the registration + culling + page-in code
    // path so we can sanity-check that without bringing up a window.
    if (useResidency && !args.interactive && args.residencyHeadlessFrames > 0) {
        metalrast::ResidencyManager::Config rc;
        rc.compressed     = args.compressed;
        rc.bitPackIndices = args.bitPacked;
        if (args.residencyPoolMiB) {
            rc.vertPoolBytes = size_t(args.residencyPoolMiB) << 20;
            rc.idxPoolBytes  = size_t(args.residencyPoolMiB) << 20;
        }
        if (args.residencyVertMiB) rc.vertPoolBytes = size_t(args.residencyVertMiB) << 20;
        if (args.residencyIdxMiB)  rc.idxPoolBytes  = size_t(args.residencyIdxMiB)  << 20;
        metalrast::ResidencyManager mgr(device, rc);
        metalrast::loadGLBToRegistry(args.glbPath, mgr, /*maxIndices*/ 0);
        simd_float3 lo, hi;
        mgr.sceneAABB(lo, hi);
        simd_float3 ctr = (simd_float3){ (lo.x + hi.x) * 0.5f,
                                         (lo.y + hi.y) * 0.5f,
                                         (lo.z + hi.z) * 0.5f };
        float radius = std::sqrt((hi.x - lo.x) * (hi.x - lo.x) +
                                 (hi.y - lo.y) * (hi.y - lo.y) +
                                 (hi.z - lo.z) * (hi.z - lo.z)) * 0.5f;
        if (radius < 1e-3f) radius = 1.0f;

        metalrast::Camera cam;
        cam.eye         = (simd_float3){ ctr.x, ctr.y, ctr.z - radius * 2.5f };
        cam.target      = ctr;
        cam.up          = (simd_float3){ 0, 1, 0 };
        cam.nearPlane   = std::max(0.01f, radius * 0.01f);
        cam.farPlane    = radius * 10.0f;
        cam.fovYRadians = 60.0f * 3.14159265f / 180.0f;

        for (int f = 0; f < args.residencyHeadlessFrames; ++f) {
            auto view = mgr.prepareFrame(cam, uint64_t(f));
            // Give the worker a brief moment to drain.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            auto s = mgr.stats();
            std::printf("[frame %3d] visible=%u resident=%u loading=%u "
                        "loadsThisFrame=%u  vert=%.0f/%.0f MiB  idx=%.0f/%.0f MiB  DCs=%u\n",
                        f, s.visibleMeshes, s.residentMeshes, s.loadingMeshes,
                        s.loadsThisFrame,
                        double(s.vertPoolUsed)/(1<<20), double(s.vertPoolCapacity)/(1<<20),
                        double(s.idxPoolUsed)/(1<<20),  double(s.idxPoolCapacity)/(1<<20),
                        view.numDrawCalls);
        }
        device->release();
        pool->release();
        return 0;
    }

    // -------- Interactive mode short-circuit -----------------------------
    // When the user asked for --interactive, hand off to InteractiveApp
    // which owns its own renderer, window, camera, and ImGui host. main()'s
    // headless code path is unreachable from here.
    if (args.interactive) {
        metalrast::InteractiveApp::Config ac;
        ac.window.width  = int(args.width);
        ac.window.height = int(args.height);
        ac.window.title  = "MetalRast";
        ac.metallibPath  = METALRAST_METALLIB_PATH;
        ac.minimalUi     = args.minimalUi;
        ac.loadCamera    = args.loadCamera;
        ac.initialHiZ    = args.hiZ;
        ac.initialCompactInstances = args.compactInstances;
        if (useResidency) {
            ac.residencyPath           = args.glbPath;
            ac.residencyCfg.compressed     = args.compressed;
            ac.residencyCfg.bitPackIndices = args.bitPacked;
            if (args.residencyPoolMiB) {
                ac.residencyCfg.vertPoolBytes = size_t(args.residencyPoolMiB) << 20;
                ac.residencyCfg.idxPoolBytes  = size_t(args.residencyPoolMiB) << 20;
            }
            if (args.residencyVertMiB)
                ac.residencyCfg.vertPoolBytes = size_t(args.residencyVertMiB) << 20;
            if (args.residencyIdxMiB)
                ac.residencyCfg.idxPoolBytes  = size_t(args.residencyIdxMiB)  << 20;
        } else {
            ac.initialScene  = std::move(scene);
            ac.initialPath   = args.glbPath;
        }
        metalrast::InteractiveApp app(device, std::move(ac));
        app.run();
        device->release();
        pool->release();
        return 0;
    }

    // -------- Renderer ----------------------------------------------------
    metalrast::RendererConfig cfg{};
    cfg.width  = args.width;
    cfg.height = args.height;
    // Bump stage2 capacity heuristically — at most every triangle could go to s2.
    // Residency mode doesn't have a Scene to inspect; size for a reasonable
    // upper bound (at the page-in cap times average triangleCount).
    uint64_t totalTrisForCfg = useResidency
        ? uint64_t(64u) * 200000u   // headroom; resized later if too small
        : scene.totalTriangles();
    cfg.stage2Capacity = std::max<uint32_t>(1u << 16,
                                            uint32_t(totalTrisForCfg / 2 + 1024));
    cfg.stage1Threadgroups = static_cast<uint32_t>(args.stage1TG);

    metalrast::MetalRastRenderer renderer(device, cfg, METALRAST_METALLIB_PATH);

    // Residency mode: stand up a manager + register meshes; the renderer
    // gets its buffers via bindFrameView each frame. Otherwise upload the
    // already-built Scene up-front.
    std::unique_ptr<metalrast::ResidencyManager> mgr;
    if (useResidency) {
        metalrast::ResidencyManager::Config rc;
        rc.compressed     = args.compressed;
        rc.bitPackIndices = args.bitPacked;
        if (args.residencyPoolMiB) {
            rc.vertPoolBytes = size_t(args.residencyPoolMiB) << 20;
            rc.idxPoolBytes  = size_t(args.residencyPoolMiB) << 20;
        }
        if (args.residencyVertMiB) rc.vertPoolBytes = size_t(args.residencyVertMiB) << 20;
        if (args.residencyIdxMiB)  rc.idxPoolBytes  = size_t(args.residencyIdxMiB)  << 20;
        mgr = std::make_unique<metalrast::ResidencyManager>(device, rc);
        if (!args.glbPath.empty()) {
            metalrast::loadGLBToRegistry(args.glbPath, *mgr, /*maxIndices*/ 0);
        } else {
            metalrast::buildZorahLikeRegistry(*mgr, uint32_t(args.zorahInstances));
        }
    } else {
        renderer.uploadScene(scene);
    }

    // -------- Camera ------------------------------------------------------
    metalrast::Camera cam;
    cam.fovYRadians = 60.0f * 3.14159265f / 180.0f;
    cam.nearPlane   = 0.05f;
    cam.farPlane    = 500.0f;
    if (!args.glbPath.empty() || (useResidency && args.zorahInstances > 0)) {
        // World-space AABB: for each draw call, build the local bbox of its
        // vertices and transform by each instance's full model matrix.
        simd_float3 lo, hi;
        if (useResidency) {
            mgr->sceneAABB(lo, hi);
        } else {
            lo = (simd_float3){  1e30f,  1e30f,  1e30f };
            hi = (simd_float3){ -1e30f, -1e30f, -1e30f };
            for (auto const& dc : scene.drawCalls) {
                // The mesh's AABB is already on the DrawCall (computed at upload).
                simd_float3 mLo = (simd_float3){ dc.aabbMinX, dc.aabbMinY, dc.aabbMinZ };
                simd_float3 mHi = (simd_float3){
                    dc.aabbMinX + dc.compressionFactorX * 65535.0f,
                    dc.aabbMinY + dc.compressionFactorY * 65535.0f,
                    dc.aabbMinZ + dc.compressionFactorZ * 65535.0f,
                };
                // Transform 8 corners by each instance and extend the world bbox.
                for (uint32_t inst = 0; inst < dc.instanceCount; ++inst) {
                    simd_float4x4 m = scene.modelMatrices[dc.firstInstance + inst];
                    simd_float3 corners[8] = {
                        {mLo.x, mLo.y, mLo.z}, {mHi.x, mLo.y, mLo.z},
                        {mLo.x, mHi.y, mLo.z}, {mHi.x, mHi.y, mLo.z},
                        {mLo.x, mLo.y, mHi.z}, {mHi.x, mLo.y, mHi.z},
                        {mLo.x, mHi.y, mHi.z}, {mHi.x, mHi.y, mHi.z},
                    };
                    for (auto const& c : corners) {
                        simd_float4 w = simd_mul(m, (simd_float4){c.x, c.y, c.z, 1.0f});
                        lo = simd_make_float3(std::min(lo.x, w.x), std::min(lo.y, w.y), std::min(lo.z, w.z));
                        hi = simd_make_float3(std::max(hi.x, w.x), std::max(hi.y, w.y), std::max(hi.z, w.z));
                    }
                }
            }
        }
        simd_float3 center = (simd_float3){ (lo.x+hi.x)*0.5f, (lo.y+hi.y)*0.5f, (lo.z+hi.z)*0.5f };
        simd_float3 size   = (simd_float3){  hi.x-lo.x,        hi.y-lo.y,        hi.z-lo.z       };
        float radius = std::max({size.x, size.y, size.z}) * 0.6f;
        if (radius < 1e-3f) radius = 1.0f;
        // Camera convention: CuRast looks +Z. We orbit around the AABB based
        // on --cam (default -z). For glTF assets you typically need to try a
        // few sides — the asset's "intended view" varies.
        float dist = radius * args.camPullback;
        if      (args.camSide == "-z") cam.eye = (simd_float3){ center.x,        center.y,        center.z - dist };
        else if (args.camSide == "+z") cam.eye = (simd_float3){ center.x,        center.y,        center.z + dist };
        else if (args.camSide == "+x") cam.eye = (simd_float3){ center.x + dist, center.y,        center.z };
        else if (args.camSide == "-x") cam.eye = (simd_float3){ center.x - dist, center.y,        center.z };
        else if (args.camSide == "+y") cam.eye = (simd_float3){ center.x,        center.y + dist, center.z };
        else if (args.camSide == "-y") cam.eye = (simd_float3){ center.x,        center.y - dist, center.z };
        else                            cam.eye = (simd_float3){ center.x,        center.y,        center.z - dist };
        cam.target = center;
        cam.up     = (simd_float3){ 0.0f, 1.0f, 0.0f };
        cam.farPlane = radius * 20.0f;
        cam.nearPlane = std::max(0.01f, radius * 0.001f);
        std::printf("GLB scene AABB: (%.2f, %.2f, %.2f) — (%.2f, %.2f, %.2f), "
                    "size (%.2f, %.2f, %.2f), framing radius %.2f\n",
                    lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
                    size.x, size.y, size.z, radius);
    } else if (args.zorahInstances > 0) {
        // Pull the camera back so the instance grid roughly fills the frame.
        // makeZorahLikeScene uses spacing 2.5 and a cube of side ceil(N^(1/3)).
        float side = std::cbrt(float(args.zorahInstances));
        float halfExtent = (side - 1.0f) * 0.5f * 2.5f + 1.5f;
        float dist = halfExtent * 2.2f;
        cam.eye    = (simd_float3){0.0f, 0.0f, -dist};
        cam.target = (simd_float3){0.0f, 0.0f,  0.0f};
        cam.up     = (simd_float3){0.0f, 1.0f,  0.0f};
        cam.farPlane = dist * 4.0f;
    } else {
        cam.eye    = (simd_float3){0.0f, 0.4f, -3.0f};
        cam.target = (simd_float3){0.0f, 0.0f,  0.0f};
        cam.up     = (simd_float3){0.0f, 1.0f,  0.0f};
    }

    // --load-camera overrides everything above. Used to deterministically
    // replay a camera the user picked in the interactive viewer.
    if (!args.loadCamera.empty()) {
        if (metalrast::loadCameraFromFile(cam, args.loadCamera)) {
            std::printf("Loaded camera from %s: eye=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) fov=%.1f deg\n",
                        args.loadCamera.c_str(),
                        cam.eye.x, cam.eye.y, cam.eye.z,
                        cam.target.x, cam.target.y, cam.target.z,
                        cam.fovYRadians * 180.0f / 3.14159265f);
        } else {
            std::fprintf(stderr, "Failed to load camera from %s; using auto-frame.\n",
                         args.loadCamera.c_str());
        }
    }

    // -------- Render loop -------------------------------------------------
    const int totalFrames  = args.benchmark ? args.frames : 1;
    const int warmupFrames = args.benchmark ? std::min(args.warmup, totalFrames - 1) : 0;

    std::vector<double> gpuTimesMs;
    gpuTimesMs.reserve(totalFrames);
    metalrast::FrameStats lastStats{};

    // Residency warm-up: run prepareFrame + sleep + bindFrameView in a loop
    // until every visible mesh is resident, so the captured render is
    // complete (no missing primitives because the worker hadn't finished).
    // Caps at 200 iterations to keep tests bounded.
    if (useResidency) {
        for (int it = 0; it < 200; ++it) {
            auto view = mgr->prepareFrame(cam, /*frameIdx*/ uint64_t(it));
            renderer.bindFrameView(view);
            auto s = mgr->stats();
            if (s.visibleMeshes > 0 &&
                s.residentMeshes >= s.visibleMeshes &&
                s.loadingMeshes == 0)
            {
                std::printf("Residency warm-up: settled at iter %d "
                            "(visible=%u resident=%u DCs=%u, vert=%.1f/%.1f MiB, idx=%.1f/%.1f MiB)\n",
                            it, s.visibleMeshes, s.residentMeshes, view.numDrawCalls,
                            double(s.vertPoolUsed)/(1<<20), double(s.vertPoolCapacity)/(1<<20),
                            double(s.idxPoolUsed)/(1<<20),  double(s.idxPoolCapacity)/(1<<20));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    renderer.setHiZOcclusion(args.hiZ);
    renderer.setUsePrefixSum(args.prefixSum);
    renderer.setHiZTwoPhase (args.hiZTwo);
    renderer.setCompactInstances(args.compactInstances);

    // -------- Counter-sample buffer (--profile-counters) ------------------
    // Discover the timestamp counter set (always supported on M-series),
    // verify dispatch-boundary sampling is supported, and allocate one
    // sample buffer reused across frames.
    MTL::CounterSampleBuffer*  tsSampleBuf  = nullptr;
    std::vector<metalrast::MetalRastRenderer::CounterReport> counterReports;
    constexpr NS::UInteger kMaxStageSamples = 32;  // start+end per labeled stage; 16 stages worst case
    if (args.profileCounters) {
        // Apple Silicon supports sampling at compute-pass (stage) boundary,
        // not at dispatch boundary. We split each labeled stage into its own
        // compute encoder so the start/end samples bracket exactly that stage.
        const bool stageOk = device->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary);
        if (!stageOk) {
            std::fprintf(stderr,
                "--profile-counters: device does not support sampling at stage boundary; "
                "falling back to plain --benchmark output\n");
        } else {
            // Find the timestamp counter set by name match.
            MTL::CounterSet* tsCounterSet = nullptr;
            NS::Array* sets = device->counterSets();
            for (NS::UInteger i = 0; sets && i < sets->count(); ++i) {
                auto* cs = static_cast<MTL::CounterSet*>(sets->object(i));
                if (cs && cs->name() &&
                    cs->name()->isEqualToString(MTL::CommonCounterSetTimestamp)) {
                    tsCounterSet = cs;
                    break;
                }
            }
            if (!tsCounterSet) {
                std::fprintf(stderr,
                    "--profile-counters: timestamp counter set not found on this device\n");
            } else {
                MTL::CounterSampleBufferDescriptor* d =
                    MTL::CounterSampleBufferDescriptor::alloc()->init();
                d->setCounterSet(tsCounterSet);
                d->setStorageMode(MTL::StorageModeShared);
                d->setSampleCount(kMaxStageSamples);
                d->setLabel(NS::String::string("MetalRast timestamps", NS::UTF8StringEncoding));
                NS::Error* err = nullptr;
                tsSampleBuf = device->newCounterSampleBuffer(d, &err);
                d->release();
                if (!tsSampleBuf) {
                    const char* msg = err ? err->localizedDescription()->utf8String() : "unknown";
                    std::fprintf(stderr,
                        "--profile-counters: newCounterSampleBuffer failed: %s\n", msg);
                } else {
                    std::printf("--profile-counters: timestamp counter set ready (%llu samples)\n",
                                (unsigned long long)kMaxStageSamples);
                }
            }
        }
        counterReports.reserve(totalFrames);
    }

    for (int f = 0; f < totalFrames; ++f) {
        if (args.spinCamera) {
            float angle = 2.0f * 3.14159265f * float(f) / float(totalFrames);
            float r     = 3.2f;
            cam.eye = (simd_float3){r * std::sin(angle), 0.4f, -r * std::cos(angle)};
        }
        renderer.setCamera(cam.uniforms(args.width, args.height));

        if (useResidency) {
            auto view = mgr->prepareFrame(cam, /*frameIdx*/ 1000ull + uint64_t(f));
            renderer.bindFrameView(view);
        }

        // GPU trace capture for the requested frame. MTLCaptureManager
        // writes a .gputrace document the user opens in Xcode for
        // performance-counter inspection (L2 hit rate, occupancy,
        // etc.). Requires `MTL_CAPTURE_ENABLED=1` in the environment.
        bool doCapture = (args.captureFrame == f + 1);
        MTL::CaptureManager* capMgr = nullptr;
        if (doCapture) {
            capMgr = MTL::CaptureManager::sharedCaptureManager();
            if (!capMgr->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
                std::fprintf(stderr,
                    "--capture-frame: GPU trace destination unsupported "
                    "(set MTL_CAPTURE_ENABLED=1 before launch)\n");
                doCapture = false;
            } else {
                MTL::CaptureDescriptor* desc = MTL::CaptureDescriptor::alloc()->init();
                desc->setCaptureObject(device);
                desc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
                NS::String* path = NS::String::string(args.captureOutput.c_str(),
                                                       NS::UTF8StringEncoding);
                desc->setOutputURL(NS::URL::fileURLWithPath(path));
                NS::Error* err = nullptr;
                if (!capMgr->startCapture(desc, &err)) {
                    const char* msg = err ? err->localizedDescription()->utf8String() : "unknown";
                    std::fprintf(stderr, "--capture-frame: startCapture failed: %s\n", msg);
                    doCapture = false;
                } else {
                    std::printf("--capture-frame: capturing frame %d → %s\n",
                                f + 1, args.captureOutput.c_str());
                }
                desc->release();
            }
        }

        metalrast::FrameStats st{};
        if (tsSampleBuf) {
            metalrast::MetalRastRenderer::CounterReport rep;
            renderer.renderAndWaitWithCounters(tsSampleBuf, rep);
            st.gpuMilliseconds = rep.gpuTotalMs;
            st.stage2Items     = rep.stage2Items;
            st.stage3Items     = rep.stage3Items;
            counterReports.push_back(std::move(rep));
        } else if (args.profile) {
            st = renderer.renderAndWaitProfiled();
        } else {
            st = renderer.renderAndWait();
        }

        if (doCapture && capMgr) {
            capMgr->stopCapture();
            std::printf("--capture-frame: wrote %s\n",
                        args.captureOutput.c_str());
        }
        gpuTimesMs.push_back(st.gpuMilliseconds);
        lastStats = st;

        if (args.benchmark) {
            if (args.profile) {
                std::printf("  frame %3d/%d  total=%7.3f  clear=%6.3f  stage1=%7.3f"
                            "  s23=%6.3f  resolve=%6.3f  s2=%u  s3=%u%s\n",
                            f + 1, totalFrames,
                            st.gpuMilliseconds, st.clearMs, st.stage1Ms,
                            st.stage23Ms, st.resolveMs,
                            st.stage2Items, st.stage3Items,
                            f < warmupFrames ? "  (warmup)" : "");
            } else {
                std::printf("  frame %3d/%d  gpu=%7.3f ms  stage2=%7u  stage3=%6u%s\n",
                            f + 1, totalFrames,
                            st.gpuMilliseconds, st.stage2Items, st.stage3Items,
                            f < warmupFrames ? "  (warmup)" : "");
            }
        }
    }

    // -------- Stats -------------------------------------------------------
    if (args.benchmark && totalFrames > warmupFrames) {
        std::vector<double> kept(gpuTimesMs.begin() + warmupFrames, gpuTimesMs.end());
        std::sort(kept.begin(), kept.end());
        double sum = std::accumulate(kept.begin(), kept.end(), 0.0);
        double mean = sum / kept.size();
        double minMs = kept.front();
        double maxMs = kept.back();
        double medMs = kept[kept.size() / 2];
        double p99   = kept[std::min<size_t>(kept.size() - 1,
                                             size_t(kept.size() * 0.99))];
        uint64_t totalTrisForMtps = useResidency
            ? renderer.totalTriangles()
            : scene.totalTriangles();
        double mtps  = double(totalTrisForMtps) * 1e-6 / (mean * 1e-3);

        std::printf("\nGPU timing across %zu measured frames"
                    " (warmup %d frames discarded):\n",
                    kept.size(), warmupFrames);
        std::printf("  min  = %7.3f ms\n", minMs);
        std::printf("  med  = %7.3f ms\n", medMs);
        std::printf("  mean = %7.3f ms\n", mean);
        std::printf("  p99  = %7.3f ms\n", p99);
        std::printf("  max  = %7.3f ms\n", maxMs);
        std::printf("Throughput @ mean: %.1f Mtri/s, %.1f fps\n",
                    mtps, 1000.0 / mean);
    } else {
        std::printf("GPU time: %.3f ms  (stage2=%u, stage3=%u)\n",
                    lastStats.gpuMilliseconds,
                    lastStats.stage2Items, lastStats.stage3Items);
    }

    // -------- Counter-sample per-stage table (--profile-counters) --------
    if (tsSampleBuf && !counterReports.empty()) {
        // Skip warmup frames (same convention as the gpuTimes stats above).
        const size_t firstIdx = size_t(warmupFrames);
        if (firstIdx >= counterReports.size()) {
            std::fprintf(stderr,
                "--profile-counters: no measured frames after %d warmup frames\n",
                warmupFrames);
        } else {
            // Aggregate by stage label across measured frames. Use std::map
            // for stable label order = first-seen order (we use vector pair).
            std::vector<std::string> labels;
            std::vector<std::vector<double>> samplesByStage; // µs per frame
            auto findOrPush = [&](const std::string& l) -> size_t {
                for (size_t i = 0; i < labels.size(); ++i) if (labels[i] == l) return i;
                labels.push_back(l);
                samplesByStage.emplace_back();
                return labels.size() - 1;
            };
            for (size_t fi = firstIdx; fi < counterReports.size(); ++fi) {
                for (auto& s : counterReports[fi].stages) {
                    size_t idx = findOrPush(s.label);
                    samplesByStage[idx].push_back(s.microseconds);
                }
            }
            std::printf("\n--profile-counters: per-stage GPU time (median over %zu measured frames)\n",
                        counterReports.size() - firstIdx);
            std::printf("  %-26s %12s %12s %12s\n", "Stage", "median µs", "min µs", "max µs");
            std::printf("  %-26s %12s %12s %12s\n", "-----", "---------", "------", "------");
            double totalMedianUs = 0.0;
            for (size_t i = 0; i < labels.size(); ++i) {
                auto v = samplesByStage[i];
                if (v.empty()) continue;
                std::sort(v.begin(), v.end());
                const double med = v[v.size() / 2];
                const double mn  = v.front();
                const double mx  = v.back();
                std::printf("  %-26s %12.1f %12.1f %12.1f\n", labels[i].c_str(), med, mn, mx);
                totalMedianUs += med;
            }
            std::printf("  %-26s %12.1f\n", "(sum of stage medians)", totalMedianUs);
        }

        if (!args.countersOutput.empty()) {
            FILE* fp = std::fopen(args.countersOutput.c_str(), "w");
            if (!fp) {
                std::fprintf(stderr, "--counters-output: could not open %s for write\n",
                             args.countersOutput.c_str());
            } else {
                std::fprintf(fp, "{\n  \"frames\": [\n");
                for (size_t fi = 0; fi < counterReports.size(); ++fi) {
                    auto& r = counterReports[fi];
                    std::fprintf(fp, "    {\"frame\": %zu, \"total_ms\": %.6f, \"warmup\": %s, \"stages\": [",
                                 fi, r.gpuTotalMs, fi < size_t(warmupFrames) ? "true" : "false");
                    for (size_t i = 0; i < r.stages.size(); ++i) {
                        if (i) std::fprintf(fp, ", ");
                        std::fprintf(fp, "{\"label\": \"%s\", \"us\": %.3f}",
                                     r.stages[i].label.c_str(), r.stages[i].microseconds);
                    }
                    std::fprintf(fp, "]}%s\n", fi + 1 == counterReports.size() ? "" : ",");
                }
                std::fprintf(fp, "  ]\n}\n");
                std::fclose(fp);
                std::printf("--counters-output: wrote %s\n", args.countersOutput.c_str());
            }
        }
        tsSampleBuf->release();
        tsSampleBuf = nullptr;
    }

    // -------- PNG output --------------------------------------------------
    if (!args.noPng) {
        auto rgba = renderer.readbackRGBA();
        int ok = stbi_write_png(args.output.c_str(),
                                int(args.width), int(args.height),
                                /*comp*/ 4, rgba.data(),
                                /*stride*/ int(args.width) * 4);
        if (!ok) {
            std::fprintf(stderr, "stbi_write_png failed for %s\n", args.output.c_str());
            return 1;
        }
        std::printf("Wrote %s (%u × %u)\n",
                    args.output.c_str(), args.width, args.height);
    }

    device->release();
    pool->release();
    return 0;
}
