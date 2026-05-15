#include "InteractiveApp.h"
#include "GLBLoader.h"

#include "imgui.h"

#include <Foundation/Foundation.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <stb_image_write.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace metalrast {

namespace {

// World-space AABB of a Scene, accounting for per-instance modelMatrices.
void sceneAABB(const Scene& s, simd_float3& lo, simd_float3& hi) {
    lo = (simd_float3){  1e30f,  1e30f,  1e30f };
    hi = (simd_float3){ -1e30f, -1e30f, -1e30f };
    for (auto const& dc : s.drawCalls) {
        simd_float3 mLo = (simd_float3){ dc.aabbMinX, dc.aabbMinY, dc.aabbMinZ };
        simd_float3 mHi = (simd_float3){
            dc.aabbMinX + dc.compressionFactorX * 65535.0f,
            dc.aabbMinY + dc.compressionFactorY * 65535.0f,
            dc.aabbMinZ + dc.compressionFactorZ * 65535.0f,
        };
        for (uint32_t i = 0; i < dc.instanceCount; ++i) {
            simd_float4x4 m = s.modelMatrices[dc.firstInstance + i];
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
    if (lo.x > hi.x) { lo = (simd_float3){-1,-1,-1}; hi = (simd_float3){1,1,1}; }
}

}  // namespace

InteractiveApp::InteractiveApp(MTL::Device* device, Config cfg)
    : device_(device), cfg_(std::move(cfg))
{
    window_  = std::make_unique<Window>(device, cfg_.window);
    rendererW_ = static_cast<uint32_t>(window_->framebufferWidth());
    rendererH_ = static_cast<uint32_t>(window_->framebufferHeight());
    rebuildRenderer(rendererW_, rendererH_);

    cam_.setViewport(rendererW_, rendererH_);

    // CLI override for the Hi-Z toggle's initial state.
    vs_.hiZOcclusion = cfg_.initialHiZ;
    vs_.compactInstances = cfg_.initialCompactInstances;

    loadRecentsFromDisk();

    if (!cfg_.residencyPath.empty()) {
        // Streaming path: register with a ResidencyManager rather than
        // uploading geometry up-front. Camera frames the world AABB
        // computed during registration.
        // Seed the Residency-panel sliders from the configured pool sizes
        // so what the user sees matches what's actually allocated.
        vs_.pendingVertPoolMiB = int(cfg_.residencyCfg.vertPoolBytes >> 20);
        vs_.pendingIdxPoolMiB  = int(cfg_.residencyCfg.idxPoolBytes  >> 20);
        residency_ = std::make_unique<ResidencyManager>(device_, cfg_.residencyCfg);
        try {
            loadGLBToRegistry(cfg_.residencyPath, *residency_, /*maxIndices*/ 0);
        } catch (std::exception& e) {
            std::fprintf(stderr, "Residency load failed: %s\n", e.what());
            residency_.reset();
        }
        if (residency_) {
            simd_float3 lo, hi;
            residency_->sceneAABB(lo, hi);
            cam_.frameAABB(lo, hi);
            vs_.lastLoadedPath = cfg_.residencyPath;
        }
    } else {
        scene_ = std::move(cfg_.initialScene);
        if (!cfg_.initialPath.empty()) {
            vs_.lastLoadedPath = cfg_.initialPath;
            auto it = std::find(vs_.recentFiles.begin(), vs_.recentFiles.end(),
                                cfg_.initialPath);
            if (it != vs_.recentFiles.end()) vs_.recentFiles.erase(it);
            vs_.recentFiles.push_front(cfg_.initialPath);
            while (vs_.recentFiles.size() > 10) vs_.recentFiles.pop_back();
            saveRecentsToDisk();
        }
        uploadAndFrame(scene_);
    }

    // After the auto-frame, optionally restore a user-saved camera so they
    // can resume orbiting from that exact angle.
    if (!cfg_.loadCamera.empty()) {
        Camera c;
        if (loadCameraFromFile(c, cfg_.loadCamera)) {
            cam_.setFromCamera(c);
            std::printf("Loaded camera from %s\n", cfg_.loadCamera.c_str());
        } else {
            std::fprintf(stderr, "Failed to load camera from %s\n",
                         cfg_.loadCamera.c_str());
        }
    }

    // ----- Hook input -------------------------------------------------------
    // ImGui swallows input first; only forward to the camera if ImGui isn't
    // capturing it AND camera-freeze isn't on.
    auto cameraActive = [this]() {
        if (vs_.freezeCamera) return false;
        ImGuiIO& io = ImGui::GetIO();
        return !io.WantCaptureMouse;
    };
    auto keyboardForUs = [this]() {
        ImGuiIO& io = ImGui::GetIO();
        return !io.WantCaptureKeyboard;
    };

    auto pushModifiers = [this]() {
        GLFWwindow* w = window_->glfwHandle();
        bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)   == GLFW_PRESS
                  || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT)  == GLFW_PRESS;
        bool ctrl  = glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS
                  || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL)== GLFW_PRESS
                  || glfwGetKey(w, GLFW_KEY_LEFT_SUPER)   == GLFW_PRESS    // ⌘ on macOS
                  || glfwGetKey(w, GLFW_KEY_RIGHT_SUPER)  == GLFW_PRESS;
        cam_.setModifiers(shift, ctrl);
    };
    window_->onCursorMove([this, cameraActive, pushModifiers](double x, double y){
        if (!cameraActive()) return;
        pushModifiers();
        cam_.onCursorMove(x, y);
    });
    window_->onMouseButton([this, cameraActive, pushModifiers](int b, int a, int m){
        if (!cameraActive()) return;
        pushModifiers();
        cam_.onMouseButton(b, a, m);
    });
    window_->onScroll([this, cameraActive](double dx, double dy){
        if (cameraActive()) cam_.onScroll(dx, dy);
    });
    window_->onKey([this, keyboardForUs](int key, int action, int /*mods*/){
        if (action != GLFW_PRESS) return;
        if (!keyboardForUs()) return;
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window_->glfwHandle(), GLFW_TRUE);
        } else if (key == GLFW_KEY_R) {
            cam_.resetView();
        } else if (key == GLFW_KEY_F1) {
            vs_.showStats = !vs_.showStats;
        } else if (key == GLFW_KEY_O) {
            std::string path = runGLBOpenPanel();
            if (!path.empty()) loadGLBFromPath(path);
        }
    });
    window_->onResize([this](int newW, int newH){
        if (newW <= 0 || newH <= 0) return;
        rebuildRenderer(newW, newH);
        cam_.setViewport(newW, newH);
    });
    window_->onDrop([this](int count, const char** paths){
        if (count < 1) return;
        loadGLBFromPath(paths[0]);
    });

    imgui_.init(device_, window_->glfwHandle(), window_->colorPixelFormat());

    appliedStage1TG_    = vs_.stage1TG;
    appliedCompressed_  = vs_.compressed;
    appliedBitPacked_   = vs_.bitPacked;
}

InteractiveApp::~InteractiveApp() {
    imgui_.shutdown();
    renderer_.reset();
    window_.reset();
}

// Build a renderer sized to (w, h). Called on init AND on window resize so the
// render texture matches the drawable.
void InteractiveApp::rebuildRenderer(uint32_t w, uint32_t h) {
    RendererConfig rc;
    rc.width  = w;
    rc.height = h;
    rc.stage1Threadgroups = static_cast<uint32_t>(vs_.stage1TG);
    // Residency mode keeps drawCalls empty (geometry pages in per frame from
    // the registry), so totalTriangles() is 0 and the default below would
    // pick the 64K floor — far too small for dense scenes (Zorah balcony
    // balusters overflow stage2 → dropped tris → visible holes). Use the
    // RendererConfig default (currently 8M) as the floor instead.
    rc.stage2Capacity = std::max<uint32_t>(rc.stage2Capacity,
                                            uint32_t(scene_.totalTriangles() / 2 + 1024));
    renderer_ = std::make_unique<MetalRastRenderer>(device_, rc, cfg_.metallibPath);
    rendererW_ = w; rendererH_ = h;
    if (!scene_.drawCalls.empty()) {
        renderer_->uploadScene(scene_);
    }
}

void InteractiveApp::uploadAndFrame(Scene& scene) {
    if (scene.drawCalls.empty()) return;
    // Compute AABB BEFORE upload (uploadScene moves out vertex/index host
    // buffers; AABB only needs DrawCalls + modelMatrices which stay).
    simd_float3 lo, hi;
    sceneAABB(scene, lo, hi);
    renderer_->uploadScene(scene);
    cam_.frameAABB(lo, hi);
}

void InteractiveApp::loadGLBFromPath(const std::string& path) {
    std::printf("Loading %s ...\n", path.c_str());
    try {
        Scene s = loadGLB(path, /*maxIndices*/ 0,
                          /*compressed*/ vs_.compressed,
                          /*bitPackIndices*/ vs_.bitPacked,
                          /*directDevice*/ device_);
        scene_ = std::move(s);
        vs_.lastLoadedPath = path;
        // Move-to-front in recent list (de-duped).
        auto it = std::find(vs_.recentFiles.begin(), vs_.recentFiles.end(), path);
        if (it != vs_.recentFiles.end()) vs_.recentFiles.erase(it);
        vs_.recentFiles.push_front(path);
        while (vs_.recentFiles.size() > 10) vs_.recentFiles.pop_back();
        saveRecentsToDisk();
        uploadAndFrame(scene_);
    } catch (std::exception& e) {
        std::fprintf(stderr, "Load failed: %s\n", e.what());
    }
}

namespace {
std::string recentsPath() {
    const char* home = std::getenv("HOME");
    std::string p = home ? home : ".";
    p += "/.metalrast_recent";
    return p;
}
}

void InteractiveApp::loadRecentsFromDisk() {
    std::ifstream f(recentsPath());
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        vs_.recentFiles.push_back(line);
        if (vs_.recentFiles.size() >= 10) break;
    }
}

void InteractiveApp::saveRecentsToDisk() const {
    std::ofstream f(recentsPath(), std::ios::trunc);
    if (!f) return;
    for (auto const& p : vs_.recentFiles) f << p << '\n';
}

void InteractiveApp::run() {
    using clock = std::chrono::steady_clock;
    auto lastTick = clock::now();

    // Single command queue, reused every frame.
    static MTL::CommandQueue* sQueue = device_->newCommandQueue();

    while (!window_->shouldClose()) {
        window_->pollEvents();

        // Per-frame autorelease pool — Metal returns autoreleased objects in
        // a couple of paths (drawables, command buffers).
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        // ---- Per-frame timestep -------------------------------------------
        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - lastTick).count();
        lastTick = now;
        if (dt > 0.25f) dt = 0.25f;

        // ---- Hot-rebuild renderer if UI knobs changed ---------------------
        if (vs_.compressed != appliedCompressed_ ||
            vs_.bitPacked  != appliedBitPacked_)
        {
            // Re-load scene with the new compression / packing setting if a
            // real GLB is the active source.
            if (!vs_.lastLoadedPath.empty()) {
                std::string p = vs_.lastLoadedPath;
                loadGLBFromPath(p);
            }
            appliedCompressed_ = vs_.compressed;
            appliedBitPacked_  = vs_.bitPacked;
        }
        if (vs_.stage1TG != appliedStage1TG_) {
            // Pure dispatch shape change — no rebuild, no re-upload, no
            // host-data dependency. Just poke the renderer's cfg_.
            renderer_->setStage1Threadgroups(static_cast<uint32_t>(vs_.stage1TG));
            appliedStage1TG_ = vs_.stage1TG;
        }

        // ---- Camera --------------------------------------------------------
        if (vs_.spinCamera) {
            cam_.bumpYaw(0.6f * dt);    // ~ 1 rev / 10 s
        }

        // WSAD/QE FPS-style fly. Speed scales with current orbit distance so
        // the controls feel right at any zoom (Zorah at 100 km / komainu at
        // 1 m alike). Shift = 4× sprint, Ctrl = 0.25× crawl.
        if (!ImGui::GetIO().WantCaptureKeyboard && !vs_.freezeCamera) {
            GLFWwindow* w = window_->glfwHandle();
            float vx = 0, vy = 0, vz = 0;
            if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) vz += 1.0f;
            if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) vz -= 1.0f;
            if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) vx += 1.0f;
            if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) vx -= 1.0f;
            if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) vy += 1.0f;
            if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) vy -= 1.0f;
            if (vx || vy || vz) {
                bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS
                          || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
                bool ctrl  = glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS
                          || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL)== GLFW_PRESS;
                float speed = cam_.distance() * 1.0f;   // 1× distance / sec
                if (shift) speed *= 4.0f;
                if (ctrl)  speed *= 0.25f;
                simd_float3 f = cam_.forwardDir();
                simd_float3 r = cam_.rightDir();
                simd_float3 d = (simd_float3){
                    (f.x * vz + r.x * vx)              * speed * dt,
                    (f.y * vz +              vy)       * speed * dt,
                    (f.z * vz + r.z * vx)              * speed * dt };
                cam_.pan(d);
            }
        }
        renderer_->setCamera(cam_.toCamera().uniforms(rendererW_, rendererH_));
        renderer_->setVisMode(uint32_t(vs_.visMode));
        renderer_->setRasterMode(
            vs_.rasterMode == 1 ? RasterMode::VisbufferIndexed
          : vs_.rasterMode == 2 ? RasterMode::VisbufferInstanced
                                : RasterMode::Auto);
        renderer_->setHiZOcclusion(vs_.hiZOcclusion);
        renderer_->setUsePrefixSum(vs_.usePrefixSum);
        renderer_->setHiZTwoPhase (vs_.hiZTwoPhase);
        renderer_->setCompactInstances(vs_.compactInstances);

        // ---- Submit + present ---------------------------------------------
        // Acquire the drawable BEFORE imgui.newFrame — the Metal backend
        // caches the colour-attachment format/sampleCount from a render-pass
        // descriptor on first NewFrame call, so we need a real target.
        CA::MetalDrawable* drawable = window_->nextDrawable();
        if (!drawable) {
            pool->release();
            continue;
        }

        // ---- ImGui frame begin (panels collect inputs before render) ------
        imgui_.newFrame(drawable->texture());

        panels::MenuActions menu;
        if (cfg_.minimalUi) {
            // Single floating "Save Camera" button — nothing else.
            static double sLastSaveTime = -10.0;
            static bool   sLastSaveOk   = false;
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_Always);
            if (ImGui::Begin("##save_cam_only", nullptr,
                             ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings))
            {
                if (ImGui::Button("Save Camera (Esc to quit)")) {
                    sLastSaveOk   = saveCameraToFile(cam_.toCamera(), defaultCameraStatePath());
                    sLastSaveTime = ImGui::GetTime();
                }
                if (ImGui::GetTime() - sLastSaveTime < 2.0) {
                    ImGui::TextColored(sLastSaveOk ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                                                   : ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                                       sLastSaveOk ? "Saved." : "Save failed.");
                }
            }
            ImGui::End();
        } else {
            panels::DrawMenuBar      (vs_, &menu);
            panels::DrawStatsWindow  (vs_, lastStats_);
            panels::DrawSceneInfoWindow(vs_, scene_);
            RendererConfig dummyCfg;          // currently unused by the panel
            panels::DrawRenderSettings(vs_, dummyCfg);
            if (residency_) panels::DrawResidencyWindow(residency_->stats(), vs_);
            panels::DrawCameraWindow (vs_, cam_);
            if (vs_.showDemo) ImGui::ShowDemoWindow(&vs_.showDemo);
        }

        if (menu.requestQuit) {
            glfwSetWindowShouldClose(window_->glfwHandle(), GLFW_TRUE);
        }
        if (menu.requestLoadDialog) {
            std::string path = runGLBOpenPanel();
            if (!path.empty()) loadGLBFromPath(path);
        }
        if (!menu.pickedRecent.empty() && menu.pickedRecent != vs_.lastLoadedPath) {
            loadGLBFromPath(menu.pickedRecent);
        }

        // Pool resize from the Residency panel. Tear down the manager (which
        // drops all paged GPU buffers), build a new one with the requested
        // pool sizes, and reload the GLB. Camera state is preserved.
        if (vs_.requestResidencyResize && !cfg_.residencyPath.empty()) {
            vs_.requestResidencyResize = false;
            cfg_.residencyCfg.vertPoolBytes = size_t(vs_.pendingVertPoolMiB) << 20;
            cfg_.residencyCfg.idxPoolBytes  = size_t(vs_.pendingIdxPoolMiB)  << 20;
            std::printf("Rebuilding residency: vert=%d MiB  idx=%d MiB\n",
                        vs_.pendingVertPoolMiB, vs_.pendingIdxPoolMiB);
            residency_.reset();
            residency_ = std::make_unique<ResidencyManager>(device_, cfg_.residencyCfg);
            try {
                loadGLBToRegistry(cfg_.residencyPath, *residency_, /*maxIndices*/ 0);
            } catch (std::exception& e) {
                std::fprintf(stderr, "Residency rebuild failed: %s\n", e.what());
                residency_.reset();
            }
            // Renderer was bound to the previous registry's buffers; rebuild
            // it so its bindings refresh, then re-bind the frame view.
            rebuildRenderer(rendererW_, rendererH_);
        }

        if (residency_) {
            auto view = residency_->prepareFrame(cam_.toCamera(), frameIdx_);
            renderer_->bindFrameView(view);
        }
        ++frameIdx_;

        MTL::CommandBuffer* cb = sQueue->commandBuffer();
        cb->retain();

        // 1. Renderer compute pipeline → outputTex_.
        renderer_->encodeRender(cb);
        // 2. Present pass → drawable->texture.
        renderer_->encodePresent(cb, drawable->texture());
        // 3. ImGui overlay on the drawable.
        imgui_.render(cb, drawable->texture(), window_->colorPixelFormat());

        cb->presentDrawable(drawable);

        cb->commit();
        cb->waitUntilCompleted();

        FrameStats st{};
        st.gpuMilliseconds = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        if (renderer_->stage2CounterBuf()) {
            st.stage2Items = *static_cast<const uint32_t*>(
                renderer_->stage2CounterBuf()->contents());
        }
        if (renderer_->stage3CounterBuf()) {
            st.stage3Items = *static_cast<const uint32_t*>(
                renderer_->stage3CounterBuf()->contents());
        }
        lastStats_ = st;
        panels::PushFrameTime(vs_, float(st.gpuMilliseconds));

        // Screenshot capture: dumps the renderer's offscreen output (no ImGui
        // overlay) and a sidecar .cam holding the exact camera that produced
        // it. Files go to <repo>/temp_images/ with a timestamp basename so
        // multiple shots don't collide.
        if (vs_.requestScreenshot) {
            // Print stage queue counts at the moment of capture so we can
            // tell if either queue is overflowing its hard cap (which would
            // cause silent drops → missing triangles in the render).
            std::printf("Screenshot stats: stage2=%u/%u  stage3=%u/%u%s%s\n",
                        st.stage2Items, renderer_->stage2Capacity(),
                        st.stage3Items, renderer_->stage3Capacity(),
                        st.stage2Items >= renderer_->stage2Capacity()
                            ? "  STAGE2 OVERFLOW" : "",
                        st.stage3Items >= renderer_->stage3Capacity()
                            ? "  STAGE3 OVERFLOW" : "");
            vs_.requestScreenshot = false;
            auto rgba = renderer_->readbackRGBA();
            // Resolve to repo root's temp_images/, not the build cwd.
            std::filesystem::path exeDir = std::filesystem::current_path();
            std::filesystem::path dir    = exeDir.parent_path() / "temp_images";
            if (!std::filesystem::exists(dir.parent_path() / "CMakeLists.txt"))
                dir = exeDir / "temp_images";  // fallback if cwd is repo root
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            auto now  = std::chrono::system_clock::now();
            auto t    = std::chrono::system_clock::to_time_t(now);
            auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count() % 1000;
            char ts[64];
            std::tm tm{}; localtime_r(&t, &tm);
            std::snprintf(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d_%03lld",
                          tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                          tm.tm_hour, tm.tm_min, tm.tm_sec,
                          (long long)ms);
            std::string base = "screenshot_" + std::string(ts);
            auto pngPath = (dir / (base + ".png")).string();
            auto camPath = (dir / (base + ".cam")).string();
            stbi_write_png(pngPath.c_str(),
                           int(rendererW_), int(rendererH_), 4,
                           rgba.data(), int(rendererW_) * 4);
            saveCameraToFile(cam_.toCamera(), camPath);
            std::printf("Screenshot → %s + %s\n",
                        pngPath.c_str(), camPath.c_str());
        }

        cb->release();
        pool->release();
    }
    // Auto-save the final camera so the next launch (or a benchmark) can
    // resume from where the user left off. Same path as the Save Camera
    // button — overwrites only at clean exit.
    if (saveCameraToFile(cam_.toCamera(), defaultCameraStatePath())) {
        std::printf("Saved camera on exit → %s\n", defaultCameraStatePath().c_str());
    }
}

}  // namespace metalrast
