// DebugPanels.cpp — ImGui windows for the interactive viewer.
// Phase 5 brings the actual UI to life. For now: forwarders that work even
// when the ImGui host isn't initialised yet.

#include "DebugPanels.h"
#include "OrbitCamera.h"

#include "imgui.h"

#include <algorithm>
#include <cstdio>

namespace metalrast::panels {

void PushFrameTime(ViewerState& vs, float ms) {
    vs.frameMsHistory[vs.frameMsHead] = ms;
    vs.frameMsHead = (vs.frameMsHead + 1) % ViewerState::kFrameWindow;
    if (vs.frameMsCount < ViewerState::kFrameWindow) ++vs.frameMsCount;
}

void DrawMenuBar(ViewerState& vs, MenuActions* out) {
    if (!ImGui::BeginMainMenuBar()) return;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open .glb...", "Ctrl+O")) {
            if (out) out->requestLoadDialog = true;
        }
        ImGui::Separator();
        if (!vs.recentFiles.empty()) {
            if (ImGui::BeginMenu("Recent")) {
                for (auto const& p : vs.recentFiles) {
                    if (ImGui::MenuItem(p.c_str())) {
                        if (out) out->pickedRecent = p;
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
        }
        if (ImGui::MenuItem("Quit", "Esc")) {
            if (out) out->requestQuit = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Stats",          nullptr, &vs.showStats);
        ImGui::MenuItem("Scene info",     nullptr, &vs.showSceneInfo);
        ImGui::MenuItem("Render settings",nullptr, &vs.showRender);
        ImGui::MenuItem("Camera",         nullptr, &vs.showCamera);
        ImGui::Separator();
        ImGui::MenuItem("ImGui demo",     nullptr, &vs.showDemo);
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
}

void DrawStatsWindow(ViewerState& vs, const FrameStats& last) {
    if (!vs.showStats) return;
    if (!ImGui::Begin("Stats", &vs.showStats, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End(); return;
    }

    float gpuMs = float(last.gpuMilliseconds);
    float fps   = (gpuMs > 0.0f) ? 1000.0f / gpuMs : 0.0f;
    ImGui::Text("GPU       : %7.3f ms (%5.1f fps)", gpuMs, fps);
    if (last.stage1Ms > 0.0 || last.resolveMs > 0.0) {
        ImGui::Separator();
        ImGui::Text("clear     : %6.3f ms", last.clearMs);
        ImGui::Text("stage 1   : %6.3f ms", last.stage1Ms);
        ImGui::Text("stage 2/3 : %6.3f ms", last.stage23Ms);
        ImGui::Text("resolve   : %6.3f ms", last.resolveMs);
    }
    ImGui::Separator();
    ImGui::Text("stage2 q  : %u", last.stage2Items);
    ImGui::Text("stage3 q  : %u", last.stage3Items);

    ImGui::Separator();
    if (vs.frameMsCount > 0) {
        ImGui::Text("frame-time history (last %d frames):", vs.frameMsCount);
        // ImGui's PlotLines wants a contiguous array; we have a ring buffer.
        // Copy into a temp.
        float tmp[ViewerState::kFrameWindow];
        int n = vs.frameMsCount;
        int start = (vs.frameMsHead - n + ViewerState::kFrameWindow)
                    % ViewerState::kFrameWindow;
        for (int i = 0; i < n; ++i) {
            tmp[i] = vs.frameMsHistory[(start + i) % ViewerState::kFrameWindow];
        }
        char overlay[64];
        std::snprintf(overlay, sizeof(overlay), "%.2f ms", gpuMs);
        ImGui::PlotLines("##frame", tmp, n, 0, overlay,
                         0.0f, FLT_MAX, ImVec2(260, 60));
    }
    ImGui::End();
}

void DrawSceneInfoWindow(ViewerState& vs, const Scene& scene) {
    if (!vs.showSceneInfo) return;
    if (!ImGui::Begin("Scene", &vs.showSceneInfo, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End(); return;
    }
    size_t verts = scene.compressed ? scene.positions.size()
                                    : scene.positionsFloat.size();
    ImGui::Text("vertices    : %zu (%s)", verts,
                scene.compressed ? "8 B compressed" : "16 B uncompressed");
    ImGui::Text("indices     : %zu", scene.indices.size());
    ImGui::Text("triangles   : %llu", (unsigned long long)scene.totalTriangles());
    ImGui::Text("draw calls  : %zu", scene.drawCalls.size());
    ImGui::Text("instances   : %u",  scene.totalInstances());
    if (!vs.lastLoadedPath.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("Source: %s", vs.lastLoadedPath.c_str());
    }
    ImGui::End();
}

void DrawRenderSettings(ViewerState& vs, RendererConfig& /*cfg*/) {
    if (!vs.showRender) return;
    if (!ImGui::Begin("Render", &vs.showRender, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End(); return;
    }
    const char* visModes[] = {
        "Normal (lambert)", "Depth", "Mesh ID", "Triangle ID", "Stage"
    };
    ImGui::Combo("Visualization", &vs.visMode, visModes, IM_ARRAYSIZE(visModes));

    // Rasterizer mode — mirrors CuRast's toolbar radios.
    ImGui::TextUnformatted("Rasterizer:");
    ImGui::SameLine();
    ImGui::RadioButton("Auto",                 &vs.rasterMode, 0); ImGui::SameLine();
    ImGui::RadioButton("Visbuffer[indexed]",   &vs.rasterMode, 1); ImGui::SameLine();
    ImGui::RadioButton("Visbuffer[instanced]", &vs.rasterMode, 2);

    ImGui::SliderInt("Stage-1 TGs", &vs.stage1TG, 8, 512);
    ImGui::Checkbox("16-bit position compression", &vs.compressed);
    ImGui::Checkbox("Bit-packed indices (var bits/mesh)", &vs.bitPacked);
    ImGui::Checkbox("Spin camera", &vs.spinCamera);
    ImGui::Checkbox("Freeze camera (no input)", &vs.freezeCamera);
    ImGui::Separator();
    ImGui::Checkbox("Hi-Z occlusion culling", &vs.hiZOcclusion);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Per-instance AABB cull against last-frame Hi-Z pyramid.\n"
                          "Skips occluded geometry before stage1.");
    ImGui::BeginDisabled(!vs.hiZOcclusion);
    ImGui::Indent();
    ImGui::Checkbox("Two-phase (disocclusion-correct)", &vs.hiZTwoPhase);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Render twice per frame: once with last-frame mask,\n"
                          "rebuild Hi-Z, then re-render the disoccluded delta.\n"
                          "Eliminates 1-frame triangle pop on fast camera moves.");
    ImGui::Checkbox("Compact instances (Hijma §6.2.2)", &vs.compactInstances);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Pack visible instances into a dense list before stage1.\n"
                          "Removes the per-iter Hi-Z bit test from stage1's instance\n"
                          "loop. Wins on LOW cull rate (every test would pass anyway);\n"
                          "loses on HIGH cull rate (compact dispatch dominates).\n"
                          "Workload-dependent — flip and watch the GPU time.");
    ImGui::Unindent();
    ImGui::EndDisabled();

    ImGui::Checkbox("Prefix-sum DC lookup (resolve)", &vs.usePrefixSum);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Resolve uses a 64K-bucket table to skip most of the\n"
                          "binary search over draw calls. Bit-identical output;\n"
                          "~8% faster resolve on Zorah-class scenes.");
    ImGui::End();
}

void DrawCameraWindow(ViewerState& vs, OrbitCamera& cam) {
    if (!vs.showCamera) return;
    if (!ImGui::Begin("Camera", &vs.showCamera, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End(); return;
    }
    simd_float3 t = cam.target();
    simd_float3 e = cam.eye();
    ImGui::Text("target  : %.3f %.3f %.3f", t.x, t.y, t.z);
    ImGui::Text("eye     : %.3f %.3f %.3f", e.x, e.y, e.z);
    ImGui::Text("dist    : %.3f", cam.distance());
    ImGui::Text("yaw/pit : %.2f deg / %.2f deg",
                cam.yaw()   * 180.0f / 3.14159265f,
                cam.pitch() * 180.0f / 3.14159265f);
    ImGui::Separator();
    float fov = cam.fovY() * 180.0f / 3.14159265f;
    if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
        cam.setFovY(fov * 3.14159265f / 180.0f);
    }

    // Near/far plane sliders. Logarithmic so the small-end of `near` is
    // controllable (0.001 → 10) and `far` covers a useful spread (10 → 500).
    float n = cam.nearPlane();
    float f = cam.farPlane();
    bool nfChanged = false;
    nfChanged |= ImGui::SliderFloat("Near plane", &n, 0.001f, 10.0f, "%.4f",
                                    ImGuiSliderFlags_Logarithmic);
    nfChanged |= ImGui::SliderFloat("Far plane",  &f, 10.0f,  500.0f, "%.2f",
                                    ImGuiSliderFlags_Logarithmic);
    if (nfChanged) cam.setNearFar(std::max(1e-4f, n),
                                  std::max(n + 1e-3f, f));

    if (ImGui::Button("Reset view")) cam.resetView();

    ImGui::Separator();
    ImGui::TextUnformatted("Presets");
    auto preset = [&](const char* label, OrbitCamera::Preset p){
        if (ImGui::Button(label)) cam.applyPreset(p);
    };
    preset("Front", OrbitCamera::Preset::Front); ImGui::SameLine();
    preset("Back",  OrbitCamera::Preset::Back);  ImGui::SameLine();
    preset("Left",  OrbitCamera::Preset::Left);  ImGui::SameLine();
    preset("Right", OrbitCamera::Preset::Right);
    preset("Top",   OrbitCamera::Preset::Top);   ImGui::SameLine();
    preset("Bottom",OrbitCamera::Preset::Bottom);ImGui::SameLine();
    preset("Iso",   OrbitCamera::Preset::Iso);

    ImGui::Separator();
    // Save current camera to disk so a headless benchmark run can replay it.
    static double sLastSaveTime = -10.0;
    static bool   sLastSaveOk   = false;
    if (ImGui::Button("Save Camera")) {
        std::string path = defaultCameraStatePath();
        sLastSaveOk   = saveCameraToFile(cam.toCamera(), path);
        sLastSaveTime = ImGui::GetTime();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("→ ~/.metalrast_camera.cam");
    if (ImGui::GetTime() - sLastSaveTime < 2.0) {
        ImGui::TextColored(sLastSaveOk ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f)
                                        : ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           sLastSaveOk ? "Saved." : "Save failed.");
    }

    // Screenshot button — flags the main loop to dump the next rendered frame
    // (PNG + sidecar .cam with the exact camera state used) into temp_images/.
    if (ImGui::Button("Screenshot")) {
        vs.requestScreenshot = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("→ temp_images/screenshot_<ts>.png + .cam");
    ImGui::End();
}

void DrawResidencyWindow(const ResidencyManager::Stats& s, ViewerState& vs) {
    if (!ImGui::Begin("Residency", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End(); return;
    }
    auto bytesToMiB = [](uint64_t b){ return double(b) / (1024.0*1024.0); };
    ImGui::Text("Meshes:    %u registered", s.totalMeshes);
    ImGui::Text("  visible: %u", s.visibleMeshes);
    ImGui::Text("  resident:%u", s.residentMeshes);
    ImGui::Text("  loading: %u", s.loadingMeshes);
    ImGui::Separator();
    auto poolBar = [&](const char* label, uint64_t used, uint64_t cap){
        float frac = cap ? float(double(used) / double(cap)) : 0.0f;
        char text[96];
        std::snprintf(text, sizeof(text), "%.1f / %.1f MiB (%.0f%%)",
                      bytesToMiB(used), bytesToMiB(cap), frac * 100.0);
        ImGui::ProgressBar(frac, ImVec2(220.f, 0.f), text);
        ImGui::SameLine(); ImGui::TextUnformatted(label);
    };
    poolBar("Vertex pool", s.vertPoolUsed, s.vertPoolCapacity);
    poolBar("Index pool",  s.idxPoolUsed,  s.idxPoolCapacity);
    ImGui::Separator();
    ImGui::Text("Loads/frame: %u   total: %llu",
                s.loadsThisFrame, (unsigned long long)s.totalLoadsLifetime);

    // Pool resize: slider sets target MiB, Apply triggers a residency rebuild
    // (drops all GPU buffers + reloads the GLB into the new pool size). Not a
    // hot-path knob — the rebuild stalls for as long as the GLB load takes.
    ImGui::Separator();
    ImGui::TextDisabled("Pool resize (rebuilds residency)");
    ImGui::SliderInt("Vertex MiB", &vs.pendingVertPoolMiB, 256, 12288, "%d MiB");
    ImGui::SliderInt("Index MiB",  &vs.pendingIdxPoolMiB,  256, 12288, "%d MiB");
    if (ImGui::Button("Apply pool sizes")) vs.requestResidencyResize = true;
    ImGui::End();
}

}  // namespace metalrast::panels
