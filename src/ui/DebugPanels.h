#pragma once

#include "MetalRastRenderer.h"
#include "MeshRegistry.h"
#include "Mesh.h"

#include <deque>
#include <string>

namespace metalrast {

class OrbitCamera;

// Holds runtime-tunable settings for the interactive viewer. The renderer
// reads these each frame; ImGui panels mutate them.
struct ViewerState {
    // Render-config knobs that can flip per-frame.
    int   stage1TG       = 128;          // override RendererConfig at runtime
    bool  compressed     = true;
    bool  bitPacked      = true;         // variable-bit-width packed indices (paper §4.6)
    int   visMode        = 0;            // 0=normal, 1=depth, 2=mesh-id, 3=tri-id, 4=stage
    int   rasterMode     = 0;            // 0=Auto, 1=Visbuffer[indexed], 2=Visbuffer[instanced]
    bool  freezeCamera   = false;
    bool  spinCamera     = false;
    bool  hiZOcclusion   = false;        // Hi-Z occlusion culling (Frostbite/Nanite-style)
    bool  usePrefixSum   = true;         // resolve uses 64K-bucket DC lookup (paper §4.5)
    bool  hiZTwoPhase    = false;        // two-phase variant — eliminates disocclusion pop
    bool  compactInstances = false;      // pack visible instances before stage1 (Hijma §6.2.2)
    bool  requestScreenshot = false;     // Camera panel button → captures next frame to temp_images/

    // Residency panel — pool resize. Slider sets target MiB; Apply triggers a
    // teardown + rebuild + GLB reload on the main thread.
    int   pendingVertPoolMiB    = 8192;
    int   pendingIdxPoolMiB     = 12288;
    bool  requestResidencyResize = false;
    bool  showDemo       = false;
    bool  showStats      = true;
    bool  showSceneInfo  = true;
    bool  showRender     = true;
    bool  showCamera     = true;
    bool  benchmarkMode  = false;        // run a hot-loop and report

    // Scene-load state (populated when a GLB is dropped).
    std::string lastLoadedPath;
    std::deque<std::string> recentFiles; // up to 10 entries

    // Frame timing rolling window (ms, last N frames). Phase 4 fills + draws.
    static constexpr int kFrameWindow = 120;
    float frameMsHistory[kFrameWindow] = {};
    int   frameMsHead     = 0;
    int   frameMsCount    = 0;
};

// All the ImGui panels. Phase 5 implementation; stubs for earlier phases.
namespace panels {

// Bridging structure for the menu-bar's Recent menu — caller fills `pickedRecent`
// with the path the user clicked on (empty if none).
struct MenuActions {
    bool        requestLoadDialog = false;
    bool        requestQuit       = false;
    std::string pickedRecent;          // non-empty if the user clicked a recent entry
};

// Top-of-window menu bar.
void DrawMenuBar(ViewerState& vs, MenuActions* out);

// Per-frame stats: GPU time, FPS, Gtri/s, optional rolling graph.
void DrawStatsWindow(ViewerState& vs, const FrameStats& last);

// Scene info: positions, indices, draw calls, instances, AABB.
void DrawSceneInfoWindow(ViewerState& vs, const Scene& scene);

// Render settings: stage1TG slider, compressed toggle, visualization mode.
void DrawRenderSettings(ViewerState& vs, RendererConfig& cfg);

// Residency-mode diagnostics: pool occupancy, resident/visible counts,
// loads/frame. Only meaningful when streaming via ResidencyManager.
void DrawResidencyWindow(const ResidencyManager::Stats& stats, ViewerState& vs);

// Camera info / presets.
void DrawCameraWindow(ViewerState& vs, OrbitCamera& cam);

// Push a new frame time into the rolling window.
void PushFrameTime(ViewerState& vs, float ms);

}  // namespace panels

}  // namespace metalrast
