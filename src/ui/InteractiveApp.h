#pragma once

#include "MetalRastRenderer.h"
#include "MeshRegistry.h"
#include "Mesh.h"
#include "Camera.h"
#include "Window.h"
#include "OrbitCamera.h"
#include "ImGuiHost.h"
#include "DebugPanels.h"

#include <memory>
#include <string>

namespace metalrast {

class InteractiveApp {
public:
    struct Config {
        Window::Config window;
        std::string    metallibPath;
        Scene          initialScene;     // already built by main().
        std::string    initialPath;      // shown in stats; empty for synthetic
        // Residency mode (CuRast-style streaming). When residencyPath is
        // non-empty, initialScene is ignored — we register that GLB with a
        // ResidencyManager and page geometry in/out per frame.
        std::string    residencyPath;
        ResidencyManager::Config residencyCfg;
        bool           minimalUi = false;   // hide all panels, show only Save Camera
        std::string    loadCamera;          // path to a saved camera file (optional)
        bool           initialHiZ = false;  // start with Hi-Z occlusion ON
        bool           initialCompactInstances = false;  // start with --compact-instances ON
    };

    InteractiveApp(MTL::Device* device, Config cfg);
    ~InteractiveApp();

    void run();

private:
    void rebuildRenderer(uint32_t width, uint32_t height);
    void uploadAndFrame(Scene& scene);
    void loadGLBFromPath(const std::string& path);
    void loadRecentsFromDisk();
    void saveRecentsToDisk() const;

    MTL::Device*    device_ = nullptr;
    Config          cfg_;
    std::unique_ptr<Window>            window_;
    std::unique_ptr<MetalRastRenderer> renderer_;
    OrbitCamera                        cam_;
    ImGuiHost                          imgui_;
    ViewerState                        vs_;
    Scene                              scene_;        // current (static-upload path)
    std::unique_ptr<ResidencyManager>  residency_;    // non-null in streaming mode
    uint64_t                           frameIdx_ = 0;
    FrameStats                         lastStats_{};

    // Mirror of vs_ knobs we last applied to the renderer; if these go out
    // of sync we rebuild the pipeline / pull in new compression mode.
    int  appliedStage1TG_ = 128;
    bool appliedCompressed_ = true;
    bool appliedBitPacked_  = true;

    uint32_t   rendererW_ = 0;
    uint32_t   rendererH_ = 0;
};

}  // namespace metalrast
