#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

struct GLFWwindow;

namespace metalrast {

// Owns the ImGui context + Metal/GLFW backends. One per window. Phase 4
// brings this to life — for Phase 1 calls are no-ops so the app builds.
class ImGuiHost {
public:
    ImGuiHost();
    ~ImGuiHost();

    bool init(MTL::Device* device, GLFWwindow* window,
              MTL::PixelFormat colorFormat);
    void shutdown();

    // Per-frame: begin, then user draws ImGui windows, then end+render.
    // `targetTex` must be the texture we'll later render onto; ImGui caches
    // its sampleCount + pixelFormat from this on first call to build its
    // render pipeline. Must not be null.
    void newFrame(MTL::Texture* targetTex);
    // Renders queued ImGui draw lists into the given pass descriptor's
    // target. Pass descriptor must already be configured to draw onto the
    // drawable (single colour attachment, load=Load so we draw on top).
    void render(MTL::CommandBuffer* cb,
                MTL::Texture* drawableTex,
                MTL::PixelFormat colorFormat);

    bool initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    void* /* ImGuiContext* */ ctx_ = nullptr;
};

}  // namespace metalrast
