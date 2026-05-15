#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <functional>
#include <string>

struct GLFWwindow;

namespace metalrast {

// GLFW window with a CAMetalLayer attached to its content view. Designed for
// our compute-rasterizer pipeline: the renderer renders to its own offscreen
// `MTL::Texture` (RGBA8), then a tiny present render-pipeline samples that
// texture onto the layer's drawable.
//
// Holding GLFWwindow as a void* keeps the public header free of GLFW.
class Window {
public:
    struct Config {
        int  width   = 1920;
        int  height  = 1080;
        bool fullscreen = false;
        std::string title = "MetalRast";
    };

    Window(MTL::Device* device, const Config& cfg);
    ~Window();

    // Returns true when the user has asked to close (clicked X or pressed ESC).
    bool shouldClose() const;
    void pollEvents();

    // Drawable acquisition. Call once per frame, between pollEvents() and the
    // present pass. May rarely return nullptr — caller should skip the frame.
    CA::MetalDrawable* nextDrawable();

    // Logical (point) size — what GLFW reports.
    int width()  const;
    int height() const;
    // Pixel size — what the drawable reports. Different on Retina.
    int framebufferWidth()  const;
    int framebufferHeight() const;

    // Pixel format of the layer (so the present pipeline can match).
    MTL::PixelFormat colorPixelFormat() const;

    // Lifecycle hooks.
    GLFWwindow* glfwHandle() const;
    CA::MetalLayer* metalLayer() const;
    MTL::Device* device() const;

    // ----- Input callback hookup -------------------------------------------

    // Cursor: x,y in window-points. Buttons: GLFW_MOUSE_BUTTON_*. action: GLFW_PRESS / GLFW_RELEASE.
    void onCursorMove (std::function<void(double x, double y)>            cb);
    void onMouseButton(std::function<void(int button, int action, int mods)> cb);
    void onScroll     (std::function<void(double dx, double dy)>          cb);
    void onKey        (std::function<void(int key, int action, int mods)> cb);
    void onDrop       (std::function<void(int count, const char** paths)> cb);
    void onResize     (std::function<void(int newW, int newH)>            cb);

    // Public so the GLFW C-style callbacks in Window.mm can reach into it.
public:
    struct Impl;
private:
    Impl* p_;
};

// Modal NSOpenPanel filtered to .glb. Returns the absolute path or empty
// string if the user cancelled. Call from the main thread only.
std::string runGLBOpenPanel();

}  // namespace metalrast
