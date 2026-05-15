// Window.mm — GLFW window wired to a CAMetalLayer.
//
// metal-cpp uses manual reference counting (-fno-objc-arc) but this file is
// compiled with ARC (per CMake's per-file flag). We bridge between the two
// via __bridge_retained / __bridge casts. Specifically: when we hand a
// `MTL::Device*` (metal-cpp, manual rc) to AppKit code that expects an
// `id<MTLDevice>` (ObjC ARC), we cast through `__bridge id` (no rc change),
// which is correct because both sides agree the device outlives the window.

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "Window.h"

#include <cstdio>
#include <cstdlib>

namespace metalrast {

struct Window::Impl {
    GLFWwindow*      glfw       = nullptr;
    CAMetalLayer*    layer      = nil;       // strong (ARC)
    MTL::Device*     device     = nullptr;   // not owned

    int              widthPt    = 0;
    int              heightPt   = 0;
    int              widthPx    = 0;
    int              heightPx   = 0;

    std::function<void(double, double)>          cbCursor;
    std::function<void(int, int, int)>           cbMouse;
    std::function<void(double, double)>          cbScroll;
    std::function<void(int, int, int)>           cbKey;
    std::function<void(int, const char**)>       cbDrop;
    std::function<void(int, int)>                cbResize;
};

// ----- GLFW callbacks: forward to Impl::cb* -----------------------------------

namespace {
Window::Impl* implFor(GLFWwindow* w) {
    return static_cast<Window::Impl*>(glfwGetWindowUserPointer(w));
}

void cb_cursor(GLFWwindow* w, double x, double y) {
    auto* p = implFor(w);
    if (p && p->cbCursor) p->cbCursor(x, y);
}
void cb_mouse(GLFWwindow* w, int button, int action, int mods) {
    auto* p = implFor(w);
    if (p && p->cbMouse) p->cbMouse(button, action, mods);
}
void cb_scroll(GLFWwindow* w, double dx, double dy) {
    auto* p = implFor(w);
    if (p && p->cbScroll) p->cbScroll(dx, dy);
}
void cb_key(GLFWwindow* w, int key, int /*scancode*/, int action, int mods) {
    auto* p = implFor(w);
    if (p && p->cbKey) p->cbKey(key, action, mods);
}
void cb_drop(GLFWwindow* w, int count, const char** paths) {
    auto* p = implFor(w);
    if (p && p->cbDrop) p->cbDrop(count, paths);
}
void cb_size(GLFWwindow* w, int newW, int newH) {
    auto* p = implFor(w);
    if (!p) return;
    p->widthPt = newW; p->heightPt = newH;
    int fbw, fbh;
    glfwGetFramebufferSize(w, &fbw, &fbh);
    p->widthPx = fbw; p->heightPx = fbh;
    if (p->layer) p->layer.drawableSize = CGSizeMake(fbw, fbh);
    if (p->cbResize) p->cbResize(fbw, fbh);
}

// One-time GLFW init. We init only the bits we need on macOS; specifically
// no GL context (we drive Metal directly via CAMetalLayer).
void ensureGLFW() {
    static bool inited = false;
    if (inited) return;
    glfwSetErrorCallback([](int code, const char* msg) {
        std::fprintf(stderr, "GLFW error %d: %s\n", code, msg);
    });
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit() failed\n");
        std::exit(1);
    }
    inited = true;
}
}  // namespace

// ----- Window impl -------------------------------------------------------------

Window::Window(MTL::Device* device, const Config& cfg) {
    p_ = new Impl();
    p_->device = device;

    ensureGLFW();

    // No GL context — the layer handles drawing.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    GLFWmonitor* mon = cfg.fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    p_->glfw = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), mon, nullptr);
    if (!p_->glfw) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        std::exit(1);
    }
    glfwSetWindowUserPointer(p_->glfw, p_);

    // Wire callbacks (caller hooks them via on*() setters).
    glfwSetCursorPosCallback     (p_->glfw, cb_cursor);
    glfwSetMouseButtonCallback   (p_->glfw, cb_mouse);
    glfwSetScrollCallback        (p_->glfw, cb_scroll);
    glfwSetKeyCallback           (p_->glfw, cb_key);
    glfwSetDropCallback          (p_->glfw, cb_drop);
    glfwSetWindowSizeCallback    (p_->glfw, cb_size);

    // Pull NSWindow from GLFW and attach a CAMetalLayer to its contentView.
    NSWindow* nswin = glfwGetCocoaWindow(p_->glfw);
    NSView*   view  = [nswin contentView];

    p_->layer = [CAMetalLayer layer];
    p_->layer.device          = (__bridge id<MTLDevice>)device;
    p_->layer.pixelFormat     = MTLPixelFormatRGBA8Unorm;
    p_->layer.framebufferOnly = YES;
    // High-quality color treatment of our linear textures.
    p_->layer.colorspace      = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    p_->layer.contentsScale   = nswin.backingScaleFactor;

    [view setWantsLayer:YES];
    [view setLayer:p_->layer];

    // Initial sizes.
    int fbw, fbh, ww, wh;
    glfwGetWindowSize     (p_->glfw, &ww,  &wh);
    glfwGetFramebufferSize(p_->glfw, &fbw, &fbh);
    p_->widthPt = ww;   p_->heightPt = wh;
    p_->widthPx = fbw;  p_->heightPx = fbh;
    p_->layer.drawableSize = CGSizeMake(fbw, fbh);
}

Window::~Window() {
    if (p_) {
        if (p_->glfw) glfwDestroyWindow(p_->glfw);
        // p_->layer is managed by ARC; will release when nslayer dies with the view.
        delete p_;
    }
}

bool Window::shouldClose() const { return glfwWindowShouldClose(p_->glfw); }
void Window::pollEvents()        { glfwPollEvents(); }

CA::MetalDrawable* Window::nextDrawable() {
    id<CAMetalDrawable> d = [p_->layer nextDrawable];
    if (!d) return nullptr;
    // __bridge cast doesn't change rc. The drawable is autoreleased; the
    // caller's autorelease pool owns the lifetime for this frame.
    return (__bridge CA::MetalDrawable*)d;
}

int Window::width()              const { return p_->widthPt; }
int Window::height()             const { return p_->heightPt; }
int Window::framebufferWidth()   const { return p_->widthPx; }
int Window::framebufferHeight()  const { return p_->heightPx; }
MTL::PixelFormat Window::colorPixelFormat() const { return MTL::PixelFormatRGBA8Unorm; }
GLFWwindow* Window::glfwHandle() const { return p_->glfw; }
CA::MetalLayer* Window::metalLayer() const {
    return (__bridge CA::MetalLayer*)p_->layer;
}
MTL::Device* Window::device() const { return p_->device; }

void Window::onCursorMove (std::function<void(double, double)>            cb) { p_->cbCursor = std::move(cb); }
void Window::onMouseButton(std::function<void(int, int, int)>             cb) { p_->cbMouse  = std::move(cb); }
void Window::onScroll     (std::function<void(double, double)>            cb) { p_->cbScroll = std::move(cb); }
void Window::onKey        (std::function<void(int, int, int)>             cb) { p_->cbKey    = std::move(cb); }
void Window::onDrop       (std::function<void(int, const char**)>         cb) { p_->cbDrop   = std::move(cb); }
void Window::onResize     (std::function<void(int, int)>                  cb) { p_->cbResize = std::move(cb); }

std::string runGLBOpenPanel() {
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        panel.allowsMultipleSelection = NO;
        panel.canChooseDirectories    = NO;
        panel.canChooseFiles          = YES;
        panel.allowedFileTypes        = @[ @"glb" ];
        panel.title                   = @"Open .glb scene";
        if ([panel runModal] != NSModalResponseOK) return {};
        NSURL* url = panel.URLs.firstObject;
        if (!url) return {};
        return std::string(url.path.UTF8String);
    }
}

}  // namespace metalrast
