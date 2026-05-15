// ImGuiHost.mm — ImGui Cocoa/Metal backend wrapper.
//
// Compiled with -fobjc-arc (per-file in CMakeLists.txt). The ImGui Metal
// backend (`imgui_impl_metal.mm`) is also ARC-only.
//
// We don't define IMGUI_IMPL_METAL_CPP here — that macro would alias the
// ObjC `MTLRenderPassDescriptor` (etc.) to metal-cpp's C++ class via a
// typedef, which collides with the real ObjC class names that come in via
// <Metal/Metal.h>. Instead we use ObjC types internally and bridge from
// metal-cpp pointers at our public API.

#include "ImGuiHost.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_metal.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#import <Metal/Metal.h>

namespace metalrast {

ImGuiHost::ImGuiHost()  = default;
ImGuiHost::~ImGuiHost() { shutdown(); }

bool ImGuiHost::init(MTL::Device* device, GLFWwindow* window,
                     MTL::PixelFormat /*colorFormat*/)
{
    if (initialized_) return true;

    IMGUI_CHECKVERSION();
    ctx_ = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // Slightly tighter style for our debug panels.
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 6.0f;
    s.FrameRounding  = 4.0f;
    s.GrabRounding   = 4.0f;
    s.WindowPadding  = ImVec2(8, 8);
    s.ItemSpacing    = ImVec2(6, 4);

    if (!ImGui_ImplGlfw_InitForOther(window, /*install_callbacks*/ true)) {
        return false;
    }
    id<MTLDevice> mdev = (__bridge id<MTLDevice>)device;
    if (!ImGui_ImplMetal_Init(mdev)) {
        return false;
    }

    initialized_ = true;
    return true;
}

void ImGuiHost::shutdown() {
    if (!initialized_) return;
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    if (ctx_) ImGui::DestroyContext(static_cast<ImGuiContext*>(ctx_));
    ctx_ = nullptr;
    initialized_ = false;
}

void ImGuiHost::newFrame(MTL::Texture* targetTex) {
    if (!initialized_) return;
    // ImGui's Metal backend caches the (sampleCount, pixelFormat) of the
    // render pass it'll draw into to build its pipeline. The descriptor
    // must therefore have a real attachment with a real texture — an empty
    // descriptor leaves sampleCount=0 and the pipeline build fails with
    // "rasterSampleCount (0) is not supported by device".
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = (__bridge id<MTLTexture>)targetTex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    ImGui_ImplMetal_NewFrame(rpd);
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiHost::render(MTL::CommandBuffer* cb,
                       MTL::Texture* drawableTex,
                       MTL::PixelFormat /*colorFormat*/)
{
    if (!initialized_) return;
    ImGui::Render();

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = (__bridge id<MTLTexture>)drawableTex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;     // overlay on top
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLCommandBuffer>        mcb = (__bridge id<MTLCommandBuffer>)cb;
    id<MTLRenderCommandEncoder> enc = [mcb renderCommandEncoderWithDescriptor:rpd];
    [enc pushDebugGroup:@"ImGui"];
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), mcb, enc);
    [enc popDebugGroup];
    [enc endEncoding];
}

}  // namespace metalrast
