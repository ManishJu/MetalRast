#pragma once

#include <simd/simd.h>

#include "Camera.h"

struct GLFWwindow;

namespace metalrast {

// Orbit camera around a target point. Mouse left-drag yaws/pitches; mouse
// right-drag pans; scroll zooms. Frame-rate-independent: deltas are scaled
// by the viewport size so the same drag gesture rotates the same amount on
// 1080p and 4K.
class OrbitCamera {
public:
    OrbitCamera() = default;

    // Frame the camera so the AABB lo..hi is fully visible from `direction`.
    // direction is a unit vector in world space pointing FROM the target TO
    // the eye (i.e. the side the camera is on).
    void frameAABB(simd_float3 lo, simd_float3 hi,
                   simd_float3 direction = (simd_float3){0, 0.2f, -1});

    // Manual setters.
    void setTarget(simd_float3 t)   { target_ = t; }
    void setDistance(float d)       { distance_ = (d > 1e-4f) ? d : 1e-4f; }
    void setFovY(float r)           { fovY_ = r; }
    void setNearFar(float n, float f){ near_ = n; far_ = f; }
    void setYaw(float r)            { yaw_ = r; }
    void setPitch(float r)          { pitch_ = clampPitch(r); }
    void bumpYaw(float dr)          { yaw_ += dr; }

    // Configure orbit state from a free-form Camera (eye/target/up/fov).
    // Inverts the (yaw, pitch, distance, target) → eye relation. Used when
    // resuming from a saved camera file.
    void setFromCamera(const Camera& c);

    // FPS-style translation. Adds `worldDelta` to the orbit target — since
    // eye = target + dir * distance, the eye moves by the same delta. Used
    // by WSAD/QE input in the host to fly the camera through the world.
    void pan(simd_float3 worldDelta) {
        target_.x += worldDelta.x;
        target_.y += worldDelta.y;
        target_.z += worldDelta.z;
    }
    // Unit-length forward vector (eye → target) in world space.
    simd_float3 forwardDir() const;
    // Unit-length right vector (perpendicular to forward and world up).
    simd_float3 rightDir()   const;

    // Snap to one of the canonical viewing angles, framing the saved AABB.
    enum class Preset { Front, Back, Left, Right, Top, Bottom, Iso };
    void applyPreset(Preset p);

    // GLFW input handlers — drive the camera. Returns true if input was
    // consumed (used to stop the host from also processing it for ImGui).
    void onCursorMove (double x, double y);
    void onMouseButton(int button, int action, int mods);
    void onScroll     (double dx, double dy);
    void setViewport  (int wPx, int hPx);
    // Modifier state at the time of the next cursor event. Blender-style
    // middle-drag uses these: bare = orbit, Shift = pan, Ctrl = drag-zoom.
    void setModifiers (bool shift, bool ctrl) { shiftDown_ = shift; ctrlDown_ = ctrl; }

    // Build the per-frame camera uniform.
    Camera toCamera() const;

    // Reset orbit angles + distance to the last `frameAABB` result.
    void resetView();

    // Read-only state for ImGui display.
    simd_float3 target()   const { return target_; }
    float       distance() const { return distance_; }
    float       yaw()      const { return yaw_; }
    float       pitch()    const { return pitch_; }
    float       fovY()     const { return fovY_; }
    float       nearPlane()const { return near_; }
    float       farPlane() const { return far_; }
    simd_float3 eye()      const;

private:
    static float clampPitch(float r) {
        constexpr float lim = 3.14159265f * 0.4995f;
        return (r < -lim) ? -lim : (r > lim ? lim : r);
    }

    simd_float3 target_   = {0, 0, 0};
    float       distance_ = 5.0f;
    float       yaw_      = 0.0f;     // around up_
    float       pitch_    = 0.0f;     // tilts from horizontal toward up_
    float       fovY_     = 60.0f * 3.14159265f / 180.0f;
    float       near_     = 0.05f;
    float       far_      = 500.0f;

    // ---- Up-aware orbit basis ----
    // up_ is the world's up axis as the user sees it. fwdRef_ is the
    // direction looked at when yaw=0, pitch=0 (orthogonal to up_).
    // rightRef_ = cross(up_, fwdRef_). Default reproduces the original
    // hardcoded +Y up / +Z forward. Updated by setFromCamera() and
    // frameAABB() to match the loaded camera's up vector — without this
    // the viewer renders +Z-up models (e.g. komainu) lying on their side.
    simd_float3 up_       = {0, 1, 0};
    simd_float3 fwdRef_   = {0, 0, 1};
    simd_float3 rightRef_ = {1, 0, 0};
    void        rebuildBasis_();
    simd_float3 dirAt_(float yaw, float pitch) const;

    // Saved for resetView().
    simd_float3 sceneLo_  = { 0, 0, 0 };
    simd_float3 sceneHi_  = { 0, 0, 0 };
    float       savedYaw_ = 0;
    float       savedPitch_ = 0;
    float       savedDist_  = 5;
    simd_float3 savedTarget_= { 0, 0, 0 };

    // Input state.
    bool   leftDown_   = false;
    bool   rightDown_  = false;
    bool   middleDown_ = false;
    bool   shiftDown_  = false;
    bool   ctrlDown_   = false;
    double prevX_ = 0, prevY_ = 0;
    int    vpW_ = 1, vpH_ = 1;
};

}  // namespace metalrast
