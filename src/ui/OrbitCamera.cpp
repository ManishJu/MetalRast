#include "OrbitCamera.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace metalrast {

namespace {
constexpr float kPi = 3.14159265358979323846f;

inline float length3(simd_float3 v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
inline simd_float3 normalize3(simd_float3 v) {
    float l = length3(v);
    if (l < 1e-8f) return (simd_float3){0, 1, 0};
    return (simd_float3){ v.x / l, v.y / l, v.z / l };
}
inline simd_float3 cross3(simd_float3 a, simd_float3 b) {
    return (simd_float3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x };
}
inline float dot3(simd_float3 a, simd_float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
}  // namespace

// Build the orthonormal orbit basis (right, up, forward) from a unit up
// vector. fwd is chosen orthogonal to up by Gram-Schmidt'ing the world axis
// least parallel to up; right = cross(up, fwd). Default up = +Y reproduces
// the original right=+X, fwd=+Z basis exactly.
void OrbitCamera::rebuildBasis_() {
    up_ = normalize3(up_);
    // Pick the world axis least aligned with up_ (largest residual after
    // projection). Using the *least parallel* canonical axis avoids the
    // degenerate case where up_ ≈ chosen axis.
    float ax = std::fabs(up_.x), ay = std::fabs(up_.y), az = std::fabs(up_.z);
    simd_float3 seed;
    if (ay <= ax && ay <= az)        seed = (simd_float3){0, 1, 0};
    else if (az <= ax && az <= ay)   seed = (simd_float3){0, 0, 1};
    else                              seed = (simd_float3){1, 0, 0};
    // Gram-Schmidt to get fwd ⟂ up.
    simd_float3 f = (simd_float3){
        seed.x - up_.x * dot3(seed, up_),
        seed.y - up_.y * dot3(seed, up_),
        seed.z - up_.z * dot3(seed, up_) };
    // For the canonical up=+Y case, seed=+Z, f=+Z (already orthogonal),
    // matching the historical behavior.
    fwdRef_   = normalize3(f);
    rightRef_ = normalize3(cross3(up_, fwdRef_));
}

simd_float3 OrbitCamera::dirAt_(float yaw, float pitch) const {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    // dir(target → eye) = sy*cp * right + cy*cp * fwd + sp * up.
    return (simd_float3){
        sy * cp * rightRef_.x + cy * cp * fwdRef_.x + sp * up_.x,
        sy * cp * rightRef_.y + cy * cp * fwdRef_.y + sp * up_.y,
        sy * cp * rightRef_.z + cy * cp * fwdRef_.z + sp * up_.z };
}

void OrbitCamera::setViewport(int wPx, int hPx) {
    vpW_ = std::max(1, wPx);
    vpH_ = std::max(1, hPx);
}

void OrbitCamera::frameAABB(simd_float3 lo, simd_float3 hi,
                            simd_float3 direction)
{
    sceneLo_ = lo; sceneHi_ = hi;

    simd_float3 center = (simd_float3){
        (lo.x + hi.x) * 0.5f,
        (lo.y + hi.y) * 0.5f,
        (lo.z + hi.z) * 0.5f
    };
    simd_float3 size = (simd_float3){ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    float radius = std::max({ size.x, size.y, size.z, 1.0f }) * 0.6f;

    target_ = center;
    // Camera distance to fit the radius at the current FOV (small slack).
    distance_ = (radius / std::tan(fovY_ * 0.5f)) * 1.6f;

    // Project `direction` (target → eye) into the (right, up, fwd) basis to
    // recover yaw/pitch. Defaults to (0, 0.2, -1) which under up=+Y means
    // looking from a slight elevation behind the AABB.
    simd_float3 d = direction;
    float dlen = length3(d);
    if (dlen < 1e-6f) d = (simd_float3){0, 0.2f, -1.0f}, dlen = 1.0f;
    d = (simd_float3){ d.x / dlen, d.y / dlen, d.z / dlen };

    rebuildBasis_();
    pitch_ = std::asin(std::clamp(dot3(d, up_), -0.999f, 0.999f));
    simd_float3 horiz = (simd_float3){
        d.x - up_.x * dot3(d, up_),
        d.y - up_.y * dot3(d, up_),
        d.z - up_.z * dot3(d, up_) };
    float hlen = length3(horiz);
    if (hlen > 1e-6f) {
        horiz = (simd_float3){ horiz.x / hlen, horiz.y / hlen, horiz.z / hlen };
        yaw_  = std::atan2(dot3(horiz, rightRef_), dot3(horiz, fwdRef_));
    } else {
        yaw_ = 0.0f;
    }

    near_ = std::max(0.001f, radius * 0.001f);
    far_  = radius * 50.0f + 10.0f;

    // Snapshot for resetView().
    savedYaw_   = yaw_;
    savedPitch_ = pitch_;
    savedDist_  = distance_;
    savedTarget_ = target_;
}

void OrbitCamera::resetView() {
    yaw_ = savedYaw_; pitch_ = savedPitch_;
    distance_ = savedDist_; target_ = savedTarget_;
}

void OrbitCamera::applyPreset(Preset p) {
    target_   = savedTarget_;
    distance_ = savedDist_;
    switch (p) {
        case Preset::Front:  yaw_ = 0.0f;        pitch_ = 0.0f;          break;
        case Preset::Back:   yaw_ = kPi;         pitch_ = 0.0f;          break;
        case Preset::Right:  yaw_ = kPi * 0.5f;  pitch_ = 0.0f;          break;
        case Preset::Left:   yaw_ = -kPi * 0.5f; pitch_ = 0.0f;          break;
        case Preset::Top:    yaw_ = 0.0f;        pitch_ = kPi * 0.4995f; break;
        case Preset::Bottom: yaw_ = 0.0f;        pitch_ = -kPi * 0.4995f;break;
        case Preset::Iso:    yaw_ = kPi * 0.25f; pitch_ = kPi * 0.18f;   break;
    }
}

simd_float3 OrbitCamera::eye() const {
    simd_float3 dir = dirAt_(yaw_, pitch_);
    return (simd_float3){
        target_.x + dir.x * distance_,
        target_.y + dir.y * distance_,
        target_.z + dir.z * distance_,
    };
}

void OrbitCamera::setFromCamera(const Camera& c) {
    target_ = c.target;
    simd_float3 diff = (simd_float3){
        c.eye.x - c.target.x, c.eye.y - c.target.y, c.eye.z - c.target.z };
    float d = length3(diff);
    distance_ = (d > 1e-4f) ? d : 1e-4f;
    simd_float3 dir = (simd_float3){ diff.x / distance_, diff.y / distance_, diff.z / distance_ };
    // Honour the loaded up vector. Falls back to +Y if the file's up was zero.
    up_ = (length3(c.up) > 1e-6f) ? normalize3(c.up) : (simd_float3){0, 1, 0};
    rebuildBasis_();
    pitch_ = clampPitch(std::asin(std::clamp(dot3(dir, up_), -0.999f, 0.999f)));
    simd_float3 horiz = (simd_float3){
        dir.x - up_.x * dot3(dir, up_),
        dir.y - up_.y * dot3(dir, up_),
        dir.z - up_.z * dot3(dir, up_) };
    float hlen = length3(horiz);
    if (hlen > 1e-6f) {
        horiz = (simd_float3){ horiz.x / hlen, horiz.y / hlen, horiz.z / hlen };
        yaw_  = std::atan2(dot3(horiz, rightRef_), dot3(horiz, fwdRef_));
    } else {
        yaw_ = 0.0f;
    }
    fovY_  = c.fovYRadians;
    near_  = c.nearPlane;
    far_   = c.farPlane;
}

simd_float3 OrbitCamera::forwardDir() const {
    // forward = -(target → eye) direction, i.e. eye → target.
    simd_float3 d = dirAt_(yaw_, pitch_);
    return (simd_float3){ -d.x, -d.y, -d.z };
}

simd_float3 OrbitCamera::rightDir() const {
    // Camera right at this orientation = cross(up_, forward) (lookAt convention).
    simd_float3 f = forwardDir();
    simd_float3 r = cross3(up_, f);
    return normalize3(r);
}

Camera OrbitCamera::toCamera() const {
    Camera c;
    c.eye    = eye();
    c.target = target_;
    c.up     = up_;
    c.fovYRadians = fovY_;
    c.nearPlane   = near_;
    c.farPlane    = far_;
    return c;
}

// ---- input ------------------------------------------------------------------

void OrbitCamera::onMouseButton(int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        leftDown_ = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        rightDown_ = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        middleDown_ = (action == GLFW_PRESS);
    }
}

void OrbitCamera::onCursorMove(double x, double y) {
    double dx = x - prevX_;
    double dy = y - prevY_;
    prevX_ = x; prevY_ = y;

    // Decide what action this drag wants. Blender-style for middle button:
    //   middle           → orbit (yaw/pitch)
    //   shift + middle   → pan
    //   ctrl  + middle   → drag-zoom (vertical = in/out)
    // Plus original: left = orbit, right = pan.
    bool wantOrbit = leftDown_   || (middleDown_ && !shiftDown_ && !ctrlDown_);
    bool wantPan   = rightDown_  || (middleDown_ &&  shiftDown_ && !ctrlDown_);
    bool wantZoom  =                (middleDown_ && !shiftDown_ &&  ctrlDown_);

    if (wantOrbit) {
        // Yaw / pitch.  Scale: a full window-width drag = ~360° yaw.
        float kx = (2.0f * kPi) / float(vpW_);
        float ky = (1.0f * kPi) / float(vpH_);
        yaw_   -= float(dx) * kx;
        pitch_ += float(dy) * ky;
        constexpr float lim = kPi * 0.4995f;
        pitch_ = std::clamp(pitch_, -lim, lim);
    }
    if (wantPan) {
        // Pan in the camera's screen plane. Camera right/up are derived
        // from the current up_-aware basis, so pan works correctly under
        // any up axis (komainu's +Z up etc.).
        simd_float3 forward = forwardDir();
        simd_float3 right   = rightDir();
        simd_float3 camUp   = cross3(forward, right);   // perpendicular to both
        camUp = normalize3(camUp);
        // Pan so 1 px ≈ 1/vpH of the visible scene height.
        float pixelToWorld = (distance_ * 2.0f * std::tan(fovY_ * 0.5f)) / float(vpH_);
        target_.x -= float(dx) * pixelToWorld * right.x + float(dy) * pixelToWorld * camUp.x;
        target_.y -= float(dx) * pixelToWorld * right.y + float(dy) * pixelToWorld * camUp.y;
        target_.z -= float(dx) * pixelToWorld * right.z + float(dy) * pixelToWorld * camUp.z;
    }
    if (wantZoom) {
        // Drag-zoom: vertical drag scales distance exponentially. Down = in.
        // 1 px ≈ 0.5% distance change keeps it gentle.
        distance_ *= std::pow(0.99f, float(dy));
        if (distance_ < 1e-4f) distance_ = 1e-4f;
    }
}

void OrbitCamera::onScroll(double /*dx*/, double dy) {
    // Exponential zoom — feels more natural than linear.
    distance_ *= std::pow(0.9f, float(dy));
    if (distance_ < 1e-4f) distance_ = 1e-4f;
}

}  // namespace metalrast
