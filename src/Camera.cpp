#include "Camera.h"

#include <cmath>
#include <cstdlib>
#include <fstream>

namespace metalrast {

simd_float4x4 makeIdentity() {
    return simd_float4x4{(simd_float4){1, 0, 0, 0},
                         (simd_float4){0, 1, 0, 0},
                         (simd_float4){0, 0, 1, 0},
                         (simd_float4){0, 0, 0, 1}};
}

simd_float4x4 makeTranslation(simd_float3 t) {
    simd_float4x4 m = makeIdentity();
    m.columns[3] = (simd_float4){t.x, t.y, t.z, 1.0f};
    return m;
}

simd_float4x4 makeRotationY(float radians) {
    float c = std::cos(radians);
    float s = std::sin(radians);
    simd_float4x4 m = makeIdentity();
    m.columns[0] = (simd_float4){ c, 0, -s, 0};
    m.columns[2] = (simd_float4){ s, 0,  c, 0};
    return m;
}

simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 target, simd_float3 up) {
    simd_float3 z_axis = simd_normalize(target - eye);
    simd_float3 x_axis = simd_normalize(simd_cross(up, z_axis));
    simd_float3 y_axis = simd_cross(z_axis, x_axis);

    float tx = -simd_dot(x_axis, eye);
    float ty = -simd_dot(y_axis, eye);
    float tz = -simd_dot(z_axis, eye);

    // Column-major: each column is (x_axis.k, y_axis.k, z_axis.k, 0); col 3 = (tx,ty,tz,1).
    simd_float4x4 m;
    m.columns[0] = (simd_float4){x_axis.x, y_axis.x, z_axis.x, 0};
    m.columns[1] = (simd_float4){x_axis.y, y_axis.y, z_axis.y, 0};
    m.columns[2] = (simd_float4){x_axis.z, y_axis.z, z_axis.z, 0};
    m.columns[3] = (simd_float4){tx,       ty,       tz,       1};
    return m;
}

CameraUniforms Camera::uniforms(uint32_t width, uint32_t height) const {
    CameraUniforms u{};
    u.viewMatrix     = makeLookAt(eye, target, up);
    u.cameraWorldPos = (simd_float4){eye.x, eye.y, eye.z, 0.0f};

    float f      = 1.0f / std::tan(fovYRadians * 0.5f);
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    u.projectionXY = (simd_float2){ f / aspect, -f };

    u.nearPlane    = nearPlane;
    u.farPlane     = farPlane;
    u.screenWidth  = width;
    u.screenHeight = height;
    u._pad0        = 0.0f;
    u._pad1        = 0.0f;
    return u;
}

// ---- Camera state on disk ------------------------------------------------
// Tiny key-value text format. Lets the interactive viewer dump a position
// the user likes, and the headless benchmark path replay it for fair
// before/after comparisons.

std::string defaultCameraStatePath() {
    const char* home = std::getenv("HOME");
    std::string p = home ? home : ".";
    p += "/.metalrast_camera.cam";
    return p;
}

bool saveCameraToFile(const Camera& cam, const std::string& path) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    f << "eye "    << cam.eye.x    << " " << cam.eye.y    << " " << cam.eye.z    << "\n";
    f << "target " << cam.target.x << " " << cam.target.y << " " << cam.target.z << "\n";
    f << "up "     << cam.up.x     << " " << cam.up.y     << " " << cam.up.z     << "\n";
    f << "fov "    << cam.fovYRadians << "\n";
    f << "near "   << cam.nearPlane   << "\n";
    f << "far "    << cam.farPlane    << "\n";
    return f.good();
}

bool loadCameraFromFile(Camera& cam, const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string key;
    auto read3 = [&](simd_float3& v) {
        float a, b, c;
        f >> a >> b >> c;
        v = (simd_float3){ a, b, c };
    };
    while (f >> key) {
        if      (key == "eye")    read3(cam.eye);
        else if (key == "target") read3(cam.target);
        else if (key == "up")     read3(cam.up);
        else if (key == "fov")    f >> cam.fovYRadians;
        else if (key == "near")   f >> cam.nearPlane;
        else if (key == "far")    f >> cam.farPlane;
        else { std::string discard; std::getline(f, discard); }
    }
    return true;
}

}  // namespace metalrast
