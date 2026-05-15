#pragma once

#include <simd/simd.h>
#include <string>
#include "SharedTypes.h"

namespace metalrast {

struct Camera {
    simd_float3 eye    = {0, 0, -3};
    simd_float3 target = {0, 0,  0};
    simd_float3 up     = {0, 1,  0};

    float fovYRadians = 1.0471975512f;   // 60°
    float nearPlane   = 0.05f;
    float farPlane    = 100.0f;

    // Build the per-frame uniform block. `width/height` is the render target size.
    CameraUniforms uniforms(uint32_t width, uint32_t height) const;
};

// Camera-at-origin-looks-+Z lookAt (paper convention; positive view-z = in front).
//   z_axis = normalize(target - eye)
//   x_axis = normalize(cross(up, z_axis))     // right
//   y_axis = cross(z_axis, x_axis)            // recomputed up
//
// Returns the world→view matrix as a column-major simd_float4x4.
simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 target, simd_float3 up);

// 4×4 identity / translation / rotation helpers — for building model matrices.
simd_float4x4 makeIdentity();
simd_float4x4 makeTranslation(simd_float3 t);
simd_float4x4 makeRotationY(float radians);

// Camera state (de)serialization: a tiny key-value text format on disk so
// users can save a camera angle from the interactive viewer and replay it
// from the headless benchmark path. See defaultCameraStatePath().
std::string defaultCameraStatePath();
bool        saveCameraToFile(const Camera& cam, const std::string& path);
bool        loadCameraFromFile(Camera& cam, const std::string& path);

}  // namespace metalrast
