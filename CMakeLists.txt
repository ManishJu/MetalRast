cmake_minimum_required(VERSION 3.20)
project(MetalRast C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT APPLE)
    message(FATAL_ERROR "CuRast Metal port requires macOS / Apple Silicon")
endif()

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build arch" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------ deps
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/metal-cpp
)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party
)

# meshoptimizer — vertex/index codec used by EXT_meshopt_compression
# (Zorah's .gltf relies on it). Decode-only subset.
add_library(meshopt STATIC
    ${CMAKE_SOURCE_DIR}/third_party/meshoptimizer/src/vertexcodec.cpp
    ${CMAKE_SOURCE_DIR}/third_party/meshoptimizer/src/indexcodec.cpp
    ${CMAKE_SOURCE_DIR}/third_party/meshoptimizer/src/vertexfilter.cpp
)
target_include_directories(meshopt PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/meshoptimizer/src
)
target_compile_options(meshopt PRIVATE -O3)

# GLFW: vendored. Build it with no extras.
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

# Dear ImGui — core + Metal + GLFW backends.
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw metal_cpp)
# imgui_impl_metal.mm REQUIRES ARC; the rest of the project uses -fno-objc-arc.
# Apply ARC to just that one file.
set_source_files_properties(
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)
target_compile_definitions(imgui PUBLIC
    GLFW_INCLUDE_NONE=1
)
# IMGUI_IMPL_METAL_CPP would re-typedef MTL types. We only want it for pure C++
# consumers — and we have none (ImGuiHost.mm uses Objective-C types directly,
# bridging from metal-cpp at the API boundary). Don't define it.

# astc-encoder — Khronos reference ASTC LDR encoder (Apache 2.0). We use it
# as the sole texture compressor: input RGBA8 → ASTC blocks (4×4 or 6×6),
# wrapped in a tiny custom MRTC container and cached on disk by content hash.
# Built with NEON enabled for Apple Silicon; SSE/AVX/SVE off.
set(ASTCENC_DIR ${CMAKE_SOURCE_DIR}/third_party/astc-encoder/Source)
add_library(astcenc STATIC
    ${ASTCENC_DIR}/astcenc_averages_and_directions.cpp
    ${ASTCENC_DIR}/astcenc_block_sizes.cpp
    ${ASTCENC_DIR}/astcenc_color_quantize.cpp
    ${ASTCENC_DIR}/astcenc_color_unquantize.cpp
    ${ASTCENC_DIR}/astcenc_compress_symbolic.cpp
    ${ASTCENC_DIR}/astcenc_compute_variance.cpp
    ${ASTCENC_DIR}/astcenc_decompress_symbolic.cpp
    ${ASTCENC_DIR}/astcenc_entry.cpp
    ${ASTCENC_DIR}/astcenc_find_best_partitioning.cpp
    ${ASTCENC_DIR}/astcenc_ideal_endpoints_and_weights.cpp
    ${ASTCENC_DIR}/astcenc_image.cpp
    ${ASTCENC_DIR}/astcenc_integer_sequence.cpp
    ${ASTCENC_DIR}/astcenc_mathlib.cpp
    ${ASTCENC_DIR}/astcenc_mathlib_softfloat.cpp
    ${ASTCENC_DIR}/astcenc_partition_tables.cpp
    ${ASTCENC_DIR}/astcenc_percentile_tables.cpp
    ${ASTCENC_DIR}/astcenc_pick_best_endpoint_format.cpp
    ${ASTCENC_DIR}/astcenc_quantization.cpp
    ${ASTCENC_DIR}/astcenc_symbolic_physical.cpp
    ${ASTCENC_DIR}/astcenc_weight_align.cpp
    ${ASTCENC_DIR}/astcenc_weight_quant_xfer_tables.cpp
)
target_include_directories(astcenc PUBLIC ${ASTCENC_DIR})
target_compile_definitions(astcenc PUBLIC
    ASTCENC_NEON=1
    ASTCENC_SSE=0
    ASTCENC_AVX=0
    ASTCENC_SVE=0
    ASTCENC_F16C=0
    ASTCENC_POPCNT=0
)
target_compile_options(astcenc PRIVATE
    -O3 -ffp-model=precise
    -Wno-unused-parameter -Wno-unused-function
    -Wno-unused-but-set-variable -Wno-unused-variable
    -Wno-sign-compare -Wno-deprecated-declarations
)

# ----------------------------------------------------------- metal shaders
set(SHADER_DIR        ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_SRC        ${SHADER_DIR}/MetalRast.metal)
set(SHADER_HDRS       ${SHADER_DIR}/SharedTypes.h)
set(SHADER_AIR        ${CMAKE_BINARY_DIR}/MetalRast.air)
set(SHADER_METALLIB   ${CMAKE_BINARY_DIR}/MetalRast.metallib)

find_program(XCRUN xcrun REQUIRED)

add_custom_command(
    OUTPUT  ${SHADER_AIR}
    COMMAND ${XCRUN} -sdk macosx metal
            -I${SHADER_DIR}
            -O3 -ffast-math
            -frecord-sources
            -c ${SHADER_SRC}
            -o ${SHADER_AIR}
    DEPENDS ${SHADER_SRC} ${SHADER_HDRS}
    COMMENT "[metal]   ${SHADER_SRC} -> ${SHADER_AIR}"
    VERBATIM
)

add_custom_command(
    OUTPUT  ${SHADER_METALLIB}
    COMMAND ${XCRUN} -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_METALLIB}
    DEPENDS ${SHADER_AIR}
    COMMENT "[metallib] ${SHADER_AIR} -> ${SHADER_METALLIB}"
    VERBATIM
)

add_custom_target(metalrast_shaders ALL DEPENDS ${SHADER_METALLIB})

# ---------------------------------------------------------------- binary
add_executable(metalrast
    src/main.cpp
    src/Mesh.cpp
    src/Camera.cpp
    src/MetalRastRenderer.cpp
    src/GLBLoader.cpp
    src/MeshRegistry.cpp
    src/TextureLoader.cpp
    # Interactive viewer (Phase 1+).
    src/ui/Window.mm
    src/ui/OrbitCamera.cpp
    src/ui/InteractiveApp.cpp
    src/ui/ImGuiHost.mm
    src/ui/DebugPanels.cpp
)
add_dependencies(metalrast metalrast_shaders)

target_include_directories(metalrast PRIVATE
    src
    shaders
)

target_link_libraries(metalrast PRIVATE metal_cpp stb glfw imgui meshopt astcenc)

target_link_libraries(metalrast PRIVATE
    "-framework Metal"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework AppKit"
    "-framework CoreGraphics"
    "-framework QuartzCore"
)

target_compile_options(metalrast PRIVATE
    -fno-objc-arc
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Wno-nullability-completeness
)

# Window.mm + ImGuiHost.mm need ARC for the Cocoa interop.
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/src/ui/Window.mm
    ${CMAKE_SOURCE_DIR}/src/ui/ImGuiHost.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

target_compile_definitions(metalrast PRIVATE
    METALRAST_METALLIB_PATH="${SHADER_METALLIB}"
    GLFW_INCLUDE_NONE=1
)
