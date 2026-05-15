<img width="1920" height="1129" alt="1" src="https://github.com/user-attachments/assets/7bd89ebf-c8cb-46e6-ad2c-eedcf3254134" />

# MetalRast

A Metal-compute software rasterizer for Apple Silicon, ported from the CUDA
reference implementation in:

> *CuRast: Cuda-Based Software Rasterization for Billions of Triangles* —
> Schütz, Lipp, Kristmann, Wimmer.
> <https://github.com/m-schuetz/CuRast>

The 3-stage pipeline (small / medium / large triangles) and the `uint64`
visibility-buffer encoding (28-bit depth ǀ 36-bit triangle id, written via
64-bit `atomic_min`) are implemented as Metal compute kernels. The host
driver is plain C++ + `metal-cpp` (header-only).

## Requirements

- macOS 13+ (Metal 3, 64-bit atomics)
- Apple Silicon (M1 or newer) — `MTLGPUFamilyApple7`
- Xcode + Command Line Tools (`xcrun metal`, `xcrun metallib`)
- CMake ≥ 3.20
- `clang++` with C++20

`atomic_min` on `device atomic_ulong*` is Apple7-only — Intel Macs and
pre-M1 GPUs are not supported.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The build compiles `MetalRast.metal` → `MetalRast.air` → `MetalRast.metallib`,
links `metalrast`, and bakes the metallib path into the binary so it can find
the shader library at runtime.

## Run

Open the interactive viewer with the synthetic default scene (sphere + ground):

```sh
./build/metalrast --interactive
```

Open with a `.glb` preloaded:

```sh
./build/metalrast --interactive --glb path/to/scene.glb
```

Headless render of a single frame:

```sh
./build/metalrast --output frame.png
```

Benchmark mode (warm-ups discarded, GPU timing reported):

```sh
./build/metalrast --benchmark --frames 30 --warmup 5 --no-png
```

See [`how_to_run.md`](how_to_run.md) for the full flag list and viewer controls.

## Test scenes

Two large public scenes are referenced in the `Launch Camera Angles/`
directory but are not bundled (they are several GB each). To reproduce
the captured angles:

- **Komainu (Kobe), 60 M tris** — Sketchfab — drop the `.glb` anywhere and
  pass `--glb <path>`.
- **Zorah, 1.6 B tris** — public Khronos asset; pass the `.gltf` path.

A `model_cache/` directory is created on first textured load to cache
ASTC-compressed mips keyed by content hash; the directory itself is
tracked (empty), the cache contents are not.

## Layout

```
CMakeLists.txt                top-level CMake (Metal toolchain + binary)
shaders/
  MetalRast.metal             clear / stage1 / stage2 / stage3 / resolve / Hi-Z kernels
  SharedTypes.h               structs shared by MSL and C++ via <simd/simd.h>
src/
  main.cpp                    CLI, scene assembly, benchmark stats, PNG output
  MetalRastRenderer.*         pipeline / buffer / dispatch driver
  GLBLoader.*                 cgltf + mmap + parallel scene loader
  TextureLoader.*             astcenc ASTC encode + on-disk .mrtc cache
  Mesh.* / Camera.*           synthetic generators + camera math
  ui/                         GLFW window, ImGui panels, interactive app loop
third_party/
  metal-cpp/                  header-only C++ wrappers for Metal
  glfw/, imgui/, astc-encoder/, meshoptimizer/, stb_*, cgltf.h  vendored deps
Launch Camera Angles/         curated camera presets (auto-loaded by GLB stem)
```

## License

Source code is MIT (see `LICENSE` if added). Vendored third-party libraries
keep their original licenses (BSD-style for GLFW, MIT for Dear ImGui, Apache 2
for ARM's astc-encoder, MIT for meshoptimizer, public domain / MIT for the
stb_* single-header libs and cgltf — see each subdirectory under `third_party/`).
