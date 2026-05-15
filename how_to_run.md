# How to run MetalRast

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

The binary lands at `build/metalrast`. The Metal shader library is built
alongside it as `build/MetalRast.metallib` and is loaded at runtime via the
`METALRAST_METALLIB_PATH` compile-time define.

Requirements: macOS on Apple Silicon (arm64). GLFW + Dear ImGui are vendored
under `third_party/`.

## Interactive viewer

Open a window with the default synthetic scene (sphere + ground plane):

```sh
./build/metalrast --interactive
```

Open with a `.glb` preloaded:

```sh
./build/metalrast --interactive --glb Resources/test_scene/komainu_kobe_60m.glb
```

Or `-i` for short.

### Controls

| Input                          | Action                              |
| ------------------------------ | ----------------------------------- |
| Left-drag / Middle-drag        | orbit (yaw + pitch)                 |
| Right-drag / Shift + Middle    | pan target                          |
| Scroll / Ctrl + Middle-drag    | zoom                                |
| `R`                            | reset to framed view                |
| `O`                            | open `.glb` via file dialog         |
| `F1`                           | toggle Stats panel                  |
| `Esc`                          | quit                                |
| Drag a `.glb` onto the window  | load that scene                     |
| File menu / Recent             | reload a previously opened scene    |

(On macOS, ⌘ also counts as Ctrl for the middle-drag zoom.)

Camera input is suppressed when ImGui has focus (clicking inside a panel) or
when **Render → Freeze camera** is on.

### Panels

- **Stats** — GPU ms, FPS, stage2/3 queue depths, rolling frame-time graph.
- **Scene** — vertex / index / triangle / draw-call / instance counts.
- **Render** — visualisation mode, rasterizer radio (`Auto` /
  `Visbuffer[indexed]` / `Visbuffer[instanced]` — 1:1 with CuRast's CUDA
  toolbar), stage-1 threadgroup count, 16-bit position compression toggle,
  spin/freeze. Stage-1 TGs and the compression toggle hot-rebuild the
  renderer; the active GLB is re-loaded under the new compression setting.
- **Camera** — yaw / pitch / distance / FOV readouts and Front / Back / Left /
  Right / Top / Bottom / Iso preset buttons.

### Visualisation modes

Set in **Render → Visualization**:

0. **Lambert** — tinted per (mesh, instance) shading (default).
1. **Depth** — 28-bit depth, gamma-tweaked greyscale.
2. **Mesh ID** — colour hashed from draw-call index.
3. **Triangle ID** — colour hashed from global triangle ID.
4. **Stage** — proxied via screen-space extent: blue (stage1) / green
   (stage2) / red (stage3).

### Recent files

The MRU list is persisted to `~/.metalrast_recent` (10 entries max).

## Headless / benchmark mode

Without `-i`, MetalRast renders a fixed number of frames and writes a PNG.

```sh
./build/metalrast --glb Resources/test_scene/komainu_kobe_60m.glb \
                  --frames 100 --warmup 8 --benchmark --no-png
```

Useful flags:

- `--width N --height N` — framebuffer size (default 1920x1080).
- `--frames N` — total frames (default 1). Use with `--benchmark` for stable timings.
- `--warmup N` — frames to exclude from stats (default 4).
- `--benchmark` — repeat render, print GPU timing stats.
- `--profile` — per-stage GPU timing breakdown (clear / stage1 / stage2+3 / resolve).
- `--spin` — rotate camera across frames.
- `--output PATH` — PNG output path (default `metalrast.png`).
- `--no-png` — skip the PNG write (benchmark-only).
- `--tg1 N` — stage-1 persistent threadgroup count (default 128, tuned for M2 Max).
- `--no-compress` / `--compress` — toggle 16-bit position compression.
- `--cam {-z,+z,+x,-x,+y,-y}` — GLB framing direction.
- `--pullback F` — camera distance multiplier (default 2.5).
- `--repeat N` — replicate the loaded GLB across an NxNxN grid.
- `--zorah N` — synthetic Zorah-style scene with ~N instances.
- `--max-tris N` — cap GLB load at N instanced triangles (memory budget).

`--help` prints the full list.
