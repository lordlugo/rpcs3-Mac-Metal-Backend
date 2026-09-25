# RPCS3 Metal (macOS)

This fork of [RPCS3](https://github.com/RPCS3/rpcs3) targets **macOS on Apple silicon only** and renders
with a **native Metal 4 backend**. Metal is the only renderer: Vulkan/MoltenVK and OpenGL are not built
(their sources stay in the tree to ease merging with upstream).

## Requirements

- A Mac with Apple silicon (M1 or newer). Intel Macs are not supported.
- macOS 26.0 or newer.
- Xcode 26 or its command line tools (`xcode-select --install`).
- [Homebrew](https://brew.sh) (any prefix; `brew` must be on your `PATH`).

## Building

```sh
git clone -b metal-backend https://github.com/lordlugo/rpcs3-Mac-Metal-Backend.git rpcs3-metal
cd rpcs3-metal
./build-macos.sh --deps
./build-macos.sh
```

`--deps` installs cmake, ninja, ccache, llvm, qt, sdl3, pkg-config and abseil with Homebrew. The second command
initialises the needed submodules, configures and builds (RelWithDebInfo). Paste commands without trailing `# ...`
comments: zsh passes them on as arguments unless `setopt interactivecomments` is on.

Options:

| Option | Effect |
| --- | --- |
| `--release` / `--debug` | Build type (default: `RelWithDebInfo`) |
| `--clean` | Delete `build-metal/` first (full rebuild) |
| `--configure-only` | Initialise submodules and run CMake without building |

Set `RPCS3_WITH_OPENCV=1` to also install and use Homebrew OpenCV (optional camera features).

The build uses Homebrew's LLVM clang (RPCS3 needs clang 19 or newer, which Apple clang does not provide) with the
macOS SDK from the selected Xcode. ccache is used automatically when installed.

Don't run `brew upgrade` while a build is running: it deletes the old LLVM from under the compiler and causes
errors like `clang-scan-deps not found` or `use of undeclared identifier 'FLT_DIG'`. After an upgrade the script
notices the new clang/SDK and starts from a clean `build-metal/` on its own. The build continues past failed
files, so a single run lists every compile error.

The app is written to:

```
build-metal/bin/rpcs3.app
```

Run it with `open build-metal/bin/rpcs3.app`, or start `build-metal/bin/rpcs3.app/Contents/MacOS/rpcs3` from a
terminal to see the log output there.

## First run

Local and CI builds are **ad-hoc signed** (with the JIT, camera and microphone entitlements from
`rpcs3/rpcs3.entitlements`), not notarized. If you downloaded the app (for example a CI artifact), macOS
Gatekeeper will block the first launch. Either:

- right-click (or Control-click) the app in Finder and choose **Open**; on current macOS versions this only shows
  the warning, then open **System Settings → Privacy & Security** and click **Open Anyway**, or
- remove the quarantine attribute: `xattr -dr com.apple.quarantine "/path/to/RPCS3 Metal.app"`

Apps you built yourself are not quarantined and start normally.

`.ci/deploy-mac.sh` packages a release archive. With `RPCS3_CODESIGN_IDENTITY` set it signs with that identity,
the hardened runtime and a secure timestamp, ready for notarization.

## Where things are stored

This fork uses its own bundle identifier (`io.github.lordlugo.rpcs3metal`) and folders, so it can be installed next
to upstream RPCS3 without sharing or overwriting anything:

- Configuration, dev_hdd0, firmware, games list, GUI settings: `~/Library/Application Support/rpcs3-metal`
- Logs and caches (shader, PPU and SPU caches): `~/Library/Caches/rpcs3-metal`

The upstream auto-updater is disabled in this fork (it would install upstream Vulkan builds over this app).

## Renderer

**Metal is the only renderer.** The renderer setting offers Metal and "Disable Video Output" (Null). Vulkan and
OpenGL settings from configurations imported from upstream RPCS3 are automatically switched to Metal.

## CI

`.github/workflows/macos-metal.yml` builds every push to `metal-backend` and every pull request on a macOS 26
arm64 runner (Release by default; other build types can be picked when starting it manually). It compiles the
Metal backend objects first (`metal-compile.log`), then the whole project
(`full-build.log`), collects all compiler errors in `errors.txt` (also shown as annotations) and uploads the
logs, plus the zipped app when the build succeeds.

## Status of this branch

Implemented (builds with Homebrew LLVM on macOS 26 and runs games on Apple silicon; still early, expect bugs):

- Native Metal 4 renderer (`rpcs3/Emu/RSX/Metal`, see `DESIGN.md` there): MTL4 command queue/allocators/command
  buffers, argument tables, residency sets, shared-event timeline, explicit barriers; RSX shaders translated
  GLSL -> SPIR-V -> MSL (SPIRV-Cross) and compiled with `MTL4Compiler` on worker threads; texture cache and surface
  cache (D24S8 stored as Depth32Float_Stencil8), zero-copy guest memory DMA, compute kernels, native overlays/UI,
  MSAA resolve, occlusion queries, CAMetalLayer presentation, MetalFX spatial upscaling (the "FSR" setting).
- Programmable blending through framebuffer fetch, feedback loops by render pass splits (counted in the debug
  overlay), shader-side sampler LOD bias on M1-M4, depth bounds on M5 (Apple10) only.
- macOS-only build: Apple silicon, macOS 26+, no Vulkan/MoltenVK/OpenGL, own bundle ID and folders, updater off,
  QoS-based thread priorities, Game Mode and local-network plist keys, entitlements.

Known gaps: shader interpreter not ported (async mode skips draws until a shader is compiled), logic ops not
emulated, wide lines drawn 1 px, last-provoking-vertex flat shading falls back to smooth, async texture streaming
off. Not done yet from the macOS-native list: Core Audio backend, GameController (DualSense), VideoToolbox, Vision,
Mach VM/exception ports, signposts, single JIT arena for notarized hardened-runtime builds.

## Defaults on this fork

New installs start with these settings, and the first start of a version that introduces a new default applies it to
an existing global configuration once (per-game configurations are left alone). Everything can still be changed in the
settings.

| Setting | Default | Why |
|---|---|---|
| VSync | Full | Presentation is paced to the display (ProMotion aware, see below) only with VSync on |
| Output Scaling | MetalFX Spatial Upscaling (+ RCAS sharpening) | Stored as "FidelityFX Super Resolution" in config.yml; the RCAS slider at 0 turns sharpening off |
| Anisotropic Filter | Automatic = 16x | Applied to every texture that can be filtered; pick a lower value to limit it, or Strict Rendering Mode for the PS3's own setting |
| Pipeline archive | On | Compiled GPU pipelines are saved next to the shader cache, so later boots skip most compiles. `RPCS3_METAL_PIPELINE_ARCHIVE=0` turns it off |
| Multithreaded RSX | On | The worker thread sleeps when idle (upstream keeps it spinning on a core forever) and only takes large copies and GPU command submission, so it no longer costs a performance core |

**ZCULL and texture semaphores.** Some games wait for a texture semaphore and then read occlusion (ZCULL) results
with the CPU; Toy Story 3 flickers black otherwise. Upstream RPCS3 only handles this in Strict Rendering Mode, with a
full GPU sync at every texture semaphore. This fork holds such a semaphore back until the results queued before it
are in memory, like the real hardware, without stalling the RSX, and only once the game is seen reading results.
Strict Rendering Mode is not needed (and keeps resolution scaling and 16x anisotropic filtering available). The log
shows `ZCULL: texture read semaphores now wait for the zcull reports queued before them` when it kicks in.

**ProMotion / frame pacing.** Each frame is held for a whole number of display refreshes with
`presentAfterMinimumDuration` (60 fps on a 120 Hz panel = every other refresh, 30 fps = every 4th), instead of
alternating between 1, 2 and 3 refreshes. Every 30 s the log gets a line starting with `Metal: presentation over`
that counts how many refreshes each frame stayed on screen.

## Metal features used (M1 and newer)

The renderer requires the Metal 4 GPU family (Apple7 = M1 and newer) and checks newer families at run time.

| Feature | Used | Notes |
|---|---|---|
| Metal 4 command queues, allocators, argument tables, `MTL4Compiler` | Yes | The whole renderer is Metal 4 |
| Residency sets, explicit barriers, shared-event timeline | Yes | |
| Pipeline binary archives (`MTL4Archive`, data-set serializer) | Yes | See Defaults |
| Framebuffer fetch / programmable blending | Yes | PS3 blending edge cases and feedback loops |
| BC1-BC3 (DXT) texture compression | Yes | PS3 compressed textures are sampled directly |
| Lossless texture compression | Yes | Automatic on Private textures; the renderer avoids the usage flag that disables it |
| 16x anisotropic filtering | Yes | |
| MSAA (2x/4x) | Yes | PS3 MSAA surfaces |
| Unified memory, no-copy buffers | Yes | Guest memory is mapped straight into GPU buffers |
| MetalFX spatial upscaling (`MTL4FXSpatialScaler`) | Yes | Replaces FSR 1 |
| `presentAfterMinimumDuration`, ProMotion refresh intervals | Yes | Frame pacing |
| Depth bounds test | M5 (Apple10) only | Ignored on older GPUs |
| Hardware sampler LOD bias | M5 (Apple10) only | Done in the shader on M1-M4 |
| MetalFX temporal upscaling, frame interpolation, denoiser | No | Need motion vectors and a jittered camera from the game; PS3 games don't provide them |
| Ray tracing, mesh shaders, tensors / ML in shaders | No | Nothing in RSX (the PS3 GPU) maps to them |
| Lossy texture compression | No | Changes pixels the game may read back; not acceptable for emulation |
| Sparse textures and buffers | No | Guest memory is already mapped with no copy |
| Raster order groups, tile shaders, memoryless targets | No | Framebuffer fetch covers the blending cases; PS3 render targets must stay in memory |
| HDR / EDR output | No | PS3 output is SDR |

## Testing the Metal renderer

Start from a terminal to see the log and enable Apple's debugging aids: Metal API and shader validation (slow, but
catches API misuse) and the Metal performance HUD.

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_HUD_ENABLED=1 \
  build-metal/bin/rpcs3.app/Contents/MacOS/rpcs3
```

To record a frame for Xcode's Metal debugger, also set `MTL_CAPTURE_ENABLED=1`. The capture layer is not enabled in
normal launches because it costs performance.

Useful settings: GPU → Renderer "Metal", Shader Mode "Async Shader Recompiler", "Debug output" / "Log shader
programs" (writes GLSL and MSL to `~/Library/Caches/rpcs3-metal/shaderlog`). The RPCS3 log is in
`~/Library/Caches/rpcs3-metal/RPCS3.log`.

## Reporting build or runtime problems

`build-macos.sh` writes everything it prints to `build-macos.log` in the repository root.

```sh
grep -n "error:" build-macos.log | head -50
grep -n "Metal\|MSL\|MTL" ~/Library/Caches/rpcs3-metal/RPCS3.log | head -100
```

The first command lists compile errors, the second the renderer's messages from the emulator log.
