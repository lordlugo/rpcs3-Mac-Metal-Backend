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
./build-macos.sh --deps   # installs cmake, ninja, ccache, llvm, qt, sdl3, pkg-config, abseil via Homebrew
./build-macos.sh          # initialises the needed submodules, configures and builds (RelWithDebInfo)
```

Options:

| Option | Effect |
| --- | --- |
| `--release` / `--debug` | Build type (default: `RelWithDebInfo`) |
| `--clean` | Delete `build-metal/` first (full rebuild) |
| `--configure-only` | Initialise submodules and run CMake without building |

Set `RPCS3_WITH_OPENCV=1` to also install and use Homebrew OpenCV (optional camera features).

The build uses Homebrew's LLVM clang (RPCS3 needs clang 19 or newer, which Apple clang does not provide) with the
macOS SDK from the selected Xcode. ccache is used automatically when installed.

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

Implemented (type-checked against metal-cpp 381 on Linux, **not yet compiled or run on a Mac**):

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

## Testing the Metal renderer

Start from a terminal to see the log and enable Apple's debugging aids:

```sh
# Metal API + shader validation (slow, catches API misuse) and the Metal performance HUD
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_HUD_ENABLED=1 \
  build-metal/bin/rpcs3.app/Contents/MacOS/rpcs3
```

Useful settings: GPU → Renderer "Metal", Shader Mode "Async Shader Recompiler", "Debug output" / "Log shader
programs" (writes GLSL and MSL to `~/Library/Caches/rpcs3-metal/shaderlog`). The RPCS3 log is in
`~/Library/Caches/rpcs3-metal/RPCS3.log`.

## Reporting build or runtime problems

```sh
./build-macos.sh 2>&1 | tee build.log
grep -n "error:" build.log | head -50      # compile errors
grep -n "Metal\|MSL\|MTL" ~/Library/Caches/rpcs3-metal/RPCS3.log | head -100   # renderer messages
```
