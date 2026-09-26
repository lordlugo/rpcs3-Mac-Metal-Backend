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
git clone https://github.com/lordlugo/rpcs3-Mac-Metal-Backend.git rpcs3-metal
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
| `--no-lto` | Skip link-time optimization: the final link takes seconds instead of minutes (slightly slower app) |
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

`.github/workflows/macos-metal.yml` builds every push to `master` and every pull request on a macOS 26
arm64 runner (Release by default; other build types can be picked when starting it manually). It compiles the
Metal backend objects first (`metal-compile.log`), then the whole project
(`full-build.log`), collects all compiler errors in `errors.txt` (also shown as annotations) and uploads the
logs, plus the zipped app when the build succeeds.

## Status

Builds with Homebrew LLVM on macOS 26 and newer and runs games on Apple silicon. Still early: expect bugs, and
report them with the log (see the end of this page).

- **Graphics:** native Metal 4 renderer (`rpcs3/Emu/RSX/Metal`, design notes in `DESIGN.md` there). Metal 4 command
  queues, allocators and argument tables, residency sets, explicit barriers. RSX shaders are translated
  GLSL -> SPIR-V -> MSL (SPIRV-Cross) and compiled with `MTL4Compiler` on worker threads. Texture and surface caches
  (D24S8 stored as Depth32Float_Stencil8), programmable blending and feedback loops through framebuffer fetch, MSAA,
  occlusion queries (ZCULL), compute kernels, native overlays, MetalFX spatial upscaling, ProMotion frame pacing.
- **Audio:** Core Audio output, lossless stereo and spatial audio surround (see [Audio](#audio)).
- **Controllers:** DualShock 3, DualShock 4, DualSense and other gamepads, through RPCS3's own handlers or SDL
  (which uses Apple's GameController framework on macOS).
- **macOS app:** Apple silicon and macOS 26+ only, no Vulkan/MoltenVK/OpenGL, own bundle ID and folders, updater off,
  QoS thread priorities, Game Mode, sRGB color management, check boxes drawn correctly on macOS 26/27.

### Known limitations of the Metal renderer

| Missing | What you may notice | Why |
|---|---|---|
| Shader interpreter | An object or effect missing for a moment the first time it appears. The shader modes with "Shader Interpreter" behave like "Async Recompiler" | Not ported to Metal yet. The renderer waits up to 8 ms per frame for a compiling shader, and the pipeline archive keeps compiled shaders for later sessions |
| Logic operations | Rare: a draw that combines colors with a logic operation (XOR, AND, ...) writes its color unchanged | Metal has no logic operation state; shader emulation not written yet |
| Wide lines | Lines are always 1 pixel wide (some debug views and UI elements) | Metal only draws 1-pixel lines |
| Flat shading (last vertex) | Flat-shaded surfaces are drawn with smooth shading (colors blend across a triangle) | Metal has no provoking-vertex setting |

The log shows a one-time warning starting with `Metal:` when a game uses one of these.

## Defaults on this fork

New installs start with these settings, and the first start of a version that introduces a new default applies it to
an existing global configuration once (per-game configurations are left alone). Everything can still be changed in the
settings.

| Setting | Default | Why |
|---|---|---|
| VSync | Full | Presentation is paced to the display (ProMotion aware, see below) only with VSync on |
| Shader Quality | Ultra | Full-precision 3-component dot products are built from fused multiply-adds, matching the RSX more closely; the cost on Apple GPUs is negligible |
| Resolution Scale | 200% (2560x1440) | Sharp on Retina displays while leaving GPU headroom; MetalFX takes it the rest of the way to the window |
| Output Scaling | MetalFX Spatial Upscaling (+ RCAS sharpening) | Stored as "FidelityFX Super Resolution" in config.yml; the RCAS slider at 0 turns sharpening off. Runs whenever the image is upscaled to the window; when the resolution scale makes the image larger than the window, it is scaled down with a bilinear draw |
| Anisotropic Filter | Automatic = 16x | Applied to the game's textures (not to render targets read by effects); pick a lower value to limit it, or Strict Rendering Mode for the PS3's own setting |
| Pipeline archive | On | Compiled GPU pipelines are saved next to the shader cache, so later boots skip most compiles. `RPCS3_METAL_PIPELINE_ARCHIVE=0` turns it off |
| Multithreaded RSX | On | The worker thread sleeps when idle (upstream keeps it spinning on a core forever) and only takes large copies and GPU command submission, so it no longer costs a performance core |
| Audio renderer | Core Audio (Spatial Audio) | Native output, see [Audio](#audio). Cubeb is still available |

Some games need specific settings to be playable. RPCS3 keeps them in its config database, which this build only
downloads when asked (Help > Download Config Database). A few titles get them built in, applied like the database:
only when the game has no custom configuration, and only for the listed settings.

| Game | Built-in settings | Fixes |
|---|---|---|
| Grand Theft Auto IV (BLES00229, BLUS30127, NPEB00882, Complete Edition) | SPU XFloat Accuracy: Accurate, MSAA: Disabled, Write Color Buffers: On, Relaxed ZCULL Sync: On, Accurate ZCULL stats: Off, Sleep Timers Accuracy: As Host | Falling through the world in the prologue (missing collision), doubled or flickering image, missing reflections |

## Audio

The Core Audio renderer (Settings > Audio > Renderer, default on macOS) outputs exactly what the emulated system
mixes: 48 kHz, 32-bit float, no resampling and no processing for stereo. When the output device supports 48 kHz it is
switched to 48 kHz while RPCS3 uses it and switched back afterwards (other apps are resampled by macOS meanwhile).

Surround: tick the formats the game may use in the list under Settings > Audio > Audio Format (Linear PCM 5.1/7.1,
Dolby Digital, DTS; the emulated system mixes Dolby and DTS as 5.1 PCM, like the real console does before encoding).
Ticking a format sets Audio Format to Manual, which is the mode that uses the list. Then:

- **AirPods and other headphones**: the 5.1/7.1 channels are rendered as a virtual speaker set around you with Apple's
  spatial audio renderer (AUSpatialMixer). Head tracking and your personalized spatial audio profile need an app signed
  with an Apple Developer team (see `rpcs3/rpcs3-spatial-audio.entitlements`); the self-built app uses the generic
  profile with the sound field fixed to your head.
- **MacBook speakers**: Apple's speaker virtualization for the built-in speakers.
- **Receivers, HDMI, multichannel interfaces**: each channel goes to the matching speaker of the layout set in
  Audio MIDI Setup > Configure Speakers. Core Audio's standard downmix is only used when the layouts differ.

Options in `config.yml` (Audio section): `Spatial Audio: Automatic | Headphones | Speakers | Off`,
`Spatial Audio Head Tracking: true`, `Switch Device To 48 kHz: true`. The log shows the chosen route
(`CoreAudio: Output route: ...`).

**ZCULL and semaphores.** Some games wait for a semaphore and then read occlusion (ZCULL) results with the CPU;
Toy Story 3 flickers black otherwise. Upstream RPCS3 handles this with a full GPU sync at such semaphores (and only in
Strict Rendering Mode for texture semaphores). This fork holds the semaphore back until the results queued before it
are in memory, like the real hardware, without stalling the RSX, once the game is seen reading results. On macOS this
matters for most games: with 16 KiB memory pages the semaphores share a page with the results, so a CPU polling a
semaphore looks like a CPU reading results (in Assassin's Creed II the RSX spent ~13% of its time waiting for the GPU
there). Strict Rendering Mode is not needed. The log shows `ZCULL: semaphores now wait for the zcull reports ...`.

**MSAA lighting (outlines along edges).** Games that light the scene in a separate pass (for example Ben 10
Ultimate Alien: Cosmic Destruction) read their anti-aliased buffers back at each pixel. A lookup at a pixel centre falls
exactly between the pixel's two samples, and float rounding used to pick either one at random, so pixels on geometry
edges mixed the depth, normal and light of different surfaces: thin colored outlines tracing the geometry. These reads
now snap to one sample, like the PS3's texture unit. Vertex positions are also computed identically in every pass
(invariant), so depth-equal multi-pass drawing cannot leave gaps either. The first boot after updating recompiles the
game's shaders once.

**GPU load.** Apple GPUs keep the frame being drawn in on-chip tile memory and write it to memory when a render pass
ends; every extra pass stores and reloads the whole frame, which at high resolution scales is most of the GPU's work.
The renderer now ends a pass for a feedback read (an effect sampling the depth or colour buffer being drawn) only when
the sampled buffer was written in that pass, instead of on every such draw. Particles, fog, heat haze and deferred
lights used to split the pass on each draw and pegged the GPU. Water, refraction and distortion are usually drawn as a
run of draws of one material (same shaders, textures, blending, viewport) that each sample and write the colour or
depth buffer. Such a run no longer ends the pass before every piece: each piece reads the buffer without the other
pieces of the run. The PS3 does not order these reads either unless the game invalidates its texture cache or waits
for idle, and either one ends the run, as do a clear, another material or a plain write. Depth writes after depth
reads (soft particles, fog) no longer split the pass: those reads are per pixel, which a tile-based GPU already orders
before later writes. Colour writes after colour reads still split it once. The next pass's geometry is now processed
while the previous pass is still being shaded. Strict Rendering Mode keeps the exact ordering, one pass split per such
read or write, if a game ever shows a difference. At the output, the letterbox bars are cleared by the
final draw itself, and the automatic 16x anisotropic filtering applies to the game's textures only, not to screen-space
buffers. Every 30 s the log gets a line
`Metal: GPU busy ... ms per frame (...% of the time), ... render passes and ... feedback splits per frame (...)` with
the reasons for the splits and the feedback reads that stayed in the pass; if the GPU time per frame rises in the same
scene after a while, the Mac is getting hot and lowering its GPU clock.

**Shader compilation.** There is no shader interpreter on Metal yet, so a draw whose shaders are still compiling is
skipped (missing geometry for a moment the first time an effect appears). The renderer now waits up to 8 ms per frame
for such pipelines before skipping, and compiled pipelines are saved to the pipeline archive, so later sessions
start with them.

**Videos and cutscenes.** Texture uploads are copied out of guest memory when the RSX reaches the draw, instead of
letting the GPU read guest memory later ("zero-copy"). Video players (Bink, the PS3's video decoder) decode the next
frame into the same buffer as soon as the RSX has read the current one; with zero-copy the GPU could read a frame that
was half overwritten, which showed as tearing, blocky frames and flicker during videos.

**Colors.** The game image is tagged as sRGB, so macOS color-matches it to the display like any other SDR content.
Before, it was shown untagged, which on P3 and XDR displays (every recent MacBook Pro, iMac and Studio Display) made
colors more saturated and contrast different from the same game on a TV or on other Macs. No setting needed.

**Check boxes on macOS 26/27.** Qt draws check boxes and radio buttons by rendering an AppKit button into the widget;
with the Liquid Glass controls (always on macOS 27, where the Info.plist compatibility key is ignored) the tick is not
drawn, so ticked boxes looked empty (game patches, settings, format lists). RPCS3 now paints these indicators itself
in the macOS look (accent color, white tick) and leaves the rest of the native style alone.

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
| Unified memory, no-copy buffers | Yes | The GPU writes back to guest memory directly (texture uploads are copied, see Videos and cutscenes) |
| MetalFX spatial upscaling (`MTL4FXSpatialScaler`) | Yes | Replaces FSR 1; used whenever the image is upscaled |
| `presentAfterMinimumDuration`, ProMotion refresh intervals | Yes | Frame pacing |
| Depth bounds test | M5 (Apple10) only | Ignored on older GPUs |
| Hardware sampler LOD bias | M5 (Apple10) only | Done in the shader on M1-M4 |
| MetalFX temporal upscaling, frame interpolation, denoiser | No | Need motion vectors and a jittered camera from the game; PS3 games don't provide them |
| Ray tracing, mesh shaders, tensors / ML in shaders | No | Nothing in RSX (the PS3 GPU) maps to them |
| Lossy texture compression | No | Changes pixels the game may read back; not acceptable for emulation |
| Sparse textures and buffers | No | Guest memory is already mapped with no copy |
| Raster order groups, tile shaders, memoryless targets | No | Framebuffer fetch covers the blending cases; PS3 render targets must stay in memory |
| HDR / EDR output | No | PS3 output is SDR; the layer is tagged sRGB so macOS color-matches it on P3/XDR displays |

## Testing the Metal renderer

Start from a terminal to see the log and enable Apple's debugging aids: Metal API and shader validation (slow, but
catches API misuse) and the Metal performance HUD.

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_HUD_ENABLED=1 \
  build-metal/bin/rpcs3.app/Contents/MacOS/rpcs3
```

To record a frame for Xcode's Metal debugger, also set `MTL_CAPTURE_ENABLED=1`. The capture layer is not enabled in
normal launches because it costs performance.

Useful settings: GPU → Shader Mode "Async Recompiler (multi-threaded)", and Debug → "Log shader programs" (writes
the GLSL and MSL of every shader to `~/Library/Caches/rpcs3-metal/shaderlog`). The RPCS3 log is in
`~/Library/Caches/rpcs3-metal/RPCS3.log`.

## Reporting build or runtime problems

`build-macos.sh` writes everything it prints to `build-macos.log` in the repository root.

```sh
grep -n "error:" build-macos.log | head -50
grep -n "Metal\|MSL\|MTL" ~/Library/Caches/rpcs3-metal/RPCS3.log | head -100
```

The first command lists compile errors, the second the renderer's messages from the emulator log.
