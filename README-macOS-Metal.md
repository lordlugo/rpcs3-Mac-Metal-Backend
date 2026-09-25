# RPCS3 Metal (macOS)

This fork of [RPCS3](https://github.com/RPCS3/rpcs3) targets **macOS on Apple silicon only** and renders
with a **native Metal 4 backend**. Metal is the only renderer: Vulkan/MoltenVK and OpenGL are not built
(their sources stay in the tree to ease merging with upstream).

## Requirements

- A Mac with Apple silicon (M1 or newer). Intel Macs are not supported.
- macOS 26.0 or newer.
- Xcode 26 or its command line tools (`xcode-select --install`).
- [Homebrew](https://brew.sh) (in `/opt/homebrew`).

## Building

```sh
git clone <this repository> rpcs3-metal
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
