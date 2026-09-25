#!/usr/bin/env bash
# Build RPCS3 Metal (native Metal 4 renderer) on macOS 26+ / Apple silicon with Homebrew.
#
#   ./build-macos.sh --deps            install/upgrade the Homebrew dependencies, then exit
#   ./build-macos.sh                   configure + build (RelWithDebInfo) into build-metal/
#   ./build-macos.sh --release|--debug choose the build type (default: RelWithDebInfo)
#   ./build-macos.sh --clean           delete build-metal/ first (full rebuild)
#   ./build-macos.sh --configure-only  init submodules and configure, but do not build (used by CI)
#
# Environment:
#   RPCS3_WITH_OPENCV=1   also install/use Homebrew OpenCV (optional camera features)
#   RPCS3_NATIVE_INSTRUCTIONS=OFF  portable binary (-march=armv8.4-a) instead of -march=native (CI artifacts)
#   BUILD_DIR=<dir>       build directory (default: build-metal)
#   JOBS=<n>              parallel jobs for ninja (default: ninja's choice)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-build-metal}"
case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$ROOT/$BUILD_DIR" ;;
esac

BUILD_TYPE="RelWithDebInfo"
DO_DEPS=0
DO_CLEAN=0
CONFIGURE_ONLY=0
MIN_MACOS_MAJOR=26
MIN_CLANG_MAJOR=19

# Homebrew formulae. qt/sdl3/llvm are required; abseil lets the bundled protobuf avoid a network fetch.
BREW_DEPS=(cmake ninja ccache llvm qt sdl3 pkg-config abseil)

usage()
{
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die()
{
    echo "error: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deps) DO_DEPS=1 ;;
        --clean) DO_CLEAN=1 ;;
        --release) BUILD_TYPE="Release" ;;
        --debug) BUILD_TYPE="Debug" ;;
        --relwithdebinfo) BUILD_TYPE="RelWithDebInfo" ;;
        --configure-only) CONFIGURE_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "unknown option '$1'" ;;
    esac
    shift
done

# Everything this script prints also goes to build-macos.log next to it (full compile errors, for bug reports)
LOG_FILE="$ROOT/build-macos.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "==> Full log: $LOG_FILE"

# ---------------------------------------------------------------------------------------------------------------------
# Host checks: macOS 26+, native arm64 (not Rosetta), Xcode command line tools, Homebrew
# ---------------------------------------------------------------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || die "this script only runs on macOS"
[[ "$(uname -m)" == "arm64" ]] || die "Apple silicon (arm64) is required (uname -m reports '$(uname -m)')"
if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
    die "this shell runs under Rosetta 2; use a native arm64 terminal"
fi

MACOS_VERSION="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VERSION%%.*}"
(( MACOS_MAJOR >= MIN_MACOS_MAJOR )) || die "macOS ${MIN_MACOS_MAJOR}.0 or newer is required (found ${MACOS_VERSION})"

xcode-select -p >/dev/null 2>&1 || die "Xcode command line tools are missing: run 'xcode-select --install' (Xcode 26 recommended)"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null)" || die "could not locate the macOS SDK (xcrun --show-sdk-path)"

command -v brew >/dev/null 2>&1 || die "Homebrew is required: https://brew.sh"
BREW_PREFIX="$(brew --prefix)"

# ---------------------------------------------------------------------------------------------------------------------
# --deps: install Homebrew dependencies and exit
# ---------------------------------------------------------------------------------------------------------------------
if (( DO_DEPS )); then
    deps=("${BREW_DEPS[@]}")
    if [[ "${RPCS3_WITH_OPENCV:-0}" == "1" ]]; then
        deps+=(opencv)
    fi
    echo "==> Installing Homebrew dependencies: ${deps[*]}"
    brew install "${deps[@]}"
    echo "==> Dependencies installed. Now run: ./build-macos.sh"
    exit 0
fi

for formula in llvm qt sdl3; do
    brew --prefix --installed "$formula" >/dev/null 2>&1 || die "Homebrew formula '$formula' is not installed; run './build-macos.sh --deps' first"
done
for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' not found; run './build-macos.sh --deps' first"
done

# ---------------------------------------------------------------------------------------------------------------------
# Submodules (same selection as .ci/build-mac.sh, minus Vulkan-only ones).
# llvm/opencv/SDL/curl/zlib come from Homebrew or the macOS SDK, feralinteractive (gamemode) is Linux-only,
# GPUOpen/VulkanMemoryAllocator is only used by the (disabled) Vulkan backend and FAudio is disabled on this fork.
# SPIRV-Cross and glslang are required by the Metal backend.
# ---------------------------------------------------------------------------------------------------------------------
cd "$ROOT"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "==> Updating git submodules"
    # shellcheck disable=SC2046
    git submodule -q update --init --depth=1 --jobs=8 $(awk '/path/ && !/llvm/ && !/opencv/ && !/libsdl-org/ && !/feralinteractive/ && !/curl/ && !/zlib/ && !/VulkanMemoryAllocator/ && !/FAudio/ { print $3 }' .gitmodules)
else
    echo "warning: not a git checkout, assuming all required submodules are already present" >&2
fi

for required in 3rdparty/SPIRV-Cross/SPIRV-Cross/CMakeLists.txt 3rdparty/glslang/glslang/CMakeLists.txt 3rdparty/metal-cpp/Metal/Metal.hpp; do
    [[ -f "$ROOT/$required" ]] || die "missing '$required' (submodule not initialised?)"
done

# ---------------------------------------------------------------------------------------------------------------------
# Toolchain: Homebrew LLVM clang (RPCS3 needs clang >= 19; Apple clang reports 17.x) + Homebrew libc++
# ---------------------------------------------------------------------------------------------------------------------
LLVM_PREFIX="$(brew --prefix llvm)"
QT_PREFIX="$(brew --prefix qt)"

export CC="$LLVM_PREFIX/bin/clang"
export CXX="$LLVM_PREFIX/bin/clang++"
[[ -x "$CXX" ]] || die "Homebrew clang not found at $CXX"

CLANG_MAJOR="$("$CXX" -dumpversion | cut -d. -f1)"
(( CLANG_MAJOR >= MIN_CLANG_MAJOR )) || die "clang >= ${MIN_CLANG_MAJOR} is required (Homebrew llvm provides $("$CXX" -dumpversion))"

# Link against Homebrew's libc++/libunwind that match the clang headers (see 'brew info llvm')
LLVM_LDFLAGS="-L$LLVM_PREFIX/lib/c++ -Wl,-rpath,$LLVM_PREFIX/lib/c++ -L$LLVM_PREFIX/lib/unwind -Wl,-rpath,$LLVM_PREFIX/lib/unwind -lunwind"

LLVM_CMAKE_DIR="$LLVM_PREFIX/lib/cmake/llvm"
[[ -f "$LLVM_CMAKE_DIR/LLVMConfig.cmake" ]] || die "LLVMConfig.cmake not found in $LLVM_CMAKE_DIR"

# Homebrew's qt may be a monolithic keg or a meta formula over split qtbase/qtmultimedia/... kegs
QT6_CMAKE_DIR=""
for candidate in "$QT_PREFIX/lib/cmake/Qt6" "$BREW_PREFIX/lib/cmake/Qt6" "$(brew --prefix qtbase 2>/dev/null || true)/lib/cmake/Qt6"; do
    if [[ -f "$candidate/Qt6Config.cmake" ]]; then
        QT6_CMAKE_DIR="$candidate"
        break
    fi
done
[[ -n "$QT6_CMAKE_DIR" ]] || die "Qt6Config.cmake not found (is Homebrew 'qt' installed?)"

MACDEPLOYQT=""
for candidate in "$QT_PREFIX/bin/macdeployqt" "$BREW_PREFIX/bin/macdeployqt" "$(command -v macdeployqt 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        MACDEPLOYQT="$candidate"
        break
    fi
done
[[ -n "$MACDEPLOYQT" ]] || die "macdeployqt not found (is Homebrew 'qt' installed?)"

USE_OPENCV=OFF
if [[ "${RPCS3_WITH_OPENCV:-0}" == "1" ]]; then
    USE_OPENCV=ON
fi

# -march=native by default (local builds); CI sets RPCS3_NATIVE_INSTRUCTIONS=OFF so artifacts run on every Apple silicon Mac
USE_NATIVE="${RPCS3_NATIVE_INSTRUCTIONS:-ON}"

# Bundle version (same scheme as .ci/build-mac.sh)
COMM_TAG="$(awk '/version{.*}/ { printf("%d.%d.%d", $5, $6, $7) }' "$ROOT/rpcs3/rpcs3_version.cpp")"
COMM_COUNT="$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 1)"

if command -v ccache >/dev/null 2>&1; then
    echo "==> ccache: $(command -v ccache) (CMake picks it up automatically)"
fi

# ---------------------------------------------------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------------------------------------------------
# `brew upgrade llvm` deletes the old Cellar version, but CMake's cache keeps pointing into it (clang-scan-deps, the
# clang resource headers, libc++). Start from a clean build directory whenever clang or the SDK changed.
TOOLCHAIN_ID="$("$CXX" --version | head -n1) | $("$CXX" -print-resource-dir) | $SDK_PATH"
TOOLCHAIN_STAMP="$BUILD_DIR/.rpcs3-toolchain"
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && [[ "$(cat "$TOOLCHAIN_STAMP" 2>/dev/null || true)" != "$TOOLCHAIN_ID" ]]; then
    echo "==> Toolchain changed since the last configure (Homebrew LLVM upgrade or new SDK?): rebuilding from scratch"
    DO_CLEAN=1
fi

if (( DO_CLEAN )) && [[ -d "$BUILD_DIR" ]]; then
    echo "==> Removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

echo "==> Configuring ($BUILD_TYPE) in $BUILD_DIR"
echo "    compiler: $CXX ($("$CXX" -dumpversion)), SDK: $SDK_PATH"
echo "    Qt6: $QT6_CMAKE_DIR, LLVM: $LLVM_CMAKE_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_EXE_LINKER_FLAGS="$LLVM_LDFLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LLVM_LDFLAGS" \
    -DCMAKE_MODULE_LINKER_FLAGS="$LLVM_LDFLAGS" \
    -DCMAKE_PREFIX_PATH="$QT_PREFIX;$BREW_PREFIX" \
    -DLLVM_DIR="$LLVM_CMAKE_DIR" \
    -DQt6_DIR="$QT6_CMAKE_DIR" \
    -DMACDEPLOYQT_EXECUTABLE="$MACDEPLOYQT" \
    -DMACOSX_BUNDLE_SHORT_VERSION_STRING="$COMM_TAG" \
    -DMACOSX_BUNDLE_BUNDLE_VERSION="$COMM_COUNT" \
    -DUSE_METAL=ON \
    -DUSE_VULKAN=OFF \
    -DUSE_SDL=ON \
    -DUSE_SYSTEM_SDL=ON \
    -DUSE_SYSTEM_FFMPEG=OFF \
    -DUSE_SYSTEM_OPENCV="$USE_OPENCV" \
    -DUSE_PRECOMPILED_HEADERS=OFF \
    -DUSE_NATIVE_INSTRUCTIONS="$USE_NATIVE" \
    -DSTATIC_LINK_LLVM=OFF \
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "$TOOLCHAIN_ID" > "$TOOLCHAIN_STAMP"

if (( CONFIGURE_ONLY )); then
    echo "==> Configured. Build with: ninja -C \"$BUILD_DIR\""
    exit 0
fi

# ---------------------------------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------------------------------
echo "==> Building"
# `-k 0`: keep building after a failed file so one run reports every compile error, not just the first few
BUILD_ARGS=()
if [[ -n "${JOBS:-}" ]]; then
    BUILD_ARGS+=(--parallel "$JOBS")
fi
if ! cmake --build "$BUILD_DIR" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} -- -k 0; then
    echo >&2
    die "build failed; see the 'error:' lines above or in $LOG_FILE (grep -n 'error:' build-macos.log)"
fi

APP="$BUILD_DIR/bin/rpcs3.app"
[[ -d "$APP" ]] || die "build finished but $APP was not found"

# The POST_BUILD step already ad-hoc signs the bundle with rpcs3/rpcs3.entitlements; verify it
codesign --verify --deep --strict "$APP" >/dev/null 2>&1 || echo "warning: code signature verification failed for $APP" >&2

echo
echo "==> RPCS3 Metal built successfully:"
echo "    $APP"
echo
echo "    Run it with:  open \"$APP\""
echo "    or (with logs in the terminal):  \"$APP/Contents/MacOS/rpcs3\""
echo "    Config/data:  ~/Library/Application Support/rpcs3-metal"
