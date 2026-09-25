# RPCS3 Metal fork: platform policy.
#
# This fork is macOS-only, Apple silicon (arm64) only, requires macOS 26.0 or newer and
# ships a native Metal 4 renderer (rpcs3/Emu/RSX/Metal) as its ONLY renderer.
# Vulkan/MoltenVK and OpenGL sources stay in the tree but are never built here.
#
# This file is included from the top-level CMakeLists.txt BEFORE project() and before the
# option() declarations:
#  - CMAKE_OSX_ARCHITECTURES / CMAKE_OSX_DEPLOYMENT_TARGET must be known before project()
#    so that the toolchain probing and every try_compile() use them.
#  - option() is a no-op when a cache entry of the same name already exists, so forcing the
#    cache values here guarantees they win over the upstream defaults declared later.
#    (Before project(), CMake already defines APPLE/CMAKE_HOST_APPLE on macOS hosts.)

include_guard(GLOBAL)

set(RPCS3_FORK_MIN_MACOS "26.0")

if(NOT APPLE AND NOT CMAKE_HOST_APPLE AND NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
	message(FATAL_ERROR "This RPCS3 fork only supports macOS ${RPCS3_FORK_MIN_MACOS}+ on Apple silicon (native Metal renderer). "
		"Use upstream RPCS3 (https://github.com/RPCS3/rpcs3) for other platforms.")
endif()

# Apple silicon only
if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
	message(WARNING "RPCS3 Metal fork: CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' is not supported, forcing 'arm64'.")
endif()
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture (RPCS3 Metal fork: Apple silicon only)" FORCE)

# Minimum macOS version: honour a higher value (cache or MACOSX_DEPLOYMENT_TARGET env), never a lower one
set(_rpcs3_fork_osx_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
if(NOT _rpcs3_fork_osx_target AND DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
	set(_rpcs3_fork_osx_target "$ENV{MACOSX_DEPLOYMENT_TARGET}")
endif()
if(NOT _rpcs3_fork_osx_target OR _rpcs3_fork_osx_target VERSION_LESS RPCS3_FORK_MIN_MACOS)
	if(_rpcs3_fork_osx_target)
		message(WARNING "RPCS3 Metal fork: deployment target ${_rpcs3_fork_osx_target} is too low, using ${RPCS3_FORK_MIN_MACOS}.")
	endif()
	set(_rpcs3_fork_osx_target "${RPCS3_FORK_MIN_MACOS}")
endif()
set(CMAKE_OSX_DEPLOYMENT_TARGET "${_rpcs3_fork_osx_target}" CACHE STRING "Minimum macOS version (RPCS3 Metal fork: ${RPCS3_FORK_MIN_MACOS} or newer)" FORCE)
unset(_rpcs3_fork_osx_target)

# Renderer / backend policy. These are declared with option() in the top-level CMakeLists.txt;
# the FORCEd cache entries below make those option() calls no-ops.
set(USE_VULKAN OFF CACHE BOOL "Vulkan render backend (always OFF: Metal is the only renderer of this fork)" FORCE)
set(USE_SYSTEM_MVK OFF CACHE BOOL "Prefer system MoltenVK (always OFF: MoltenVK is not used by this fork)" FORCE)
set(USE_SYSTEM_VULKAN_MEMORY_ALLOCATOR OFF CACHE BOOL "Prefer system Vulkan Memory Allocator (always OFF: Vulkan is not built)" FORCE)
set(USE_FAUDIO OFF CACHE BOOL "FAudio audio backend (always OFF on this fork, Cubeb/CoreAudio is used)" FORCE)

option(USE_METAL "Native Metal 4 render backend (the only renderer of this fork)" ON)
if(NOT USE_METAL)
	message(FATAL_ERROR "USE_METAL=OFF is not supported: Metal is the only renderer of this RPCS3 fork.")
endif()

message(STATUS "RPCS3 Metal fork: macOS ${CMAKE_OSX_DEPLOYMENT_TARGET}+ / ${CMAKE_OSX_ARCHITECTURES}, renderer: Metal (Vulkan/OpenGL disabled)")
