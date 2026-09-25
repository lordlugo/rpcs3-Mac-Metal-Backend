# RPCS3 Metal fork: keep Homebrew's shared include directory from shadowing bundled and SDK headers.
#
# Imported targets from Homebrew (SDL3, abseil, ...) add ${HOMEBREW_PREFIX}/include to the compile line as -isystem,
# ahead of bundled 3rdparty headers that are also -isystem (protobuf, libpng, SoundTouch, miniupnpc, stb). Any
# Homebrew formula that installs the same library (protobuf comes with Qt) then silently replaces the bundled headers,
# e.g. "Protobuf C++ gencode is built with an incompatible version of Protobuf C++ headers/runtime".
#
# Declare the directory as compiler-implicit so CMake never emits it, and search it with -idirafter instead: after every
# -I/-isystem directory, libc++, the clang resource headers and the macOS SDK. Headers that only Homebrew provides
# (<SDL3/...>, <absl/...>) are still found.
#
# Must be included after project() (which sets CMAKE_<LANG>_IMPLICIT_INCLUDE_DIRECTORIES) and before any target or
# add_subdirectory() so every directory scope inherits it.

if(NOT APPLE)
	return()
endif()

set(_rpcs3_brew_include "")
if(DEFINED ENV{HOMEBREW_PREFIX} AND IS_DIRECTORY "$ENV{HOMEBREW_PREFIX}/include")
	set(_rpcs3_brew_include "$ENV{HOMEBREW_PREFIX}/include")
elseif(IS_DIRECTORY "/opt/homebrew/include")
	set(_rpcs3_brew_include "/opt/homebrew/include")
endif()

if(_rpcs3_brew_include)
	message(STATUS "RPCS3 Metal fork: searching ${_rpcs3_brew_include} last (-idirafter)")
	foreach(_lang C CXX OBJC OBJCXX)
		list(APPEND CMAKE_${_lang}_IMPLICIT_INCLUDE_DIRECTORIES "${_rpcs3_brew_include}")
	endforeach()
	add_compile_options("SHELL:-idirafter \"${_rpcs3_brew_include}\"")
endif()

unset(_rpcs3_brew_include)
