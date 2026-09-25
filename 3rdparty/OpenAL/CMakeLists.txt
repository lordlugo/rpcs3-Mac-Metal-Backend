# OpenAL
if(USE_SYSTEM_OPENAL)
    if(WIN32)
      find_package(OpenAL CONFIG REQUIRED)
    else()
      find_package(OpenAL REQUIRED)
    endif()
    add_library(3rdparty_openal INTERFACE)
    target_link_libraries(3rdparty_openal INTERFACE OpenAL::OpenAL)
    set_target_properties(OpenAL::OpenAL PROPERTIES IMPORTED_GLOBAL ON)
else()
    option(ALSOFT_UTILS "Build utility programs" OFF)
    option(ALSOFT_EXAMPLES "Build example programs" OFF)
    # openal-soft 1.25 uses C++20 modules when CMake+Ninja+Clang allow it, which needs clang-scan-deps. The macOS
    # fork builds with CMAKE_CXX_SCAN_FOR_MODULES=OFF (Homebrew LLVM does not always ship clang-scan-deps), so use
    # openal-soft's header fallback; every `import` there is guarded by HAVE_CXXMODULES.
    if(NOT CMAKE_CXX_SCAN_FOR_MODULES)
        set(ALSOFT_ENABLE_MODULES OFF CACHE BOOL "Enable use of C++20 modules when supported" FORCE)
    endif()
    set(LIBTYPE "STATIC")
    add_subdirectory(openal-soft EXCLUDE_FROM_ALL)
    add_library(3rdparty_openal INTERFACE)
    target_link_libraries(3rdparty_openal INTERFACE OpenAL::OpenAL)
endif()
