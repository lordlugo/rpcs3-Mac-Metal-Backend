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

    # openal-soft 1.25 passes &std::unique_ptr<...>::operator* to std::views::transform. Taking the address of a
    # standard library member function is unspecified, and libc++ 23 (Homebrew LLVM 23) rejects it ("no matching
    # function for call to object of type '__transform::__fn'"). Patch the only use built on macOS with an equivalent
    # lambda. Idempotent; leaves the submodule untouched when the pattern is absent.
    set(_alsoft_alc "${CMAKE_CURRENT_SOURCE_DIR}/openal-soft/alc/alc.cpp")
    if(EXISTS "${_alsoft_alc}")
        file(READ "${_alsoft_alc}" _alsoft_src)
        set(_alsoft_old "std::views::transform(&ContextBase::EffectSlotCluster::operator*)")
        set(_alsoft_new "std::views::transform([](ContextBase::EffectSlotCluster &cluster) -> auto& { return *cluster; })")
        string(FIND "${_alsoft_src}" "${_alsoft_old}" _alsoft_pos)
        if(NOT _alsoft_pos EQUAL -1)
            string(REPLACE "${_alsoft_old}" "${_alsoft_new}" _alsoft_src "${_alsoft_src}")
            file(WRITE "${_alsoft_alc}" "${_alsoft_src}")
            message(STATUS "RPCS3: patched openal-soft alc/alc.cpp for libc++ 23")
        endif()
    endif()

    add_subdirectory(openal-soft EXCLUDE_FROM_ALL)
    add_library(3rdparty_openal INTERFACE)
    target_link_libraries(3rdparty_openal INTERFACE OpenAL::OpenAL)
endif()
