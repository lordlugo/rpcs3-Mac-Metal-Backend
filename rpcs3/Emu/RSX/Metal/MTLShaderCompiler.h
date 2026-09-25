#pragma once

// GLSL (Vulkan semantics) -> SPIR-V (glslang) -> MSL (SPIRV-Cross) -> MTL::Library (MTL4Compiler).
//
// The translation half is pure CPU work and is exposed separately so it can be reused by tooling (shader cache
// inspection, offline validation). Everything here is thread-safe and may run on pipe-compiler worker threads.
// The implementation TU is the only one in the backend built with exceptions (SPIRV-Cross throws on errors).

#include "MTLProgramPipeline.h"
#include "util/atomic.hpp"

#include <array>
#include <string>
#include <string_view>

namespace mtl::glsl
{
	struct msl_translation_result
	{
		std::string msl;                         // Generated Metal Shading Language source
		std::string entry_point;                 // Entry point function name inside `msl`
		bool needs_buffer_size_buffer = false;   // MSL reads spvBufferSizeConstants at layout.buffer_count
		std::array<u32, 3> workgroup_size{ 1, 1, 1 }; // Compute only: GLSL local_size (dispatch must use it)
	};

	// Unique (per stage) MSL entry point names given to every translated shader.
	std::string_view get_entry_point_name(::glsl::program_domain domain);

	// Translate Vulkan-flavoured GLSL to MSL using `layout` for every (set, binding) -> Metal index mapping.
	// Returns false and logs the GLSL (and whatever MSL could be produced) on failure.
	bool translate_glsl_to_msl(
		::glsl::program_domain domain,
		const std::string& glsl_source,
		const binding_layout& layout,
		msl_translation_result& result);

	// Build a Metal library from MSL through the device's MTL4Compiler. Returns an owned (+1) library or nullptr
	// (errors, including the MSL, are logged). `fast_math` selects MTL::MathModeFast vs MTL::MathModeSafe.
	MTL::Library* compile_msl_library(const std::string& msl, std::string_view label, bool fast_math);

	// Debug switch: when set, every generated MSL source is written to the log (notice level).
	// Independently, "Log shader programs" (g_cfg.video.log_programs) dumps the MSL to shaderlog/*.msl.
	extern atomic_t<bool> g_dump_msl_to_log;
}
