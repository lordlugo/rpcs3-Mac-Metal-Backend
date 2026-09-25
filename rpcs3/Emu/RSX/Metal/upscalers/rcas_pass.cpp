#include "stdafx.h"

#include "rcas_pass.h"

#include "../mtlutils/device.h"

#include "Utilities/StrUtil.h"

#include <algorithm>
#include <string_view>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wunused-function"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#define A_CPU 1
#include "3rdparty/GPUOpen/include/ffx_a.h"
#include "3rdparty/GPUOpen/include/ffx_fsr1.h"
#undef A_CPU

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace mtl
{
	namespace FidelityFX
	{
		std::string build_rcas_shader_source()
		{
			// Just use AMD-provided source with minimal modification (same sources as the VK backend)
			std::string source =
				#include "Emu/RSX/Program/Upscalers/FSR1/fsr_ubershader.glsl"
			;

			const char* ffx_a_contents =
				#include "Emu/RSX/Program/Upscalers/FSR1/fsr_ffx_a_flattened.inc"
			;

			const char* ffx_fsr_contents =
				#include "Emu/RSX/Program/Upscalers/FSR1/fsr_ffx_fsr1_flattened.inc"
			;

			// Metal has no robust buffer/texture access (see MTLCompute.h). Patch the ubershader before the FFX libraries
			// are inserted (fmt::replace_all never rescans the text it inserts):
			//  - clamp every texelFetch (RCAS reads a 1-texel cross around the output texel) to the input texture;
			//  - wrap the per-texel filter in a bounds check against the output image (the dispatch covers whole
			//    16x16 tiles, so the last row/column of groups may reach past the image).
			const std::pair<std::string_view, std::string> metal_patches[] =
			{
				{ "texelFetch(InputTexture, ASU2(p), 0)", "texelFetch(InputTexture, clamp(ASU2(p), ASU2(0), textureSize(InputTexture, 0) - ASU2(1)), 0)" },
				{ "void CurrFilter(AU2 pos)", "void CurrFilter_unchecked(AU2 pos)" },
				{ "layout(local_size_x=64) in;",
					"void CurrFilter(AU2 pos)\n"
					"{\n"
					"	if (any(greaterThanEqual(pos, AU2(imageSize(OutputTexture)))))\n"
					"		return;\n"
					"\n"
					"	CurrFilter_unchecked(pos);\n"
					"}\n"
					"\n"
					"layout(local_size_x=64) in;" },
			};

			source = fmt::replace_all(source, metal_patches);

			const auto count = [&source](std::string_view needle)
			{
				usz n = 0;
				for (usz pos = source.find(needle); pos != std::string::npos; pos = source.find(needle, pos + needle.size()))
				{
					n++;
				}
				return n;
			};

			const usz texel_fetches = count("texelFetch(");
			if (!texel_fetches ||
				texel_fetches != count("texelFetch(InputTexture, clamp(") ||
				count("void CurrFilter_unchecked(AU2 pos)") != 1 ||
				count("CurrFilter_unchecked(pos);") != 1)
			{
				rsx_log.error("MetalFX: the FSR1 ubershader does not match the RCAS bounds-check patches. RCAS sharpening is unavailable.");
				return {};
			}

			const std::pair<std::string_view, std::string> replacement_table[] =
			{
				{ "%FFX_DEFINITIONS%",
					"#define SAMPLE_RCAS 1\n"
					"#define SAMPLE_EASU 0\n"
					"#define SAMPLE_BILINEAR 0\n"
					"#define SAMPLE_SLOW_FALLBACK 1" },
				{ "%FFX_A_IMPORT%", ffx_a_contents },
				{ "%FFX_FSR_IMPORT%", ffx_fsr_contents },
				{ "%push_block%", "push_constant" }
			};

			return fmt::replace_all(source, replacement_table);
		}

		rcas_pass::rcas_pass()
		{
			m_src = build_rcas_shader_source();

			// No ssbo usage
			ssbo_count = 0;

			// Const0 + Sample (2 x uvec4)
			use_push_constants = true;
			push_constants_size = 32;

			// local_size_x in fsr_ubershader.glsl (baked into the kernel)
			workgroup_size_x = 64;

			create();
		}

		std::vector<glsl::program_input> rcas_pass::get_inputs()
		{
			auto result = compute_task::get_inputs();

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"InputTexture",
				glsl::input_type_texture,
				glsl::binding_set_index_compute,
				0));

			result.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"OutputTexture",
				glsl::input_type_storage_texture,
				glsl::binding_set_index_compute,
				1));

			return result;
		}

		bool rcas_pass::prepare()
		{
			if (m_program)
			{
				return true;
			}

			if (m_build_failed || m_src.empty())
			{
				return false;
			}

			// Same build as compute_task::load_program(), minus the fatal error: sharpening is optional
			m_program = glsl::create_compute_program(m_src, get_inputs());

			if (!m_program || m_program->max_total_threads_per_threadgroup() < workgroup_size_x * workgroup_size_y * workgroup_size_z)
			{
				rsx_log.error("MetalFX: failed to build the RCAS sharpening kernel. Sharpening is disabled.");
				m_program.reset(); // Never used by the GPU
				m_build_failed = true;
				return false;
			}

			return true;
		}

		void rcas_pass::bind_resources(mtl::command_list& /*cmd*/)
		{
			if (!m_sampler)
			{
				// RCAS only uses texelFetch, but the combined image sampler slot still gets a valid sampler
				sampler_create_info info{};
				info.min_filter = MTL::SamplerMinMagFilterNearest;
				info.mag_filter = MTL::SamplerMinMagFilterNearest;
				m_sampler = std::make_unique<mtl::sampler>(*g_render_device, info);
			}

			push_constants(0, push_constants_size, m_constants.data());

			m_program->bind_uniform(glsl::image_binding_info(m_input, m_sampler->value), glsl::binding_set_index_compute, 0);
			m_program->bind_uniform(glsl::image_binding_info(m_output, nullptr), glsl::binding_set_index_compute, 1);
		}

		void rcas_pass::run(mtl::command_list& cmd, mtl::image* src, mtl::image* dst, u32 sharpening_intensity)
		{
			ensure(m_program, "RCAS: prepare() must succeed before run()");
			ensure(src && dst && src->width() == dst->width() && src->height() == dst->height());

			m_input = src->value;
			m_output = dst->value;

			// 0 is actually the sharpest with 2 being the chosen limit. Each progressive unit 'halves' the sharpening intensity.
			const f32 cas_attenuation = 2.f - (static_cast<f32>(std::min(sharpening_intensity, 100u)) / 50.f);

			// Sample.x = 0: the output is not squared (display-referred input and output)
			m_constants.fill(0);
			FsrRcasCon(m_constants.data(), cas_attenuation);

			// Each group of 64 threads filters one 16x16 tile (four 8x8 quadrants), see main() in fsr_ubershader.glsl
			constexpr u32 tile_size = 16;
			const u32 groups_x = utils::aligned_div(dst->width(), tile_size);
			const u32 groups_y = utils::aligned_div(dst->height(), tile_size);

			// Binds the pipeline on cmd.compute() (intra-encoder barrier / queue barrier of a new encoder) and dispatches
			compute_task::run(cmd, groups_x, groups_y, 1);
		}
	}
}
