#pragma once

// FidelityFX RCAS (robust contrast adaptive sharpening), run after MetalFX spatial upscaling.
// Port of vk::FidelityFX::rcas_pass (VK/upscalers/fsr_pass.h): same shared FSR1 ubershader, same configuration
// (SAMPLE_RCAS, 32-bit "slow fallback" path, 32 bytes of push constants, 64 threads per group, one 16x16 tile per group)
// and the same sharpening curve driven by g_cfg.video.rcas_sharpening_intensity.
//
// Metal differences (no robust buffer/texture access, see MTLCompute.h):
//  - the RCAS taps (a 1-texel cross around every output texel) are clamped to the input texture;
//  - output texels outside the destination (partial 16x16 tiles) are skipped.
//
// Resources: src needs MTL::TextureUsageShaderRead, dst needs MTL::TextureUsageShaderWrite. Both must have the same size.
// Ordering: the kernel is recorded through cmd.compute() like every compute task, i.e. after an intra-encoder barrier
// (or the queue barrier of a new encoder), so it reads what earlier commands wrote.

#include "../MTLCompute.h"
#include "../mtlutils/sampler.h"

#include <array>
#include <memory>
#include <string>

namespace mtl
{
	namespace FidelityFX
	{
		// The kernel source (GLSL, Vulkan semantics). Empty if the shared shader no longer matches the Metal patch
		// points (then the pass is unavailable). Exposed for offline shader validation.
		std::string build_rcas_shader_source();

		class rcas_pass : public compute_task
		{
			std::unique_ptr<mtl::sampler> m_sampler;
			MTL::Texture* m_input = nullptr;
			MTL::Texture* m_output = nullptr;
			std::array<u32, 8> m_constants{}; // Const0, Sample

			bool m_build_failed = false;

		protected:
			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(mtl::command_list& cmd) override;

		public:
			rcas_pass();

			// Builds the pipeline if needed (synchronously, once). Returns false, without throwing, if the kernel is
			// unavailable; run() must not be called then.
			bool prepare();

			// Sharpens src into dst. sharpening_intensity: 1..100 (0 = off is handled by the caller).
			void run(mtl::command_list& cmd, mtl::image* src, mtl::image* dst, u32 sharpening_intensity);
		};
	}
}
