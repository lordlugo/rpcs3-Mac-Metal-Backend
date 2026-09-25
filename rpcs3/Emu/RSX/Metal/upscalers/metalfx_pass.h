#pragma once

// MetalFX spatial upscaling (replaces the VK backend's FidelityFX FSR1 pass, output_scaling_mode::fsr).
//
// The scaler is an MTL4FX::SpatialScaler encoded straight into the Metal 4 command buffer. It is (re)created
// whenever the input texture format/size or the output size changes. If MetalFX is unavailable or the scaler
// cannot be created, the pass behaves like bilinear_upscale_pass.
//
// Texture usage requirements:
//  - input (src): scaler->colorTextureUsage() — MTL::TextureUsageShaderRead on current systems. The flip image
//    must therefore be created with ShaderRead (surfaces and textures always are). If the input lacks a required
//    bit, the pass logs once and falls back to bilinear.
//  - output: created by this pass with scaler->outputTextureUsage() | ShaderRead | RenderTarget, so it can feed
//    video_out_calibration_pass or be drawn into the drawable.

#include "upscaling.h"

#include <memory>

namespace mtl
{
	class metalfx_upscale_pass : public upscaler
	{
		std::unique_ptr<mtl::viewable_image> m_output_left;
		std::unique_ptr<mtl::viewable_image> m_output_right;

		MTL4FX::SpatialScaler* m_scaler = nullptr;   // Owned (+1)
		MTL::Fence* m_fence = nullptr;               // Owned (+1). Orders the scaler against the work before it

		struct scaler_config
		{
			MTL::PixelFormat input_format = MTL::PixelFormatInvalid;
			MTL::PixelFormat output_format = MTL::PixelFormatInvalid;
			u32 input_width = 0;
			u32 input_height = 0;
			u32 output_width = 0;
			u32 output_height = 0;

			bool operator==(const scaler_config&) const = default;
		}
		m_config{};

		bool m_unsupported = false;       // MetalFX not available on this system
		bool m_logged_fallback = false;

		void dispose_images();
		void dispose_scaler();
		bool initialize(mtl::viewable_image* src, u32 output_w, u32 output_h, rsx::flags32_t mode);

		static MTL::PixelFormat get_output_format(MTL::PixelFormat input_format);

		// Makes the scaler (encoded as its own passes) wait for all previously recorded work
		void signal_fence(mtl::command_list& cmd);

	public:
		metalfx_upscale_pass() = default;
		~metalfx_upscale_pass() override;

		metalfx_upscale_pass(const metalfx_upscale_pass&) = delete;
		metalfx_upscale_pass& operator=(const metalfx_upscale_pass&) = delete;

		mtl::viewable_image* scale_output(
			mtl::command_list& cmd,                 // CB
			mtl::viewable_image* src,               // Source input
			MTL::Texture* present_surface,          // Present target. May be nullptr for some passes
			const areai& src_area,                  // Scaling request: source region
			const areai& dst_area,                  // Scaling request: destination region
			rsx::flags32_t mode                     // Mode
		) override;
	};
}
