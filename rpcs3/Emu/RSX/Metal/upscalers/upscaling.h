#pragma once

// Output upscalers (port of VK/upscalers/upscaling.h).
//
// VK -> Metal substitutions:
//  - VkImage present_surface + VkImageLayout  ->  MTL::Texture* present_surface (e.g. the CAMetalDrawable texture;
//    needs MTL::TextureUsageRenderTarget, which CAMetalLayer drawables always have)
//  - const VkImageBlit& request                ->  src_area / dst_area (VkImageBlit srcOffsets / dstOffsets;
//    x1 > x2 or y1 > y2 mirrors the image, exactly like vkCmdBlitImage)

#include "../mtlutils/commands.h"
#include "../mtlutils/image.h"
#include "Emu/system_config_types.h"
#include "Utilities/geometry.h"

#include <memory>

namespace mtl
{
	namespace upscaling_flags_
	{
		enum upscaling_flags
		{
			UPSCALE_DEFAULT_VIEW = (1 << 0),
			UPSCALE_LEFT_VIEW    = (1 << 0),
			UPSCALE_RIGHT_VIEW   = (1 << 1),
			UPSCALE_AND_COMMIT   = (1 << 2)
		};
	}

	using namespace upscaling_flags_;

	struct upscaler
	{
		virtual ~upscaler() {}

		// Metal addition. Called once per presented frame before the output (drawable) is acquired. One-time or
		// otherwise slow setup belongs here (or on a worker thread), never in scale_output().
		virtual void prepare() {}

		// Without UPSCALE_AND_COMMIT: returns an image holding the upscaled src (or src itself if the upscaler cannot
		// do better than the sampler in the following pass). With UPSCALE_AND_COMMIT: draws src_area of the result
		// into dst_area of present_surface and returns nullptr.
		virtual mtl::viewable_image* scale_output(
			mtl::command_list& cmd,                 // CB
			mtl::viewable_image* src,               // Source input
			MTL::Texture* present_surface,          // Present target. May be nullptr for some passes
			const areai& src_area,                  // Scaling request: source region (VkImageBlit::srcOffsets)
			const areai& dst_area,                  // Scaling request: destination region (VkImageBlit::dstOffsets)
			rsx::flags32_t mode                     // Mode
		) = 0;
	};

	// Scaled draw of src_area of `src` into dst_area of `dst` (render target texture). Replaces vkCmdBlitImage for
	// the present path. Implemented in MTLBlit.cpp with mtl::blit_pass.
	void upscale_blit(mtl::command_list& cmd, mtl::viewable_image* src, MTL::Texture* dst,
		const areai& src_area, const areai& dst_area, bool linear_filter);

	// Creates the upscaler for the output scaling setting. output_scaling_mode::fsr maps to MetalFX spatial upscaling
	// followed by optional RCAS sharpening (falls back to bilinear at runtime whenever MetalFX cannot be used).
	// Implemented in upscalers/metalfx_pass.cpp.
	std::unique_ptr<upscaler> create_upscaler(output_scaling_mode mode);
}
