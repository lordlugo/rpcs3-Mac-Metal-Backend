#pragma once

#include "upscaling.h"

namespace mtl
{
	struct nearest_upscale_pass : public upscaler
	{
		mtl::viewable_image* scale_output(
			mtl::command_list& cmd,                 // CB
			mtl::viewable_image* src,               // Source input
			MTL::Texture* present_surface,          // Present target. May be nullptr for some passes
			const areai& src_area,                  // Scaling request: source region
			const areai& dst_area,                  // Scaling request: destination region
			rsx::flags32_t mode                     // Mode
		) override
		{
			if (mode & UPSCALE_AND_COMMIT)
			{
				ensure(present_surface);

				upscale_blit(cmd, src, present_surface, src_area, dst_area, false);
				return nullptr;
			}

			// Upscaling source only is unsupported
			return src;
		}
	};
}
