#pragma once

// Port of VK/VKFormats.h: RSX/GCM formats -> Metal pixel formats, component maps and sampler helpers.
// Apple silicon only: no D24S8 (always Depth32Float_Stencil8), native 16-bit packed formats, BC1-3 where supported.

#include "mtlutils/mtl_api.h"
#include "Emu/RSX/gcm_enums.h"

#include <array>
#include <utility>

namespace mtl
{
	class image;

	struct minification_filter
	{
		MTL::SamplerMinMagFilter filter;
		MTL::SamplerMipFilter mipmap_mode;
		bool sample_mipmaps;
	};

	// Metal only has 3 fixed border colors. Exact matches for 0x00000000, 0xFFFFFFFF and 0xFF000000 (ARGB8), otherwise
	// the closest of the three (see mtl::border_color_t).
	MTL::SamplerBorderColor get_border_color(u32 color);

	MTL::PixelFormat get_compatible_depth_surface_format(rsx::surface_depth_format2 format);
	MTL::PixelFormat get_compatible_sampler_format(u32 format);
	MTL::PixelFormat get_compatible_srgb_format(MTL::PixelFormat rgb_format);   // PixelFormatInvalid if none
	MTL::PixelFormat get_compatible_snorm_format(MTL::PixelFormat rgb_format);  // PixelFormatInvalid if none
	u8 get_format_texel_width(MTL::PixelFormat format);                         // RSX-side texel width (D32F = D16F surface = 2)
	std::pair<u8, u8> get_format_element_size(MTL::PixelFormat format);         // {ELEMENT_SIZE, NUM_ELEMENTS_PER_TEXEL}
	std::pair<bool, u32> get_format_convert_flags(MTL::PixelFormat format);     // {requires byteswap, swap word size}
	bool formats_are_bitcast_compatible(MTL::PixelFormat format1, MTL::PixelFormat format2);
	bool formats_are_bitcast_compatible(image* image1, image* image2);

	minification_filter get_min_filter(rsx::texture_minify_filter min_filter);
	MTL::SamplerMinMagFilter get_mag_filter(rsx::texture_magnify_filter mag_filter);
	MTL::SamplerAddressMode mtl_wrap_mode(rsx::texture_wrap_mode gcm_wrap);    // == mtl::get_sampler_address_mode (sampler.h)
	float max_aniso(rsx::texture_max_anisotropy gcm_aniso);
	std::array<MTL::TextureSwizzle, 4> get_component_mapping(u32 format);       // ARGB order, like the VK helper

	// ---- Metal-specific helpers ------------------------------------------------------------------------------------

	// Physical memory layout of one copy "block" as seen by MTL blit copies (buffer <-> texture).
	// `aspect` selects the plane of a combined depth-stencil format (depth = 4 bytes float, stencil = 1 byte).
	struct format_block_info
	{
		u8 block_width = 1;
		u8 block_height = 1;
		u8 bytes_per_block = 4;
	};

	format_block_info get_format_block_info(MTL::PixelFormat format, u32 aspect = 0);
	bool is_compressed_format(MTL::PixelFormat format);
	bool format_is_renderable(MTL::PixelFormat format);       // Can be a render pass attachment on Apple GPUs
	bool format_supports_shader_write(MTL::PixelFormat format); // Supports MTLTextureUsageShaderWrite on Apple GPUs
}
