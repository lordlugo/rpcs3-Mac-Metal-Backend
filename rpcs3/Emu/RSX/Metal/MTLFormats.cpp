#include "stdafx.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

namespace mtl
{
	// NOTE on the 16-bit packed formats: Metal names packed components from the least significant bit upwards, Vulkan
	// from the most significant bit downwards. The bit layouts therefore map 1:1 (this is also what MoltenVK does):
	//   VK_FORMAT_R5G6B5_UNORM_PACK16   == MTLPixelFormatB5G6R5Unorm   (R 15:11, G 10:5, B 4:0)
	//   VK_FORMAT_R5G5B5A1_UNORM_PACK16 == MTLPixelFormatA1BGR5Unorm   (R 15:11, G 10:6, B 5:1, A 0)
	//   VK_FORMAT_A1R5G5B5_UNORM_PACK16 == MTLPixelFormatBGR5A1Unorm   (A 15, R 14:10, G 9:5, B 4:0)
	//   VK_FORMAT_R4G4B4A4_UNORM_PACK16 == MTLPixelFormatABGR4Unorm    (R 15:12, G 11:8, B 7:4, A 3:0)
	// All four are supported (sample, filter, render, blend) on every Apple GPU family, so the component maps below
	// are identical to the VK backend. The VK "__APPLE__ && ARCH_X64" BGRA8 fallbacks (Intel/AMD Macs) are dropped.

	MTL::PixelFormat get_compatible_depth_surface_format(rsx::surface_depth_format2 format)
	{
		switch (format)
		{
		case rsx::surface_depth_format2::z16_uint:
			return MTL::PixelFormatDepth16Unorm;
		case rsx::surface_depth_format2::z16_float:
			// Same as VK (D32_SFLOAT). The RSX-side texel width of this format is 2 (see get_format_texel_width).
			return MTL::PixelFormatDepth32Float;
		case rsx::surface_depth_format2::z24s8_uint:
			// No D24S8 on Apple GPUs. Memory transfers pack/unpack through the d24x8 <-> d32x8 compute kernels.
			return MTL::PixelFormatDepth32Float_Stencil8;
		case rsx::surface_depth_format2::z24s8_float:
			return MTL::PixelFormatDepth32Float_Stencil8;
		default:
			break;
		}
		fmt::throw_exception("Invalid format (0x%x)", static_cast<u32>(format));
	}

	minification_filter get_min_filter(rsx::texture_minify_filter min_filter)
	{
		switch (min_filter)
		{
		case rsx::texture_minify_filter::nearest: return { MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, false };
		case rsx::texture_minify_filter::linear: return { MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterNearest, false };
		case rsx::texture_minify_filter::nearest_nearest: return { MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterNearest, true };
		case rsx::texture_minify_filter::linear_nearest: return { MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterNearest, true };
		case rsx::texture_minify_filter::nearest_linear: return { MTL::SamplerMinMagFilterNearest, MTL::SamplerMipFilterLinear, true };
		case rsx::texture_minify_filter::linear_linear: return { MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterLinear, true };
		case rsx::texture_minify_filter::convolution_min: return { MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterNearest, false };
		default:
			fmt::throw_exception("Invalid min filter");
		}
	}

	MTL::SamplerMinMagFilter get_mag_filter(rsx::texture_magnify_filter mag_filter)
	{
		switch (mag_filter)
		{
		case rsx::texture_magnify_filter::nearest: return MTL::SamplerMinMagFilterNearest;
		case rsx::texture_magnify_filter::linear: return MTL::SamplerMinMagFilterLinear;
		case rsx::texture_magnify_filter::convolution_mag: return MTL::SamplerMinMagFilterLinear;
		default:
			break;
		}

		fmt::throw_exception("Invalid mag filter (0x%x)", static_cast<u32>(mag_filter));
	}

	MTL::SamplerBorderColor get_border_color(u32 color)
	{
		switch (color)
		{
		case 0x00000000:
			return MTL::SamplerBorderColorTransparentBlack;
		case 0xFFFFFFFF:
			return MTL::SamplerBorderColorOpaqueWhite;
		case 0xFF000000:
			return MTL::SamplerBorderColorOpaqueBlack;
		default:
		{
			// No custom border colors in Metal. Pick the closest fixed color (ARGB8 encoding).
			const color4f decoded(
				static_cast<f32>((color >> 16) & 0xFF) / 255.f,
				static_cast<f32>((color >> 8) & 0xFF) / 255.f,
				static_cast<f32>(color & 0xFF) / 255.f,
				static_cast<f32>((color >> 24) & 0xFF) / 255.f);
			return border_color_t(decoded).value;
		}
		}
	}

	MTL::SamplerAddressMode mtl_wrap_mode(rsx::texture_wrap_mode gcm_wrap)
	{
		return get_sampler_address_mode(gcm_wrap);
	}

	float max_aniso(rsx::texture_max_anisotropy gcm_aniso)
	{
		switch (gcm_aniso)
		{
		case rsx::texture_max_anisotropy::x1: return 1.0f;
		case rsx::texture_max_anisotropy::x2: return 2.0f;
		case rsx::texture_max_anisotropy::x4: return 4.0f;
		case rsx::texture_max_anisotropy::x6: return 6.0f;
		case rsx::texture_max_anisotropy::x8: return 8.0f;
		case rsx::texture_max_anisotropy::x10: return 10.0f;
		case rsx::texture_max_anisotropy::x12: return 12.0f;
		case rsx::texture_max_anisotropy::x16: return 16.0f;
		default:
			break;
		}

		fmt::throw_exception("Texture anisotropy error: bad max aniso (%d)", static_cast<u32>(gcm_aniso));
	}

	std::array<MTL::TextureSwizzle, 4> get_component_mapping(u32 format)
	{
		// Component map in ARGB format
		std::array<MTL::TextureSwizzle, 4> mapping = {};

		constexpr auto R = MTL::TextureSwizzleRed;
		constexpr auto G = MTL::TextureSwizzleGreen;
		constexpr auto B = MTL::TextureSwizzleBlue;
		constexpr auto A = MTL::TextureSwizzleAlpha;
		constexpr auto ONE = MTL::TextureSwizzleOne;

		switch (format)
		{
		case CELL_GCM_TEXTURE_A1R5G5B5:
		case CELL_GCM_TEXTURE_R5G5B5A1:
		case CELL_GCM_TEXTURE_R6G5B5:
		case CELL_GCM_TEXTURE_R5G6B5:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
			mapping = { A, R, G, B }; break;

		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			mapping = { R, R, R, R }; break;

		case CELL_GCM_TEXTURE_A4R4G4B4:
			mapping = { R, G, B, A }; break;

		case CELL_GCM_TEXTURE_G8B8:
			mapping = { G, R, G, R }; break;

		case CELL_GCM_TEXTURE_B8:
			mapping = { ONE, R, R, R }; break;

		case CELL_GCM_TEXTURE_X16:
			mapping = { R, ONE, R, ONE }; break;

		case CELL_GCM_TEXTURE_X32_FLOAT:
			mapping = { R, R, R, R }; break;

		case CELL_GCM_TEXTURE_Y16_X16:
			mapping = { G, R, G, R }; break;

		case CELL_GCM_TEXTURE_Y16_X16_FLOAT:
			mapping = { R, G, R, G }; break;

		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			mapping = { A, R, G, B }; break;

		case CELL_GCM_TEXTURE_D8R8G8B8:
			mapping = { ONE, R, G, B }; break;

		case CELL_GCM_TEXTURE_D1R5G5B5:
			mapping = { ONE, R, G, B }; break;

		case CELL_GCM_TEXTURE_COMPRESSED_HILO8:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8:
			mapping = { R, G, R, G }; break;

		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8:
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8:
			mapping = { A, R, G, B }; break;

		case CELL_GCM_TEXTURE_A8R8G8B8:
			mapping = { A, R, G, B }; break;

		default:
			fmt::throw_exception("Invalid or unsupported component mapping for texture format (0x%x)", format);
		}

		return mapping;
	}

	MTL::PixelFormat get_compatible_sampler_format(u32 format)
	{
		const auto renderer = get_current_renderer();
		const bool supports_dxt = renderer && renderer->caps().bc_texture_compression;

		switch (format)
		{
		case CELL_GCM_TEXTURE_R5G6B5: return MTL::PixelFormatB5G6R5Unorm;
		case CELL_GCM_TEXTURE_R6G5B5: return MTL::PixelFormatB5G6R5Unorm; // Expand, discard high bit?
		case CELL_GCM_TEXTURE_R5G5B5A1: return MTL::PixelFormatA1BGR5Unorm;
		case CELL_GCM_TEXTURE_D1R5G5B5: return MTL::PixelFormatBGR5A1Unorm;
		case CELL_GCM_TEXTURE_A1R5G5B5: return MTL::PixelFormatBGR5A1Unorm;
		case CELL_GCM_TEXTURE_A4R4G4B4: return MTL::PixelFormatABGR4Unorm;
		case CELL_GCM_TEXTURE_B8: return MTL::PixelFormatR8Unorm;
		case CELL_GCM_TEXTURE_A8R8G8B8: return MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1: return supports_dxt ? MTL::PixelFormatBC1_RGBA : MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23: return supports_dxt ? MTL::PixelFormatBC2_RGBA : MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45: return supports_dxt ? MTL::PixelFormatBC3_RGBA : MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_G8B8: return MTL::PixelFormatRG8Unorm;
		case CELL_GCM_TEXTURE_DEPTH24_D8: return MTL::PixelFormatDepth32Float_Stencil8;
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT: return MTL::PixelFormatDepth32Float_Stencil8;
		case CELL_GCM_TEXTURE_DEPTH16: return MTL::PixelFormatDepth16Unorm;
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT: return MTL::PixelFormatDepth32Float;
		case CELL_GCM_TEXTURE_X16: return MTL::PixelFormatR16Unorm;
		case CELL_GCM_TEXTURE_Y16_X16: return MTL::PixelFormatRG16Unorm;
		case CELL_GCM_TEXTURE_Y16_X16_FLOAT: return MTL::PixelFormatRG16Float;
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT: return MTL::PixelFormatRGBA16Float;
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT: return MTL::PixelFormatRGBA32Float;
		case CELL_GCM_TEXTURE_X32_FLOAT: return MTL::PixelFormatR32Float;
		case CELL_GCM_TEXTURE_D8R8G8B8: return MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8: return MTL::PixelFormatRG8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8: return MTL::PixelFormatRG8Snorm;
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8: return MTL::PixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8: return MTL::PixelFormatBGRA8Unorm;
		default:
			break;
		}
		fmt::throw_exception("Invalid or unsupported sampler format for texture format (0x%x)", format);
	}

	MTL::PixelFormat get_compatible_srgb_format(MTL::PixelFormat rgb_format)
	{
		switch (rgb_format)
		{
		// 8-bit
		case MTL::PixelFormatR8Unorm:
			return MTL::PixelFormatR8Unorm_sRGB;
		case MTL::PixelFormatRG8Unorm:
			return MTL::PixelFormatRG8Unorm_sRGB;
		case MTL::PixelFormatRGBA8Unorm:
			return MTL::PixelFormatRGBA8Unorm_sRGB;
		case MTL::PixelFormatBGRA8Unorm:
			return MTL::PixelFormatBGRA8Unorm_sRGB;
		// DXT
		case MTL::PixelFormatBC1_RGBA:
			return MTL::PixelFormatBC1_RGBA_sRGB;
		case MTL::PixelFormatBC2_RGBA:
			return MTL::PixelFormatBC2_RGBA_sRGB;
		case MTL::PixelFormatBC3_RGBA:
			return MTL::PixelFormatBC3_RGBA_sRGB;
		default:
			return MTL::PixelFormatInvalid;
		}
	}

	MTL::PixelFormat get_compatible_snorm_format(MTL::PixelFormat rgb_format)
	{
		switch (rgb_format)
		{
		// 8-bit
		case MTL::PixelFormatR8Unorm:
			return MTL::PixelFormatR8Snorm;
		case MTL::PixelFormatRG8Unorm:
			return MTL::PixelFormatRG8Snorm;
		case MTL::PixelFormatRGBA8Unorm:
			return MTL::PixelFormatRGBA8Snorm;
		// NOTE: Metal has no BGRA8Snorm. The shader-side renormalization path handles BGRA8 sources instead.
		// 16-bit
		case MTL::PixelFormatR16Unorm:
			return MTL::PixelFormatR16Snorm;
		case MTL::PixelFormatRG16Unorm:
			return MTL::PixelFormatRG16Snorm;
		default:
			return MTL::PixelFormatInvalid;
		}
	}

	u8 get_format_texel_width(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Unorm:
		case MTL::PixelFormatR8Unorm_sRGB:
		case MTL::PixelFormatR8Snorm:
			return 1;
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR16Float:
		case MTL::PixelFormatR16Unorm:
		case MTL::PixelFormatR16Snorm:
		case MTL::PixelFormatRG8Unorm:
		case MTL::PixelFormatRG8Unorm_sRGB:
		case MTL::PixelFormatRG8Snorm:
		case MTL::PixelFormatBGR5A1Unorm:
		case MTL::PixelFormatABGR4Unorm:
		case MTL::PixelFormatB5G6R5Unorm:
		case MTL::PixelFormatA1BGR5Unorm:
			return 2;
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatR32Float:
		case MTL::PixelFormatRG16Unorm:
		case MTL::PixelFormatRG16Snorm:
		case MTL::PixelFormatRG16Float:
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRGBA8Unorm_sRGB:
		case MTL::PixelFormatRGBA8Snorm:
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatBGRA8Unorm_sRGB:
		case MTL::PixelFormatBC1_RGBA:
		case MTL::PixelFormatBC2_RGBA:
		case MTL::PixelFormatBC3_RGBA:
		case MTL::PixelFormatBC1_RGBA_sRGB:
		case MTL::PixelFormatBC2_RGBA_sRGB:
		case MTL::PixelFormatBC3_RGBA_sRGB:
			return 4;
		case MTL::PixelFormatRGBA16Float:
			return 8;
		case MTL::PixelFormatRGBA32Float:
			return 16;
		case MTL::PixelFormatDepth16Unorm:
		case MTL::PixelFormatDepth32Float:
			return 2;
		case MTL::PixelFormatDepth32Float_Stencil8: // RSX D24S8 / D24FS8 surface
		case MTL::PixelFormatDepth24Unorm_Stencil8:
			return 4;
		default:
			break;
		}

		fmt::throw_exception("Unexpected MTLPixelFormat 0x%X", static_cast<u32>(format));
	}

	std::pair<u8, u8> get_format_element_size(MTL::PixelFormat format)
	{
		// Return value is {ELEMENT_SIZE, NUM_ELEMENTS_PER_TEXEL}
		// NOTE: Due to endianness issues, coalesced larger types are preferred
		// e.g UINT1 to hold 4x1 bytes instead of UBYTE4 to hold 4x1

		switch (format)
		{
		//8-bit
		case MTL::PixelFormatR8Unorm:
		case MTL::PixelFormatR8Unorm_sRGB:
		case MTL::PixelFormatR8Snorm:
			return{ 1, 1 };
		case MTL::PixelFormatRG8Unorm:
		case MTL::PixelFormatRG8Unorm_sRGB:
		case MTL::PixelFormatRG8Snorm:
			return{ 2, 1 }; //UNSIGNED_SHORT_8_8
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRGBA8Unorm_sRGB:
		case MTL::PixelFormatRGBA8Snorm:
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatBGRA8Unorm_sRGB:
			return{ 4, 1 }; //UNSIGNED_INT_8_8_8_8
		//16-bit
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR16Float:
		case MTL::PixelFormatR16Unorm:
		case MTL::PixelFormatR16Snorm:
			return{ 2, 1 }; //UNSIGNED_SHORT and HALF_FLOAT
		case MTL::PixelFormatRG16Unorm:
		case MTL::PixelFormatRG16Snorm:
		case MTL::PixelFormatRG16Float:
			return{ 2, 2 }; //HALF_FLOAT
		case MTL::PixelFormatRGBA16Float:
			return{ 2, 4 }; //HALF_FLOAT
		case MTL::PixelFormatBGR5A1Unorm:
		case MTL::PixelFormatABGR4Unorm:
		case MTL::PixelFormatB5G6R5Unorm:
		case MTL::PixelFormatA1BGR5Unorm:
			return{ 2, 1 }; //UNSIGNED_SHORT_X_Y_Z_W
		//32-bit
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatR32Float:
			return{ 4, 1 }; //FLOAT
		case MTL::PixelFormatRGBA32Float:
			return{ 4, 4 }; //FLOAT
		//DXT
		case MTL::PixelFormatBC1_RGBA:
		case MTL::PixelFormatBC2_RGBA:
		case MTL::PixelFormatBC3_RGBA:
		case MTL::PixelFormatBC1_RGBA_sRGB:
		case MTL::PixelFormatBC2_RGBA_sRGB:
		case MTL::PixelFormatBC3_RGBA_sRGB:
			return{ 4, 1 };
		//Depth
		case MTL::PixelFormatDepth16Unorm:
		case MTL::PixelFormatDepth32Float:
			return{ 2, 1 };
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
			return{ 4, 1 };
		default:
			break;
		}

		fmt::throw_exception("Unexpected MTLPixelFormat 0x%X", static_cast<u32>(format));
	}

	std::pair<bool, u32> get_format_convert_flags(MTL::PixelFormat format)
	{
		switch (format)
		{
			//8-bit
		case MTL::PixelFormatR8Unorm:
		case MTL::PixelFormatR8Unorm_sRGB:
		case MTL::PixelFormatR8Snorm:
			return{ false, 1 };
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatBGRA8Unorm_sRGB:
		case MTL::PixelFormatRGBA8Unorm_sRGB:
		case MTL::PixelFormatRGBA8Snorm:
			return{ true, 4 };
			//16-bit
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR16Float:
		case MTL::PixelFormatR16Unorm:
		case MTL::PixelFormatR16Snorm:
		case MTL::PixelFormatRG8Unorm:
		case MTL::PixelFormatRG8Unorm_sRGB:
		case MTL::PixelFormatRG8Snorm:
		case MTL::PixelFormatRG16Unorm:
		case MTL::PixelFormatRG16Snorm:
		case MTL::PixelFormatRG16Float:
		case MTL::PixelFormatRGBA16Float:
		case MTL::PixelFormatBGR5A1Unorm:
		case MTL::PixelFormatABGR4Unorm:
		case MTL::PixelFormatB5G6R5Unorm:
		case MTL::PixelFormatA1BGR5Unorm:
			return{ true, 2 };
			//32-bit
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatR32Float:
		case MTL::PixelFormatRGBA32Float:
			return{ true, 4 };
			//DXT
		case MTL::PixelFormatBC1_RGBA:
		case MTL::PixelFormatBC2_RGBA:
		case MTL::PixelFormatBC3_RGBA:
		case MTL::PixelFormatBC1_RGBA_sRGB:
		case MTL::PixelFormatBC2_RGBA_sRGB:
		case MTL::PixelFormatBC3_RGBA_sRGB:
			return{ false, 1 };
			//Depth
		case MTL::PixelFormatDepth16Unorm:
		case MTL::PixelFormatDepth32Float:
			return{ true, 2 };
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
			return{ true, 4 };
		default:
			break;
		}

		fmt::throw_exception("Unknown MTLPixelFormat 0x%x", static_cast<u32>(format));
	}

	bool formats_are_bitcast_compatible(MTL::PixelFormat format1, MTL::PixelFormat format2)
	{
		if (format1 == format2) [[likely]]
		{
			return true;
		}

		// Formats are compatible if the following conditions are met:
		// 1. Texel sizes must match
		// 2. Both formats require no transforms (basic memcpy) or...
		// 3. Both formats have the same transform (e.g RG16_UNORM to RG16_SFLOAT, both are down and uploaded with a 2-byte byteswap)

		if (get_format_texel_width(format1) != get_format_texel_width(format2))
		{
			return false;
		}

		const auto transform_a = get_format_convert_flags(format1);
		const auto transform_b = get_format_convert_flags(format2);

		if (transform_a.first == transform_b.first)
		{
			return !transform_a.first || (transform_a.second == transform_b.second);
		}

		return false;
	}

	bool formats_are_bitcast_compatible(image* image1, image* image2)
	{
		if (const u32 transfer_class = image1->format_class() | image2->format_class();
			transfer_class & RSX_FORMAT_CLASS_DEPTH_FLOAT_MASK)
		{
			// If any one of the two images is a depth float, the other must match exactly or bust
			return (image1->format_class() == image2->format_class());
		}

		return formats_are_bitcast_compatible(image1->format(), image2->format());
	}

	// ---- Metal-specific helpers ------------------------------------------------------------------------------------

	bool is_compressed_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatBC1_RGBA:
		case MTL::PixelFormatBC2_RGBA:
		case MTL::PixelFormatBC3_RGBA:
		case MTL::PixelFormatBC1_RGBA_sRGB:
		case MTL::PixelFormatBC2_RGBA_sRGB:
		case MTL::PixelFormatBC3_RGBA_sRGB:
			return true;
		default:
			return false;
		}
	}

	format_block_info get_format_block_info(MTL::PixelFormat format, u32 aspect)
	{
		switch (format)
		{
		case MTL::PixelFormatBC1_RGBA:
		case MTL::PixelFormatBC1_RGBA_sRGB:
			return { 4, 4, 8 };
		case MTL::PixelFormatBC2_RGBA:
		case MTL::PixelFormatBC2_RGBA_sRGB:
		case MTL::PixelFormatBC3_RGBA:
		case MTL::PixelFormatBC3_RGBA_sRGB:
			return { 4, 4, 16 };
		case MTL::PixelFormatDepth16Unorm:
			return { 1, 1, 2 };
		case MTL::PixelFormatDepth32Float:
			return { 1, 1, 4 };
		case MTL::PixelFormatStencil8:
		case MTL::PixelFormatX32_Stencil8:
		case MTL::PixelFormatX24_Stencil8:
			return { 1, 1, 1 };
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
			// Blits of combined depth-stencil formats address one plane at a time (MTLBlitOption*FromDepthStencil)
			ensure(aspect == aspect_depth || aspect == aspect_stencil, "Depth-stencil blits must select exactly one plane");
			return { 1, 1, static_cast<u8>(aspect == aspect_stencil ? 1 : 4) };
		default:
			return { 1, 1, get_format_texel_width(format) };
		}
	}

	bool format_is_renderable(MTL::PixelFormat format)
	{
		// Every format we create is color/depth renderable on Apple GPUs except the block-compressed ones
		// (the 16-bit packed formats are renderable on the Apple GPU families).
		return !is_compressed_format(format);
	}

	bool format_supports_shader_write(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Unorm:
		case MTL::PixelFormatR8Snorm:
		case MTL::PixelFormatRG8Unorm:
		case MTL::PixelFormatRG8Snorm:
		case MTL::PixelFormatR16Unorm:
		case MTL::PixelFormatR16Snorm:
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR16Float:
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatR32Float:
		case MTL::PixelFormatRG16Unorm:
		case MTL::PixelFormatRG16Snorm:
		case MTL::PixelFormatRG16Float:
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRGBA8Snorm:
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatRGBA16Float:
		case MTL::PixelFormatRGBA32Float:
			return true;
		default:
			// sRGB, 16-bit packed, compressed and depth/stencil formats are not shader-writable
			return false;
		}
	}

	// ---- Declared in MTLHelpers.h (texture helper section) ---------------------------------------------------------

	std::pair<MTL::PixelFormat, MTL::TextureSwizzleChannels> get_compatible_surface_format(rsx::surface_color_format color_format)
	{
		const auto o_rgb = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleBlue, MTL::TextureSwizzleOne);
		const auto z_rgb = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleBlue, MTL::TextureSwizzleZero);

		switch (color_format)
		{
		case rsx::surface_color_format::r5g6b5:
			return std::make_pair(MTL::PixelFormatB5G6R5Unorm, swizzle_identity);

		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			return std::make_pair(MTL::PixelFormatBGR5A1Unorm, o_rgb);

		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			return std::make_pair(MTL::PixelFormatBGR5A1Unorm, z_rgb);

		case rsx::surface_color_format::a8r8g8b8:
			return std::make_pair(MTL::PixelFormatBGRA8Unorm, swizzle_identity);

		case rsx::surface_color_format::a8b8g8r8:
			return std::make_pair(MTL::PixelFormatRGBA8Unorm, swizzle_identity);

		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			return std::make_pair(MTL::PixelFormatRGBA8Unorm, o_rgb);

		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			return std::make_pair(MTL::PixelFormatRGBA8Unorm, z_rgb);

		case rsx::surface_color_format::x8r8g8b8_z8r8g8b8:
			return std::make_pair(MTL::PixelFormatBGRA8Unorm, z_rgb);

		case rsx::surface_color_format::x8r8g8b8_o8r8g8b8:
			return std::make_pair(MTL::PixelFormatBGRA8Unorm, o_rgb);

		case rsx::surface_color_format::w16z16y16x16:
			return std::make_pair(MTL::PixelFormatRGBA16Float, swizzle_identity);

		case rsx::surface_color_format::w32z32y32x32:
			return std::make_pair(MTL::PixelFormatRGBA32Float, swizzle_identity);

		case rsx::surface_color_format::b8:
		{
			const auto no_alpha = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleOne);
			return std::make_pair(MTL::PixelFormatR8Unorm, no_alpha);
		}

		case rsx::surface_color_format::g8b8:
		{
			const auto gb_rg = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen);
			return std::make_pair(MTL::PixelFormatRG8Unorm, gb_rg);
		}

		case rsx::surface_color_format::x32:
		{
			const auto rrrr = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed);
			return std::make_pair(MTL::PixelFormatR32Float, rrrr);
		}

		default:
			rsx_log.error("Surface color buffer: Unsupported surface color format (0x%x)", static_cast<u32>(color_format));
			return std::make_pair(MTL::PixelFormatBGRA8Unorm, swizzle_identity);
		}
	}
}
