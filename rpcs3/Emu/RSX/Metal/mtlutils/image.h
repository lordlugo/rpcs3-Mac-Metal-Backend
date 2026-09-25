#pragma once

#include "mtl_api.h"
#include "device.h"
#include "buffer_object.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Utils/color_utils.hpp"

#include <memory>
#include <unordered_map>

using namespace ::rsx::format_class_;

namespace mtl
{
	enum image_aspect : u32
	{
		aspect_none    = 0,
		aspect_color   = 1,
		aspect_depth   = 2,
		aspect_stencil = 4,
		aspect_depth_stencil = aspect_depth | aspect_stencil,
	};

	enum : u32 // special remap encodings, mirrors the VK backend
	{
		MTL_REMAP_IDENTITY = 0xCAFEBABE,          // Identity view that ignores the native component map
		MTL_REMAP_VIEW_MULTISAMPLED = 0xDEADBEEF, // Multisampled view
	};

	inline const MTL::TextureSwizzleChannels swizzle_identity = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleBlue, MTL::TextureSwizzleAlpha);

	inline bool operator==(const MTL::TextureSwizzleChannels& a, const MTL::TextureSwizzleChannels& b)
	{
		return a.red == b.red && a.green == b.green && a.blue == b.blue && a.alpha == b.alpha;
	}

	u32 get_format_aspect(MTL::PixelFormat format);
	bool is_depth_format(MTL::PixelFormat format);
	bool is_stencil_format(MTL::PixelFormat format);

	struct image_create_info
	{
		MTL::TextureType type = MTL::TextureType2D;
		MTL::PixelFormat format = MTL::PixelFormatRGBA8Unorm;
		u32 width = 1;
		u32 height = 1;
		u32 depth = 1;
		u32 mipmaps = 1;
		u32 layers = 1;        // array length (cube maps: layers = 6 * cube count, type = Cube/CubeArray)
		u8 samples = 1;
		MTL::TextureUsage usage = MTL::TextureUsageShaderRead;
		memory_location storage = memory_location::device_local;
		rsx::format_class format_class = RSX_FORMAT_CLASS_UNDEFINED;
	};

	// Owned MTL::Texture. Always created with MTLTextureUsagePixelFormatView so that views may reinterpret format
	// (RSX aliases surfaces aggressively) and with untracked hazard mode (Metal 4 model).
	class image
	{
		std::string m_debug_name;
		rsx::format_class m_format_class = RSX_FORMAT_CLASS_UNDEFINED;

	protected:
		image() = default;
		void create_impl(const render_device& dev, const image_create_info& info);

	public:
		MTL::Texture* value = nullptr;
		image_create_info info{};

		// Channel order of the underlying data relative to RGBA (like vk::image::native_component_map)
		MTL::TextureSwizzleChannels native_component_map = swizzle_identity;

		image(const render_device& dev, const image_create_info& info);
		virtual ~image();

		image(const image&) = delete;
		image& operator=(const image&) = delete;

		MTL::Texture* handle() const { return value; }

		u32 width() const { return info.width; }
		u32 height() const { return info.height; }
		u32 depth() const { return info.depth; }
		u32 mipmaps() const { return info.mipmaps; }
		u32 layers() const { return info.layers; }
		u8 samples() const { return info.samples; }
		MTL::PixelFormat format() const { return info.format; }
		MTL::TextureType type() const { return info.type; }
		u32 aspect() const { return get_format_aspect(info.format); }
		rsx::format_class format_class() const { return m_format_class; }

		const std::string& debug_name() const { return m_debug_name; }
		void set_debug_name(const std::string& name);
	};

	struct image_view_info
	{
		MTL::PixelFormat format = MTL::PixelFormatInvalid;      // Invalid = same as image
		MTL::TextureType type = static_cast<MTL::TextureType>(~0ull); // ~0 = same as image
		MTL::TextureSwizzleChannels swizzle = swizzle_identity;
		u32 base_level = 0;
		u32 level_count = ~0u;   // ~0 = all
		u32 base_layer = 0;
		u32 layer_count = ~0u;   // ~0 = all
		u32 aspect = aspect_color | aspect_depth;
	};

	class image_view
	{
		std::unordered_map<MTL::PixelFormat, std::unique_ptr<image_view>> m_subviews;
		mtl::image* m_resource = nullptr;
		image_view* m_root_view = nullptr;

		void create_impl();

	public:
		MTL::Texture* value = nullptr;
		image_view_info info{};

		image_view(mtl::image* resource, const image_view_info& view_info = {});
		~image_view();

		image_view(const image_view&) = delete;
		image_view& operator=(const image_view&) = delete;

		// Returns a view of the same subresources reinterpreted as `format` (cached, owned by the root view)
		image_view* as(MTL::PixelFormat format);

		mtl::image* image() const { return m_resource; }
		MTL::PixelFormat format() const { return info.format; }
		MTL::Texture* handle() const { return value; }
		u32 encoded_component_map() const { return 0; }
	};

	class viewable_image : public image
	{
	protected:
		std::unordered_map<u64, std::unique_ptr<mtl::image_view>> views;

	public:
		using image::image;

		// View with RSX channel remapping applied on top of the native component layout
		virtual image_view* get_view(const rsx::texture_channel_remap_t& remap, u32 aspect_mask = aspect_color | aspect_depth);

		// Full-resource identity view (render target binding, copies)
		image_view* get_identity_view(u32 aspect_mask = aspect_color | aspect_depth);

		void set_native_component_layout(const MTL::TextureSwizzleChannels& new_layout);
		void release_views();
	};

	// Applies an RSX remap vector on top of a base channel layout (a, r, g, b order like the VK helper)
	MTL::TextureSwizzleChannels apply_swizzle_remap(const std::array<MTL::TextureSwizzle, 4>& base_remap_argb, const rsx::texture_channel_remap_t& remap);
}
