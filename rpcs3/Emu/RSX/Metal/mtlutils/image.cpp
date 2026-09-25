#include "stdafx.h"
#include "image.h"
#include "buffer_object.h"
#include "garbage_collector.h"

namespace mtl
{
	bool is_depth_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatDepth16Unorm:
		case MTL::PixelFormatDepth32Float:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
			return true;
		default:
			return false;
		}
	}

	bool is_stencil_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatStencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatX32_Stencil8:
		case MTL::PixelFormatX24_Stencil8:
			return true;
		default:
			return false;
		}
	}

	u32 get_format_aspect(MTL::PixelFormat format)
	{
		u32 result = 0;
		if (is_depth_format(format)) result |= aspect_depth;
		if (is_stencil_format(format)) result |= aspect_stencil;
		return result ? result : static_cast<u32>(aspect_color);
	}

	// Linear <-> sRGB twins. Metal does not require MTLTextureUsagePixelFormatView for views that only toggle sRGB.
	static MTL::PixelFormat get_linear_format(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Unorm_sRGB: return MTL::PixelFormatR8Unorm;
		case MTL::PixelFormatRG8Unorm_sRGB: return MTL::PixelFormatRG8Unorm;
		case MTL::PixelFormatRGBA8Unorm_sRGB: return MTL::PixelFormatRGBA8Unorm;
		case MTL::PixelFormatBGRA8Unorm_sRGB: return MTL::PixelFormatBGRA8Unorm;
		case MTL::PixelFormatBGR10_XR_sRGB: return MTL::PixelFormatBGR10_XR;
		case MTL::PixelFormatBGRA10_XR_sRGB: return MTL::PixelFormatBGRA10_XR;
		case MTL::PixelFormatBC1_RGBA_sRGB: return MTL::PixelFormatBC1_RGBA;
		case MTL::PixelFormatBC2_RGBA_sRGB: return MTL::PixelFormatBC2_RGBA;
		case MTL::PixelFormatBC3_RGBA_sRGB: return MTL::PixelFormatBC3_RGBA;
		case MTL::PixelFormatBC7_RGBAUnorm_sRGB: return MTL::PixelFormatBC7_RGBAUnorm;
		default: return format;
		}
	}

	// True if a view of `view_format` on a texture of `image_format` changes the component layout, i.e. needs the
	// texture to be created with MTLTextureUsagePixelFormatView. Not needed (MTLTextureUsage.pixelFormatView docs,
	// same rules as MoltenVK): same format (swizzle, texture type, level/slice range only) and linear <-> sRGB views.
	// Stencil-plane views of combined depth-stencil formats are treated as reinterpreting (MoltenVK does the same).
	static bool is_reinterpreting_view(MTL::PixelFormat image_format, MTL::PixelFormat view_format)
	{
		return get_linear_format(image_format) != get_linear_format(view_format);
	}

	static bool is_combined_depth_stencil_format(MTL::PixelFormat format)
	{
		return (get_format_aspect(format) & aspect_depth_stencil) == aspect_depth_stencil;
	}

	void image::create_impl(const render_device& dev, const image_create_info& create_info)
	{
		info = create_info;
		m_format_class = create_info.format_class;

		autorelease_scope pool;
		auto desc = ref(MTL::TextureDescriptor::alloc()->init());
		desc->setTextureType(info.type);
		desc->setPixelFormat(info.format);
		desc->setWidth(info.width);
		desc->setHeight(info.height);
		desc->setDepth(info.type == MTL::TextureType3D ? info.depth : 1);
		desc->setMipmapLevelCount(std::max(1u, info.mipmaps));
		desc->setSampleCount(std::max<u8>(1, info.samples));

		u32 array_length = std::max(1u, info.layers);
		if (info.type == MTL::TextureTypeCube || info.type == MTL::TextureTypeCubeArray)
		{
			ensure(array_length % 6 == 0);
			array_length /= 6;
		}
		desc->setArrayLength(array_length);

		// Lossless compression: Apple GPUs compress Private textures transparently (allowGPUOptimizedContents, left at
		// its default of true) unless the usage forbids it. MTLTextureUsagePixelFormatView is one of the things that
		// disables it, so it is only set where a view with another component layout can be created:
		//  - requested by the creator through info.usage (texture cache images that may be sampled through an snorm
		//    view, see MTLTextureCache.cpp),
		//  - always for combined depth-stencil formats, whose stencil is sampled through an X32_Stencil8 view.
		// RSX surface aliasing / typeless transfers never use views (they copy through buffers, see MTLTexture.cpp), and
		// swizzle, sRGB, texture type and subresource range views do not need the flag.
		if (is_combined_depth_stencil_format(info.format))
		{
			info.usage |= MTL::TextureUsagePixelFormatView;
		}

		desc->setUsage(info.usage);
		desc->setStorageMode(info.storage == memory_location::host_visible ? MTL::StorageModeShared : MTL::StorageModePrivate);
		desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);

		value = dev.handle()->newTexture(desc.get());
		if (!(value))
		{
			fmt::throw_exception("Metal: failed to create texture %ux%ux%u fmt=%d mips=%u layers=%u samples=%u", info.width, info.height, info.depth, static_cast<int>(info.format), info.mipmaps, info.layers, info.samples);
		}

		g_render_device->make_resident(value);
	}

	image::image(const render_device& dev, const image_create_info& create_info)
	{
		create_impl(dev, create_info);
	}

	image::~image()
	{
		if (value)
		{
			if (g_render_device)
			{
				g_render_device->evict(value);
			}

			value->release();
			value = nullptr;
		}
	}

	void image::set_debug_name(const std::string& name)
	{
		m_debug_name = name;
		if (value)
		{
			autorelease_scope pool;
			value->setLabel(ns_str(name));
		}
	}

	// ---------------------------------------------------------------------------------------------

	image_view::image_view(mtl::image* resource, const image_view_info& view_info)
		: m_resource(resource), info(view_info)
	{
		create_impl();
	}

	void image_view::create_impl()
	{
		ensure(m_resource && m_resource->value);

		if (info.format == MTL::PixelFormatInvalid)
		{
			info.format = m_resource->format();
		}

		if (info.type == static_cast<MTL::TextureType>(~0ull))
		{
			info.type = m_resource->type();
		}

		if (info.level_count == ~0u)
		{
			info.level_count = m_resource->mipmaps() - info.base_level;
		}

		if (info.layer_count == ~0u)
		{
			info.layer_count = m_resource->layers() - info.base_layer;
		}

		// Depth-stencil formats: select the plane the shader will sample.
		MTL::PixelFormat view_format = info.format;
		const u32 fmt_aspect = get_format_aspect(view_format);
		if ((fmt_aspect & aspect_depth_stencil) == aspect_depth_stencil && (info.aspect & aspect_depth_stencil) == aspect_stencil)
		{
			// Stencil-only view
			view_format = (view_format == MTL::PixelFormatDepth24Unorm_Stencil8) ? MTL::PixelFormatX24_Stencil8 : MTL::PixelFormatX32_Stencil8;
		}

		info.format = view_format;

		if (!(m_resource->info.usage & MTL::TextureUsagePixelFormatView) &&
			is_reinterpreting_view(m_resource->format(), view_format)) [[unlikely]]
		{
			// Caller bug: the image must be created with MTLTextureUsagePixelFormatView (see image::create_impl).
			// Without it the view may read the (losslessly compressed) texture with the wrong layout.
			rsx_log.error("Metal: view format %d of texture '%s' (format %d) needs MTLTextureUsagePixelFormatView, which the texture was not created with",
				static_cast<int>(view_format), m_resource->debug_name(), static_cast<int>(m_resource->format()));
		}

		value = m_resource->value->newTextureView(
			view_format,
			info.type,
			NS::Range::Make(info.base_level, info.level_count),
			NS::Range::Make(info.base_layer, info.layer_count),
			info.swizzle);

		if (!(value))
		{
			fmt::throw_exception("Metal: failed to create texture view (fmt=%d type=%d)", static_cast<int>(view_format), static_cast<int>(info.type));
		}
	}

	image_view::~image_view()
	{
		m_subviews.clear();

		if (value)
		{
			value->release();
			value = nullptr;
		}
	}

	image_view* image_view::as(MTL::PixelFormat format)
	{
		if (info.format == format)
		{
			return this;
		}

		auto self = m_root_view ? m_root_view : this;
		if (auto found = self->m_subviews.find(format); found != self->m_subviews.end())
		{
			return found->second.get();
		}

		image_view_info sub_info = info;
		sub_info.format = format;

		auto view = std::make_unique<image_view>(m_resource, sub_info);
		view->m_root_view = self;

		auto result = view.get();
		self->m_subviews.emplace(format, std::move(view));
		return result;
	}

	// ---------------------------------------------------------------------------------------------

	MTL::TextureSwizzleChannels apply_swizzle_remap(const std::array<MTL::TextureSwizzle, 4>& base_remap_argb, const rsx::texture_channel_remap_t& remap)
	{
		const auto final_mapping = remap.remap(base_remap_argb, MTL::TextureSwizzleZero, MTL::TextureSwizzleOne);
		MTL::TextureSwizzleChannels result;
		result.red = final_mapping[1];
		result.green = final_mapping[2];
		result.blue = final_mapping[3];
		result.alpha = final_mapping[0];
		return result;
	}

	image_view* viewable_image::get_view(const rsx::texture_channel_remap_t& remap, u32 aspect_mask)
	{
		u32 remap_encoding = remap.encoded;
		if (remap_encoding == MTL_REMAP_IDENTITY && native_component_map == swizzle_identity)
		{
			remap_encoding = RSX_TEXTURE_REMAP_IDENTITY;
		}

		const u64 storage_key = remap_encoding | (static_cast<u64>(aspect_mask) << 32);
		if (auto found = views.find(storage_key); found != views.end())
		{
			return found->second.get();
		}

		MTL::TextureSwizzleChannels mapping;
		switch (remap_encoding)
		{
		case MTL_REMAP_IDENTITY:
			mapping = swizzle_identity;
			break;
		case RSX_TEXTURE_REMAP_IDENTITY:
			mapping = native_component_map;
			break;
		default:
			mapping = apply_swizzle_remap(
				{ native_component_map.alpha, native_component_map.red, native_component_map.green, native_component_map.blue },
				remap);
			break;
		}

		image_view_info view_info{};
		view_info.swizzle = mapping;
		view_info.aspect = aspect() & aspect_mask;
		ensure(view_info.aspect);

		auto view = std::make_unique<mtl::image_view>(this, view_info);
		auto result = view.get();
		views.emplace(storage_key, std::move(view));
		return result;
	}

	image_view* viewable_image::get_identity_view(u32 aspect_mask)
	{
		rsx::texture_channel_remap_t identity{};
		identity.encoded = MTL_REMAP_IDENTITY;
		identity.control_map = { CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP, CELL_GCM_TEXTURE_REMAP_REMAP };
		identity.channel_map = { 0, 1, 2, 3 };
		return get_view(identity, aspect_mask);
	}

	void viewable_image::set_native_component_layout(const MTL::TextureSwizzleChannels& new_layout)
	{
		if (!(new_layout == native_component_map))
		{
			native_component_map = new_layout;

			// Views depend on the native layout; drop them (deferred, GPU may still reference them)
			release_views();
		}
	}

	void viewable_image::release_views()
	{
		// Metal 4 command buffers do not retain resources and argument tables hold raw resource IDs,
		// so a view may still be referenced by in-flight work. Defer destruction through the GC.
		if (auto gc = get_gc())
		{
			for (auto& [key, view] : views)
			{
				gc->dispose(view);
			}
		}
		views.clear();
	}
}
