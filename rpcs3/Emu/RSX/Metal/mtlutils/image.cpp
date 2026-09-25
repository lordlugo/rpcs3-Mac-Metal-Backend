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

		// Always allow format reinterpretation; RSX aliases surfaces between formats all the time.
		desc->setUsage(info.usage | MTL::TextureUsagePixelFormatView);
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
