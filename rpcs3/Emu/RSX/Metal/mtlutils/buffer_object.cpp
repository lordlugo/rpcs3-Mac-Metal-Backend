#include "stdafx.h"
#include "buffer_object.h"

namespace mtl
{
	static MTL::ResourceOptions get_resource_options(memory_location location)
	{
		switch (location)
		{
		case memory_location::device_local:
			return MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked;
		case memory_location::host_visible:
		default:
			// Write-combined is intentionally NOT used: RSX reads back from these buffers (reports, labels, DMA).
			return MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeDefaultCache | MTL::ResourceHazardTrackingModeUntracked;
		}
	}

	buffer::buffer(const render_device& dev, u64 size, memory_location location, std::string_view label)
		: m_size(size), m_location(location)
	{
		ensure(size > 0);
		m_buffer = dev.handle()->newBuffer(size, get_resource_options(location));
		if (!(m_buffer))
		{
			fmt::throw_exception("Metal: failed to allocate buffer of %llu bytes", size);
		}

		if (!label.empty())
		{
			autorelease_scope pool;
			m_buffer->setLabel(ns_str(label));
		}

		g_render_device->make_resident(m_buffer);
	}

	buffer::buffer(const render_device& dev, void* host_pointer, u64 size, std::string_view label)
		: m_size(size), m_location(memory_location::host_visible), m_host_import(true)
	{
		ensure(host_pointer && size);
		m_buffer = dev.handle()->newBuffer(host_pointer, size,
			MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked,
			nullptr);
		if (!(m_buffer))
		{
			fmt::throw_exception("Metal: newBufferWithBytesNoCopy failed (ptr=%p, size=0x%llx)", host_pointer, size);
		}

		if (!label.empty())
		{
			autorelease_scope pool;
			m_buffer->setLabel(ns_str(label));
		}

		g_render_device->make_resident(m_buffer);
	}

	buffer::~buffer()
	{
		if (m_buffer)
		{
			if (g_render_device)
			{
				g_render_device->evict(m_buffer);
			}

			m_buffer->release();
			m_buffer = nullptr;
		}
	}

	void* buffer::map(u64 offset) const
	{
		ensure(m_location == memory_location::host_visible, "Metal: attempted to map a private buffer");
		return static_cast<u8*>(m_buffer->contents()) + offset;
	}

	u32 get_texel_buffer_element_size(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Uint:
		case MTL::PixelFormatR8Unorm:
			return 1;
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR16Float:
		case MTL::PixelFormatRG8Uint:
			return 2;
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatR32Float:
		case MTL::PixelFormatRGBA8Uint:
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRG16Uint:
			return 4;
		case MTL::PixelFormatRG32Uint:
		case MTL::PixelFormatRG32Float:
		case MTL::PixelFormatRGBA16Uint:
			return 8;
		case MTL::PixelFormatRGBA32Uint:
		case MTL::PixelFormatRGBA32Float:
			return 16;
		default:
			break;
		}

		fmt::throw_exception("Metal: unsupported texel buffer format %d", static_cast<int>(format));
	}

	u64 get_texel_buffer_alignment(const render_device& dev, MTL::PixelFormat format)
	{
		return std::max<u64>(16, dev.handle()->minimumTextureBufferAlignmentForPixelFormat(format));
	}

	buffer_view::buffer_view(const render_device& dev, const buffer& buf, MTL::PixelFormat format_, u64 offset_, u64 size_)
		: parent(buf.value()), format(format_), offset(offset_), size(size_)
	{
		const u32 texel_size = get_texel_buffer_element_size(format);
		const u64 width = size / texel_size;
		ensure(width > 0);

		auto desc = ref(MTL::TextureDescriptor::textureBufferDescriptor(format, width,
			buf.location() == memory_location::host_visible
				? (MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked)
				: (MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked),
			MTL::TextureUsageShaderRead)->retain());

		value = parent->newTexture(desc.get(), offset, width * texel_size);
		if (!(value))
		{
			fmt::throw_exception("Metal: failed to create texel buffer view (fmt=%d, offset=%llu, size=%llu)", static_cast<int>(format), offset, size);
		}
		(void)dev;
	}

	buffer_view::~buffer_view()
	{
		if (value)
		{
			value->release();
			value = nullptr;
		}
	}

	bool buffer_view::in_range(u32 address, u32 range, u32& out_offset) const
	{
		if (address < offset)
		{
			return false;
		}

		const u32 _offset = address - static_cast<u32>(offset);
		if (size >= _offset + range)
		{
			out_offset = _offset;
			return true;
		}

		return false;
	}
}
