#pragma once

#include "mtl_api.h"
#include "device.h"

namespace mtl
{
	enum class memory_location
	{
		device_local,  // MTLStorageModePrivate: GPU only (render targets, textures, GPU-written scratch)
		host_visible,  // MTLStorageModeShared: CPU+GPU on unified memory (rings, readback, labels)
	};

	// Owned MTL::Buffer registered in the device residency set.
	class buffer
	{
		MTL::Buffer* m_buffer = nullptr;
		MTL::GPUAddress m_gpu_address = 0; // Fixed for the buffer's lifetime; cached because draws bind buffers by address
		u64 m_size = 0;
		memory_location m_location = memory_location::host_visible;
		bool m_host_import = false;

	public:
		buffer(const render_device& dev, u64 size, memory_location location, std::string_view label = {});

		// Zero-copy import of host memory (newBufferWithBytesNoCopy). `host_pointer` and `size` must be
		// page aligned (16 KiB on Apple silicon) and lie inside one VM region. The memory is NOT freed by us.
		buffer(const render_device& dev, void* host_pointer, u64 size, std::string_view label = {});

		~buffer();

		buffer(const buffer&) = delete;
		buffer& operator=(const buffer&) = delete;

		MTL::Buffer* value() const { return m_buffer; }
		operator MTL::Buffer*() const { return m_buffer; }

		u64 size() const { return m_size; }
		memory_location location() const { return m_location; }
		bool is_host_import() const { return m_host_import; }

		// Persistent CPU pointer (host_visible buffers only). Unified memory: no map/unmap cost.
		void* map(u64 offset = 0) const;

		MTL::GPUAddress gpu_address(u64 offset = 0) const { return m_gpu_address + offset; }
	};

	// A typed texel view of a buffer (MSL texture_buffer<T>), used for vertex pulling and texel-buffer inputs.
	struct buffer_view
	{
		MTL::Texture* value = nullptr;
		MTL::ResourceID resource_id{};   // value->gpuResourceID(), fixed for the view's lifetime (bound by resource ID)
		MTL::Buffer* parent = nullptr;
		MTL::PixelFormat format = MTL::PixelFormatInvalid;
		u64 offset = 0;
		u64 size = 0;

		// `offset` must be aligned to device->minimumTextureBufferAlignmentForPixelFormat(format)
		buffer_view(const render_device& dev, const buffer& buf, MTL::PixelFormat format, u64 offset, u64 size);
		~buffer_view();

		buffer_view(const buffer_view&) = delete;
		buffer_view& operator=(const buffer_view&) = delete;

		bool in_range(u32 address, u32 size, u32& offset) const;
	};

	// Bytes per texel for the formats used with buffer views
	u32 get_texel_buffer_element_size(MTL::PixelFormat format);
	u64 get_texel_buffer_alignment(const render_device& dev, MTL::PixelFormat format);
}
