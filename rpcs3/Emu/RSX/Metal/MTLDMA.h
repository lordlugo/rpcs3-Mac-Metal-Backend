#pragma once

// Port of VK/VKDMA.h: guest memory DMA blocks (64 KiB granularity, chained for larger ranges).
//
// Two block types, selected per block by create_dma_block():
//  - dma_block_EXT (passthrough): a zero-copy MTL::Buffer wrapping the guest "super pointer" range
//    (newBufferWithBytesNoCopy). GPU writes land directly in guest memory; load/flush are no-ops.
//    Requires backend_config.supports_passthrough_dma (R sets it from mtl::is_passthrough_dma_supported()).
//  - dma_block: host_visible (shared) copy of the range; load()/flush() memcpy between guest memory and the copy.

#include "mtlutils/buffer_object.h"
#include "Utilities/address_range.h"

#include <memory>
#include <utility>

namespace mtl
{
	using dma_mapping_handle = std::pair<u32, mtl::buffer*>;

	// True on Apple silicon: shared-storage buffers can alias page-aligned host memory without copies.
	bool is_passthrough_dma_supported();

	dma_mapping_handle map_dma(u32 local_address, u32 length);
	void load_dma(u32 local_address, u32 length);
	void flush_dma(u32 local_address, u32 length);
	void unmap_dma(u32 local_address, u32 length);

	void clear_dma_resources();

	class dma_block
	{
	protected:
		struct
		{
			dma_block* parent = nullptr;
			u32 block_offset = 0;
		}
		inheritance_info;

		u32 base_address = 0;
		u8* memory_mapping = nullptr;
		std::unique_ptr<buffer> allocated_memory;

		virtual void allocate(const render_device& dev, usz size);
		virtual void free();
		virtual void* map_range(const utils::address_range32& range);
		virtual void unmap();

	public:

		dma_block() = default;
		virtual ~dma_block();

		virtual void init(const render_device& dev, u32 addr, usz size);
		virtual void init(dma_block* parent, u32 addr, usz size);
		virtual void flush(const utils::address_range32& range);
		virtual void load(const utils::address_range32& range);
		std::pair<u32, buffer*> get(const utils::address_range32& range);

		u32 start() const;
		u32 end() const;
		u32 size() const;

		dma_block* head();
		const dma_block* head() const;
		virtual void set_parent(dma_block* parent);
		virtual void extend(const render_device& dev, usz new_size);
	};

	class dma_block_EXT: public dma_block
	{
	private:
		void allocate(const render_device& dev, usz size) override;
		void* map_range(const utils::address_range32& range) override;
		void unmap() override;

	public:
		void flush(const utils::address_range32& range) override;
		void load(const utils::address_range32& range) override;
	};

	void create_dma_block(std::unique_ptr<dma_block>& block, u32 base_address, usz expected_length);
}
