#pragma once

#include "Emu/RSX/Common/ring_buffer_helper.h"
#include "buffer_object.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#include <memory>
#include <unordered_map>
#include <initializer_list>
#include <functional>
#include <type_traits>

namespace mtl
{
	enum data_heap_pool_flags
	{
		heap_pool_default    = 0,
		heap_pool_fixed_size = (1 << 1),
	};

	// Ring buffer on a persistently mapped, CPU/GPU shared MTL::Buffer (unified memory: no staging, no flushes).
	class data_heap : public rsx::data_heap
	{
	private:
		usz initial_size = 0;
		rsx::flags32_t m_flags = heap_pool_default;
		bool notify_on_grow = false;
		u8* m_ptr = nullptr;

	protected:
		bool grow(usz size) override;

	public:
		std::unique_ptr<buffer> heap;

		void create(usz size, const char* name, usz guard = 0x10000, bool notify = false, rsx::flags32_t flags = heap_pool_default);
		void destroy();

		template <typename T = void>
		T* map(usz offset, usz /*size*/)
		{
			return reinterpret_cast<T*>(m_ptr + offset);
		}

		void unmap(bool /*force*/ = false) {}

		template<int Alignment, typename T = char>
			requires std::is_trivially_destructible_v<T>
		std::pair<usz, T*> alloc_and_map(usz count)
		{
			const auto size_bytes = count * sizeof(T);
			const auto addr = alloc<Alignment>(size_bytes);
			return { addr, reinterpret_cast<T*>(map(addr, size_bytes)) };
		}

		MTL::Buffer* value() const { return heap->value(); }
		MTL::GPUAddress gpu_address(usz offset = 0) const { return heap->gpu_address(offset); }

		// Compatibility with the VK heap-manager interface (no shadow copies on unified memory)
		bool is_dirty() const { return false; }
	};

	namespace data_heap_manager
	{
		using managed_heap_snapshot_t = std::unordered_map<const mtl::data_heap*, s64>;

		void register_ring_buffer(mtl::data_heap& heap);
		void register_ring_buffers(std::initializer_list<std::reference_wrapper<mtl::data_heap>> heaps);
		managed_heap_snapshot_t get_heap_snapshot();
		void restore_snapshot(const managed_heap_snapshot_t& snapshot);
		void reset_heap_allocations();
		void reset();
	}
}
