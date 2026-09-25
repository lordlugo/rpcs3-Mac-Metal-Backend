#include "stdafx.h"
#include "data_heap.h"
#include "garbage_collector.h"
#include "Emu/RSX/Metal/MTLHelpers.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/IdManager.h"

#include <unordered_set>

namespace mtl
{
	void data_heap::create(usz size, const char* name, usz guard, bool notify, rsx::flags32_t flags)
	{
		rsx::data_heap::init(size, name, guard);

		heap = std::make_unique<buffer>(*g_render_device, size, memory_location::host_visible, name ? name : "ring buffer");
		m_ptr = static_cast<u8*>(heap->map());
		initial_size = size;
		notify_on_grow = notify;
		m_flags = flags;
	}

	void data_heap::destroy()
	{
		m_ptr = nullptr;
		heap.reset();
	}

	bool data_heap::grow(usz size)
	{
		if (m_flags & heap_pool_fixed_size)
		{
			return false;
		}

		// Sizes are aligned up by 64M, up to 1GiB
		const usz size_limit = 1024 * 0x100000;
		usz aligned_new_size = utils::align(m_size + size, 64 * 0x100000);

		if (aligned_new_size >= size_limit)
		{
			rsx_log.error("[%s] Pool limit was reached. Will attempt to swap out the current heap.", m_name);
			aligned_new_size = size_limit;
		}

		// Wait for DMA activity to end before swapping the backing store
		g_fxo->get<rsx::dma_manager>().sync();

		rsx::data_heap::init(aligned_new_size, m_name, m_min_guard_size);

		// The old buffer may still be read by in-flight GPU work: dispose it through the GC.
		get_gc()->dispose(heap);
		heap = std::make_unique<buffer>(*g_render_device, aligned_new_size, memory_location::host_visible, m_name);
		m_ptr = static_cast<u8*>(heap->map());

		if (notify_on_grow)
		{
			raise_status_interrupt(mtl::heap_changed);
		}

		return true;
	}

	namespace data_heap_manager
	{
		static std::unordered_set<mtl::data_heap*> g_managed_heaps;

		void register_ring_buffer(mtl::data_heap& heap)
		{
			g_managed_heaps.insert(&heap);
		}

		void register_ring_buffers(std::initializer_list<std::reference_wrapper<mtl::data_heap>> heaps)
		{
			for (auto&& heap : heaps)
			{
				register_ring_buffer(heap);
			}
		}

		managed_heap_snapshot_t get_heap_snapshot()
		{
			managed_heap_snapshot_t result{};
			for (auto& heap : g_managed_heaps)
			{
				result[heap] = heap->get_current_put_pos_minus_one();
			}
			return result;
		}

		void restore_snapshot(const managed_heap_snapshot_t& snapshot)
		{
			for (auto& heap : g_managed_heaps)
			{
				const auto found = snapshot.find(heap);
				if (found == snapshot.end())
				{
					continue;
				}

				heap->set_get_pos(found->second);
				heap->notify();
			}
		}

		void reset_heap_allocations()
		{
			for (auto& heap : g_managed_heaps)
			{
				heap->reset_allocation_stats();
			}
		}

		void reset()
		{
			for (auto& heap : g_managed_heaps)
			{
				heap->destroy();
			}

			g_managed_heaps.clear();
		}
	}
}
