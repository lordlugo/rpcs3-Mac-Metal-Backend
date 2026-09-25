#pragma once

#include <util/types.hpp>
#include "Emu/Cell/lv2/sys_rsx.h"
#include "Emu/RSX/GCM.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "RSXIOMap.hpp"

namespace rsx
{
	namespace gcm
	{
		enum limits
		{
			tiles_count = 15,
			zculls_count = 8
		};
	}

	struct GCM_tile_reference
	{
		u32 base_address = 0;
		const GcmTileInfo* tile = nullptr;

		operator bool() const
		{
			return !!tile;
		}

		utils::address_range32 tile_align(const rsx::address_range32& range) const;
	};

	struct GCM_context
	{
		RsxDmaControl* ctrl = nullptr;
		u32 dma_address{ 0 };
		rsx_iomap_table iomap_table;

		GcmTileInfo tiles[gcm::limits::tiles_count];
		GcmZcullInfo zculls[gcm::limits::zculls_count];

		RsxDisplayInfo display_buffers[8];
		u32 display_buffers_count{ 0 };
		u32 current_display_buffer{ 0 };

		shared_mutex sys_rsx_mtx;
		u32 device_addr{ 0 };
		u32 label_addr{ 0 };
		u32 main_mem_size{ 0 };
		u32 local_mem_size{ 0 };
		u32 rsx_event_port{ 0 };
		u32 driver_info{ 0 };

		atomic_t<u64> unsent_gcm_events = 0; // Unsent event bits when aborting RSX/VBLANK thread (will be sent on savestate load)

		// Set when this context fetches its FIFO commands from RSX local memory instead of IO-mapped main memory.
		// libgcm's system mode (e.g. GTA IV 1.00) queues its first commands in local memory before any IO is mapped.
		atomic_t<bool> fifo_in_local_memory = false;

		// Effective address of a FIFO offset (GET/PUT, JUMP/CALL targets), umax when unmapped
		u32 fifo_offset_to_ea(u32 offset) const
		{
			if (fifo_in_local_memory && offset < local_mem_size) [[unlikely]]
			{
				return rsx::constants::local_mem_base + offset;
			}

			return iomap_table.get_addr(offset);
		}

		GCM_tile_reference get_tiled_memory_region(const utils::address_range32& range) const;
	};
}
