#include "stdafx.h"
#include "MTLCommandStream.h"
#include "MTLGSRenderTypes.hpp"

#include "Emu/IdManager.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/system_config.h"

namespace mtl
{
	// Global submit guard to prevent races between commit order and timeline signal order
	static shared_mutex g_submit_mutex;

	void acquire_global_submit_lock()
	{
		g_submit_mutex.lock();
	}

	void release_global_submit_lock()
	{
		g_submit_mutex.unlock();
	}

	u64 queue_submit_now(mtl::command_list& commands, const submit_info_t& info)
	{
		autorelease_scope pool;

		acquire_global_submit_lock();
		const u64 value = commands.submit(info);
		release_global_submit_lock();

		return value;
	}

	FORCE_INLINE
	static void queue_submit_impl(command_buffer_chunk* commands, const submit_info_t& info)
	{
		ensure(commands);

		{
			std::lock_guard lock(commands->guard_mutex);
			queue_submit_now(*commands, info);
		}

		// Signal "flushed"
		commands->submit_queued.release(false);
	}

	void queue_submit(command_buffer_chunk* commands, const submit_info_t& info, bool flush)
	{
		if (auto renderer = rsx::get_current_renderer())
		{
			renderer->get_stats().submit_count++;
		}

		// Access to this method must be externally synchronized.
		// Offloader is guaranteed to never call this for async flushes.
		commands->submit_queued = true;

		if (!flush && g_cfg.video.multithreaded_rsx)
		{
			auto packet = new queue_submit_t{ commands, info };
			g_fxo->get<rsx::dma_manager>().backend_ctrl(rctrl_queue_submit, packet);
		}
		else
		{
			queue_submit_impl(commands, info);
		}
	}

	void queue_submit(const queue_submit_t* packet)
	{
		// Flush-only version used by asynchronous submit processing (MTRSX)
		queue_submit_impl(packet->commands, packet->info);
	}
}
