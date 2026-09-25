#pragma once

#include "mtlutils/commands.h"

namespace mtl
{
	struct command_buffer_chunk;

	enum rctrl_command : u32 // callback commands
	{
		rctrl_queue_submit = 0x80000000,
	};

	// A deferred queue submission (mirrors vk::queue_submit_t / submit_packet).
	// The packet copies the submit info; the caller must keep any drawable alive until the list has been flushed.
	struct queue_submit_t
	{
		command_buffer_chunk* commands = nullptr;
		submit_info_t info{};
	};

	// Global submit guard. Metal 4 queues are thread-safe, but committing a batch and signaling the shared timeline must
	// happen atomically so that timeline values are signaled in commit order.
	void acquire_global_submit_lock();
	void release_global_submit_lock();

	// Submit a closed command list. With MTRSX enabled and !flush, the commit is performed by the RSX offloader thread
	// (in FIFO order with other offloaded work) and the list is marked as `submit_queued` until then.
	void queue_submit(command_buffer_chunk* commands, const submit_info_t& info, bool flush);

	// Flush-only version used by asynchronous submit processing (MTRSX)
	void queue_submit(const queue_submit_t* packet);

	// Immediately commit any command_list (auxiliary lists, e.g. secondary chain). Serialized with the global lock.
	u64 queue_submit_now(mtl::command_list& commands, const submit_info_t& info = {});
}
