#pragma once

#include "mtl_api.h"
#include "device.h"
#include "util/atomic.hpp"

namespace mtl
{
	// A monotonically increasing GPU timeline backed by MTLSharedEvent (Metal's timeline semaphore).
	// Queues signal it after each committed batch; the CPU polls or waits on values.
	class timeline
	{
		MTL::SharedEvent* m_event = nullptr;
		atomic_t<u64> m_next_value{ 0 };

	public:
		timeline() = default;
		~timeline();

		timeline(const timeline&) = delete;
		timeline& operator=(const timeline&) = delete;

		void create(const render_device& dev, std::string_view label);
		void destroy();

		MTL::SharedEvent* handle() const { return m_event; }

		// Reserve the next value and enqueue a signal of it on `queue` (after previously committed work).
		u64 signal(MTL4::CommandQueue* queue);

		// Enqueue a GPU-side wait on `queue` until the timeline reaches `value`.
		void gpu_wait(MTL4::CommandQueue* queue, u64 value) const;

		u64 completed_value() const { return m_event ? m_event->signaledValue() : 0; }
		u64 last_signaled_value() const { return m_next_value.load(); }
		bool is_complete(u64 value) const { return completed_value() >= value; }

		// CPU wait. timeout_us == 0 means wait forever. Returns false on timeout.
		bool wait(u64 value, u64 timeout_us = 0) const;
	};

	// Host-side "fence" for a single submitted batch.
	struct fence
	{
		const timeline* owner = nullptr;
		u64 value = 0;
		atomic_t<bool> flushed = false;

		void reset() { value = 0; flushed = false; }
		bool signaled() const { return !value || (owner && owner->is_complete(value)); }
		bool wait(u64 timeout_us = 0) const { return !value || !owner || owner->wait(value, timeout_us); }
	};
}
