#include "stdafx.h"
#include "sync.h"

#include <thread>

namespace mtl
{
	timeline::~timeline()
	{
		destroy();
	}

	void timeline::create(const render_device& dev, std::string_view label)
	{
		ensure(!m_event);
		autorelease_scope pool;

		m_event = dev.handle()->newSharedEvent();
		ensure(m_event, "Metal: failed to create shared event");
		m_event->setLabel(ns_str(label));
		m_event->setSignaledValue(0);
		m_next_value = 0;
	}

	void timeline::destroy()
	{
		if (m_event)
		{
			m_event->release();
			m_event = nullptr;
		}
	}

	u64 timeline::signal(MTL4::CommandQueue* queue)
	{
		const u64 value = ++m_next_value;
		queue->signalEvent(m_event, value);
		return value;
	}

	void timeline::gpu_wait(MTL4::CommandQueue* queue, u64 value) const
	{
		queue->wait(m_event, value);
	}

	bool timeline::wait(u64 value, u64 timeout_us) const
	{
		if (!m_event || m_event->signaledValue() >= value)
		{
			return true;
		}

		if (timeout_us == 0)
		{
			// Wait "forever" in large chunks so a hung GPU is still diagnosable in the log.
			while (!m_event->waitUntilSignaledValue(value, 10000))
			{
				rsx_log.error("Metal: waited more than 10s for GPU timeline value %llu (current %llu)", value, m_event->signaledValue());
			}
			return true;
		}

		const u64 timeout_ms = std::max<u64>(1, timeout_us / 1000);
		return m_event->waitUntilSignaledValue(value, timeout_ms);
	}
}
