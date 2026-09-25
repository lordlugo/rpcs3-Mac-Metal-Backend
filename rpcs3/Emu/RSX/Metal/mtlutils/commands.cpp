#include "stdafx.h"
#include "commands.h"

#include <mutex>

namespace mtl
{
	command_list::~command_list()
	{
		destroy();
	}

	void command_list::create(const render_device& dev, MTL4::CommandQueue* queue, timeline& tl, std::string_view label)
	{
		ensure(!m_allocator);
		autorelease_scope pool;

		m_device = &dev;
		m_queue = queue;
		m_timeline = &tl;
		m_label = std::string(label);

		m_allocator = dev.handle()->newCommandAllocator();
		ensure(m_allocator, "Metal: failed to create command allocator");

		m_commands = dev.handle()->newCommandBuffer();
		ensure(m_commands, "Metal: failed to create MTL4CommandBuffer");

		auto desc = ref(MTL4::ArgumentTableDescriptor::alloc()->init());
		desc->setMaxBufferBindCount(gpu_capabilities::max_buffers_per_stage);
		desc->setMaxTextureBindCount(64);
		desc->setMaxSamplerStateBindCount(gpu_capabilities::max_samplers_per_stage);
		desc->setInitializeBindings(true);
		desc->setSupportAttributeStrides(false);

		for (u32 i = 0; i < table_count; ++i)
		{
			NS::Error* error = nullptr;
			m_argument_tables[i] = dev.handle()->newArgumentTable(desc.get(), &error);
			if (!(m_argument_tables[i]))
			{
				fmt::throw_exception("Metal: failed to create argument table: %s", to_string(error));
			}
		}

		m_submit_fence.owner = m_timeline;
	}

	void command_list::destroy()
	{
		if (!m_allocator)
		{
			return;
		}

		if (m_is_pending)
		{
			wait();
		}

		for (auto& table : m_argument_tables)
		{
			if (table)
			{
				table->release();
				table = nullptr;
			}
		}

		if (m_commands)
		{
			m_commands->release();
			m_commands = nullptr;
		}

		m_allocator->release();
		m_allocator = nullptr;
	}

	void command_list::begin()
	{
		ensure(!m_is_open);

		if (m_is_pending)
		{
			// The allocator's memory is still in use by the GPU.
			wait();
		}

		m_allocator->reset();
		m_commands->beginCommandBuffer(m_allocator);

		if (!m_label.empty())
		{
			autorelease_scope pool;
			m_commands->setLabel(ns_str(m_label));
		}

		m_is_open = true;
		m_pending_full_barrier = true;
		m_compute_commands_since_barrier = 0;
	}

	void command_list::end()
	{
		ensure(m_is_open);
		end_encoder();
		m_commands->endCommandBuffer();
		m_is_open = false;
	}

	u64 command_list::submit(const submit_info_t& info)
	{
		if (m_is_open)
		{
			end();
		}

		ensure(!m_is_pending);

		// Residency changes made while recording must be visible before the GPU runs this work.
		g_render_device->commit_residency();

		// Waits + commit + timeline signal must be atomic across threads, otherwise two submitters could interleave
		// and signal timeline values out of order relative to their work.
		static std::mutex s_submit_mutex;
		std::lock_guard lock(s_submit_mutex);

		if (info.wait_drawable)
		{
			m_queue->wait(info.wait_drawable);
		}

		if (info.wait_event)
		{
			m_queue->wait(info.wait_event, info.wait_value);
		}

		const MTL4::CommandBuffer* buffers[] = { m_commands };
		m_queue->commit(buffers, 1);

		if (info.signal_drawable)
		{
			m_queue->signalDrawable(info.signal_drawable);
		}

		const u64 value = m_timeline->signal(m_queue);
		m_submit_fence.value = value;
		m_submit_fence.flushed = true;
		m_is_pending = true;
		return value;
	}

	void command_list::open_encoder_barrier(MTL4::CommandEncoder* encoder, MTL::Stages before_stages)
	{
		// Conservative: wait for every stage of all prior work on this queue, make results device-visible.
		encoder->barrierAfterQueueStages(stages_all_work, before_stages, MTL4::VisibilityOptionDevice);
		m_pending_full_barrier = false;
	}

	MTL4::RenderCommandEncoder* command_list::begin_render_pass(const MTL4::RenderPassDescriptor* desc, MTL4::RenderEncoderOptions options)
	{
		ensure(m_is_open);
		end_encoder();

		m_render_encoder = m_commands->renderCommandEncoder(desc, options);
		ensure(m_render_encoder, "Metal: failed to begin render pass");

		open_encoder_barrier(m_render_encoder, stages_render);
		return m_render_encoder;
	}

	void command_list::end_render_pass()
	{
		if (m_render_encoder)
		{
			m_render_encoder->endEncoding();
			m_render_encoder = nullptr;
		}
	}

	MTL4::ComputeCommandEncoder* command_list::compute_unordered()
	{
		ensure(m_is_open);

		if (!m_compute_encoder)
		{
			end_render_pass();
			m_compute_encoder = m_commands->computeCommandEncoder();
			ensure(m_compute_encoder, "Metal: failed to begin compute encoder");
			open_encoder_barrier(m_compute_encoder, stages_compute);
			m_compute_commands_since_barrier = 0;
		}

		m_compute_commands_since_barrier++;
		return m_compute_encoder;
	}

	MTL4::ComputeCommandEncoder* command_list::compute()
	{
		ensure(m_is_open);

		if (m_compute_encoder && m_compute_commands_since_barrier > 0)
		{
			// Commands inside an MTL4 compute encoder execute concurrently; serialize.
			m_compute_encoder->barrierAfterEncoderStages(stages_compute, stages_compute, MTL4::VisibilityOptionDevice);
			m_compute_commands_since_barrier = 0;
		}

		return compute_unordered();
	}

	void command_list::end_encoder()
	{
		end_render_pass();

		if (m_compute_encoder)
		{
			m_compute_encoder->endEncoding();
			m_compute_encoder = nullptr;
			m_compute_commands_since_barrier = 0;
		}
	}

	command_list::encoder_type command_list::active_encoder() const
	{
		if (m_render_encoder) return encoder_type::render;
		if (m_compute_encoder) return encoder_type::compute;
		return encoder_type::none;
	}

	void command_list::full_barrier()
	{
		end_encoder();
		m_pending_full_barrier = true;
	}

	void command_list::push_debug_group(std::string_view name)
	{
		autorelease_scope pool;
		if (m_render_encoder) m_render_encoder->pushDebugGroup(ns_str(name));
		else if (m_compute_encoder) m_compute_encoder->pushDebugGroup(ns_str(name));
		else if (m_is_open) m_commands->pushDebugGroup(ns_str(name));
	}

	void command_list::pop_debug_group()
	{
		if (m_render_encoder) m_render_encoder->popDebugGroup();
		else if (m_compute_encoder) m_compute_encoder->popDebugGroup();
		else if (m_is_open) m_commands->popDebugGroup();
	}

	bool command_list::poke()
	{
		if (!m_is_pending)
		{
			return true;
		}

		if (m_submit_fence.signaled())
		{
			m_is_pending = false;
			m_submit_fence.reset();
			return true;
		}

		return false;
	}

	bool command_list::wait(u64 timeout_us)
	{
		if (!m_is_pending)
		{
			return true;
		}

		if (!m_submit_fence.wait(timeout_us))
		{
			return false;
		}

		m_is_pending = false;
		m_submit_fence.reset();
		return true;
	}
}
