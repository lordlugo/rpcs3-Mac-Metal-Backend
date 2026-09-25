#pragma once

#include "mtl_api.h"
#include "device.h"
#include "sync.h"
#include "Utilities/mutex.h"

#include <array>

namespace mtl
{
	enum argument_table_slot : u32
	{
		table_vertex = 0,
		table_fragment = 1,
		table_compute = 2,
		table_count = 3
	};

	struct submit_info_t
	{
		// Optional GPU-side waits before this batch executes
		const MTL::Event* wait_event = nullptr;
		u64 wait_value = 0;
		const MTL::Drawable* wait_drawable = nullptr;     // Wait for the drawable to be available before rendering into it
		const MTL::Drawable* signal_drawable = nullptr;   // Signal drawable completion after this batch (then call present())
		bool flush = false;                               // Hint only; Metal commits are always flushed
	};

	// Metal 4 command recording unit (equivalent of vk::command_buffer + command pool).
	//
	// Hazard model (Metal 4 resources are untracked):
	//  - Every new encoder starts with a consumer barrier on ALL previously committed/encoded work of the queue
	//    (barrierAfterQueueStages(all, <encoder stages>)). Passes are therefore serialized like a tracked queue.
	//  - Every command recorded through compute() after the first one in the same compute encoder is preceded by an
	//    intra-encoder barrier (dispatch|blit -> dispatch|blit), because MTL4 compute encoders run commands concurrently.
	//  - Inside a render pass nothing can be waited on (Apple GPUs do not support fragment->fragment barriers inside a
	//    pass); to read a render target that is being written, end the pass (feedback loop handling lives in the renderer).
	// This is conservative and correct; it can be relaxed later with finer stage masks.
	class command_list
	{
	public:
		enum access_type_hint
		{
			flush_only, // Only to be submitted/opened/closed via command flush
			all         // Auxiliary, can be submitted/opened/closed at any time
		}
		access_hint = flush_only;

		enum command_buffer_data_flag : u32
		{
			cb_has_occlusion_task     = 0x01,
			cb_has_blit_transfer      = 0x02,
			cb_has_dma_transfer       = 0x04,
			cb_has_open_query         = 0x08,
			cb_load_occluson_task     = 0x10,
			cb_has_conditional_render = 0x20,
			cb_reload_dynamic_state   = 0x40
		};
		u32 flags = 0;

		enum class encoder_type
		{
			none,
			render,
			compute
		};

	protected:
		const render_device* m_device = nullptr;
		MTL4::CommandQueue* m_queue = nullptr;
		timeline* m_timeline = nullptr;

		MTL4::CommandAllocator* m_allocator = nullptr;
		MTL4::CommandBuffer* m_commands = nullptr;

		MTL4::RenderCommandEncoder* m_render_encoder = nullptr;   // Not owned (lifetime of the encoding)
		MTL4::ComputeCommandEncoder* m_compute_encoder = nullptr; // Not owned
		u32 m_compute_commands_since_barrier = 0;
		bool m_pending_full_barrier = true;

		std::array<MTL4::ArgumentTable*, table_count> m_argument_tables{};

		bool m_is_open = false;
		bool m_is_pending = false;
		fence m_submit_fence{};

		std::string m_label;

		void open_encoder_barrier(MTL4::CommandEncoder* encoder, MTL::Stages before_stages);

	public:
		command_list() = default;
		virtual ~command_list();

		command_list(const command_list&) = delete;
		command_list& operator=(const command_list&) = delete;

		void create(const render_device& dev, MTL4::CommandQueue* queue, timeline& tl, std::string_view label);
		void destroy();

		// Begin recording. The allocator is reset; caller must ensure previous work finished (see wait()/poke()).
		void begin();
		// Close any open encoder and end the command buffer.
		void end();
		// Commit to the queue and signal the timeline. Returns the timeline value that marks completion.
		u64 submit(const submit_info_t& info = {});

		bool is_recording() const { return m_is_open; }
		bool is_pending() const { return m_is_pending; }
		const fence& get_fence() const { return m_submit_fence; }

		// --- Encoders -------------------------------------------------------------------------------------------
		// Begin a render pass. Ends any active encoder first.
		MTL4::RenderCommandEncoder* begin_render_pass(const MTL4::RenderPassDescriptor* desc, MTL4::RenderEncoderOptions options = 0);
		bool is_render_pass_open() const { return m_render_encoder != nullptr; }
		MTL4::RenderCommandEncoder* render_encoder() const { return m_render_encoder; }
		void end_render_pass();

		// Returns an open compute/blit encoder (ending a render pass if needed) and inserts the intra-encoder
		// barrier required before the next command. Call once per recorded command.
		MTL4::ComputeCommandEncoder* compute();
		// Same as compute() but without the automatic barrier (caller guarantees independence of the next command).
		MTL4::ComputeCommandEncoder* compute_unordered();

		void end_encoder();
		encoder_type active_encoder() const;

		// Force the next encoder to wait for everything encoded before it (default behaviour anyway).
		void full_barrier();

		// --- Argument tables (one per stage slot, reused across encoders; contents captured at draw/dispatch) ---
		MTL4::ArgumentTable* argument_table(argument_table_slot slot) const { return m_argument_tables[slot]; }

		// --- Misc -----------------------------------------------------------------------------------------------
		MTL4::CommandBuffer* handle() const { return m_commands; }
		MTL4::CommandQueue* queue() const { return m_queue; }
		const render_device& device() const { return *m_device; }

		void push_debug_group(std::string_view name);
		void pop_debug_group();

		void clear_flags() { flags = 0; }
		void set_flag(command_buffer_data_flag flag) { flags |= flag; }

		// Completion tracking (called by the renderer's command-buffer ring)
		bool poke();                     // Non-blocking; returns true if no work is pending
		bool wait(u64 timeout_us = 0);   // Blocking; returns false on timeout
	};
}
