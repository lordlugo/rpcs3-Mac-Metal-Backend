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

	// Why a render pass had to be ended early (telemetry)
	enum class pass_split_reason : u32
	{
		read_after_write = 0, // a draw samples an attachment written by the open pass
		write_after_read,     // strict mode only: a write follows reads of the same attachment
		depth_compare,        // depth compare emulated by sampling the bound depth buffer after it was written
		vertex_read,          // a vertex program samples an image written by earlier fragment work
		read_through_copy,    // a draw samples a converted/copied view of a bound surface (made outside the pass)
		count
	};

	// Telemetry for the renderer's periodic log line: GPU time of committed work (union of the start-end intervals
	// reported by Metal 4 commit feedback, so overlapping work is not counted twice) and render passes begun.
	struct gpu_stats_t
	{
		u64 busy_ns = 0;
		u64 render_passes = 0;
		u64 draw_render_passes = 0;     // passes of the renderer's framebuffer (the rest: copies, clears, overlays)
		u64 feedback_splits = 0;
		std::array<u64, static_cast<u32>(pass_split_reason::count)> splits_by_reason{};
		u64 feedback_reads_in_pass = 0; // feedback reads served without a split (see render_target feedback streaks)
	};

	gpu_stats_t get_gpu_stats_and_reset();
	void count_feedback_split(pass_split_reason reason = pass_split_reason::read_after_write);
	void count_feedback_read_in_pass();
	void count_draw_render_pass();

	// Contents of one argument table as last written through update_*(). Metal takes a snapshot of a table's bindings
	// when a draw/dispatch is encoded; the table itself is a plain object whose contents persist across encoders and
	// command buffers. So when every write goes through here (glsl::program::bind() is the only writer), a slot whose
	// value already matches needs no write. A table holds values (GPU addresses, resource IDs), not objects, so an equal
	// value means identical GPU-visible state even if the object behind it was recreated.
	struct argument_table_shadow
	{
		static constexpr u32 max_textures = 64; // Table size, see command_list::create()

		std::array<MTL::GPUAddress, gpu_capabilities::max_buffers_per_stage> buffers{};
		std::array<u64, max_textures> textures{};                                 // MTL::ResourceID::_impl
		std::array<u64, gpu_capabilities::max_samplers_per_stage> samplers{};     // MTL::ResourceID::_impl
		u32 buffers_known = 0;   // Bit i: buffers[i] is what the table holds
		u64 textures_known = 0;
		u32 samplers_known = 0;

		// Each returns true, and records the value, if the table does not hold it yet (the caller writes it then)
		bool update_buffer(u32 index, MTL::GPUAddress address)
		{
			return update(buffers[index], buffers_known, u32{1} << index, address);
		}

		bool update_texture(u32 index, MTL::ResourceID id)
		{
			return update(textures[index], textures_known, u64{1} << index, id._impl);
		}

		bool update_sampler(u32 index, MTL::ResourceID id)
		{
			return update(samplers[index], samplers_known, u32{1} << index, id._impl);
		}

		// Every slot unknown: the next bind writes all the slots it uses
		void invalidate()
		{
			buffers_known = 0;
			textures_known = 0;
			samplers_known = 0;
		}

	private:
		template <typename M>
		static bool update(u64& slot, M& known, M bit, u64 value)
		{
			if ((known & bit) && slot == value)
			{
				return false;
			}

			slot = value;
			known |= bit;
			return true;
		}
	};

	// State glsl::program::bind() (the only code setting pipeline states and argument tables) set on the open render
	// encoder. Encoder state lasts until the encoder ends; reset whenever a render pass begins.
	struct render_encoder_bindings
	{
		u64 program_uid = 0;  // glsl::program whose pipeline state is set (0: none)
		u32 tables_set = 0;   // Bit (1 << argument_table_slot): that table is set for its stage
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
	//  - Every new compute encoder starts with a consumer barrier on ALL previously committed/encoded work of the queue
	//    (barrierAfterQueueStages(all, <encoder stages>)).
	//  - A render pass orders its fragment work (shading, attachment loads) after all earlier work, and its vertex
	//    work after all earlier work except fragment work: binning of the next pass overlaps the shading of the
	//    previous one, which matters with many short passes (feedback loops). Vertex programs reading images written
	//    by fragment work (render targets as vertex textures) call require_vertex_after_fragment() for the full barrier.
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
		u64 m_pass_serial = 0;                                     // Unique id of the open render pass (all lists)
		MTL4::ComputeCommandEncoder* m_compute_encoder = nullptr; // Not owned
		u32 m_compute_commands_since_barrier = 0;
		bool m_pending_full_barrier = true;
		bool m_next_pass_orders_vertex = false;  // the next render pass also orders its vertex work after fragment work
		bool m_pass_orders_vertex = false;       // the open render pass was begun that way

		std::array<MTL4::ArgumentTable*, table_count> m_argument_tables{};
		std::array<argument_table_shadow, table_count> m_argument_table_shadows{};
		render_encoder_bindings m_render_bindings{};

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

		// Unique id (across all command lists) of the open render pass, 0 when none is open. Attachments written by a
		// pass reach memory when it ends: feedback reads compare a surface's last writing pass with this id.
		u64 open_pass_serial() const { return m_render_encoder ? m_pass_serial : 0; }
		MTL4::RenderCommandEncoder* render_encoder() const { return m_render_encoder; }
		void end_render_pass();

		// The current draw reads, in its vertex stage, an image written by earlier fragment work. Ends the open pass if
		// it was begun without ordering vertex work after fragment work (returns true then); the next pass is.
		bool require_vertex_after_fragment();

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
		// What the table of `slot` holds. Whoever writes a table directly must update (or invalidate) this.
		argument_table_shadow& argument_table_contents(argument_table_slot slot) { return m_argument_table_shadows[slot]; }
		// Pipeline state and tables set on the open render encoder
		render_encoder_bindings& render_bindings() { return m_render_bindings; }

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
