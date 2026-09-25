#pragma once

#include "mtlutils/commands.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/image.h"
#include "MTLResourceManager.h"

#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/rsx_cache.h"
#include "Utilities/mutex.h"
#include "util/asm.hpp"

#include <optional>
#include <string>
#include <thread>

// Initial heap allocation values. The heaps are growable and will automatically increase in size to accomodate demands
#define MTL_ATTRIB_RING_BUFFER_SIZE_M 64
#define MTL_TEXTURE_UPLOAD_RING_BUFFER_SIZE_M 64
#define MTL_UBO_RING_BUFFER_SIZE_M 16
#define MTL_TRANSFORM_CONSTANTS_BUFFER_SIZE_M 16
#define MTL_FRAGMENT_CONSTANTS_BUFFER_SIZE_M 16
#define MTL_INDEX_RING_BUFFER_SIZE_M 16
#define MTL_SCRATCH_RING_BUFFER_SIZE_M 16

// Number of command lists in each ring. Every command list owns an MTL4CommandAllocator + 3 argument tables.
#define MTL_MAX_ASYNC_CB_COUNT 128

// Number of frames that can be queued for presentation (matches CAMetalLayer.maximumDrawableCount)
#define MTL_MAX_DRAWABLE_COUNT 3

#ifndef TEARDOWN_WAIT_TIMEOUT
#define TEARDOWN_WAIT_TIMEOUT 5000000ull // 5 seconds: renderer destruction never blocks the UI thread forever on a hung GPU
#endif

#ifndef FRAME_PRESENT_TIMEOUT
#define FRAME_PRESENT_TIMEOUT 10000000ull // 10 seconds
#endif

#ifndef GENERAL_WAIT_TIMEOUT
#define GENERAL_WAIT_TIMEOUT  2000000ull  // 2 seconds
#endif

namespace CA
{
	class MetalDrawable;
}

namespace mtl
{
	struct buffer_view;
	struct program_cache;
	struct pipeline_props;

	using vertex_cache = rsx::vertex_cache::default_vertex_cache<rsx::vertex_cache::uploaded_range>;
	using weak_vertex_cache = rsx::vertex_cache::weak_vertex_cache;
	using null_vertex_cache = vertex_cache;

	using shader_cache = rsx::shaders_cache<mtl::pipeline_props, mtl::program_cache>;

	// Fixed-function state that Vulkan bakes into the graphics pipeline but Metal sets on the render encoder
	struct rasterizer_state
	{
		MTL::CullMode cull_mode = MTL::CullModeNone;
		MTL::Winding front_face = MTL::WindingCounterClockwise;
		MTL::DepthClipMode depth_clip_mode = MTL::DepthClipModeClip;
		MTL::TriangleFillMode fill_mode = MTL::TriangleFillModeFill;
		bool cull_all_polygons = false;   // cull_face::front_and_back (Metal has no FrontAndBack cull mode)
		u64 depth_stencil_key = 0;        // Packed MTLDepthStencilDescriptor contents (incl. stencil masks)
	};

	// Encoder state last applied to the renderer's main pass (invalidated whenever a pass is opened)
	struct encoder_state
	{
		u64 pass_id = umax;
		const MTL::RenderPipelineState* pipeline = nullptr;
		const MTL::DepthStencilState* depth_stencil = nullptr;
	};

	struct vertex_upload_info
	{
		MTL::PrimitiveType primitive;
		u32 vertex_draw_count;
		u32 allocated_vertex_count;
		u32 first_vertex;
		u32 vertex_index_base;
		u32 vertex_index_offset;
		u32 persistent_window_offset;
		u32 volatile_window_offset;
		std::optional<std::tuple<u64, MTL::IndexType>> index_info; // { offset in index ring, index type }
	};

	// A command list of the renderer's rings (vk::command_buffer_chunk equivalent).
	// Adds eid tagging (GC scopes), a reset counter used by occlusion queries and support for deferred submission
	// through the RSX offloader thread (MTRSX). All completion paths report the event id to the resource manager.
	struct command_buffer_chunk : public mtl::command_list
	{
		u64 eid_tag = 0;
		u64 reset_id = 0;
		shared_mutex guard_mutex;

		// Set while the submission of this list is queued on the offloader thread (see mtl::queue_submit)
		atomic_t<bool> submit_queued = false;

		command_buffer_chunk() = default;

		inline void tag()
		{
			eid_tag = mtl::get_event_id();
		}

		// Waits for any previous use of this list to complete. Must be called before begin().
		void reset()
		{
			if (!poke())
			{
				wait(FRAME_PRESENT_TIMEOUT);
			}

			++reset_id;
		}

		// Non-blocking completion check. Returns true if the list is idle (not queued, not in flight).
		bool poke()
		{
			if (submit_queued)
			{
				// Not even committed yet
				return false;
			}

			reader_lock lock(guard_mutex);

			if (!m_is_pending)
			{
				return true;
			}

			if (!m_submit_fence.signaled())
			{
				return false;
			}

			lock.upgrade();

			if (m_is_pending)
			{
				m_is_pending = false;
				m_submit_fence.reset();
				mtl::on_event_completed(eid_tag);
				eid_tag = 0;
			}

			return true;
		}

		// Blocking wait. timeout_us == 0 waits forever. Returns false on timeout (the list stays pending).
		bool wait(u64 timeout_us = 0ull)
		{
			flush();

			reader_lock lock(guard_mutex);

			if (!m_is_pending)
			{
				return true;
			}

			if (!m_submit_fence.wait(timeout_us))
			{
				rsx_log.error("Metal: timed out waiting for command list '%s' (timeline value %llu)", m_label, m_submit_fence.value);
				return false;
			}

			lock.upgrade();

			if (m_is_pending)
			{
				m_is_pending = false;
				m_submit_fence.reset();
				mtl::on_event_completed(eid_tag);
				eid_tag = 0;
			}

			return true;
		}

		// Waits until the list has been handed to the Metal queue (deferred submissions only).
		void flush() const
		{
			utils::spin_wait(submit_queued, [](auto v)
			{
				return !v;
			});
		}

		// True if all work recorded before the given reset_id has completed on the GPU.
		bool is_complete(u64 sync_reset_id)
		{
			if (reset_id != sync_reset_id)
			{
				// The list has been recycled, which implies completion
				return true;
			}

			return !is_recording() && poke();
		}

		const std::string& label() const { return m_label; }

		// Teardown only (no thread will ever process this list again): a fatal error inside queue_submit() leaves
		// submit_queued set forever, and flush()/wait() would spin on it.
		void abandon_queued_submit()
		{
			if (submit_queued)
			{
				rsx_log.error("Metal: command list '%s' was never submitted (the RSX thread stopped mid-submission)", m_label);
				submit_queued.release(false);
			}
		}
	};

	struct occlusion_data
	{
		rsx::simple_array<u32> indices;
		command_buffer_chunk* command_buffer_to_wait = nullptr;
		u64 command_buffer_sync_id = 0;

		bool is_current(command_buffer_chunk* cmd) const
		{
			return (command_buffer_to_wait == cmd && command_buffer_sync_id == cmd->reset_id);
		}

		void set_sync_command_buffer(command_buffer_chunk* cmd)
		{
			command_buffer_to_wait = cmd;
			command_buffer_sync_id = cmd->reset_id;
		}

		void sync()
		{
			if (command_buffer_to_wait->reset_id == command_buffer_sync_id)
			{
				// Visibility results are only written once the render pass has been executed; wait for the list
				command_buffer_to_wait->wait();
			}
		}
	};

	struct frame_context_t
	{
		// The drawable acquired for this frame. Holds a +1 reference between nextDrawable() and present().
		CA::MetalDrawable* drawable = nullptr;

		rsx::flags32_t flags = 0;

		// The list whose completion ends the frame: the present list (present queue) of a presented frame, otherwise
		// the frame's last main-queue list. Everything the frame used may be recycled once it has completed.
		command_buffer_chunk* swap_command_buffer = nullptr;
		u64 swap_timeline_value = 0; // RSX timeline value signaled once the frame's main-queue work completes (0 if unknown)

		// Presentation through the present queue (MTLPresent.cpp). The present passes of the frame composite the output
		// into present_image (same size/format as the drawable) on the main queue; present_command_buffer waits for that
		// work and for the drawable on the present queue, then copies the image into the drawable. Both are owned by the
		// frame context and reused once swap_command_buffer has completed.
		std::unique_ptr<viewable_image> present_image;
		std::unique_ptr<command_buffer_chunk> present_command_buffer;

		data_heap_manager::managed_heap_snapshot_t heap_snapshot;
		u64 last_frame_sync_time = 0;

		// Copy shareable information
		void grab_resources(frame_context_t& other)
		{
			flags = other.flags;
			heap_snapshot = other.heap_snapshot;
		}

		void tag_frame_end()
		{
			heap_snapshot = data_heap_manager::get_heap_snapshot();
			last_frame_sync_time = rsx::get_shared_tag();
		}

		void reset_heap_ptrs()
		{
			last_frame_sync_time = 0;
			heap_snapshot.clear();
		}
	};

	struct flush_request_task
	{
		atomic_t<bool> pending_state{ false };  //Flush request status; true if rsx::thread is yet to service this request
		atomic_t<int> num_waiters{ 0 };  //Number of threads waiting for this request to be serviced
		bool hard_sync = false;

		flush_request_task() = default;

		void post(bool _hard_sync)
		{
			hard_sync = (hard_sync || _hard_sync);
			pending_state = true;
			num_waiters++;
		}

		void remove_one()
		{
			num_waiters--;
		}

		void clear_pending_flag()
		{
			hard_sync = false;
			pending_state.store(false);
		}

		bool pending() const
		{
			return pending_state.load();
		}

		void consumer_wait() const
		{
			utils::spin_wait(num_waiters, [](auto v)
			{
				return v == 0;
			});
		}

		void producer_wait() const
		{
			utils::spin_wait(pending_state, [](auto v)
			{
				return !v;
			});
		}
	};

	struct present_surface_info
	{
		u32 address;
		u32 format;
		u32 width;
		u32 height;
		u32 pitch;
		u8  eye;
	};

	struct draw_call_t
	{
		u32 subdraw_id;
	};

	template<int Count>
	class command_buffer_chain
	{
		atomic_t<u32> m_current_index = 0;
		std::array<mtl::command_buffer_chunk, Count> m_cb_list;

	public:
		command_buffer_chain() = default;

		void create(const mtl::render_device& dev, MTL4::CommandQueue* queue, mtl::timeline& tl, std::string_view label, mtl::command_list::access_type_hint access)
		{
			u32 index = 0;
			for (auto& cb : m_cb_list)
			{
				cb.create(dev, queue, tl, fmt::format("%s #%u", label, index++));
				cb.access_hint = access;
			}
		}

		void destroy()
		{
			for (auto& cb : m_cb_list)
			{
				cb.abandon_queued_submit();
				cb.wait(TEARDOWN_WAIT_TIMEOUT);
				cb.destroy();
			}
		}

		void abandon_queued_submits()
		{
			for (auto& cb : m_cb_list)
			{
				cb.abandon_queued_submit();
			}
		}

		void poke_all()
		{
			for (auto& cb : m_cb_list)
			{
				cb.poke();
			}
		}

		bool wait_all(u64 timeout_us = 0)
		{
			bool result = true;
			for (auto& cb : m_cb_list)
			{
				result = cb.wait(timeout_us) && result;
			}
			return result;
		}

		inline command_buffer_chunk* next()
		{
			const auto result_id = ++m_current_index % Count;
			auto result = &m_cb_list[result_id];

			if (!result->poke())
			{
				rsx_log.error("CB chain has run out of free entries!");
			}

			return result;
		}

		inline command_buffer_chunk* get()
		{
			return &m_cb_list[m_current_index % Count];
		}
	};
}
