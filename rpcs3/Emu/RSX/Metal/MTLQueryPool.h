#pragma once

#include "mtlutils/mtl_api.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Utilities/mutex.h"

#include <deque>
#include <memory>
#include <vector>

namespace mtl
{
	class buffer;
	class command_list;
	class render_device;
	struct command_buffer_chunk;

	// Occlusion query storage: a host-visible buffer used as the render pass visibility result buffer.
	// Every slot is 8 bytes (MTLVisibilityResultModeCounting writes a 64-bit sample count).
	// Render passes are created with MTLVisibilityResultTypeAccumulate so that a query survives render pass splits;
	// slots therefore start at zero (the pool is cleared on creation/reuse, never while in flight).
	class query_pool : public rsx::ref_counted
	{
		std::unique_ptr<mtl::buffer> m_buffer;
		u64* m_results = nullptr;
		u32 m_size = 0;

	public:
		static constexpr u32 slot_size = 8;

		query_pool(const mtl::render_device& dev, u32 size);
		~query_pool();

		query_pool(const query_pool&) = delete;
		query_pool& operator=(const query_pool&) = delete;

		const mtl::buffer* get() const { return m_buffer.get(); }
		MTL::Buffer* value() const;

		u64 result(u32 index) const { return m_results[index]; }

		// Clears all slots. Only call when the GPU no longer uses the pool.
		void reset();

		// Re-initializes a recycled pool: clears results and takes 'size' references again
		void rearm();

		inline u32 size() const
		{
			return m_size;
		}
	};

	class query_pool_manager
	{
		struct query_slot_info
		{
			query_pool* pool;
			bool any_passed;
			bool active;
			bool ready;
			u32 data;

			// Command list that recorded the query; results are final once it completes
			command_buffer_chunk* owner;
			u64 owner_sync_id;
		};

		class query_pool_ref
		{
			std::unique_ptr<query_pool> m_object;
			query_pool_manager* m_pool_man;

		public:
			query_pool_ref(query_pool_manager* pool_man, std::unique_ptr<query_pool>& pool)
				: m_object(std::move(pool))
				, m_pool_man(pool_man)
			{}

			~query_pool_ref();
		};

		std::vector<std::unique_ptr<query_pool>> m_consumed_pools;
		std::unique_ptr<query_pool> m_current_query_pool;
		std::deque<u32> m_available_slots;
		u32 m_pool_lifetime_counter = 0;

		// A slot may only be used once per pool (results accumulate and are never reset on the GPU timeline).
		// m_slot_generation[i] == m_pool_generation means slot i was already handed out from the current pool.
		std::vector<u32> m_slot_generation;
		u32 m_pool_generation = 0;

		std::deque<std::unique_ptr<query_pool>> m_query_pool_cache;
		shared_mutex m_query_pool_cache_lock;

		MTL::VisibilityResultMode m_result_mode = MTL::VisibilityResultModeCounting;

		const mtl::render_device* owner = nullptr;
		std::vector<query_slot_info> query_slot_status;

		bool poke_query(query_slot_info& query, u32 index);
		void allocate_new_pool();
		void reallocate_pool();
		void run_pool_cleanup();

	public:
		query_pool_manager(const mtl::render_device& dev, u32 num_entries);
		~query_pool_manager();

		// precise = count samples, otherwise a boolean "any sample passed" is enough
		void set_control_flags(bool precise);

		// Buffer to attach as the visibilityResultBuffer of every render pass that can run queries (may be null before
		// the first allocation). Changes only in allocate_query(), which ends the active render pass when it does.
		const MTL::Buffer* get_visibility_result_buffer() const;

		// `active_pass` is the render encoder of the main pass if one is open on `cmd`, otherwise nullptr.
		// A query that is open while a render pass (re)starts must be re-armed with resume_query().
		void begin_query(command_buffer_chunk& cmd, MTL4::RenderCommandEncoder* active_pass, u32 index);
		void end_query(command_buffer_chunk& cmd, MTL4::RenderCommandEncoder* active_pass, u32 index);
		void resume_query(MTL4::RenderCommandEncoder* active_pass, u32 index);

		bool check_query_status(u32 index);
		u32  get_query_result(u32 index);

		// GPU copy of `count` consecutive slots (8 bytes each) into dst. Ends the active render pass (results are only
		// written when a pass completes).
		void get_query_result_indirect(mtl::command_list& cmd, u32 index, u32 count, const mtl::buffer* dst, u64 dst_offset, u64 bytes_per_slot = query_pool::slot_size);

		// `active_pass` is ended (via cmd) when a new pool has to be started.
		u32 allocate_query(mtl::command_list& cmd);
		void free_query(mtl::command_list& /*cmd*/, u32 index);

		void on_query_pool_released(std::unique_ptr<mtl::query_pool>& pool);

		template<typename T>
			requires std::ranges::range<T> && std::same_as<std::ranges::range_value_t<T>, u32> // List of u32
		void free_queries(mtl::command_list& cmd, T& list)
		{
			for (const auto index : list)
			{
				free_query(cmd, index);
			}
		}
	};
}
