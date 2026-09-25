#pragma once

#include "mtlutils/garbage_collector.h"
#include "mtlutils/sampler.h"
#include "Utilities/mutex.h"

#include <list>
#include <memory>
#include <vector>

namespace mtl
{
	// Event ids ("eid") tag GPU work submissions. Resources disposed while eid N is current are destroyed once
	// the command list tagged with eid >= N has completed (mirrors the VK backend's scheme).
	u64 get_event_id();
	u64 current_event_id();
	u64 last_completed_event_id();
	void on_event_completed(u64 event_id);

	struct eid_scope_t
	{
		u64 eid;
		std::vector<disposable_t> m_disposables;

		explicit eid_scope_t(u64 _eid) : eid(_eid) {}
		~eid_scope_t() { discard(); }

		void discard() { m_disposables.clear(); }
	};

	class resource_manager : public garbage_collector
	{
	private:
		sampler_pool_t m_sampler_pool;

		std::list<eid_scope_t> m_eid_map;
		mutable shared_mutex m_eid_map_lock;

		std::vector<std::function<void()>> m_exit_handlers;

		eid_scope_t& get_current_eid_scope();

	public:
		resource_manager() = default;
		~resource_manager() override = default;

		void destroy();
		void flush();

		// Reference-counted sampler cache. Pass the previously bound sampler (or nullptr) so its ref is dropped.
		mtl::sampler* get_sampler(const mtl::render_device& dev, mtl::sampler* previous, const sampler_create_info& info);

		void add_exit_callback(std::function<void()> callback) override;
		void dispose(mtl::disposable_t& disposable) override;
		using garbage_collector::dispose;

		void push_down_current_scope();
		void eid_completed(u64 eid);
		void trim();
	};

	resource_manager* get_resource_manager();
}
