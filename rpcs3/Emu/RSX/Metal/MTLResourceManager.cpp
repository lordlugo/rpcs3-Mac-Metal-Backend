#include "stdafx.h"
#include "MTLResourceManager.h"

namespace mtl
{
	static resource_manager g_resource_manager;
	static atomic_t<u64> g_event_ctr;
	static atomic_t<u64> g_last_completed_event;

	resource_manager* get_resource_manager()
	{
		return &g_resource_manager;
	}

	garbage_collector* get_gc()
	{
		return &g_resource_manager;
	}

	u64 get_event_id()
	{
		return g_event_ctr++;
	}

	u64 current_event_id()
	{
		return g_event_ctr.load();
	}

	u64 last_completed_event_id()
	{
		return g_last_completed_event.load();
	}

	void on_event_completed(u64 event_id)
	{
		g_resource_manager.eid_completed(event_id);

		g_last_completed_event.atomic_op([event_id](u64& value)
		{
			value = std::max(event_id, value);
		});
	}

	eid_scope_t& resource_manager::get_current_eid_scope()
	{
		const auto eid = current_event_id();
		if (m_eid_map.empty() || m_eid_map.back().eid != eid)
		{
			m_eid_map.emplace_back(eid);
		}
		return m_eid_map.back();
	}

	void resource_manager::destroy()
	{
		flush();

		for (const auto& callback : m_exit_handlers)
		{
			callback();
		}
		m_exit_handlers.clear();
	}

	void resource_manager::flush()
	{
		std::list<eid_scope_t> dispose_queue;
		{
			std::lock_guard lock(m_eid_map_lock);
			dispose_queue.splice(dispose_queue.begin(), m_eid_map);
		}

		m_sampler_pool.clear();
	}

	mtl::sampler* resource_manager::get_sampler(const mtl::render_device& dev, mtl::sampler* previous, const sampler_create_info& info)
	{
		if (previous)
		{
			auto as_cached_object = static_cast<cached_sampler_object_t*>(previous);
			ensure(as_cached_object->has_refs());
			as_cached_object->release();
		}

		if (const auto found = m_sampler_pool.find(info))
		{
			found->add_ref();
			return found;
		}

		auto result = std::make_unique<cached_sampler_object_t>(dev, info);
		auto ret = m_sampler_pool.emplace(result);
		ret->add_ref();
		return ret;
	}

	void resource_manager::add_exit_callback(std::function<void()> callback)
	{
		m_exit_handlers.push_back(std::move(callback));
	}

	void resource_manager::dispose(mtl::disposable_t& disposable)
	{
		std::lock_guard lock(m_eid_map_lock);
		get_current_eid_scope().m_disposables.emplace_back(std::move(disposable));
	}

	void resource_manager::push_down_current_scope()
	{
		std::lock_guard lock(m_eid_map_lock);
		get_current_eid_scope().eid++;
	}

	void resource_manager::eid_completed(u64 eid)
	{
		std::list<eid_scope_t> discarded_scopes;
		{
			reader_lock lock(m_eid_map_lock);

			auto newest_it = m_eid_map.begin();
			while (newest_it != m_eid_map.end() && newest_it->eid <= eid)
			{
				newest_it++;
			}

			if (newest_it == m_eid_map.begin())
			{
				return;
			}

			lock.upgrade();
			discarded_scopes.splice(discarded_scopes.end(), m_eid_map, m_eid_map.begin(), newest_it);
		}

		// Destruction (and residency eviction) happens here, outside the lock.
	}

	void resource_manager::trim()
	{
		// Keep the number of idle samplers bounded
		constexpr usz max_idle_samplers = 1024;
		if (m_sampler_pool.size() > max_idle_samplers)
		{
			auto unused = m_sampler_pool.collect([](const cached_sampler_object_t& sampler)
			{
				return !sampler.has_refs();
			});

			for (auto& object : unused)
			{
				dispose(object);
			}
		}
	}
}
