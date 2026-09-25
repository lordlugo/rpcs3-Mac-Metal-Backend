#pragma once

#include <util/types.hpp>
#include <functional>
#include <memory>
#include <utility>

namespace mtl
{
	// Type-erased owned object whose destruction is deferred until the GPU no longer references it.
	// Metal 4 command buffers do NOT retain resources, so anything referenced by in-flight work must go through here.
	class disposable_t
	{
		void* ptr;
		void (*deleter)(void*);

		disposable_t(void* ptr_, void (*deleter_)(void*)) :
			ptr(ptr_), deleter(deleter_) {}

	public:
		disposable_t() = delete;
		disposable_t(const disposable_t&) = delete;
		disposable_t& operator=(const disposable_t&) = delete;

		disposable_t(disposable_t&& other) noexcept :
			ptr(std::exchange(other.ptr, nullptr)),
			deleter(other.deleter)
		{}

		~disposable_t()
		{
			if (ptr)
			{
				deleter(ptr);
				ptr = nullptr;
			}
		}

		template <typename T>
		static disposable_t make(T* raw)
		{
			return disposable_t(raw, [](void* p)
			{
				delete static_cast<T*>(p);
			});
		}
	};

	struct garbage_collector
	{
		virtual ~garbage_collector() = default;
		virtual void dispose(mtl::disposable_t& object) = 0;
		virtual void add_exit_callback(std::function<void()> callback) = 0;

		template<typename T>
		void dispose(std::unique_ptr<T>& object)
		{
			if (!object)
			{
				return;
			}

			auto ptr = mtl::disposable_t::make(object.release());
			dispose(ptr);
		}
	};

	garbage_collector* get_gc();
}
