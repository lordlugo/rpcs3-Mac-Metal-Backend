#pragma once

// Single include point for metal-cpp (macOS 27 drop, 381.0.0) inside the Metal backend.
// The backend targets Metal 4 (MTL4*) on macOS 26+ / Apple silicon only.
//
// Conventions (see Metal/DESIGN.md):
//  - Objects returned by new*/alloc/copy are owned (+1) and must be released.
//  - Everything else is autoreleased; every thread that calls into Metal wraps work in mtl::autorelease_scope.
//  - The backend uses Metal 4 command queues: resources are UNTRACKED, so all hazards are handled with explicit
//    barriers (see mtl::command_list) and every allocation must be registered with the device residency set.

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#endif

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <MetalFX/MetalFX.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "util/types.hpp"
#include <string>
#include <string_view>

namespace mtl
{
	// Autorelease pool RAII. Metal-cpp does not have ARC; autoreleased temporaries leak without a pool.
	class autorelease_scope
	{
		NS::AutoreleasePool* m_pool;

	public:
		autorelease_scope() : m_pool(NS::AutoreleasePool::alloc()->init()) {}
		~autorelease_scope() { m_pool->release(); }

		autorelease_scope(const autorelease_scope&) = delete;
		autorelease_scope& operator=(const autorelease_scope&) = delete;
	};

	// Build an NS::String. Result is autoreleased (requires an active autorelease_scope).
	inline NS::String* ns_str(std::string_view str)
	{
		const std::string tmp(str);
		return NS::String::string(tmp.c_str(), NS::UTF8StringEncoding);
	}

	inline std::string to_string(const NS::String* str)
	{
		if (!str)
		{
			return {};
		}
		const char* c = str->utf8String();
		return c ? std::string(c) : std::string{};
	}

	inline std::string to_string(const NS::Error* err)
	{
		if (!err)
		{
			return "no error";
		}
		return to_string(err->localizedDescription());
	}

	// Minimal owning smart pointer for metal-cpp objects (+1 references).
	template <typename T>
	class ref
	{
		T* m_ptr = nullptr;

	public:
		ref() = default;
		explicit ref(T* owned) : m_ptr(owned) {}                  // Adopts a +1 reference
		ref(const ref& other) : m_ptr(other.m_ptr) { if (m_ptr) m_ptr->retain(); }
		ref(ref&& other) noexcept : m_ptr(other.m_ptr) { other.m_ptr = nullptr; }
		~ref() { reset(); }

		ref& operator=(const ref& other)
		{
			if (this != &other)
			{
				if (other.m_ptr) other.m_ptr->retain();
				reset();
				m_ptr = other.m_ptr;
			}
			return *this;
		}

		ref& operator=(ref&& other) noexcept
		{
			if (this != &other)
			{
				reset();
				m_ptr = other.m_ptr;
				other.m_ptr = nullptr;
			}
			return *this;
		}

		static ref retain(T* unowned)
		{
			if (unowned) unowned->retain();
			return ref(unowned);
		}

		void reset(T* owned = nullptr)
		{
			if (m_ptr) m_ptr->release();
			m_ptr = owned;
		}

		T* release_ownership()
		{
			T* p = m_ptr;
			m_ptr = nullptr;
			return p;
		}

		T* get() const { return m_ptr; }
		T* operator->() const { return m_ptr; }
		explicit operator bool() const { return m_ptr != nullptr; }
	};

	// Stage masks used by the conservative barrier model.
	constexpr MTL::Stages stages_render = MTL::StageVertex | MTL::StageFragment | MTL::StageTile;
	constexpr MTL::Stages stages_compute = MTL::StageDispatch | MTL::StageBlit;
	constexpr MTL::Stages stages_all_work = stages_render | stages_compute;
	// "After" scope for queue barriers: everything that may have been encoded before, including work encoded by
	// frameworks (MetalFX may use machine-learning / resource-state stages).
	constexpr MTL::Stages stages_all_producers = stages_all_work | MTL::StageMachineLearning | MTL::StageResourceState;
	// Fragment shading and attachment load/store of render passes
	constexpr MTL::Stages stages_fragment_work = MTL::StageFragment | MTL::StageTile;
}
