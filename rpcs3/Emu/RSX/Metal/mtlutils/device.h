#pragma once

#include "mtl_api.h"
#include "Utilities/mutex.h"

#include <string>
#include <vector>

namespace mtl
{
	struct gpu_capabilities
	{
		bool metal4 = false;                 // MTLGPUFamilyMetal4 (Apple7+/M1+) — required
		bool apple8 = false;                 // M2 class
		bool apple9 = false;                 // M3/M4 class
		bool apple10 = false;                // M5/A19 class
		bool depth_bounds = false;           // setDepthTestBounds (Apple10 only)
		bool bc_texture_compression = false; // BC1-7 on Apple silicon Macs
		bool unified_memory = true;
		bool placement_sparse = false;
		u32 max_texture_size_2d = 16384;
		u32 max_texture_size_3d = 2048;
		u32 max_threads_per_threadgroup = 1024;
		u32 max_compile_tasks = 2;
		u64 max_buffer_length = 0;
		u64 recommended_working_set = 0;
		// Metal limits we must respect when laying out argument tables
		static constexpr u32 max_buffers_per_stage = 31;
		static constexpr u32 max_textures_per_stage = 128;
		static constexpr u32 max_samplers_per_stage = 16;
	};

	// Owns the Metal device, the Metal 4 command queue(s), the shader compiler and the global residency set.
	class render_device
	{
		MTL::Device* m_device = nullptr;
		MTL4::CommandQueue* m_queue = nullptr;        // Graphics + compute + copy (one queue keeps ordering simple)
		MTL4::CommandQueue* m_async_queue = nullptr;  // Optional async compute/transfer queue
		MTL4::Compiler* m_compiler = nullptr;
		MTL::ResidencySet* m_residency = nullptr;

		shared_mutex m_residency_lock;
		bool m_residency_dirty = false;
		u64 m_residency_count = 0;
		// Allocations removed from m_residency whose removal has not been committed yet. The residency set may still
		// reference them until commit(), so they are kept alive (retained) until then.
		std::vector<MTL::Allocation*> m_pending_evictions;

		gpu_capabilities m_caps{};
		std::string m_name;

	public:
		render_device() = default;
		~render_device();

		render_device(const render_device&) = delete;
		render_device& operator=(const render_device&) = delete;

		// Creates device, queues, compiler and residency set. Returns false if Metal 4 is unavailable.
		bool create();
		void destroy();

		MTL::Device* handle() const { return m_device; }
		MTL4::CommandQueue* queue() const { return m_queue; }
		MTL4::CommandQueue* async_queue() const { return m_async_queue; }
		MTL4::Compiler* compiler() const { return m_compiler; }
		MTL::ResidencySet* residency_set() const { return m_residency; }

		const gpu_capabilities& caps() const { return m_caps; }
		const std::string& name() const { return m_name; }

		// Residency management. Every MTL::Buffer / MTL::Texture / MTL::Heap the GPU touches must be registered.
		// Views and texture-buffers created from a registered parent do not need registering.
		void make_resident(const MTL::Allocation* allocation);
		void evict(const MTL::Allocation* allocation);   // Call only once the GPU has finished with the allocation
		void commit_residency();                          // Called by command_list::submit before committing work
		void attach_residency_set(const MTL::ResidencySet* set); // Extra sets (e.g. CAMetalLayer::residencySet)

		u64 allocated_bytes() const;
	};

	// Set by MTLGSRender during initialization; valid for the renderer's lifetime.
	extern render_device* g_render_device;

	inline render_device& get_device() { return *g_render_device; }
}
