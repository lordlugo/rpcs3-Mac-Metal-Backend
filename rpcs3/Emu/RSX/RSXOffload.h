#pragma once

#include "util/types.hpp"
#include "Utilities/address_range.h"
#include "gcm_enums.h"

#include <vector>

template <typename T>
class named_thread;

namespace rsx
{
	class dma_manager
	{
		enum op
		{
			raw_copy = 0,
			vector_copy = 1,
			index_emulate = 2,
			callback = 3
		};

		struct transport_packet
		{
			op type{};
			std::vector<u8> opt_storage{};
			void* src{};
			void* dst{};
			u32 length{};
			u32 aux_param0{};
			u32 aux_param1{};

			transport_packet(void *_dst, void *_src, u32 len)
				: type(op::raw_copy), src(_src), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, std::vector<u8>& _src, u32 len)
				: type(op::vector_copy), opt_storage(std::move(_src)), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, rsx::primitive_type prim, u32 len)
				: type(op::index_emulate), dst(_dst), length(len), aux_param0(static_cast<u8>(prim))
			{}

			transport_packet(u32 command, void* args)
				: type(op::callback), src(args), aux_param0(command)
			{}

			transport_packet(const transport_packet&) = delete;
			transport_packet& operator=(const transport_packet&) = delete;
		};

		atomic_t<bool> m_mem_fault_flag = false;

		struct offload_thread;
		std::shared_ptr<named_thread<offload_thread>> m_thread;

#if defined(__APPLE__) && defined(ARCH_ARM64)
		// Apple silicon copies tens of GB/s per core, so small copies finish long before a hand-over to the worker pays
		// off (queue node allocation, cache line transfers between cores, and a wake-up when the worker sleeps).
		// Only large copies are offloaded; the bar is higher while the worker sleeps (waking it costs microseconds).
		static constexpr u32 max_immediate_transfer_size = 16 * 1024;
		static constexpr u32 max_immediate_transfer_size_idle = 128 * 1024;
#else
		// TODO: Improved benchmarks here; value determined by profiling on a Ryzen CPU, rounded to the nearest 512 bytes
		static constexpr u32 max_immediate_transfer_size = 3584;
		static constexpr u32 max_immediate_transfer_size_idle = 3584;
#endif

		// True if a transfer of this size should be done by the calling thread
		bool is_immediate_transfer(u32 length) const;

	public:
		dma_manager() = default;

		// initialization
		void init();

		// General tranport
		void copy(void *dst, std::vector<u8>& src, u32 length) const;
		void copy(void *dst, void *src, u32 length) const;

		// Vertex utilities
		void emulate_as_indexed(void *dst, rsx::primitive_type primitive, u32 count);

		// Renderer callback
		void backend_ctrl(u32 request_code, void* args);

		// Synchronization
		bool is_current_thread() const;
		bool sync() const;
		void join();
		void set_mem_fault_flag();
		void clear_mem_fault_flag();

		// Fault recovery
		utils::address_range32 get_fault_range(bool writing) const;
	};
}
