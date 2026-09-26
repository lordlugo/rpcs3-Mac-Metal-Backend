#pragma once

// Port of VK/VKPipelineCompiler: a pool of "RSX.W" worker threads that translate shaders (GLSL -> MSL -> MTLLibrary)
// and build Metal 4 pipeline state objects, either inline (COMPILE_INLINE) or deferred with a completion callback.
// Deferred jobs go to one queue shared by all workers.

#include "MTLProgramPipeline.h"
#include "MTLPipelineArchive.h" // Pipeline persistence hooks used by MTLGSRender (initialize/preloaded/flush)
#include "Utilities/lockless.h"
#include "Emu/RSX/Common/simple_array.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mtl
{
	class render_device;

	void destroy_pipe_compiler();

	// The only primitive topology class a render pipeline declares is Point (see build_graphics_program): line and
	// triangle pipelines are identical. Pipeline keys (graphics_pipeline_state::topology_class) therefore use
	// MTL::PrimitiveTopologyClassTriangle for every non-point topology, so the same pipeline is not built twice.
	inline u8 get_pipeline_topology_class(u8 topology_class)
	{
		return topology_class == static_cast<u8>(MTL::PrimitiveTopologyClassPoint)
			? topology_class
			: static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
	}

	// Vertex shaders that declare input attributes (GLSL `layout(location = N) in ...`, e.g. overlay passes) get an
	// automatic MTLVertexDescriptor: all attributes interleaved in ONE buffer, ascending location order, tightly packed
	// (4-byte aligned), stride = packed size, per-vertex step. Bind that buffer after program::bind() with
	// cmd.argument_table(table_vertex)->setAddress(address, stage_in_vertex_buffer_index).
	// RSX vertex programs pull vertices from texel buffers and never use this.
	constexpr u32 stage_in_vertex_buffer_index = 30;

	class pipe_compiler
	{
	public:
		enum op_flag_bits
		{
			COMPILE_DEFAULT = 0,
			COMPILE_INLINE = 1,
			COMPILE_DEFERRED = 2,
			SEPARATE_SHADER_OBJECTS = 4,   // Accepted for parity with VK; Metal stages always have separate binding tables
			USE_LAST_PROVOKING_VERTEX = 8  // Unsupported by Metal (first vertex provokes); ignored, logged once
		};

		using op_flags = rsx::flags32_t;

		using callback_t = std::function<void(std::unique_ptr<glsl::program>&)>;

		pipe_compiler();
		~pipe_compiler();

		void initialize(const mtl::render_device* pdev);

		// Compute pipeline. The shader must have been create()d; it is translated (once) by whichever thread gets
		// to it first. The shader object must outlive any deferred job referencing it.
		std::unique_ptr<glsl::program> compile(
			glsl::shader* cs,
			op_flags flags, callback_t callback = {},
			const std::vector<glsl::program_input>& cs_inputs = {});

		// Graphics pipeline from two created shaders + baked fixed-function state (RSX pipelines).
		// Deferred jobs return nullptr immediately and deliver the program (or nullptr on failure) through `callback`
		// on the worker thread. Inline jobs return the program (nullptr on failure) and ignore `callback`.
		std::unique_ptr<glsl::program> compile(
			const mtl::pipeline_props& create_info,
			glsl::shader* vs,
			glsl::shader* fs,
			op_flags flags, callback_t callback = {},
			const std::vector<glsl::program_input>& vs_inputs = {},
			const std::vector<glsl::program_input>& fs_inputs = {});

		// Worker thread entry
		void operator()();

		// Number of deferred jobs finished so far (built or failed). It is bumped after the job's callback returned,
		// so a pipeline that was not in the cache when this was sampled is there once the count has changed.
		static u32 get_completed_job_count();

		// Waits until the completed job count differs from `count`, at most `timeout_us`. May return early.
		static void wait_for_completed_job(u32 count, u64 timeout_us);

		// A thread is about to wait for a pipeline of this shader pair: its queued jobs move to the front of the queue
		// and run at emulation thread priority. No effect on jobs that are already running.
		static void prioritize_jobs(const glsl::shader* vs, const glsl::shader* fs);

	private:
		friend void destroy_pipe_compiler();

		struct pipe_compiler_job
		{
			bool is_graphics_job;
			bool urgent = false;          // Waited for (prioritize_jobs): runs at emulation thread priority
			callback_t callback_func;

			mtl::pipeline_props graphics_data{};
			glsl::shader* shaders[2]{};   // [vs, fs] or [cs, nullptr]
			std::vector<glsl::program_input> inputs[2];

			op_flags flags;

			pipe_compiler_job(
				const mtl::pipeline_props& props,
				glsl::shader* vs,
				glsl::shader* fs,
				const std::vector<glsl::program_input>& vs_in,
				const std::vector<glsl::program_input>& fs_in,
				op_flags flags_,
				callback_t func)
				: is_graphics_job(true)
				, callback_func(std::move(func))
				, graphics_data(props)
				, shaders{ vs, fs }
				, inputs{ vs_in, fs_in }
				, flags(flags_)
			{
			}

			pipe_compiler_job(
				glsl::shader* cs,
				const std::vector<glsl::program_input>& cs_in,
				op_flags flags_,
				callback_t func)
				: is_graphics_job(false)
				, callback_func(std::move(func))
				, shaders{ cs, nullptr }
				, inputs{ cs_in, {} }
				, flags(flags_)
			{
			}
		};

		// Deferred jobs of all workers (MTLPipelineCompiler.cpp). One queue, so that an idle worker never waits
		// behind a slow compile that happened to be queued on another worker.
		struct job_queue;
		static job_queue s_queue;

		const mtl::render_device* m_device = nullptr;

		std::unique_ptr<glsl::program> int_compile_compute_pipe(
			glsl::shader* cs,
			const std::vector<glsl::program_input>& cs_inputs,
			op_flags flags);

		std::unique_ptr<glsl::program> int_compile_graphics_pipe(
			const mtl::pipeline_props& create_info,
			glsl::shader* vs,
			glsl::shader* fs,
			const std::vector<glsl::program_input>& vs_inputs,
			const std::vector<glsl::program_input>& fs_inputs,
			op_flags flags);
	};

	// 0 (or negative) = automatic count (one worker per concurrent Metal compilation task, see the .cpp). Always capped
	// by device caps().max_compile_tasks.
	void initialize_pipe_compiler(int num_worker_threads = -1);
	void destroy_pipe_compiler(); // Joins the workers and drops the jobs that were still queued (callbacks never run)
	pipe_compiler* get_pipe_compiler();

	// ---- Synchronous building blocks (no worker threads needed; used by the workers and glsl::create_*_program) ----
	// Translate the shader(s) if needed and build the program. nullptr (logged) on failure. Thread-safe.
	std::unique_ptr<glsl::program> build_compute_program(
		glsl::shader& cs,
		const std::vector<glsl::program_input>& cs_inputs);

	std::unique_ptr<glsl::program> build_graphics_program(
		glsl::shader& vs,
		glsl::shader& fs,
		const glsl::graphics_pipeline_state& state,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs);
}
