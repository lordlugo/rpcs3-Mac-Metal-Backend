#pragma once

// Port of VK/VKPipelineCompiler: a pool of "RSX.W" worker threads that translate shaders (GLSL -> MSL -> MTLLibrary)
// and build Metal 4 pipeline state objects, either inline (COMPILE_INLINE) or deferred with a completion callback.

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

	private:
		struct pipe_compiler_job
		{
			bool is_graphics_job;
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

		const mtl::render_device* m_device = nullptr;
		lf_queue<pipe_compiler_job> m_work_queue;

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

	// 0 (or negative) = automatic count (same heuristic as VK). Always capped by device caps().max_compile_tasks.
	void initialize_pipe_compiler(int num_worker_threads = -1);
	void destroy_pipe_compiler();
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
