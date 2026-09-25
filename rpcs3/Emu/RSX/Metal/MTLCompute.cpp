#include "stdafx.h"
#include "MTLCompute.h"
#include "MTLHelpers.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/device.h"

namespace mtl
{
	std::unordered_map<u32, std::unique_ptr<mtl::compute_task>> g_compute_tasks;

	std::vector<glsl::program_input> compute_task::get_inputs()
	{
		std::vector<glsl::program_input> result;
		for (unsigned i = 0; i < ssbo_count; ++i)
		{
			result.push_back(glsl::program_input::make
			(
				::glsl::glsl_compute_program,
				"ssbo" + std::to_string(i),
				glsl::program_input_type::input_type_storage_buffer,
				glsl::binding_set_index_compute,
				i
			));
		}

		if (use_push_constants && push_constants_size > 0)
		{
			result.push_back(glsl::program_input::make
			(
				::glsl::glsl_compute_program,
				"push_constants",
				glsl::program_input_type::input_type_push_constant,
				glsl::binding_set_index_compute,
				umax,
				glsl::push_constant_ref{ .offset = 0, .size = push_constants_size }
			));
		}

		return result;
	}

	void compute_task::create()
	{
		if (!initialized)
		{
			// Apple GPUs (VK: MVK / HONEYKRISP branch). SIMD width is 32; 256-wide groups keep occupancy high
			// for these memory-bound kernels.
			unroll_loops = true;
			optimal_kernel_size = 1;
			optimal_group_size = 256;

			if (g_render_device)
			{
				optimal_group_size = std::min(optimal_group_size, g_render_device->caps().max_threads_per_threadgroup);
			}

			// Metal has no small per-dimension grid limit, but keep the VK split heuristic for very large 1D jobs
			max_invocations_x = 65535;

			initialized = true;
		}
	}

	void compute_task::destroy()
	{
		if (initialized)
		{
			m_program.reset();
			initialized = false;
		}
	}

	void compute_task::push_constants(u32 offset, u32 size, const void* data)
	{
		ensure(m_program);
		m_program->push_constants(glsl::binding_set_index_compute, offset, size, data);
	}

	void compute_task::load_program(mtl::command_list& cmd)
	{
		if (!m_program)
		{
			m_program = glsl::create_compute_program(m_src, get_inputs());
			if (!m_program)
			{
				fmt::throw_exception("Metal: failed to build compute task.\n%s", m_src);
			}

			const u32 group_x = workgroup_size_x ? workgroup_size_x : optimal_group_size;
			const u32 threads_per_group = group_x * workgroup_size_y * workgroup_size_z;
			if (threads_per_group > m_program->max_total_threads_per_threadgroup())
			{
				// The GLSL local_size is baked into the kernel; a smaller dispatch would break gl_WorkGroupSize math.
				fmt::throw_exception("Metal: compute task requires %u threads per threadgroup but the pipeline supports only %u",
					threads_per_group, m_program->max_total_threads_per_threadgroup());
			}
		}

		bind_resources(cmd);

		// Sets the pipeline on cmd.compute() (ends any open render pass, orders against the previous compute command),
		// uploads push constants and binds the argument table.
		m_program->bind(cmd, get_scratch_heap());
	}

	void compute_task::run(mtl::command_list& cmd, u32 invocations_x, u32 invocations_y, u32 invocations_z)
	{
		if (!invocations_x || !invocations_y || !invocations_z)
		{
			return;
		}

		load_program(cmd);

		const u32 group_x = workgroup_size_x ? workgroup_size_x : optimal_group_size;

		// The barrier for this dispatch was recorded by program::bind() (cmd.compute()); do not add another one.
		cmd.compute_unordered()->dispatchThreadgroups(
			MTL::Size(invocations_x, invocations_y, invocations_z),
			MTL::Size(group_x, workgroup_size_y, workgroup_size_z));
	}

	void compute_task::run(mtl::command_list& cmd, u32 num_invocations)
	{
		u32 invocations_x, invocations_y;
		if (num_invocations > max_invocations_x)
		{
			// Split the 1D job into 2 dimensions (kernels linearize gl_GlobalInvocationID themselves)
			invocations_x = static_cast<u32>(floor(std::sqrt(num_invocations)));
			invocations_y = invocations_x;

			if (num_invocations % invocations_x) invocations_y++;
		}
		else
		{
			invocations_x = num_invocations;
			invocations_y = 1;
		}

		run(cmd, invocations_x, invocations_y, 1);
	}

	cs_shuffle_base::cs_shuffle_base()
	{
		// Metal has no robust buffer access: every shuffle kernel bounds-checks against the job length (params[0].x bytes)
		use_push_constants = true;
		push_constants_size = 16;

		variables =
			"	uint block_length = params[0].x >> 2;\n";

		work_kernel =
			"		if (index >= block_length)\n"
			"			return;\n"
			"\n"
			"		value = data[index];\n"
			"		data[index] = %f(value);\n";

		loop_advance =
			"		index++;\n";

		suffix =
			"}\n";
	}

	void cs_shuffle_base::build(const char* function_name, u32 _kernel_size)
	{
		// Initialize to allow detecting optimal settings
		create();

		kernel_size = _kernel_size? _kernel_size : optimal_kernel_size;

		m_src =
		#include "../Program/GLSLSnippets/ShuffleBytes.glsl"
		;

		const auto parameters_size = utils::align(push_constants_size, 16) / 16;
		const std::pair<std::string_view, std::string> syntax_replace[] =
		{
			{ "%loc", "0" },
			{ "%set", "set = 0"},
			{ "%ws", std::to_string(optimal_group_size) },
			{ "%ks", std::to_string(kernel_size) },
			{ "%vars", variables },
			{ "%f", function_name },
			{ "%md", method_declarations },
			{ "%ub", use_push_constants? "layout(push_constant) uniform ubo{ uvec4 params[" + std::to_string(parameters_size) + "]; };\n" : "" },
		};

		m_src = fmt::replace_all(m_src, syntax_replace);
		work_kernel = fmt::replace_all(work_kernel, syntax_replace);

		if (kernel_size <= 1)
		{
			m_src += "	{\n" + work_kernel + "	}\n";
		}
		else if (unroll_loops)
		{
			work_kernel += loop_advance + "\n";

			m_src += std::string
			(
				"	//Unrolled loop\n"
				"	{\n"
			);

			// Assemble body with manual loop unroll to try loweing GPR usage
			for (u32 n = 0; n < kernel_size; ++n)
			{
				m_src += work_kernel;
			}

			m_src += "	}\n";
		}
		else
		{
			m_src += "	for (int loop = 0; loop < KERNEL_SIZE; ++loop)\n";
			m_src += "	{\n";
			m_src += work_kernel;
			m_src += loop_advance;
			m_src += "	}\n";
		}

		m_src += suffix;
	}

	void cs_shuffle_base::bind_resources(mtl::command_list& cmd)
	{
		set_parameters(cmd);
		m_program->bind_uniform({ m_data, m_data_offset, m_data_length }, glsl::binding_set_index_compute, 0);
	}

	void cs_shuffle_base::set_parameters(mtl::command_list& /*cmd*/)
	{
		if (!m_params.empty())
		{
			ensure(use_push_constants);
			push_constants(0, m_params.size_bytes32(), m_params.data());
		}
	}

	void cs_shuffle_base::run(mtl::command_list& cmd, const mtl::buffer* data, u32 data_length, u32 data_offset)
	{
		m_params = { data_length, 0, 0, 0 };
		run_impl(cmd, data, data_length, data_offset);
	}

	void cs_shuffle_base::run_impl(mtl::command_list& cmd, const mtl::buffer* data, u32 data_length, u32 data_offset)
	{
		m_data = data;
		m_data_offset = data_offset;
		m_data_length = data_length;

		const auto num_bytes_per_invocation = optimal_group_size * kernel_size * 4;
		const auto num_bytes_to_process = rsx::align2(data_length, num_bytes_per_invocation);
		const auto num_invocations = num_bytes_to_process / num_bytes_per_invocation;

		if ((data_length + data_offset) > data->size())
		{
			// No robust buffer access on Metal: the kernel bounds check uses data_length, so the range itself must fit
			rsx_log.error("Inadequate buffer length submitted for a compute operation."
				"Required=%d bytes, Available=%d bytes", data_length + data_offset, data->size());
		}

		compute_task::run(cmd, num_invocations);
	}

	cs_interleave_task::cs_interleave_task()
	{
		use_push_constants = true;
		push_constants_size = 16;

		variables =
			"	uint block_length = params[0].x >> 2;\n"
			"	uint z_offset = params[0].y >> 2;\n"
			"	uint s_offset = params[0].z >> 2;\n"
			"	uint depth;\n"
			"	uint stencil;\n"
			"	uint stencil_shift;\n"
			"	uint stencil_offset;\n";
	}

	void cs_interleave_task::bind_resources(mtl::command_list& cmd)
	{
		set_parameters(cmd);
		m_program->bind_uniform({ m_data, m_data_offset, m_ssbo_length }, glsl::binding_set_index_compute, 0);
	}

	void cs_interleave_task::run(mtl::command_list& cmd, const mtl::buffer* data, u32 data_offset, u32 data_length, u32 zeta_offset, u32 stencil_offset)
	{
		m_params = { data_length, zeta_offset - data_offset, stencil_offset - data_offset, 0 };

		ensure(stencil_offset > data_offset);
		m_ssbo_length = stencil_offset + (data_length / 4) - data_offset;
		cs_shuffle_base::run_impl(cmd, data, data_length, data_offset);
	}

	cs_scatter_d24x8::cs_scatter_d24x8()
	{
		work_kernel =
			"		if (index >= block_length)\n"
			"			return;\n"
			"\n"
			"		value = data[index];\n"
			"		data[index + z_offset] = (value >> 8);\n"
			"		stencil_offset = (index / 4);\n"
			"		stencil_shift = (index % 4) * 8;\n"
			"		stencil = (value & 0xFF) << stencil_shift;\n"
			"		atomicOr(data[stencil_offset + s_offset], stencil);\n";

		cs_shuffle_base::build("");
	}

	cs_aggregator::cs_aggregator()
	{
		ssbo_count = 2;

		// MSL runtime-sized arrays have no length(); pass the element count explicitly
		use_push_constants = true;
		push_constants_size = 16;

		create();

		m_src =
			"#version 450\n"
			"layout(local_size_x = %ws, local_size_y = 1, local_size_z = 1) in;\n\n"

			"layout(set=0, binding=0, std430) readonly buffer ssbo0{ uint src[]; };\n"
			"layout(set=0, binding=1, std430) writeonly buffer ssbo1{ uint result; };\n"
			"layout(push_constant) uniform aggregator_params{ uint word_count; };\n\n"

			"void main()\n"
			"{\n"
			"	if (gl_GlobalInvocationID.x < word_count)\n"
			"	{\n"
			"		atomicAdd(result, src[gl_GlobalInvocationID.x]);\n"
			"	}\n"
			"}\n";

		const std::pair<std::string_view, std::string> syntax_replace[] =
		{
			{ "%ws", std::to_string(optimal_group_size) },
		};

		m_src = fmt::replace_all(m_src, syntax_replace);
	}

	void cs_aggregator::bind_resources(mtl::command_list& /*cmd*/)
	{
		const u32 params[4] = { word_count, 0, 0, 0 };
		push_constants(0, sizeof(params), params);

		m_program->bind_uniform({ src, 0, block_length }, glsl::binding_set_index_compute, 0);
		m_program->bind_uniform({ dst, 0, 4 }, glsl::binding_set_index_compute, 1);
	}

	void cs_aggregator::run(mtl::command_list& cmd, const mtl::buffer* dst, const mtl::buffer* src, u32 num_words)
	{
		this->dst = dst;
		this->src = src;
		word_count = num_words;
		block_length = num_words * 4;

		const u32 linear_invocations = utils::aligned_div(word_count, optimal_group_size);
		compute_task::run(cmd, linear_invocations);
	}

	void reset_compute_tasks()
	{
		// Compute tasks own no per-frame resources (parameters live in the scratch heap)
	}

	void destroy_compute_tasks()
	{
		for (auto& [key, task] : g_compute_tasks)
		{
			task->destroy();
		}

		g_compute_tasks.clear();
	}
}
