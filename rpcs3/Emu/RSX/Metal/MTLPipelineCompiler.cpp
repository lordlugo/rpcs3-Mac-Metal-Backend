#include "stdafx.h"
#include "MTLPipelineCompiler.h"
#include "mtlutils/device.h"

#include "Emu/system_config.h"
#include "Utilities/Thread.h"
#include "util/sysinfo.hpp"
#include "util/asm.hpp"
#include "util/fnv_hash.hpp"

#include <algorithm>
#include <deque>
#include <mutex>
#include <optional>

namespace mtl
{
	// Global list of worker threads
	static std::unique_ptr<named_thread_group<pipe_compiler>> g_pipe_compilers;

	struct pipe_compiler::job_queue
	{
		std::mutex mutex;
		std::deque<pipe_compiler_job> jobs;   // Oldest first, except for jobs moved to the front by prioritize_jobs()
		atomic_t<u32> pushed = 0;             // Bumped by every push; idle workers wait on it
		atomic_t<u32> completed = 0;          // Bumped after every finished deferred job (see get_completed_job_count)

		void push(pipe_compiler_job&& job)
		{
			{
				std::lock_guard lock(mutex);
				jobs.push_back(std::move(job));
			}

			pushed++;
			pushed.notify_one();
		}

		std::optional<pipe_compiler_job> pop()
		{
			std::lock_guard lock(mutex);
			if (jobs.empty())
			{
				return std::nullopt;
			}

			std::optional<pipe_compiler_job> result(std::move(jobs.front()));
			jobs.pop_front();
			return result;
		}
	};

	pipe_compiler::job_queue pipe_compiler::s_queue;

	namespace
	{
		bool use_fast_math()
		{
			return !g_cfg.video.disable_msl_fast_math;
		}

		// FNV-1a over 64-bit words (then bytes), length first
		u64 hash_bytes(u64 hash, const void* data, usz size)
		{
			const auto* bytes = static_cast<const u8*>(data);
			hash = rpcs3::hash64(hash, u64{size});

			for (; size >= sizeof(u64); bytes += sizeof(u64), size -= sizeof(u64))
			{
				u64 word;
				std::memcpy(&word, bytes, sizeof(word));
				hash = rpcs3::hash64(hash, word);
			}

			for (; size; bytes++, size--)
			{
				hash = rpcs3::hash64(hash, *bytes);
			}

			return hash;
		}

		// Pipeline key for the pipeline archive (mtl::new_render_pipeline_state): the descriptors built below are a
		// deterministic function of the translated shaders (MSL, which reflects the binding layout) and `state`
		// (fast math has separate archives). Never 0, which means "unknown".
		u64 make_pipeline_key(u64 kind, std::initializer_list<const std::string*> msl, const void* state = nullptr, usz state_size = 0)
		{
			u64 hash = rpcs3::hash64(rpcs3::fnv_seed, kind);

			for (const std::string* source : msl)
			{
				hash = hash_bytes(hash, source->data(), source->size());
			}

			hash = hash_bytes(hash, state, state_size);
			return hash ? hash : 1;
		}

		// Size in bytes of the vertex fetch formats produced by the translator's stage_in reflection
		u32 get_vertex_format_size(MTL::VertexFormat format)
		{
			switch (format)
			{
			case MTL::VertexFormatHalf: return 2;
			case MTL::VertexFormatHalf2: return 4;
			case MTL::VertexFormatHalf3: return 6;
			case MTL::VertexFormatHalf4: return 8;
			case MTL::VertexFormatFloat: case MTL::VertexFormatInt: case MTL::VertexFormatUInt: return 4;
			case MTL::VertexFormatFloat2: case MTL::VertexFormatInt2: case MTL::VertexFormatUInt2: return 8;
			case MTL::VertexFormatFloat3: case MTL::VertexFormatInt3: case MTL::VertexFormatUInt3: return 12;
			case MTL::VertexFormatFloat4: case MTL::VertexFormatInt4: case MTL::VertexFormatUInt4: return 16;
			default: return 0;
			}
		}

		// Builds the implicit vertex descriptor for vertex shaders with [[stage_in]] attributes (see
		// stage_in_vertex_buffer_index) from the attribute list reflected once at translation time (no MTLFunction
		// reflection per pipeline). Returns an empty ref if the shader fetches no attributes (all RSX programs).
		mtl::ref<MTL::VertexDescriptor> make_stage_in_descriptor(const glsl::shader& vs, const glsl::binding_layout& vs_layout, bool& error)
		{
			error = false;

			const auto& attributes = vs.vertex_attributes();
			if (attributes.empty())
			{
				return {};
			}

			if (vs_layout.buffer_count > stage_in_vertex_buffer_index)
			{
				rsx_log.error("[MSL] Vertex shader uses input attributes and %u buffers; buffer %u is reserved for vertex fetch",
					vs_layout.buffer_count, stage_in_vertex_buffer_index);
				error = true;
				return {};
			}

			auto descriptor = mtl::ref(MTL::VertexDescriptor::alloc()->init());
			u32 offset = 0;

			// Sorted by location: interleaved, tightly packed (4-byte aligned) in location order
			for (const auto& [location, format] : attributes)
			{
				const u32 size = get_vertex_format_size(format);
				if (!size)
				{
					rsx_log.error("[MSL] Unsupported vertex attribute format %u at location %u", static_cast<u32>(format), location);
					error = true;
					return {};
				}

				auto attribute_desc = descriptor->attributes()->object(location);
				attribute_desc->setFormat(format);
				attribute_desc->setOffset(offset);
				attribute_desc->setBufferIndex(stage_in_vertex_buffer_index);
				offset = utils::align(offset + size, 4u);
			}

			auto layout_desc = descriptor->layouts()->object(stage_in_vertex_buffer_index);
			layout_desc->setStride(offset);
			layout_desc->setStepFunction(MTL::VertexStepFunctionPerVertex);
			layout_desc->setStepRate(1);

			return descriptor;
		}

		mtl::ref<MTL4::LibraryFunctionDescriptor> make_function_descriptor(const glsl::shader& shader)
		{
			auto descriptor = mtl::ref(MTL4::LibraryFunctionDescriptor::alloc()->init());
			descriptor->setLibrary(shader.library());
			descriptor->setName(mtl::ns_str(shader.entry_point()));
			return descriptor;
		}
	}

	std::unique_ptr<glsl::program> build_compute_program(
		glsl::shader& cs,
		const std::vector<glsl::program_input>& cs_inputs)
	{
		ensure(g_render_device);
		mtl::autorelease_scope autorelease;

		const auto layout = glsl::build_binding_layout(cs_inputs);
		if (!cs.compile(layout, use_fast_math()))
		{
			return {};
		}

		auto function = make_function_descriptor(cs);

		auto descriptor = mtl::ref(MTL4::ComputePipelineDescriptor::alloc()->init());
		descriptor->setComputeFunctionDescriptor(function.get());
		descriptor->setLabel(mtl::ns_str(cs.entry_point()));

		// Through the pipeline archive (reuses binaries from earlier sessions, records new ones) when it is enabled
		NS::Error* error = nullptr;
		MTL::ComputePipelineState* pipeline = mtl::new_compute_pipeline_state(descriptor.get(), &error, make_pipeline_key(2, { &cs.get_msl() }));

		if (!pipeline)
		{
			rsx_log.error("[MSL] Failed to create compute pipeline: %s", mtl::to_string(error));
			rsx_log.error("[MSL] Compute MSL:\n%s", cs.get_msl());
			return {};
		}

		auto result = std::make_unique<glsl::program>(pipeline, cs_inputs);
		if (cs.needs_buffer_size_buffer())
		{
			result->enable_buffer_size_table(glsl::binding_set_index_compute);
		}

		return result;
	}

	std::unique_ptr<glsl::program> build_graphics_program(
		glsl::shader& vs,
		glsl::shader& fs,
		const glsl::graphics_pipeline_state& state,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs)
	{
		ensure(g_render_device);
		mtl::autorelease_scope autorelease;

		const bool fast_math = use_fast_math();
		const bool rasterization_enabled = !!state.rasterization_enabled;

		const auto vs_layout = glsl::build_binding_layout(vs_inputs);
		if (!vs.compile(vs_layout, fast_math))
		{
			return {};
		}

		// Metal requires a nil fragment function when rasterization is disabled
		if (rasterization_enabled && !fs.compile(glsl::build_binding_layout(fs_inputs), fast_math))
		{
			return {};
		}

		auto descriptor = mtl::ref(MTL4::RenderPipelineDescriptor::alloc()->init());

		auto vs_function = make_function_descriptor(vs);
		descriptor->setVertexFunctionDescriptor(vs_function.get());

		mtl::ref<MTL4::LibraryFunctionDescriptor> fs_function;
		if (rasterization_enabled)
		{
			fs_function = make_function_descriptor(fs);
			descriptor->setFragmentFunctionDescriptor(fs_function.get());
		}

		bool vertex_desc_error = false;
		auto vertex_descriptor = make_stage_in_descriptor(vs, vs_layout, vertex_desc_error);
		if (vertex_desc_error)
		{
			return {};
		}

		if (vertex_descriptor)
		{
			descriptor->setVertexDescriptor(vertex_descriptor.get());
		}

		descriptor->setRasterizationEnabled(rasterization_enabled);
		// RSX vertex programs always write gl_PointSize ([[point_size]] in MSL). Metal rejects such a vertex function
		// in a pipeline declared as line/triangle ("Vertex shader writes point size but inputPrimitiveTopology is
		// MTLPrimitiveTopologyClassTriangle"), so only point pipelines declare their class. Unspecified is Metal's
		// default and is only required to be explicit for layered rendering, which RSX never uses. Keys store the class
		// as get_pipeline_topology_class() normalizes it, so this never builds a line and a triangle variant.
		descriptor->setInputPrimitiveTopology(state.topology_class == static_cast<u8>(MTL::PrimitiveTopologyClassPoint)
			? MTL::PrimitiveTopologyClassPoint
			: MTL::PrimitiveTopologyClassUnspecified);
		descriptor->setRasterSampleCount(std::max<u32>(1u, state.sample_count));
		descriptor->setAlphaToCoverageState(state.alpha_to_coverage ? MTL4::AlphaToCoverageStateEnabled : MTL4::AlphaToCoverageStateDisabled);
		descriptor->setAlphaToOneState(state.alpha_to_one ? MTL4::AlphaToOneStateEnabled : MTL4::AlphaToOneStateDisabled);
		descriptor->setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingStateIdentity);

		const u32 color_count = std::min<u32>(state.color_count, ::size32(state.color));
		for (u32 i = 0; i < color_count; ++i)
		{
			const auto& color = state.color[i];
			if (!color.pixel_format)
			{
				continue;
			}

			auto attachment = descriptor->colorAttachments()->object(i);
			attachment->setPixelFormat(static_cast<MTL::PixelFormat>(color.pixel_format));
			attachment->setWriteMask(static_cast<MTL::ColorWriteMask>(color.write_mask & MTL::ColorWriteMaskAll));

			if (color.blend_enable)
			{
				attachment->setBlendingState(MTL4::BlendStateEnabled);
				attachment->setSourceRGBBlendFactor(static_cast<MTL::BlendFactor>(color.src_rgb));
				attachment->setDestinationRGBBlendFactor(static_cast<MTL::BlendFactor>(color.dst_rgb));
				attachment->setRgbBlendOperation(static_cast<MTL::BlendOperation>(color.op_rgb));
				attachment->setSourceAlphaBlendFactor(static_cast<MTL::BlendFactor>(color.src_a));
				attachment->setDestinationAlphaBlendFactor(static_cast<MTL::BlendFactor>(color.dst_a));
				attachment->setAlphaBlendOperation(static_cast<MTL::BlendOperation>(color.op_a));
			}
			else
			{
				attachment->setBlendingState(MTL4::BlendStateDisabled);
			}
		}

		// Through the pipeline archive (reuses binaries from earlier sessions, records new ones) when it is enabled.
		// The descriptor must be a deterministic function of the shaders and `state`: it is the archive lookup key.
		NS::Error* error = nullptr;
		const std::string no_fragment_shader;
		const u64 key = make_pipeline_key(1, { &vs.get_msl(), rasterization_enabled ? &fs.get_msl() : &no_fragment_shader }, &state, sizeof(state));
		MTL::RenderPipelineState* pipeline = mtl::new_render_pipeline_state(descriptor.get(), &error, key);

		if (!pipeline)
		{
			rsx_log.error("[MSL] Failed to create render pipeline (%u color attachments, %u samples, topology class %u): %s",
				color_count, state.sample_count, state.topology_class, mtl::to_string(error));
			rsx_log.error("[MSL] Vertex MSL:\n%s", vs.get_msl());

			if (rasterization_enabled)
			{
				rsx_log.error("[MSL] Fragment MSL:\n%s", fs.get_msl());
			}

			return {};
		}

		auto result = std::make_unique<glsl::program>(pipeline, vs_inputs, fs_inputs);

		if (vs.needs_buffer_size_buffer())
		{
			result->enable_buffer_size_table(glsl::binding_set_index_vertex);
		}

		if (rasterization_enabled && fs.needs_buffer_size_buffer())
		{
			result->enable_buffer_size_table(glsl::binding_set_index_fragment);
		}

		return result;
	}

	pipe_compiler::pipe_compiler() = default;

	pipe_compiler::~pipe_compiler() = default;

	void pipe_compiler::initialize(const mtl::render_device* pdev)
	{
		m_device = pdev;
	}

	void pipe_compiler::operator()()
	{
		// Pipeline builds are background work: run below the emulation threads (named threads start at
		// QOS_CLASS_USER_INTERACTIVE on macOS; -1 is QOS_CLASS_USER_INITIATED, which still gets performance cores). The
		// Metal compiler service inherits the QoS of the calling thread, so this applies to the MTLLibrary and pipeline
		// builds as well. A job the RSX thread is blocked on (prioritize_jobs) runs at emulation priority instead: the
		// RSX thread is idle meanwhile.
		bool urgent_priority = false;
		thread_ctrl::set_native_priority(-1);

		while (thread_ctrl::state() != thread_state::aborting)
		{
			// Sampled before the queue is checked: a job pushed after the check changes it and ends the wait
			const u32 pushed = s_queue.pushed.load();

			auto job = s_queue.pop();
			if (!job)
			{
				thread_ctrl::wait_on(s_queue.pushed, pushed);
				continue;
			}

			if (job->urgent != urgent_priority)
			{
				urgent_priority = job->urgent;
				thread_ctrl::set_native_priority(urgent_priority ? 1 : -1);
			}

			{
				// Every job gets its own pool: Metal descriptors, errors and strings are autoreleased
				mtl::autorelease_scope autorelease;

				std::unique_ptr<glsl::program> compiled;
				if (job->is_graphics_job)
				{
					compiled = int_compile_graphics_pipe(job->graphics_data, job->shaders[0], job->shaders[1], job->inputs[0], job->inputs[1], job->flags);
				}
				else
				{
					compiled = int_compile_compute_pipe(job->shaders[0], job->inputs[0], job->flags);
				}

				if (job->callback_func)
				{
					job->callback_func(compiled);
				}
			}

			// After the callback: the pipeline (or the failure) is visible to whoever waits for this count to change
			s_queue.completed++;
			s_queue.completed.notify_all();
		}
	}

	u32 pipe_compiler::get_completed_job_count()
	{
		return s_queue.completed.load();
	}

	void pipe_compiler::wait_for_completed_job(u32 count, u64 timeout_us)
	{
		if (!timeout_us)
		{
			return;
		}

		// Plain atomic wait (not thread_ctrl::wait_on): the caller is the RSX thread, whose thread notifications and task
		// queue would end the wait early; it is bounded by the timeout anyway.
		s_queue.completed.wait(count, atomic_wait_timeout{ timeout_us * 1000 });
	}

	void pipe_compiler::prioritize_jobs(const glsl::shader* vs, const glsl::shader* fs)
	{
		std::lock_guard lock(s_queue.mutex);

		// Stable, so the queue order is kept among the moved jobs and among the others
		const auto first_other = std::stable_partition(s_queue.jobs.begin(), s_queue.jobs.end(), [&](const pipe_compiler_job& job)
		{
			return job.is_graphics_job && job.shaders[0] == vs && job.shaders[1] == fs;
		});

		for (auto it = s_queue.jobs.begin(); it != first_other; ++it)
		{
			it->urgent = true;
		}
	}

	std::unique_ptr<glsl::program> pipe_compiler::int_compile_compute_pipe(
		glsl::shader* cs,
		const std::vector<glsl::program_input>& cs_inputs,
		op_flags /*flags*/)
	{
		ensure(cs);
		return build_compute_program(*cs, cs_inputs);
	}

	std::unique_ptr<glsl::program> pipe_compiler::int_compile_graphics_pipe(
		const mtl::pipeline_props& create_info,
		glsl::shader* vs,
		glsl::shader* fs,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs,
		op_flags flags)
	{
		ensure(vs && fs);

		if (flags & USE_LAST_PROVOKING_VERTEX)
		{
			// Metal has no provoking-vertex control. The renderer is expected to report
			// supports_last_provoking_vertex = false so this never happens; keep going with the default convention.
			static atomic_t<bool> s_reported = false;
			if (!s_reported.exchange(true))
			{
				rsx_log.warning("[MSL] Last provoking vertex requested but unsupported by Metal; flat shading may differ.");
			}
		}

		return build_graphics_program(*vs, *fs, create_info.state, vs_inputs, fs_inputs);
	}

	std::unique_ptr<glsl::program> pipe_compiler::compile(
		glsl::shader* cs,
		op_flags flags, callback_t callback,
		const std::vector<glsl::program_input>& cs_inputs)
	{
		if (flags & COMPILE_INLINE)
		{
			return int_compile_compute_pipe(cs, cs_inputs, flags);
		}

		s_queue.push(pipe_compiler_job(cs, cs_inputs, flags, std::move(callback)));
		return {};
	}

	std::unique_ptr<glsl::program> pipe_compiler::compile(
		const mtl::pipeline_props& create_info,
		glsl::shader* vs,
		glsl::shader* fs,
		op_flags flags, callback_t callback,
		const std::vector<glsl::program_input>& vs_inputs,
		const std::vector<glsl::program_input>& fs_inputs)
	{
		if (flags & COMPILE_INLINE)
		{
			return int_compile_graphics_pipe(create_info, vs, fs, vs_inputs, fs_inputs, flags);
		}

		s_queue.push(pipe_compiler_job(create_info, vs, fs, vs_inputs, fs_inputs, flags, std::move(callback)));
		return {};
	}

	void initialize_pipe_compiler(int num_worker_threads)
	{
		ensure(g_render_device); // "Cannot initialize pipe compiler before creating a logical device"

		// Library/pipeline builds dominate each job and Metal only runs max_compile_tasks of them concurrently
		const u32 compile_tasks = std::max(1u, g_render_device->caps().max_compile_tasks);
		const int max_workers = static_cast<int>(compile_tasks);

		if (num_worker_threads <= 0)
		{
			// Most of a job is spent in the Metal compiler service, which runs up to maximumConcurrentCompilationTaskCount
			// builds at a time (raised by setShouldMaximizeConcurrentCompilation, see render_device::create): one worker
			// per task keeps it busy without queueing work inside the service. The workers' own CPU work (GLSL ->
			// SPIR-V -> MSL) runs below emulation priority (see operator()). At least 2 workers, so that one slow
			// compile never holds back the others; at most all but two host threads (and 16).
			const u32 hw_threads = utils::get_thread_count();
			const u32 cpu_limit = std::min(16u, std::max(2u, hw_threads > 2 ? hw_threads - 2 : 0u));
			num_worker_threads = static_cast<int>(std::clamp(compile_tasks, 2u, cpu_limit));

			rsx_log.notice("Async pipeline compiler auto-selected %d worker(s) (Metal concurrent compilation tasks: %u, host threads: %u).",
				num_worker_threads, compile_tasks, hw_threads);
		}

		if (num_worker_threads > max_workers)
		{
			rsx_log.notice("Pipeline compiler worker count capped from %d to %d (Metal concurrent compilation limit).",
				num_worker_threads, max_workers);
			num_worker_threads = max_workers;
		}

		ensure(num_worker_threads >= 1);

		// Create the thread pool
		g_pipe_compilers = std::make_unique<named_thread_group<pipe_compiler>>("RSX.W", num_worker_threads);

		// Initialize the workers. At least one inline compiler shall exist (doesn't actually run)
		for (pipe_compiler& compiler : *g_pipe_compilers.get())
		{
			compiler.initialize(g_render_device);
		}
	}

	void destroy_pipe_compiler()
	{
		g_pipe_compilers.reset();

		// No worker is left: drop what was never started (the callbacks reference the program cache, which goes next)
		std::lock_guard lock(pipe_compiler::s_queue.mutex);
		pipe_compiler::s_queue.jobs.clear();
	}

	pipe_compiler* get_pipe_compiler()
	{
		ensure(g_pipe_compilers);

		// Deferred jobs go to the shared queue, so any worker object will do
		return g_pipe_compilers.get()->begin();
	}
}
