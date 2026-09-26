#pragma once
#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "MTLPipelineCompiler.h"
#include "../Program/ProgramStateCache.h"

#include "Emu/system_config.h"
#include "Utilities/mutex.h"
#include "util/fnv_hash.hpp"

#include <unordered_set>

namespace mtl
{
	// RSX pipelines that could not be built. program_state_cache keeps the null placeholder it inserted for them, which
	// looks exactly like a pipeline that is still being compiled; this set tells the two apart, so that nobody waits
	// for (or announces the compilation of) a pipeline that will never exist.
	class pipeline_failure_registry
	{
	public:
		struct key_type
		{
			u32 vertex_program_id;
			u32 fragment_program_id;
			mtl::pipeline_props properties; // Validated (MTLTraits::make_failure_key)

			bool operator==(const key_type&) const = default;
		};

		void add(const key_type& key)
		{
			std::lock_guard lock(m_mutex);
			if (m_keys.insert(key).second)
			{
				// Once per pipeline: the program cache never rebuilds it (the reason was logged by the builder)
				rsx_log.error("Metal: the pipeline for vp id %u / fp id %u (%u color attachments, %u samples) could not be built. Draws that use it are skipped.",
					key.vertex_program_id, key.fragment_program_id, key.properties.state.color_count, key.properties.state.sample_count);
			}
		}

		bool contains(const key_type& key) const
		{
			reader_lock lock(m_mutex);
			return m_keys.contains(key);
		}

		void clear()
		{
			std::lock_guard lock(m_mutex);
			m_keys.clear();
		}

	private:
		struct key_hash
		{
			usz operator()(const key_type& key) const
			{
				return rpcs3::hash64(rpcs3::hash64(rpcs3::hash_struct(key.properties), key.vertex_program_id), key.fragment_program_id);
			}
		};

		mutable shared_mutex m_mutex;
		std::unordered_set<key_type, key_hash> m_keys;
	};

	struct MTLTraits
	{
		using vertex_program_type = MTLVertexProgram;
		using fragment_program_type = MTLFragmentProgram;
		using pipeline_type = mtl::glsl::program;
		using pipeline_storage_type = std::unique_ptr<mtl::glsl::program>;
		using pipeline_properties = mtl::pipeline_props;

		static
			void recompile_fragment_program(const RSXFragmentProgram& RSXFP, fragment_program_type& fragmentProgramData, usz ID)
		{
			fragmentProgramData.Decompile(RSXFP);
			fragmentProgramData.id = static_cast<u32>(ID);
			fragmentProgramData.Compile();
		}

		static
			void recompile_vertex_program(const RSXVertexProgram& RSXVP, vertex_program_type& vertexProgramData, usz ID)
		{
			vertexProgramData.Decompile(RSXVP);
			vertexProgramData.id = static_cast<u32>(ID);
			vertexProgramData.Compile();
		}

		static
			void validate_pipeline_properties(const MTLVertexProgram&, const MTLFragmentProgram& fp, mtl::pipeline_props& properties)
		{
			// Explicitly disable writing to undefined registers
			for (u32 i = 0; i < ::size32(properties.state.color); ++i)
			{
				properties.state.color[i].write_mask &= static_cast<u8>(fp.output_color_masks[i]);
			}
		}

		// program_state_cache builds a new program pair with the properties as they are, but validates them first for
		// programs it already had. Validation is idempotent, so validating here gives one key for both cases.
		static
			pipeline_failure_registry::key_type make_failure_key(const vertex_program_type& vp, const fragment_program_type& fp, mtl::pipeline_props properties)
		{
			validate_pipeline_properties(vp, fp, properties);
			return { vp.id, fp.id, properties };
		}

		static
			pipeline_type* build_pipeline(
				const vertex_program_type& vertexProgramData,
				const fragment_program_type& fragmentProgramData,
				const mtl::pipeline_props& pipelineProperties,
				bool compile_async,
				std::function<pipeline_type*(pipeline_storage_type&)> callback,
				pipeline_failure_registry& failures)
		{
			mtl::pipe_compiler::op_flags compiler_flags = compile_async ? mtl::pipe_compiler::COMPILE_DEFERRED : mtl::pipe_compiler::COMPILE_INLINE;
			compiler_flags |= mtl::pipe_compiler::SEPARATE_SHADER_OBJECTS;

			if (vertexProgramData.Flags() & RSX_SHADER_CONTROL_FLAT_SHADING)
			{
				compiler_flags |= mtl::pipe_compiler::USE_LAST_PROVOKING_VERTEX;
			}

			// The cache's callback leaves the null placeholder in place when the build failed: record the failure
			// before it runs, so that a lookup never sees the placeholder without the failure after a failed build
			auto on_built = [&failures, failure_key = make_failure_key(vertexProgramData, fragmentProgramData, pipelineProperties),
				callback = std::move(callback)](pipeline_storage_type& pipeline) -> pipeline_type*
			{
				if (!pipeline)
				{
					failures.add(failure_key);
				}

				return callback(pipeline);
			};

			auto compiler = mtl::get_pipe_compiler();
			auto result = compiler->compile(
				pipelineProperties,
				vertexProgramData.handle,
				fragmentProgramData.handle,
				compiler_flags, on_built,
				vertexProgramData.uniforms,
				fragmentProgramData.uniforms);

			if (compile_async)
			{
				// Queued: a worker hands the result to on_built (the null result here is not a failure)
				return nullptr;
			}

			return on_built(result);
		}
	};

	struct program_cache : public program_state_cache<MTLTraits>
	{
		using base_type = program_state_cache<MTLTraits>;

		program_cache(decompiler_callback_t callback)
		{
			notify_pipeline_compiled = callback;
		}

		// MTL4 render pipelines do not bake depth/stencil formats, so graphics_pipeline_state::depth_stencil_format is
		// removed from the effective cache key: it is zeroed while the pipeline is looked up / built / stored in the
		// shader cache (the field stays in the POD for shader-cache layout compatibility) and restored afterwards so
		// the caller's own state comparisons are unaffected. Hides program_state_cache::get_graphics_pipeline.
		// Also tells a failed pipeline from one that is still compiling (check_pipeline_failed).
		template <typename... Args>
		auto get_graphics_pipeline(
			rsx::program_cache_hint_t* cache_hint,
			const RSXVertexProgram& vertex_shader,
			const RSXFragmentProgram& fragment_shader,
			mtl::pipeline_props& pipeline_properties,
			bool compile_async,
			bool allow_notification,
			Args&& ...args)
		{
			const u32 depth_stencil_format = std::exchange(pipeline_properties.state.depth_stencil_format, 0u);

			auto result = base_type::get_graphics_pipeline(cache_hint, vertex_shader, fragment_shader, pipeline_properties,
				compile_async, allow_notification, m_failed_pipelines, std::forward<Args>(args)...);

			m_last_pipeline_failed = false;
			if (const auto& [pipeline, vp, fp] = result; !pipeline)
			{
				// Same key as the build (depth/stencil format still zeroed)
				m_last_pipeline_failed = m_failed_pipelines.contains(MTLTraits::make_failure_key(*vp, *fp, pipeline_properties));

				if (m_last_pipeline_failed)
				{
					// Nothing is being compiled: no shader compilation notification
					m_cache_miss_flag = false;
				}
			}

			pipeline_properties.state.depth_stencil_format = depth_stencil_format;
			return result;
		}

		u64 get_hash(const mtl::pipeline_props& props)
		{
			return rpcs3::hash_struct<mtl::pipeline_props>(props);
		}

		u64 get_hash(const RSXVertexProgram& prog)
		{
			return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(prog);
		}

		u64 get_hash(const RSXFragmentProgram& prog)
		{
			return program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(prog);
		}

		template <typename... Args>
		void add_pipeline_entry(RSXVertexProgram& vp, RSXFragmentProgram& fp, mtl::pipeline_props& props, Args&& ...args)
		{
			normalize_cached_programs(vp, fp);
			normalize_cached_pipeline(fp, props);
			get_graphics_pipeline(nullptr, vp, fp, props, false, false, std::forward<Args>(args)...);
		}

		void preload_programs(rsx::program_cache_hint_t* cache_hint, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			// Same programs as add_pipeline_entry() will look up, except for the A2C flag (it depends on the pipeline
			// state, which is not passed here)
			if (needs_msaa_normalization(vp, fp))
			{
				RSXVertexProgram vp_ = vp;
				RSXFragmentProgram fp_ = fp;
				normalize_cached_programs(vp_, fp_);

				search_vertex_program(cache_hint, vp_);
				search_fragment_program(cache_hint, fp_);
				return;
			}

			search_vertex_program(cache_hint, vp);
			search_fragment_program(cache_hint, fp);
		}

		bool check_cache_missed() const
		{
			return m_cache_miss_flag;
		}

		// The last get_graphics_pipeline() returned no pipeline because it could not be built (not because it is
		// still compiling). Such a pipeline is never retried.
		bool check_pipeline_failed() const
		{
			return m_last_pipeline_failed;
		}

		// Hides program_state_cache::clear
		void clear()
		{
			base_type::clear();
			m_failed_pipelines.clear();
			m_last_pipeline_failed = false;
		}

	private:
		pipeline_failure_registry m_failed_pipelines;
		bool m_last_pipeline_failed = false; // Like m_cache_miss_flag: describes the last lookup

		// ---- Shader cache entries --------------------------------------------------------------------------------
		// Entries keep the state they were recorded with. The parts that depend on renderer settings instead of the
		// guest are brought to what this session produces for the same draw; otherwise the preload builds programs
		// and pipelines that no draw can use (e.g. 4-sample A2C pipelines recorded with MSAA on, now that it is off).

		static bool msaa_disabled()
		{
			// Same condition as MTLGSRender (backend_config.supports_hw_msaa/a2c/a2one) and
			// surface_cache_traits::create_new_surface (one sample per surface)
			return g_cfg.video.antialiasing_level == msaa_level::none;
		}

		static bool needs_msaa_normalization(const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			return msaa_disabled() &&
				(vp.texture_state.multisampled_textures || fp.texture_state.multisampled_textures || (fp.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED));
		}

		static void normalize_cached_programs(RSXVertexProgram& vp, RSXFragmentProgram& fp)
		{
			if (!msaa_disabled())
			{
				return;
			}

			// Without hardware MSAA the RSX never flags multisampled textures or multisampled ROP output
			// (backend_config.supports_hw_msaa = false)
			vp.texture_state.multisampled_textures = 0;
			fp.texture_state.multisampled_textures = 0;
			fp.ctrl &= ~RSX_SHADER_CONTROL_ROP_MULTISAMPLED;
		}

		static void normalize_cached_pipeline(RSXFragmentProgram& fp, mtl::pipeline_props& props)
		{
			auto& state = props.state;
			state.topology_class = get_pipeline_topology_class(state.topology_class);

			if (!msaa_disabled())
			{
				return;
			}

			// Without hardware A2C the RSX has the fragment program emulate it (RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE).
			// decode_rsx_state only enables hardware A2C when A2C is on and the target is multisampled, which is
			// exactly when the RSX leaves the emulation flag out.
			if (state.alpha_to_coverage)
			{
				fp.ctrl |= RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE;
			}

			state.sample_count = 1;
			state.alpha_to_coverage = 0;
			state.alpha_to_one = 0;
		}
	};
}
