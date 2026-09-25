#pragma once
#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "MTLPipelineCompiler.h"
#include "../Program/ProgramStateCache.h"

namespace mtl
{
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

		static
			pipeline_type* build_pipeline(
				const vertex_program_type& vertexProgramData,
				const fragment_program_type& fragmentProgramData,
				const mtl::pipeline_props& pipelineProperties,
				bool compile_async,
				std::function<pipeline_type*(pipeline_storage_type&)> callback)
		{
			mtl::pipe_compiler::op_flags compiler_flags = compile_async ? mtl::pipe_compiler::COMPILE_DEFERRED : mtl::pipe_compiler::COMPILE_INLINE;
			compiler_flags |= mtl::pipe_compiler::SEPARATE_SHADER_OBJECTS;

			if (vertexProgramData.Flags() & RSX_SHADER_CONTROL_FLAT_SHADING)
			{
				compiler_flags |= mtl::pipe_compiler::USE_LAST_PROVOKING_VERTEX;
			}

			auto compiler = mtl::get_pipe_compiler();
			auto result = compiler->compile(
				pipelineProperties,
				vertexProgramData.handle,
				fragmentProgramData.handle,
				compiler_flags, callback,
				vertexProgramData.uniforms,
				fragmentProgramData.uniforms);

			return callback(result);
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
				compile_async, allow_notification, std::forward<Args>(args)...);

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
			get_graphics_pipeline(nullptr, vp, fp, props, false, false, std::forward<Args>(args)...);
		}

		void preload_programs(rsx::program_cache_hint_t* cache_hint, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			search_vertex_program(cache_hint, vp);
			search_fragment_program(cache_hint, fp);
		}

		bool check_cache_missed() const
		{
			return m_cache_miss_flag;
		}
	};
}
