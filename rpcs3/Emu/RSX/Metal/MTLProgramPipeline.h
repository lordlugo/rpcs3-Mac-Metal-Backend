#pragma once

// CONTRACT HEADER — shared by the shader/pipeline, compute/overlay and renderer-core components.
// Public declarations here must not change without updating every user.
//
// Shader path: RSX ucode -> (shared decompilers) Vulkan-flavoured GLSL 450 -> glslang SPIR-V -> SPIRV-Cross MSL
//              -> MTL4Compiler library -> MTL4 render/compute pipeline.
// Binding model: every stage owns an MTL4ArgumentTable (see mtl::command_list). Each GLSL (set, binding) is mapped to
// Metal buffer/texture/sampler indices by build_binding_layout(), which is used BOTH when translating to MSL
// (SPIRV-Cross resource bindings) and when binding resources at draw time, so the two can never disagree.

#include "mtlutils/mtl_api.h"
#include "mtlutils/commands.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/image.h"
#include "Emu/RSX/Program/GLSLTypes.h"
#include "Emu/RSX/Common/simple_array.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mtl
{
	class data_heap;
	struct sampler;

	namespace glsl
	{
		enum program_input_type : u32
		{
			input_type_uniform_buffer = 0,
			input_type_texel_buffer,
			input_type_texture,
			input_type_storage_buffer,
			input_type_storage_texture,
			input_type_push_constant,
			input_type_attachment,        // subpassInput -> [[color(n)]] framebuffer fetch (no binding)

			input_type_max_enum,
			input_type_undefined = 0xffff'ffff
		};

		struct push_constant_ref
		{
			u32 offset = 0;
			u32 size = 0;
		};

		// Same shape as vk::glsl::program_input so decompiler code ports 1:1.
		struct program_input
		{
			::glsl::program_domain domain = ::glsl::glsl_invalid_program;
			program_input_type type = input_type_undefined;
			push_constant_ref push_constant{};
			u32 set = 0;
			u32 location = umax;       // GLSL binding (or input attachment index for input_type_attachment)
			u32 array_size = 1;        // > 1 for sampler arrays (interpreter)
			std::string name = "undefined";

			static program_input make(
				::glsl::program_domain domain,
				const std::string& name,
				program_input_type type,
				u32 set,
				u32 location,
				const push_constant_ref& push_constant = {},
				u32 array_size = 1)
			{
				return program_input
				{
					.domain = domain,
					.type = type,
					.push_constant = push_constant,
					.set = set,
					.location = location,
					.array_size = array_size,
					.name = name
				};
			}
		};

		enum binding_set_index : u32
		{
			binding_set_index_vertex = 0,
			binding_set_index_fragment = 1,

			binding_set_index_compute = 0,
			binding_set_index_unified = 0,

			binding_set_index_max_enum = 2,
		};

		// Metal slot assigned to a GLSL (set, binding). umax = unused.
		struct resource_slot
		{
			program_input_type type = input_type_undefined;
			u32 buffer_index = umax;   // uniform/storage buffers and push constants
			u32 texture_index = umax;  // textures, texel buffers, storage textures (first element for arrays)
			u32 sampler_index = umax;  // samplers for combined image samplers (first element for arrays)
			u32 array_size = 1;
		};

		// Per-stage binding layout. Deterministic function of the input list.
		struct binding_layout
		{
			std::unordered_map<u32, resource_slot> slots; // key: GLSL binding location within this set
			u32 push_constant_buffer_index = umax;         // Buffer index used for the push constant block
			u32 push_constant_size = 0;
			u32 buffer_count = 0;
			u32 texture_count = 0;
			u32 sampler_count = 0;
		};

		// Assigns Metal indices for ONE stage (all inputs of the list, whatever their set):
		//  - buffers (UBO, SSBO) 0.. in input order, push-constant block last (index 30 max);
		//  - sampled textures (input_type_texture) first 0.., then texel buffers and storage textures (63 max);
		//  - sampler index == texture index for the first 16 sampled texture slots, in input order. Later sampled
		//    textures get sampler_index = umax and the MSL gives them a constant nearest / clamp-to-border(black) /
		//    LOD 0 sampler (exactly the stencil-mirror sampler), so list textures that need real samplers first;
		//  - input_type_attachment gets no slot (framebuffer fetch [[color(n)]]).
		// Keys are GLSL binding locations; a location may only be declared once per stage (fatal: programming error).
		// Running out of buffer/texture slots is not fatal: the slot stays umax and the shader translation fails.
		// If a stage reads SSBO lengths, its buffer-size table lives at index buffer_count (see shader/program).
		binding_layout build_binding_layout(const std::vector<program_input>& inputs);

		// Resources handed to program::bind_uniform
		struct buffer_binding_info
		{
			const mtl::buffer* buffer = nullptr;   // or raw:
			MTL::Buffer* raw_buffer = nullptr;
			u64 offset = 0;
			u64 range = 0;

			buffer_binding_info() = default;
			buffer_binding_info(const mtl::buffer* buf, u64 off, u64 len) : buffer(buf), offset(off), range(len) {}
			buffer_binding_info(MTL::Buffer* buf, u64 off, u64 len) : raw_buffer(buf), offset(off), range(len) {}

			MTL::GPUAddress gpu_address() const;
		};

		struct image_binding_info
		{
			MTL::Texture* texture = nullptr;          // A view (mtl::image_view::value) or full texture
			MTL::SamplerState* sampler = nullptr;     // nullptr for storage images / texelFetch-only textures

			image_binding_info() = default;
			image_binding_info(MTL::Texture* tex, MTL::SamplerState* smp) : texture(tex), sampler(smp) {}
			image_binding_info(const mtl::image_view* view, const mtl::sampler* smp);
		};

		// A translated shader stage. GLSL source is kept for the shader cache / debugging.
		class shader
		{
			::glsl::program_domain m_type = ::glsl::program_domain::glsl_vertex_program;
			std::string m_source;       // GLSL (Vulkan semantics)
			std::string m_msl;          // Generated MSL
			std::string m_entry_point;  // MSL entry point name
			MTL::Library* m_library = nullptr;

			std::mutex m_compile_lock;          // compile() may race between pipe-compiler workers sharing a shader
			bool m_compile_failed = false;      // Do not retry (and re-log) a translation that already failed
			bool m_needs_buffer_sizes = false;  // MSL reads spvBufferSizeConstants (GLSL SSBO .length())
			std::vector<std::pair<u32, MTL::VertexFormat>> m_vertex_attributes; // Vertex stage inputs (location, format)

		public:
			shader() = default;
			~shader();

			shader(const shader&) = delete;
			shader& operator=(const shader&) = delete;

			void create(::glsl::program_domain domain, const std::string& source);

			// GLSL -> SPIR-V -> MSL using the given layout, then MTL4Compiler::newLibrary. Thread-safe.
			// Returns false (and logs the MSL/GLSL) on failure.
			bool compile(const binding_layout& layout, bool fast_math);

			void destroy();

			::glsl::program_domain domain() const { return m_type; }
			const std::string& get_source() const { return m_source; }
			const std::string& get_msl() const { return m_msl; }
			const std::string& entry_point() const { return m_entry_point; }
			MTL::Library* library() const { return m_library; }
			bool is_compiled() const { return m_library != nullptr; }

			// True if the MSL expects a buffer-size table (uint per Metal buffer index) at binding_layout::buffer_count.
			// Only set for shaders using GLSL SSBO .length(); pipeline builders forward it to program.
			bool needs_buffer_size_buffer() const { return m_needs_buffer_sizes; }

			// Vertex shaders only: `layout(location = N) in` attributes (reflected at translation time, sorted by
			// location). Empty for RSX vertex programs, which pull their inputs from texel buffers.
			const std::vector<std::pair<u32, MTL::VertexFormat>>& vertex_attributes() const { return m_vertex_attributes; }
		};

		// Fixed-function state baked into a Metal 4 render pipeline. POD; hashed and serialized raw (shader cache).
		struct color_attachment_state
		{
			u32 pixel_format = 0;       // MTL::PixelFormat (0 = unused)
			u8 write_mask = 0xF;        // MTL::ColorWriteMask bits (R=8,G=4,B=2,A=1)
			u8 blend_enable = 0;
			u8 src_rgb = 1;             // MTL::BlendFactor
			u8 dst_rgb = 0;
			u8 op_rgb = 0;              // MTL::BlendOperation
			u8 src_a = 1;
			u8 dst_a = 0;
			u8 op_a = 0;
		};

		struct graphics_pipeline_state
		{
			std::array<color_attachment_state, 4> color{};
			u32 color_count = 0;
			u32 depth_stencil_format = 0; // informational only (MTL4 pipelines do not bake depth/stencil formats)
			u8 sample_count = 1;
			u8 alpha_to_coverage = 0;
			u8 alpha_to_one = 0;
			u8 topology_class = 3;        // MTL::PrimitiveTopologyClass (1=point, 2=line, 3=triangle)
			u8 rasterization_enabled = 1;
			u8 pad[3]{};
		};

		static_assert(std::is_trivially_copyable_v<graphics_pipeline_state>);

		// A linked pipeline + its binding layouts + current bindings.
		class program
		{
			MTL::RenderPipelineState* m_render_pipeline = nullptr;
			MTL::ComputePipelineState* m_compute_pipeline = nullptr;

			std::array<std::vector<program_input>, binding_set_index_max_enum> m_inputs;
			std::array<binding_layout, binding_set_index_max_enum> m_layouts;

			struct stage_bindings
			{
				std::array<MTL::GPUAddress, 31> buffers{};
				std::array<MTL::ResourceID, 64> textures{};
				std::array<MTL::ResourceID, 16> samplers{};
				std::vector<u8> push_constants;
				bool dirty = true;

				std::array<u32, 31> buffer_sizes{};     // Bound ranges, uploaded when the stage needs a buffer-size table
				bool needs_buffer_sizes = false;
			};
			std::array<stage_bindings, binding_set_index_max_enum> m_bindings;

			u32 m_compute_threads_per_group = 1;

			// (set, binding) -> bitmask of stages (m_inputs/m_layouts index) that declare it
			static constexpr u32 max_binding_locations = 128;
			std::array<std::array<u8, max_binding_locations>, binding_set_index_max_enum> m_binding_stage_mask{};
			bool m_missing_binding_reported = false;

			void init_layouts();
			template <typename F> void for_each_bound_slot(u32 set_id, u32 binding_point, F&& func);

		public:
			program(MTL::RenderPipelineState* pipeline,
				const std::vector<program_input>& vertex_inputs,
				const std::vector<program_input>& fragment_inputs);

			program(MTL::ComputePipelineState* pipeline,
				const std::vector<program_input>& compute_inputs);

			program(const program&) = delete;
			program(program&&) = delete;
			~program();

			bool is_compute() const { return m_compute_pipeline != nullptr; }
			MTL::RenderPipelineState* render_pipeline() const { return m_render_pipeline; }
			MTL::ComputePipelineState* compute_pipeline() const { return m_compute_pipeline; }
			u32 max_total_threads_per_threadgroup() const;

			bool has_uniform(program_input_type type, std::string_view uniform_name) const;
			std::pair<u32, u32> get_uniform_location(::glsl::program_domain domain, program_input_type type, std::string_view uniform_name) const; // {set, location}

			void bind_uniform(const buffer_binding_info& buffer, u32 set_id, u32 binding_point);     // UBO / SSBO
			void bind_uniform(const image_binding_info& image, u32 set_id, u32 binding_point);       // sampled / storage image
			void bind_uniform(const mtl::buffer_view* view, u32 set_id, u32 binding_point);         // texel buffer
			void bind_uniform_array(std::span<const image_binding_info> images, u32 set_id, u32 binding_point);
			// Vulkan semantics: [offset, offset + size) addresses the push-constant space shared by all stages; every
			// stage whose push block covers it receives the bytes (set_id is accepted for parity, not needed).
			void push_constants(u32 set_id, u32 offset, u32 size, const void* data);

			// Sets the pipeline state on the active encoder of `cmd` (render encoder for graphics programs, compute()
			// for compute programs), uploads push constants to `scratch` and writes + sets the stage argument tables.
			// For graphics programs a render pass must already be open.
			void bind(mtl::command_list& cmd, mtl::data_heap& scratch);

			// For pipeline builders: the translated shader of `stage_index` (0 = vertex/compute, 1 = fragment) reads
			// SSBO lengths, so bind() uploads the bound storage-buffer ranges to binding_layout::buffer_count.
			void enable_buffer_size_table(u32 stage_index);
		};

		// ---- Program creation helpers (for static passes: compute kernels, overlays, blits) ----------------------
		// Synchronous build from GLSL (Vulkan semantics). Returns nullptr and logs on failure.
		std::unique_ptr<program> create_compute_program(const std::string& compute_glsl, const std::vector<program_input>& inputs);

		std::unique_ptr<program> create_graphics_program(
			const std::string& vertex_glsl, const std::vector<program_input>& vertex_inputs,
			const std::string& fragment_glsl, const std::vector<program_input>& fragment_inputs,
			const graphics_pipeline_state& state);
	}

	// Key for RSX graphics pipelines (program_state_cache<MTLTraits>::pipeline_properties). POD, hashed with
	// rpcs3::hash_struct and stored raw in the shader cache.
	struct pipeline_props
	{
		glsl::graphics_pipeline_state state{};

		bool operator==(const pipeline_props& other) const
		{
			return std::memcmp(&state, &other.state, sizeof(state)) == 0;
		}
	};

	static_assert(std::is_trivially_copyable_v<pipeline_props>);

	// Global scratch ring for push constants and small per-draw uniforms (created by MTLGSRender, 16 MiB, grows).
	mtl::data_heap& get_scratch_heap();
}
