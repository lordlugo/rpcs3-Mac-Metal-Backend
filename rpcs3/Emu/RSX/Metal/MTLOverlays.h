#pragma once

// Full-screen / partial-screen draw passes (port of VK/VKOverlays.h) plus the scaled-copy blit passes.
//
// Every run() opens its own MTL4 render pass on the given target (loadAction Load unless the pass overwrites the
// whole target), draws, and closes the pass again. Nothing is left open for the caller.
//
// Shader conventions (the GLSL is Vulkan-flavoured like the VK backend):
//  - Vertex stage resources use set 0 (glsl::binding_set_index_vertex), fragment stage resources use set 1
//    (glsl::binding_set_index_fragment), matching the RSX program layout; shared snippets that hard-code set=0 in
//    fragment shaders are remapped when the pass is built.
//  - Push constants keep Vulkan semantics: one address space shared by the stages (glsl::program::push_constants
//    routes by offset), so a fragment block that follows a vertex block uses layout(offset=N) as in the VK backend.
//  - There are no vertex descriptors: geometry is generated from gl_VertexIndex or pulled from a storage buffer
//    that lives in mtl::get_scratch_heap() and is bound by GPU address.
//  - Clip space follows Vulkan conventions; the shader translator flips Y (SPIRV-Cross flip_vert_y) for every
//    vertex shader, so the VK passes work unchanged.

#include "../Common/simple_array.hpp"
#include "../Overlays/overlay_controls.h"

#include "MTLProgramPipeline.h"
#include "MTLHelpers.h"

#include "mtlutils/data_heap.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

#include "Emu/IdManager.h"

#include <array>
#include <unordered_map>

namespace rsx
{
	namespace overlays
	{
		enum class texture_sampling_mode;
		struct overlay;
	}
}

namespace mtl
{
	// Depth/stencil fixed-function state of a pass. One MTL::DepthStencilState is cached per unique configuration.
	struct depth_stencil_config
	{
		bool depth_test = false;
		bool depth_write = false;
		MTL::CompareFunction depth_compare = MTL::CompareFunctionAlways;

		bool stencil_test = false;
		MTL::StencilOperation stencil_fail = MTL::StencilOperationKeep;
		MTL::StencilOperation depth_fail = MTL::StencilOperationKeep;
		MTL::StencilOperation stencil_pass = MTL::StencilOperationKeep;
		MTL::CompareFunction stencil_compare = MTL::CompareFunctionAlways;
		u32 stencil_read_mask = 0xFF;
		u32 stencil_write_mask = 0xFF;
		u32 stencil_ref = 0;           // Dynamic state (setStencilReferenceValue), not part of the cached object

		u64 key() const;
	};

	// Returns a cached depth-stencil state object (owned by the cache, valid until destroy_overlay_passes()).
	MTL::DepthStencilState* get_depth_stencil_state(const depth_stencil_config& config);

	// Moves fragment shader resources declared in descriptor set 0 (VK overlay convention) to set 1
	// (glsl::binding_set_index_fragment) so shared GLSL snippets can be used unchanged.
	std::string remap_fragment_set(const std::string& source);

	// Pipeline configuration of an overlay pass (replaces vk::graphics_pipeline_state for these passes).
	// Attachment pixel formats and the sample count are filled in from the render target at draw time.
	struct overlay_pipeline_config
	{
		glsl::graphics_pipeline_state state{};
		depth_stencil_config ds{};
		MTL::PrimitiveType primitive = MTL::PrimitiveTypeTriangleStrip;
		MTL::CullMode cull_mode = MTL::CullModeNone;

		overlay_pipeline_config();

		void set_primitive_type(MTL::PrimitiveType type);
		void set_attachment_count(u32 count);
		void set_color_mask(u32 index, bool r, bool g, bool b, bool a);
		void enable_blend(u32 index,
			MTL::BlendFactor src_factor_rgb, MTL::BlendFactor src_factor_a,
			MTL::BlendFactor dst_factor_rgb, MTL::BlendFactor dst_factor_a,
			MTL::BlendOperation equation_rgb, MTL::BlendOperation equation_a);
		void disable_blend(u32 index);
		void set_depth_mask(bool enable);
		void enable_depth_test(MTL::CompareFunction op);
		void disable_depth_test();
		void enable_stencil_test(MTL::StencilOperation fail, MTL::StencilOperation zfail, MTL::StencilOperation pass,
			MTL::CompareFunction func, u32 func_mask, u32 ref);
		void disable_stencil_test();
		void set_stencil_mask(u32 write_mask);
	};

	// Render target of an overlay pass. `color` and/or `depth_stencil` may be raw textures (e.g. a CAMetalDrawable
	// texture) or the MTL::Texture of an mtl::image. Textures must have MTL::TextureUsageRenderTarget.
	struct overlay_target
	{
		MTL::Texture* color = nullptr;          // Color attachment 0
		MTL::Texture* depth_stencil = nullptr;  // Depth and/or stencil attachment (chosen by pixel format)
		u32 level = 0;                          // Mip level rendered to
		u32 slice = 0;                          // Array slice / cube face rendered to

		// Optional: begin the pass by clearing the whole color attachment (loadAction Clear), e.g. the drawable before
		// video_out_calibration_pass draws the letterboxed image. Cheaper than a separate clear on Apple GPUs.
		bool clear_color_on_load = false;
		color4f clear_color_value = { 0.f, 0.f, 0.f, 1.f };

		overlay_target() = default;
		overlay_target(MTL::Texture* texture, u32 level = 0, u32 slice = 0);                 // Color or depth by format
		overlay_target(const mtl::image* image, u32 level = 0, u32 slice = 0);               // Color or depth by format
		overlay_target(const mtl::image* color_image, const mtl::image* depth_stencil_image); // Both (either may be null)

		MTL::Texture* primary() const { return color ? color : depth_stencil; }
		u32 width() const;
		u32 height() const;
		u32 samples() const;
		bool has_depth() const;
		bool has_stencil() const;
	};

	// TODO: Refactor text print class to inherit from this base class
	struct overlay_pass
	{
		MTL::SamplerMinMagFilter m_sampler_filter = MTL::SamplerMinMagFilterLinear;
		u32 m_num_usable_samplers = 1;
		u32 m_num_input_attachments = 0;
		u32 m_num_uniform_buffers = 1;

		std::unordered_map<u64, std::unique_ptr<glsl::program>> m_program_cache;
		std::array<std::unique_ptr<mtl::sampler>, 2> m_samplers; // [nearest, linear]
		const mtl::render_device* m_device = nullptr;

		std::string vs_src;
		std::string fs_src;

		overlay_pipeline_config renderpass_config;

		bool initialized = false;

		// Attachments this pass fully overwrites inside the viewport (loadAction DontCare instead of Load when the
		// viewport covers the whole target).
		bool m_overwrites_color = false;
		bool m_overwrites_depth = false;
		bool m_overwrites_stencil = false;

		u32 num_drawable_elements = 4;
		u32 first_vertex = 0;

		// Uniform and vertex data live in mtl::get_scratch_heap() and are bound by GPU address
		u32 m_ubo_length = 128;
		u64 m_ubo_offset = 0;
		u64 m_vao_offset = 0;
		u64 m_vao_length = 0;

		overlay_pass();
		virtual ~overlay_pass();

		overlay_pass(const overlay_pass&) = delete;
		overlay_pass& operator=(const overlay_pass&) = delete;

		u64 get_pipeline_key(const glsl::graphics_pipeline_state& state) const;

		virtual void update_uniforms(mtl::command_list& /*cmd*/, glsl::program* /*program*/) {}

		virtual std::vector<glsl::program_input> get_vertex_inputs();
		virtual std::vector<glsl::program_input> get_fragment_inputs();

		int sampler_location(int index) const { return static_cast<int>(m_num_uniform_buffers) + index; }
		int input_attachment_location(int index) const { return static_cast<int>(m_num_uniform_buffers + m_num_usable_samplers) + index; }

		// Copies vertex data to the scratch heap; the vertex shader reads it from a storage buffer (set 0, binding 0)
		template <typename T>
		void upload_vertex_data(const T* data, u32 count)
		{
			upload_vertex_data_raw(data, count * sizeof(T));
		}

		void upload_vertex_data_raw(const void* data, usz size);
		void upload_uniform_data(const void* data, usz size);

		glsl::program* build_pipeline(u64 storage_key, const glsl::graphics_pipeline_state& state);
		glsl::program* load_program(mtl::command_list& cmd, const overlay_target& target, const std::vector<mtl::image_view*>& src);

		virtual void create(const mtl::render_device& dev);
		virtual void destroy();

		void free_resources();

		mtl::sampler* get_sampler(bool linear);

		// Render pass management. begin_pass/end_pass bracket one or more draw() calls on the same target.
		void begin_pass(mtl::command_list& cmd, const overlay_target& target, const areau& viewport);
		void end_pass(mtl::command_list& cmd);
		void draw(mtl::command_list& cmd, const areau& viewport, const overlay_target& target, const std::vector<mtl::image_view*>& src);

		// Hook to change attachment load/clear actions of the pass descriptor (called by begin_pass)
		virtual void configure_attachments(MTL4::RenderPassDescriptor* /*desc*/, const overlay_target& /*target*/, bool /*covers_target*/) {}

		virtual void emit_geometry(mtl::command_list& cmd, glsl::program* program);

		virtual void set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h);

		// Opens a render pass on `target`, draws once and closes the pass.
		void run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target, const std::vector<mtl::image_view*>& src);
		void run(mtl::command_list& cmd, const areau& viewport, mtl::image* target, const std::vector<mtl::image_view*>& src);
		void run(mtl::command_list& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src);
	};

	struct ui_overlay_renderer : public overlay_pass
	{
		f32 m_time = 0.f;
		f32 m_blur_strength = 0.f;
		color4f m_scale_offset;
		color4f m_color;
		bool m_pulse_glow = false;
		bool m_clip_enabled = false;
		bool m_disable_vertex_snap = false;
		rsx::overlays::texture_sampling_mode m_texture_type;
		areaf m_clip_region;
		coordf m_viewport;

		rsx::overlays::compiled_resource::sdf_config_t m_sdf_config{};

		std::vector<std::unique_ptr<mtl::image>> resources;
		std::unordered_map<u64, std::unique_ptr<mtl::image>> font_cache;
		std::unordered_map<u64, std::unique_ptr<mtl::image_view>> view_cache;
		std::unordered_map<u64, std::pair<u32, std::unique_ptr<mtl::image>>> temp_image_cache;
		std::unordered_map<u64, std::unique_ptr<mtl::image_view>> temp_view_cache;
		rsx::overlays::primitive_type m_current_primitive_type = rsx::overlays::primitive_type::quad_list;

		// Scratch for primitive expansion (quads / fans -> triangle lists)
		std::vector<::vertex> m_expanded_verts;

		static constexpr u32 vertex_push_constants_size = 68;
		static constexpr u32 fragment_push_constants_size = 60;

		ui_overlay_renderer();

		void upload_simple_texture(mtl::image* tex, mtl::command_list& cmd,
			mtl::data_heap& upload_heap, u32 w, u32 h, u32 layers, bool font, const void* pixel_src);

		mtl::image_view* upload_simple_texture(const mtl::render_device& dev, mtl::command_list& cmd,
			mtl::data_heap& upload_heap, u64 key, u32 w, u32 h, u32 layers, bool font, bool temp, const void* pixel_src, u32 owner_uid);

		void init(mtl::command_list& cmd, mtl::data_heap& upload_heap);

		void destroy() override;

		void remove_temp_resources(u32 key);

		mtl::image_view* find_font(const rsx::overlays::font* font, mtl::command_list& cmd, mtl::data_heap& upload_heap);
		mtl::image_view* find_temp_image(const rsx::overlays::image_info_base* desc, mtl::command_list& cmd, mtl::data_heap& upload_heap, u32 owner_uid);

		std::vector<glsl::program_input> get_vertex_inputs() override;
		std::vector<glsl::program_input> get_fragment_inputs() override;

		void update_uniforms(mtl::command_list& cmd, glsl::program* program) override;

		void set_primitive_type(rsx::overlays::primitive_type type);

		void emit_geometry(mtl::command_list& cmd, glsl::program* program) override;

		// Renders `ui` into `target` (e.g. the drawable texture). Texture uploads use `upload_heap` and are recorded
		// before the render pass is opened; all draw commands share one render pass.
		void run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target,
			mtl::data_heap& upload_heap, rsx::overlays::overlay& ui);
	};

	struct attachment_clear_pass : public overlay_pass
	{
		color4f clear_color = { 0.f, 0.f, 0.f, 0.f };
		color4f colormask = { 1.f, 1.f, 1.f, 1.f };
		coordu region = {};

		static constexpr u32 vertex_push_constants_size = 32;
		static_assert(vertex_push_constants_size == (sizeof(clear_color) + sizeof(colormask)));

		attachment_clear_pass();

		std::vector<glsl::program_input> get_vertex_inputs() override;

		void update_uniforms(mtl::command_list& cmd, glsl::program* program) override;

		void set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h) override;

		void configure_attachments(MTL4::RenderPassDescriptor* desc, const overlay_target& target, bool covers_target) override;

		// Clears `rect` of the color target (color attachment only) with `color`, honouring the RSX clear mask
		// (0x10 R, 0x20 G, 0x40 B, 0x80 A). For MRT, call once per color target.
		void run(mtl::command_list& cmd, const overlay_target& target, const coordu& rect, u32 clearmask, color4f color);
	};

	struct stencil_clear_pass : public overlay_pass
	{
		coordu region = {};

		stencil_clear_pass();

		void set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h) override;

		// Partial stencil clear of a depth-stencil target through the stencil write mask
		void run(mtl::command_list& cmd, const overlay_target& target, const coordu& rect, u32 stencil_clear, u32 stencil_write_mask);
	};

	struct video_out_calibration_pass : public overlay_pass
	{
		union config_t
		{
			struct
			{
				float gamma;
				int   limit_range;
				int   stereo_display_mode;
				int   stereo_image_count;
				color4_base<float> left_anaglyph_matrix[3];
				color4_base<float> right_anaglyph_matrix[3];
			};

			float data[(
				sizeof(gamma) +
				sizeof(limit_range) +
				sizeof(stereo_display_mode) +
				sizeof(stereo_image_count) +
				sizeof(left_anaglyph_matrix) +
				sizeof(right_anaglyph_matrix)
				) / sizeof(float)];
		}
		config = {};

		static constexpr u32 fragment_push_constants_size = 112;

		video_out_calibration_pass();

		std::vector<glsl::program_input> get_fragment_inputs() override;

		void update_uniforms(mtl::command_list& cmd, glsl::program* /*program*/) override;

		void run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target,
			const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
			bool stereo_enabled);
	};

	// Scaled copy of a texture region into a render target region (Metal has no vkCmdBlitImage).
	// The base class writes color; the derived classes write depth ([[depth]]), stencil ([[stencil]]) or both.
	struct blit_pass : public overlay_pass
	{
		u32 m_dst_aspect = aspect_color;
		f32 m_src_rect[4] = { 0.f, 0.f, 1.f, 1.f }; // Normalized source origin (xy) and extent (zw), negative extent mirrors

		static constexpr u32 vertex_push_constants_size = 16;

		explicit blit_pass(u32 dst_aspect = aspect_color);

		std::vector<glsl::program_input> get_vertex_inputs() override;

		void update_uniforms(mtl::command_list& cmd, glsl::program* program) override;

		// Draws `src_area` (texel coordinates of the base level of src[0]) into `dst_area` (texel coordinates of the
		// target level). Either area may be mirrored (x1 > x2 or y1 > y2). src holds one view, or {depth, stencil}
		// views for depth_stencil_blit_pass. Views must be single-sampled 2D views.
		void run(mtl::command_list& cmd, const overlay_target& target, const areai& dst_area,
			const std::vector<mtl::image_view*>& src, const areai& src_area, bool linear_filter);
	};

	struct depth_blit_pass : public blit_pass
	{
		depth_blit_pass() : blit_pass(aspect_depth) {}
	};

	struct stencil_blit_pass : public blit_pass
	{
		stencil_blit_pass() : blit_pass(aspect_stencil) {}
	};

	struct depth_stencil_blit_pass : public blit_pass
	{
		depth_stencil_blit_pass() : blit_pass(aspect_depth_stencil) {}
	};

	// TODO: Replace with a proper manager
	extern std::unordered_map<u32, std::unique_ptr<mtl::overlay_pass>> g_overlay_passes;

	template<class T>
	T* get_overlay_pass()
	{
		const u32 index = stx::typeindex<id_manager::typeinfo, T>();
		auto& e = g_overlay_passes[index];

		if (!e)
		{
			e = std::make_unique<T>();
			e->create(*mtl::get_current_renderer());
		}

		return static_cast<T*>(e.get());
	}

	// Per-frame hook (VK parity; overlay data lives in the scratch heap, nothing to reset)
	void reset_overlay_passes();

	// Destroys all overlay passes, their pipelines/samplers/textures, the depth-stencil state cache, the fallback
	// textures and the blit scratch images. Call on shutdown once the GPU is idle, before the resource manager and
	// the device are destroyed.
	void destroy_overlay_passes();

	// Releases the scratch images used by copy_scaled_image (part of destroy_overlay_passes). Implemented in MTLBlit.cpp.
	void destroy_blit_resources();
}
