#include "stdafx.h"
#include "../Common/BufferUtils.h"
#include "../Program/GLSLCommon.h"
#include "../rsx_methods.h"

#include "MTLFormats.h"
#include "MTLGSRender.h"
#include "MTLRenderPass.h"
#include "mtlutils/buffer_object.h"

#include "Emu/RSX/NV47/HW/context_accessors.define.h"

#include <map>

namespace mtl
{
	// MTLGSRender.cpp
	u8 get_topology_class(MTL::PrimitiveType type);

	MTL::TextureType get_view_type(rsx::texture_dimension_extended type)
	{
		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			// The shader translator emits 2D (height 1) textures for 1D samplers
		case rsx::texture_dimension_extended::texture_dimension_2d:
			return MTL::TextureType2D;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			return MTL::TextureTypeCube;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			return MTL::TextureType3D;
		default: fmt::throw_exception("Unreachable");
		}
	}

	MTL::StencilOperation get_stencil_op(rsx::stencil_op op)
	{
		switch (op)
		{
		case rsx::stencil_op::keep: return MTL::StencilOperationKeep;
		case rsx::stencil_op::zero: return MTL::StencilOperationZero;
		case rsx::stencil_op::replace: return MTL::StencilOperationReplace;
		case rsx::stencil_op::incr: return MTL::StencilOperationIncrementClamp;
		case rsx::stencil_op::decr: return MTL::StencilOperationDecrementClamp;
		case rsx::stencil_op::invert: return MTL::StencilOperationInvert;
		case rsx::stencil_op::incr_wrap: return MTL::StencilOperationIncrementWrap;
		case rsx::stencil_op::decr_wrap: return MTL::StencilOperationDecrementWrap;
		default:
			fmt::throw_exception("Unknown stencil op: 0x%x", static_cast<u32>(op));
		}
	}

	MTL::Winding get_front_face(rsx::front_face ffv)
	{
		// 1:1 with VkFrontFace: the shader translator flips clip-space Y (SPIRV-Cross flip_vert_y) exactly like MoltenVK,
		// so the framebuffer-space winding seen by Metal matches what Vulkan sees.
		switch (ffv)
		{
		case rsx::front_face::cw: return MTL::WindingClockwise;
		case rsx::front_face::ccw: return MTL::WindingCounterClockwise;
		default:
			fmt::throw_exception("Unknown front face value: 0x%x", static_cast<u32>(ffv));
		}
	}

	// ---- Depth-stencil state keys -----------------------------------------------------------------------------------
	// Metal has no dynamic stencil masks, so the full MTLDepthStencilDescriptor is part of the key:
	// [0..2] depth compare | [3] depth write | [4] stencil enable |
	// [8..35] front face (fail, zfail, pass, func: 3 bits each, read mask, write mask: 8 bits each) | [36..63] back face
	struct stencil_face_desc
	{
		MTL::StencilOperation fail = MTL::StencilOperationKeep;
		MTL::StencilOperation zfail = MTL::StencilOperationKeep;
		MTL::StencilOperation pass = MTL::StencilOperationKeep;
		MTL::CompareFunction func = MTL::CompareFunctionAlways;
		u8 read_mask = 0xFF;
		u8 write_mask = 0xFF;

		u64 encode() const
		{
			return (static_cast<u64>(fail) & 7) |
				((static_cast<u64>(zfail) & 7) << 3) |
				((static_cast<u64>(pass) & 7) << 6) |
				((static_cast<u64>(func) & 7) << 9) |
				(static_cast<u64>(read_mask) << 12) |
				(static_cast<u64>(write_mask) << 20);
		}

		static stencil_face_desc decode(u64 bits)
		{
			stencil_face_desc result;
			result.fail = static_cast<MTL::StencilOperation>(bits & 7);
			result.zfail = static_cast<MTL::StencilOperation>((bits >> 3) & 7);
			result.pass = static_cast<MTL::StencilOperation>((bits >> 6) & 7);
			result.func = static_cast<MTL::CompareFunction>((bits >> 9) & 7);
			result.read_mask = static_cast<u8>(bits >> 12);
			result.write_mask = static_cast<u8>(bits >> 20);
			return result;
		}
	};

	constexpr u64 ds_key_depth_write = (1ull << 3);
	constexpr u64 ds_key_stencil_enable = (1ull << 4);

	u64 make_depth_stencil_key(MTL::CompareFunction depth_func, bool depth_write, const stencil_face_desc* front, const stencil_face_desc* back)
	{
		u64 key = static_cast<u64>(depth_func) & 7;
		key |= depth_write ? ds_key_depth_write : 0ull;

		if (front && back)
		{
			key |= ds_key_stencil_enable;
			key |= (front->encode() << 8);
			key |= (back->encode() << 36);
		}

		return key;
	}

	u64 encode_depth_stencil_state(const rsx::context* ctx)
	{
		MTL::CompareFunction depth_func = MTL::CompareFunctionAlways;
		bool depth_write = false;

		if (REGS(ctx)->depth_test_enabled())
		{
			//NOTE: Like stencil, depth write is meaningless without depth test
			depth_func = mtl::get_compare_function(REGS(ctx)->depth_func());
			depth_write = REGS(ctx)->depth_write_enabled();
		}

		if (!REGS(ctx)->stencil_test_enabled())
		{
			return make_depth_stencil_key(depth_func, depth_write, nullptr, nullptr);
		}

		stencil_face_desc front, back;
		front.fail = get_stencil_op(REGS(ctx)->stencil_op_fail());
		front.zfail = get_stencil_op(REGS(ctx)->stencil_op_zfail());
		front.pass = get_stencil_op(REGS(ctx)->stencil_op_zpass());
		front.func = mtl::get_compare_function(REGS(ctx)->stencil_func());
		front.read_mask = REGS(ctx)->stencil_func_mask();
		front.write_mask = REGS(ctx)->stencil_mask();

		if (REGS(ctx)->two_sided_stencil_test_enabled())
		{
			back.fail = get_stencil_op(REGS(ctx)->back_stencil_op_fail());
			back.zfail = get_stencil_op(REGS(ctx)->back_stencil_op_zfail());
			back.pass = get_stencil_op(REGS(ctx)->back_stencil_op_zpass());
			back.func = mtl::get_compare_function(REGS(ctx)->back_stencil_func());
			back.read_mask = REGS(ctx)->back_stencil_func_mask();
			back.write_mask = REGS(ctx)->back_stencil_mask();
		}
		else
		{
			back = front;
		}

		return make_depth_stencil_key(depth_func, depth_write, &front, &back);
	}

	mtl::rasterizer_state decode_rasterizer_state(const rsx::context* ctx)
	{
		mtl::rasterizer_state state{};
		state.front_face = get_front_face(REGS(ctx)->front_face_mode());
		state.depth_clip_mode = (REGS(ctx)->depth_clamp_enabled() || !REGS(ctx)->depth_clip_enabled())
			? MTL::DepthClipModeClamp
			: MTL::DepthClipModeClip;
		state.fill_mode = MTL::TriangleFillModeFill;

		if (REGS(ctx)->cull_face_enabled())
		{
			switch (REGS(ctx)->cull_face_mode())
			{
			case rsx::cull_face::back:
				state.cull_mode = MTL::CullModeBack;
				break;
			case rsx::cull_face::front:
				state.cull_mode = MTL::CullModeFront;
				break;
			case rsx::cull_face::front_and_back:
				// Metal has no FrontAndBack; polygons are skipped by the renderer (lines/points still draw)
				state.cull_mode = MTL::CullModeNone;
				state.cull_all_polygons = true;
				break;
			default:
				fmt::throw_exception("Unknown cull face value: 0x%x", static_cast<u32>(REGS(ctx)->cull_face_mode()));
			}
		}

		state.depth_stencil_key = encode_depth_stencil_state(ctx);
		return state;
	}

	// Linear filtering of depth formats: always legal through comparison samplers (sample_compare).
	// Plain sampling of 32-bit float depth can only be filtered on Apple9+ (M3 and later); Depth16Unorm is always filterable.
	static bool is_depth_format_filterable(MTL::PixelFormat format, bool shader_compare, bool apple9)
	{
		switch (format)
		{
		case MTL::PixelFormatDepth16Unorm:
			return true;
		case MTL::PixelFormatDepth32Float:
		case MTL::PixelFormatDepth32Float_Stencil8:
			return shader_compare || apple9;
		case MTL::PixelFormatX32_Stencil8:
		case MTL::PixelFormatStencil8:
			return false;
		default:
			// Colour reinterpretation of depth data (e.g. RGBA8 views)
			return !is_depth_format(format) && !is_stencil_format(format);
		}
	}

	// ---- In-pass attachment clears ------------------------------------------------------------------------------------
	// Metal has no vkCmdClearAttachments. Scissored or channel-masked clears draw a quad inside the renderer's main pass
	// (no pass split): the pipeline writes the clear colour with the RSX channel mask and/or the clear depth through
	// gl_Position.z, and stencil through a Replace stencil op with the RSX stencil write mask.
	namespace inpass_clear
	{
		struct request
		{
			u8 color_write_mask = 0;      // MTL::ColorWriteMask bits applied to every colour attachment
			color4f color{};
			bool depth = false;
			bool stencil = false;
			f32 depth_value = 1.f;
			u8 stencil_value = 0;
			u8 stencil_write_mask = 0xFF;
		};

		struct push_constants_t
		{
			f32 color[4];
			f32 depth;
			f32 pad[3];
		};

		static_assert(sizeof(push_constants_t) == 32);

		static std::map<std::pair<u64, u32>, std::unique_ptr<glsl::program>> g_programs;

		static std::vector<glsl::program_input> get_inputs(::glsl::program_domain domain)
		{
			return
			{
				glsl::program_input::make(
					domain,
					"push_constants_block",
					glsl::input_type_push_constant,
					domain == ::glsl::glsl_vertex_program ? glsl::binding_set_index_vertex : glsl::binding_set_index_fragment,
					umax,
					glsl::push_constant_ref{ .offset = 0, .size = sizeof(push_constants_t) })
			};
		}

		static glsl::program* get_program(const mtl::framebuffer_info& fbo, u64 renderpass_key, u8 color_write_mask)
		{
			const auto key = std::make_pair(renderpass_key, u32{ color_write_mask } | (fbo.color_count << 8));
			if (auto found = g_programs.find(key); found != g_programs.end())
			{
				return found->second.get();
			}

			const std::string vs_src =
				"#version 450\n"
				"layout(push_constant) uniform push_constants_block\n"
				"{\n"
				"	vec4 clear_color;\n"
				"	float clear_depth;\n"
				"};\n\n"
				"void main()\n"
				"{\n"
				"	const vec2 positions[4] = vec2[4](vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.));\n"
				"	gl_Position = vec4(positions[gl_VertexIndex & 3], clear_depth, 1.);\n"
				"}\n";

			std::string fs_src =
				"#version 450\n"
				"layout(push_constant) uniform push_constants_block\n"
				"{\n"
				"	vec4 clear_color;\n"
				"	float clear_depth;\n"
				"};\n\n";

			for (u32 i = 0; i < fbo.color_count; ++i)
			{
				fs_src += fmt::format("layout(location=%u) out vec4 ocol%u;\n", i, i);
			}

			fs_src += "\nvoid main()\n{\n";
			for (u32 i = 0; i < fbo.color_count; ++i)
			{
				fs_src += fmt::format("	ocol%u = clear_color;\n", i);
			}
			fs_src += "}\n";

			glsl::graphics_pipeline_state state{};
			state.color_count = fbo.color_count;
			for (u32 i = 0; i < fbo.color_count; ++i)
			{
				state.color[i].pixel_format = static_cast<u32>(fbo.color[i]->format());
				state.color[i].write_mask = color_write_mask;
				state.color[i].blend_enable = 0;
			}

			state.depth_stencil_format = fbo.depth_stencil ? static_cast<u32>(fbo.depth_stencil->format()) : 0u;
			state.sample_count = std::max<u8>(1, fbo.samples);
			state.topology_class = static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
			state.rasterization_enabled = 1;

			auto program = glsl::create_graphics_program(
				vs_src, get_inputs(::glsl::glsl_vertex_program),
				fs_src, get_inputs(::glsl::glsl_fragment_program),
				state);

			if (!program)
			{
				rsx_log.error("Metal: failed to build the attachment clear pipeline");
				return nullptr;
			}

			auto result = program.get();
			g_programs.emplace(key, std::move(program));
			return result;
		}
	}

	void destroy_inpass_clear_programs()
	{
		inpass_clear::g_programs.clear();
	}
}

bool MTLGSRender::is_render_pass_open() const
{
	return m_render_pass.is_open(*m_current_command_buffer);
}

MTL4::RenderCommandEncoder* MTLGSRender::get_render_encoder() const
{
	return m_render_pass.encoder(*m_current_command_buffer);
}

void MTLGSRender::update_render_pass_descriptor()
{
	m_draw_fbo_textures = m_draw_fbo.get_textures();
	m_draw_pass_visibility_buffer = m_occlusion_query_manager->get_visibility_result_buffer();
	m_draw_pass_desc = mtl::ref<MTL4::RenderPassDescriptor>(mtl::create_render_pass_descriptor(m_draw_fbo, m_draw_pass_visibility_buffer));
}

void MTLGSRender::begin_render_pass(const mtl::attachment_clear_info* clear)
{
	const bool has_clear = clear && !clear->empty();
	if (!has_clear && is_render_pass_open())
	{
		return;
	}

	ensure(m_draw_pass_desc, "Render pass requested without a valid surface configuration");

	// End the current pass (ours or another component's) before starting a new one
	close_render_pass();

	if (m_draw_fbo.get_textures() != m_draw_fbo_textures)
	{
		// A bound surface replaced its texture (spill/unspill, destructive clone). Rebuild the attachment set.
		update_render_pass_descriptor();
	}

	// The visibility result buffer belongs to the descriptor; it changes when the query pool is replaced
	if (const auto visibility_buffer = m_occlusion_query_manager->get_visibility_result_buffer();
		visibility_buffer != m_draw_pass_visibility_buffer)
	{
		mtl::set_visibility_result_buffer(m_draw_pass_desc.get(), visibility_buffer);
		m_draw_pass_visibility_buffer = visibility_buffer;
	}

	if (has_clear)
	{
		mtl::apply_clear_load_ops(m_draw_pass_desc.get(), m_draw_fbo, *clear);
	}

	auto encoder = m_render_pass.begin(*m_current_command_buffer, m_draw_pass_desc.get());

	if (has_clear)
	{
		mtl::restore_load_ops(m_draw_pass_desc.get(), m_draw_fbo);
	}

	on_render_pass_begin(encoder);
}

void MTLGSRender::on_render_pass_begin(MTL4::RenderCommandEncoder* encoder)
{
	// Starting a new renderpass clobbers all dynamic state
	m_current_command_buffer->flags |= mtl::command_list::cb_reload_dynamic_state;
	m_encoder_state = { .pass_id = m_render_pass.pass_id() };

	// An open occlusion query must count in this pass too (visibility mode is encoder state)
	if ((m_current_command_buffer->flags & mtl::command_list::cb_has_open_query) && m_active_query_info)
	{
		const auto open_query = m_occlusion_map[m_active_query_info->driver_handle].indices.back();
		m_occlusion_query_manager->resume_query(encoder, open_query);
	}
}

void MTLGSRender::close_render_pass()
{
	// Ends the active render encoder whether it is our main pass or a pass opened by another component
	if (m_current_command_buffer->is_render_pass_open())
	{
		m_current_command_buffer->end_render_pass();
	}

	m_render_pass.reset();
}

void MTLGSRender::invalidate_render_pass()
{
	// Vulkan regenerates the render pass here (feedback loop layouts). Metal cannot make attachment writes visible to
	// texture reads inside a pass, so the pass is ended; the next draw reopens it behind a full queue barrier.
	split_render_pass();
}

void MTLGSRender::split_render_pass()
{
	if (is_render_pass_open())
	{
		close_render_pass();
		mtl::g_feedback_loop_pass_splits++;
	}
}

MTL::DepthStencilState* MTLGSRender::get_depth_stencil_state(u64 key)
{
	if (auto found = m_depth_stencil_states.find(key); found != m_depth_stencil_states.end())
	{
		return found->second.get();
	}

	mtl::autorelease_scope pool;

	auto desc = mtl::ref(MTL::DepthStencilDescriptor::alloc()->init());
	desc->setDepthCompareFunction(static_cast<MTL::CompareFunction>(key & 7));
	desc->setDepthWriteEnabled(!!(key & mtl::ds_key_depth_write));

	if (key & mtl::ds_key_stencil_enable)
	{
		const auto setup_face = [](u64 bits)
		{
			const auto face = mtl::stencil_face_desc::decode(bits);
			auto stencil = mtl::ref(MTL::StencilDescriptor::alloc()->init());
			stencil->setStencilCompareFunction(face.func);
			stencil->setStencilFailureOperation(face.fail);
			stencil->setDepthFailureOperation(face.zfail);
			stencil->setDepthStencilPassOperation(face.pass);
			stencil->setReadMask(face.read_mask);
			stencil->setWriteMask(face.write_mask);
			return stencil;
		};

		const auto front = setup_face(key >> 8);
		const auto back = setup_face(key >> 36);
		desc->setFrontFaceStencil(front.get());
		desc->setBackFaceStencil(back.get());
	}

	auto state = m_device->handle()->newDepthStencilState(desc.get());
	ensure(state, "Metal: failed to create depth-stencil state");

	auto result = state;
	m_depth_stencil_states.emplace(key, mtl::ref<MTL::DepthStencilState>(state));
	return result;
}

mtl::image_view* MTLGSRender::get_null_texture_view(rsx::texture_dimension_extended type, bool is_depth)
{
	// Shadow samplers are declared as depth2d<> (1D/2D) or depthcube<> which cannot take colour views.
	// NOTE: There is no 3D depth texture type in MSL; 3D shadow samplers cannot be declared and fall back to colour.
	if (is_depth && type != rsx::texture_dimension_extended::texture_dimension_3d)
	{
		const bool is_cube = (type == rsx::texture_dimension_extended::texture_dimension_cubemap);
		auto& null_texture = m_null_depth_textures[is_cube ? 1 : 0];

		if (!null_texture)
		{
			mtl::image_create_info info{};
			info.type = is_cube ? MTL::TextureTypeCube : MTL::TextureType2D;
			info.format = MTL::PixelFormatDepth32Float;
			info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
			info.format_class = RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32;
			info.layers = is_cube ? 6 : 1;

			null_texture = std::make_unique<mtl::viewable_image>(*m_device, info);
			null_texture->set_debug_name(is_cube ? "null depth cube texture" : "null depth texture");

			mtl::image_clear_value value{};
			value.depth = 1.f;
			mtl::clear_image(*m_current_command_buffer, null_texture.get(), value);
		}

		return null_texture->get_identity_view(mtl::aspect_depth);
	}

	return mtl::null_image_view(*m_current_command_buffer, mtl::get_view_type(type));
}

void MTLGSRender::update_draw_state()
{
	m_profiler.start();

	auto encoder = ensure(get_render_encoder());
	const auto& regs = rsx::method_registers;

	// Wide lines are not supported by Metal (1px lines)
	if (regs.current_draw_clause.primitive >= rsx::primitive_type::points &&
		regs.current_draw_clause.primitive <= rsx::primitive_type::line_strip &&
		!m_wide_lines_warning_logged &&
		regs.line_width() * resolution_scaling_config.scale_factor() > 1.f)
	{
		m_wide_lines_warning_logged = true;
		rsx_log.warning("Metal: wide lines are not supported; lines are rendered 1px wide.");
	}

	if (regs.blend_enabled_mask())
	{
		// Update blend constants
		auto blend_colors = rsx::get_constant_blend_colors();
		encoder->setBlendColor(blend_colors[0], blend_colors[1], blend_colors[2], blend_colors[3]);
	}

	// Fixed-function state that Vulkan keeps in the pipeline object is encoder state on Metal
	m_rasterizer_state = mtl::decode_rasterizer_state(m_ctx);
	encoder->setCullMode(m_rasterizer_state.cull_mode);
	encoder->setFrontFacingWinding(m_rasterizer_state.front_face);
	encoder->setDepthClipMode(m_rasterizer_state.depth_clip_mode);
	encoder->setTriangleFillMode(m_rasterizer_state.fill_mode);

	// Depth/stencil tests without the corresponding attachment are meaningless (and rejected by Metal validation)
	if (!m_draw_fbo.depth_stencil)
	{
		m_rasterizer_state.depth_stencil_key = mtl::make_depth_stencil_key(MTL::CompareFunctionAlways, false, nullptr, nullptr);
	}
	else if (!m_draw_fbo.has_stencil())
	{
		m_rasterizer_state.depth_stencil_key &= (mtl::ds_key_depth_write | 0x7ull);
	}

	if (auto state = get_depth_stencil_state(m_rasterizer_state.depth_stencil_key);
		state != m_encoder_state.depth_stencil)
	{
		encoder->setDepthStencilState(state);
		m_encoder_state.depth_stencil = state;
	}

	if (regs.stencil_test_enabled())
	{
		const bool two_sided_stencil = regs.two_sided_stencil_test_enabled();
		const u32 front_ref = regs.stencil_func_ref();
		const u32 back_ref = two_sided_stencil ? regs.back_stencil_func_ref() : front_ref;
		encoder->setStencilReferenceValues(front_ref, back_ref);
	}

	// The remaining dynamic state should only be set once and we have signals to enable/disable mid-renderpass
	if (!(m_current_command_buffer->flags & mtl::command_list::cb_reload_dynamic_state))
	{
		// Dynamic state already set
		m_frame_stats.setup_time += m_profiler.duration();
		return;
	}

	if (regs.poly_offset_fill_enabled())
	{
		// offset_bias is the constant factor, multiplied by the implementation factor R
		// offst_scale is the slope factor, multiplied by the triangle slope factor M
		// Depth is always 32-bit float on Metal, which behaves like the VK float path.
		encoder->setDepthBias(regs.poly_offset_bias(), regs.poly_offset_scale(), 0.f);
	}
	else
	{
		// Zero bias value - disables depth bias
		encoder->setDepthBias(0.f, 0.f, 0.f);
	}

	if (m_device->caps().depth_bounds)
	{
		f32 bounds_min, bounds_max;
		if (regs.depth_bounds_test_enabled())
		{
			// Update depth bounds min/max
			bounds_min = regs.depth_bounds_min();
			bounds_max = regs.depth_bounds_max();
		}
		else
		{
			// Avoid special case where min=max and depth bounds (incorrectly) fails
			bounds_min = std::min(0.f, regs.clip_min());
			bounds_max = std::max(1.f, regs.clip_max());
		}

		// No unrestricted depth range on Metal
		bounds_min = std::clamp(bounds_min, 0.f, 1.f);
		bounds_max = std::clamp(bounds_max, 0.f, 1.f);

		encoder->setDepthTestBounds(bounds_min, bounds_max);
	}
	else if (regs.depth_bounds_test_enabled() && !m_depth_bounds_warning_logged)
	{
		m_depth_bounds_warning_logged = true;
		rsx_log.warning("Metal: depth bounds test requested but not supported by this GPU (Apple10+ only). Ignored.");
	}

	bind_viewport();

	m_current_command_buffer->flags &= ~mtl::command_list::cb_reload_dynamic_state;
	m_graphics_state.clear(rsx::pipeline_state::polygon_offset_state_dirty | rsx::pipeline_state::depth_bounds_state_dirty);
	m_frame_stats.setup_time += m_profiler.duration();
}

void MTLGSRender::load_texture_env()
{
	// Load textures
	bool check_for_cyclic_refs = false;
	auto check_surface_cache_sampler_valid = [&](auto descriptor, const auto& tex)
	{
		if (!m_texture_cache.test_if_descriptor_expired(*m_current_command_buffer, m_rtts, descriptor, tex))
		{
			check_for_cyclic_refs |= descriptor->is_cyclic_reference;
			return true;
		}

		return false;
	};

	auto get_border_color = [&](const rsx::Texture auto& tex, bool remap_colorspace)
	{
		// Metal border colours bypass the view swizzle like Vulkan's without custom border colour remapping
		return rsx::decode_border_color(tex.border_color(remap_colorspace));
	};

	std::lock_guard lock(m_sampler_mutex);

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!fs_sampler_state[i])
		{
			fs_sampler_state[i] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.fragment_textures[i];
		const auto previous_format_class = fs_sampler_state[i]->format_class;

		if (!m_samplers_dirty &&
			!m_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_textures_dirty[i];
		m_textures_dirty[i] = false;

		if (!tex.enabled())
		{
			*sampler_state = {};
			m_fs_lod_bias[i] = 0.f;
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(*m_current_command_buffer, tex, m_rtts);
		if (!sampler_state->validate())
		{
			m_fs_lod_bias[i] = 0.f;
			continue;
		}

		if (sampler_state->is_cyclic_reference)
		{
			check_for_cyclic_refs |= true;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::fragment_program_state_dirty;
			}

			if (sampler_state->format_ex)
			{
				// Nothing to change, use cached sampler
				continue;
			}
		}

		sampler_state->format_ex = tex.format_ex();

		if (sampler_state->format_ex.texel_remap_control &&
			sampler_state->image_handle &&
			sampler_state->upload_context == rsx::texture_upload_context::shader_read &&
			(current_fp_metadata.bx2_texture_reads_mask & (1u << i)) == 0 &&
			!g_cfg.video.disable_hardware_texel_remapping) [[ unlikely ]]
		{
			// Check if we need to override the view format
			const auto mtl_format = sampler_state->image_handle->format();
			MTL::PixelFormat format_override = mtl_format;
			rsx::flags32_t flags_to_erase = 0u;
			rsx::flags32_t host_flags_to_set = 0u;

			if (sampler_state->format_ex.hw_SNORM_possible())
			{
				format_override = mtl::get_compatible_snorm_format(mtl_format);
				flags_to_erase = rsx::texture_control_bits::SEXT_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SNORM;
			}
			else if (sampler_state->format_ex.hw_SRGB_possible())
			{
				format_override = mtl::get_compatible_srgb_format(mtl_format);
				flags_to_erase = rsx::texture_control_bits::GAMMA_CTRL_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SRGB;
			}

			if (format_override != MTL::PixelFormatInvalid && format_override != mtl_format)
			{
				sampler_state->image_handle = sampler_state->image_handle->as(format_override);
				sampler_state->format_ex.texel_remap_control &= (~flags_to_erase);
				sampler_state->format_ex.host_features |= host_flags_to_set;
			}
		}

		MTL::SamplerMinMagFilter mag_filter;
		mtl::minification_filter min_filter;
		f32 min_lod = 0.f, max_lod = 0.f;
		f32 lod_bias = 0.f;

		const u32 texture_format = sampler_state->format_ex.format();
		bool compare_enabled = false;
		MTL::CompareFunction depth_compare_mode = MTL::CompareFunctionNever;

		// Mirror RSXThread's shadow_textures decision (the decompiler emits sample_compare only for those units).
		// A compare sampler must only be attached when the shader actually compares.
		if (texture_format >= CELL_GCM_TEXTURE_DEPTH24_D8 && texture_format <= CELL_GCM_TEXTURE_DEPTH16_FLOAT &&
			sampler_state->format_class != RSX_FORMAT_CLASS_COLOR &&
			!tex.alpha_kill_enabled() &&
			tex.zfunc() > rsx::comparison_function::never &&
			tex.zfunc() < rsx::comparison_function::always)
		{
			compare_enabled = true;
			depth_compare_mode = mtl::get_compare_function(tex.zfunc(), true);
		}

		const f32 af_level = mtl::max_aniso(tex.max_aniso());
		const auto wrap_s = mtl::mtl_wrap_mode(tex.wrap_s());
		const auto wrap_t = mtl::mtl_wrap_mode(tex.wrap_t());
		const auto wrap_r = mtl::mtl_wrap_mode(tex.wrap_r());

		// NOTE: Metal border colours are not swizzled. Only 3 fixed colours exist (see mtl::border_color_t).
		const bool sext_conv_required = (sampler_state->format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
		mtl::border_color_t border_color(MTL::SamplerBorderColorOpaqueBlack);

		if (rsx::is_border_clamped_texture(tex))
		{
			auto color_value = get_border_color(tex, sext_conv_required);
			if (sampler_state->format_ex.host_snorm_format_active())
			{
				// Convert the border color in host space (2N - 1)
				// HW does the conversion in integer space as (x - 128) / 127 which introduces a biasing error.
				const float bias_v = 128.f / 255.f;
				const float scale_v = 255.f / 127.f;

				color4f scale{ 1.f }, bias{ 0.f };
				const auto snorm_mask = tex.argb_signed();
				if (snorm_mask & 1) { scale.a = scale_v; bias.a = -bias_v; }
				if (snorm_mask & 2) { scale.r = scale_v; bias.r = -bias_v; }
				if (snorm_mask & 4) { scale.g = scale_v; bias.g = -bias_v; }
				if (snorm_mask & 8) { scale.b = scale_v; bias.b = -bias_v; }
				color_value = (color_value + bias) * scale;
			}

			border_color = mtl::border_color_t(color_value);
		}

		// Check if non-point filtering can even be used on this format
		bool can_sample_linear;
		if (sampler_state->format_class == RSX_FORMAT_CLASS_COLOR) [[likely]]
		{
			// Most PS3-like formats can be linearly filtered without problem
			// Exclude textures that require SNORM conversion however
			can_sample_linear = !sext_conv_required;
		}
		else if (sampler_state->format_class != rsx::classify_format(texture_format) &&
			(texture_format == CELL_GCM_TEXTURE_A8R8G8B8 || texture_format == CELL_GCM_TEXTURE_D8R8G8B8))
		{
			// Depth format redirected to BGRA8 resample stage. Do not filter to avoid bits leaking
			can_sample_linear = false;
		}
		else
		{
			// Pre-Apple9 GPUs cannot filter 32-bit float depth outside of comparison sampling
			const auto mtl_format = sampler_state->image_handle ? sampler_state->image_handle->image()->format() :
				mtl::get_compatible_sampler_format(sampler_state->external_subresource_desc.gcm_format);

			can_sample_linear = mtl::is_depth_format_filterable(mtl_format, compare_enabled, m_device->caps().apple9);
		}

		const auto mipmap_count = tex.get_exact_mipmap_count();
		min_filter = mtl::get_min_filter(tex.min_filter());

		if (can_sample_linear)
		{
			mag_filter = mtl::get_mag_filter(tex.mag_filter());
		}
		else
		{
			mag_filter = MTL::SamplerMinMagFilterNearest;
			min_filter.filter = MTL::SamplerMinMagFilterNearest;
			min_filter.mipmap_mode = MTL::SamplerMipFilterNearest;
		}

		if (min_filter.sample_mipmaps && mipmap_count > 1)
		{
			f32 actual_mipmaps;
			if (sampler_state->upload_context == rsx::texture_upload_context::shader_read)
			{
				actual_mipmaps = static_cast<f32>(mipmap_count);
			}
			else if (sampler_state->external_subresource_desc.op != rsx::deferred_request_command::nop)
			{
				actual_mipmaps = sampler_state->external_subresource_desc.exact_mip_count();
			}
			else
			{
				actual_mipmaps = 1.f;
			}

			if (actual_mipmaps > 1.f)
			{
				min_lod = tex.min_lod();
				max_lod = tex.max_lod();
				lod_bias = tex.bias();

				min_lod = std::min(min_lod, actual_mipmaps - 1.f);
				max_lod = std::min(max_lod, actual_mipmaps - 1.f);

				if (min_filter.mipmap_mode == MTL::SamplerMipFilterNearest)
				{
					// Round to nearest 0.5 to work around some broken games
					// Unlike openGL, sampler parameters cannot be dynamically changed on Metal, leading to many permutations
					lod_bias = std::floor(lod_bias * 2.f + 0.5f) * 0.5f;
				}
			}
			else
			{
				min_lod = max_lod = lod_bias = 0.f;
				min_filter.mipmap_mode = MTL::SamplerMipFilterNearest;
			}
		}

		mtl::sampler_create_info info{};
		info.clamp_u = wrap_s;
		info.clamp_v = wrap_t;
		info.clamp_w = wrap_r;
		info.unnormalized_coordinates = false;
		info.mip_lod_bias = lod_bias;
		info.max_anisotropy = af_level;

		// Pre-Apple10 samplers ignore mip_lod_bias; programs built with requires_lod_bias add it in the shader
		m_fs_lod_bias[i] = lod_bias;
		info.min_lod = min_lod;
		info.max_lod = max_lod;
		info.min_filter = min_filter.filter;
		info.mag_filter = mag_filter;
		info.mip_filter = min_filter.mipmap_mode;
		info.border_color = border_color;
		info.depth_compare = compare_enabled;
		info.compare_function = depth_compare_mode;

		if (fs_sampler_handles[i] && fs_sampler_handles[i]->info == info)
		{
			continue;
		}

		fs_sampler_handles[i] = mtl::get_resource_manager()->get_sampler(*m_device, fs_sampler_handles[i], info);
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!vs_sampler_state[i])
		{
			vs_sampler_state[i] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.vertex_textures[i];
		const auto previous_format_class = sampler_state->format_class;

		if (!m_samplers_dirty &&
			!m_vertex_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_vertex_textures_dirty[i];
		m_vertex_textures_dirty[i] = false;

		if (!rsx::method_registers.vertex_textures[i].enabled())
		{
			*sampler_state = {};
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(*m_current_command_buffer, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (sampler_state->is_cyclic_reference || sampler_state->external_subresource_desc.do_not_cache)
		{
			check_for_cyclic_refs |= true;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::vertex_program_state_dirty;
			}

			if (vs_sampler_handles[i])
			{
				continue;
			}
		}

		const bool unnormalized_coords = !!(tex.format() & CELL_GCM_TEXTURE_UN);
		const auto min_lod = tex.min_lod();
		const auto max_lod = tex.max_lod();
		const auto wrap_s = mtl::mtl_wrap_mode(tex.wrap_s());
		const auto wrap_t = mtl::mtl_wrap_mode(tex.wrap_t());

		const auto border_color = is_border_clamped_texture(tex)
			? mtl::border_color_t(get_border_color(tex, false))
			: mtl::border_color_t(MTL::SamplerBorderColorOpaqueBlack);

		mtl::sampler_create_info info{};
		info.clamp_u = wrap_s;
		info.clamp_v = wrap_t;
		info.clamp_w = MTL::SamplerAddressModeRepeat;
		info.unnormalized_coordinates = unnormalized_coords;
		info.mip_lod_bias = 0.f;
		info.max_anisotropy = 1.f;
		info.min_lod = min_lod;
		info.max_lod = max_lod;
		info.min_filter = MTL::SamplerMinMagFilterNearest;
		info.mag_filter = MTL::SamplerMinMagFilterNearest;
		info.mip_filter = MTL::SamplerMipFilterNearest;
		info.border_color = border_color;

		if (vs_sampler_handles[i] && vs_sampler_handles[i]->info == info)
		{
			continue;
		}

		vs_sampler_handles[i] = mtl::get_resource_manager()->get_sampler(*m_device, vs_sampler_handles[i], info);
	}

	m_samplers_dirty.store(false);

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
	{
		// No depth framebuffer fetch on Metal: the depth buffer is sampled as a texture. End the pass so that the
		// sampling pass is ordered after every previous depth write.
		auto ds = ensure(m_rtts.m_bound_depth_stencil.second, "Invalid FS export configuration.");
		ds->texture_barrier(*m_current_command_buffer);

		check_for_cyclic_refs = true;
	}

	if (check_for_cyclic_refs)
	{
		// Feedback loop: end the render pass (counted as a pass split)
		invalidate_render_pass();
	}
}

bool MTLGSRender::bind_texture_env()
{
	bool out_of_memory = false;

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		if (m_fs_binding_table->ftex_location[i] == umax)
		{
			// Corrupt shader table
			break;
		}

		mtl::image_view* view = nullptr;
		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());

		if (rsx::method_registers.fragment_textures[i].enabled() &&
			sampler_state->validate())
		{
			if (view = sampler_state->image_handle; !view)
			{
				//Requires update, copy subresource
				if (!(view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, sampler_state->external_subresource_desc)))
				{
					out_of_memory = true;
				}
			}
		}

		if (view) [[likely]]
		{
			m_program->bind_uniform({ view, fs_sampler_handles[i] },
				mtl::glsl::binding_set_index_fragment,
				m_fs_binding_table->ftex_location[i]);

			if (current_fragment_program.texture_state.redirected_textures & (1 << i))
			{
				// Stencil mirror required
				auto root_image = static_cast<mtl::viewable_image*>(view->image());
				auto stencil_view = root_image->get_view(rsx::default_remap_vector, mtl::aspect_stencil);

				if (!m_stencil_mirror_sampler)
				{
					mtl::sampler_create_info info{};
					info.clamp_u = MTL::SamplerAddressModeClampToBorderColor;
					info.clamp_v = MTL::SamplerAddressModeClampToBorderColor;
					info.clamp_w = MTL::SamplerAddressModeClampToBorderColor;
					info.min_lod = 0.f;
					info.max_lod = 0.f;
					info.min_filter = MTL::SamplerMinMagFilterNearest;
					info.mag_filter = MTL::SamplerMinMagFilterNearest;
					info.mip_filter = MTL::SamplerMipFilterNearest;
					info.border_color = mtl::border_color_t(MTL::SamplerBorderColorOpaqueBlack);

					m_stencil_mirror_sampler = std::make_unique<mtl::sampler>(*m_device, info);
				}

				m_program->bind_uniform({ stencil_view, m_stencil_mirror_sampler.get() },
					mtl::glsl::binding_set_index_fragment,
					m_fs_binding_table->ftex_stencil_location[i]);
			}
		}
		else
		{
			const bool is_shadow = !!(current_fragment_program.texture_state.shadow_textures & (1u << i));
			const auto null_view = get_null_texture_view(current_fragment_program.get_texture_dimension(i), is_shadow);
			const mtl::glsl::image_binding_info desc = { null_view, mtl::null_sampler() };

			m_program->bind_uniform(desc,
				mtl::glsl::binding_set_index_fragment,
				m_fs_binding_table->ftex_location[i]);

			if (current_fragment_program.texture_state.redirected_textures & (1 << i))
			{
				m_program->bind_uniform(desc,
					mtl::glsl::binding_set_index_fragment,
					m_fs_binding_table->ftex_stencil_location[i]);
			}
		}
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		if (m_vs_binding_table->vtex_location[i] == umax)
		{
			// Corrupt shader
			break;
		}

		if (!rsx::method_registers.vertex_textures[i].enabled())
		{
			const auto null_view = get_null_texture_view(current_vertex_program.get_texture_dimension(i), false);
			m_program->bind_uniform({ null_view, mtl::null_sampler() },
				mtl::glsl::binding_set_index_vertex,
				m_vs_binding_table->vtex_location[i]);

			continue;
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		auto image_ptr = sampler_state->image_handle;

		if (!image_ptr && sampler_state->validate())
		{
			if (!(image_ptr = m_texture_cache.create_temporary_subresource(*m_current_command_buffer, sampler_state->external_subresource_desc)))
			{
				out_of_memory = true;
			}
		}

		if (!image_ptr)
		{
			rsx_log.error("Texture upload failed to vtexture index %d. Binding null sampler.", i);
			const auto null_view = get_null_texture_view(current_vertex_program.get_texture_dimension(i), false);

			m_program->bind_uniform({ null_view, mtl::null_sampler() },
				mtl::glsl::binding_set_index_vertex,
				m_vs_binding_table->vtex_location[i]);

			continue;
		}

		m_program->bind_uniform({ image_ptr, vs_sampler_handles[i] },
			mtl::glsl::binding_set_index_vertex,
			m_vs_binding_table->vtex_location[i]);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
	{
		auto ds = ensure(m_rtts.m_bound_depth_stencil.second);
		auto view = ds->get_view(rsx::default_remap_vector, mtl::aspect_depth);
		m_program->bind_uniform({ view, mtl::null_sampler() }, mtl::glsl::binding_set_index_fragment, m_fs_binding_table->frag_depth_input_location);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		// Framebuffer fetch ([[color(n)]]): the attachments are read in the tile, nothing to bind and no barrier.
		ensure(current_fragment_program.mrt_buffers_count == m_draw_buffers.size());
	}

	return out_of_memory;
}

void MTLGSRender::emit_geometry(u32 sub_index)
{
	auto &draw_call = rsx::method_registers.current_draw_clause;
	m_profiler.start();

	const rsx::flags32_t vertex_state_mask = rsx::vertex_base_changed | rsx::vertex_arrays_changed;
	const rsx::flags32_t state_flags = (sub_index == 0) ? rsx::vertex_arrays_changed : draw_call.execute_pipeline_dependencies(m_ctx);

	if (state_flags & rsx::vertex_arrays_changed)
	{
		m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	}
	else if (state_flags & rsx::vertex_base_changed)
	{
		// Rebase vertex bases instead of
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
			const auto vertex_base_offset = rsx::method_registers.vertex_data_base_offset();
			info->real_offset_address = rsx::get_address(rsx::get_vertex_offset_from_base(vertex_base_offset, info->base_offset), info->memory_location);
		}
	}
	else
	{
		// Discard cached results
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
		}
	}

	if ((state_flags & vertex_state_mask) && !m_vertex_layout.validate())
	{
		// No vertex inputs enabled
		// Execute remainining pipeline barriers with NOP draw
		do
		{
			draw_call.execute_pipeline_dependencies(m_ctx);
		}
		while (draw_call.next());

		draw_call.end();
		return;
	}

	// Programs data is dependent on vertex state
	auto upload_info = upload_vertex_data();
	if (!upload_info.vertex_draw_count)
	{
		// Malformed vertex setup; abort
		return;
	}

	m_frame_stats.vertex_upload_time += m_profiler.duration();

	// Faults are allowed during vertex upload. Ensure consistent CB state after uploads.
	if (m_current_command_buffer->flags & mtl::command_list::cb_load_occluson_task)
	{
		u32 occlusion_id = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
		if (occlusion_id == umax)
		{
			// Force flush
			rsx_log.warning("[Performance Warning] Out of free occlusion slots. Forcing hard sync.");
			ZCULL_control::sync(this);

			occlusion_id = m_occlusion_query_manager->allocate_query(*m_current_command_buffer);
			if (occlusion_id == umax)
			{
				//rsx_log.error("Occlusion pool overflow");
				if (m_current_task) m_current_task->result = 1;
			}
		}

		if (occlusion_id != umax)
		{
			// Begin query. If the main pass is not open yet, on_render_pass_begin() arms it.
			m_occlusion_query_manager->begin_query(*m_current_command_buffer, get_render_encoder(), occlusion_id);

			auto& data = m_occlusion_map[m_active_query_info->driver_handle];
			data.indices.push_back(occlusion_id);
			data.set_sync_command_buffer(m_current_command_buffer);

			m_current_command_buffer->flags &= ~mtl::command_list::cb_load_occluson_task;
			m_current_command_buffer->flags |= (mtl::command_list::cb_has_occlusion_task | mtl::command_list::cb_has_open_query);
		}
	}

	const mtl::buffer_view* persistent_buffer = m_persistent_attribute_storage ? m_persistent_attribute_storage.get() : null_buffer_view.get();
	const mtl::buffer_view* volatile_buffer = m_volatile_attribute_storage ? m_volatile_attribute_storage.get() : null_buffer_view.get();
	bool update_descriptors = false;

	if (m_current_draw.subdraw_id == 0)
	{
		update_descriptors = true;

		// Allocate stream layout memory for this batch
		const u64 alloc_size = rsx::method_registers.current_draw_clause.pass_count() * 168;
		m_vertex_layout_dynamic_offset = m_vertex_layout_ring_info.alloc<8>(alloc_size);
	}

	// Update vertex fetch parameters
	update_vertex_env(sub_index, upload_info);

	if (update_descriptors)
	{
		m_program->bind_uniform(persistent_buffer, mtl::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location);
		m_program->bind_uniform(volatile_buffer, mtl::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location + 1);

		// The layout ring may have been swapped (grown) by the allocation above
		m_program->bind_uniform(mtl::glsl::buffer_binding_info(m_vertex_layout_ring_info.heap.get(), 0, m_vertex_layout_ring_info.size()),
			mtl::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location + 2);
	}

	if (sub_index && m_vs_binding_table->cbuf_location != umax)
	{
		// Transform constants may have been patched (and their ring grown) between sub-draws
		m_program->bind_uniform(mtl::glsl::buffer_binding_info(m_transform_constants_ring_info.heap.get(), 0, m_transform_constants_ring_info.size()),
			mtl::glsl::binding_set_index_vertex, m_vs_binding_table->cbuf_location);
	}

	bool reload_state = (!m_current_draw.subdraw_id++);

	// (Re)open the main pass. It may have been ended by a copy/compute operation, a pass of another component, a
	// feedback-loop split or a submit.
	if (!is_render_pass_open())
	{
		begin_render_pass();
		reload_state = true;
	}

	// Programmable blending uses framebuffer fetch: no input attachments and no barriers inside the pass.

	// Bind pipeline and resources. MTL4 argument tables are captured at draw time, so bind before every draw.
	auto encoder = ensure(get_render_encoder());
	m_program->bind(*m_current_command_buffer, mtl::get_scratch_heap());
	m_encoder_state.pipeline = m_program->render_pipeline();

	if (reload_state)
	{
		update_draw_state();
	}

	m_frame_stats.setup_time += m_profiler.duration();

	if (m_rasterizer_state.cull_all_polygons &&
		mtl::get_topology_class(upload_info.primitive) == static_cast<u8>(MTL::PrimitiveTopologyClassTriangle))
	{
		// cull_face::front_and_back: every polygon is discarded
		m_frame_stats.draw_exec_time += m_profiler.duration();
		return;
	}

	if (const auto scissor = get_clamped_scissor(); !scissor.width || !scissor.height)
	{
		// Empty scissor: nothing is rasterized (Metal does not accept empty scissor rectangles)
		m_frame_stats.draw_exec_time += m_profiler.duration();
		return;
	}

	// NOTE: No multi-draw on Metal; sub-ranges are looped (backend_config.supports_multidraw = false)
	if (!upload_info.index_info)
	{
		if (draw_call.is_trivial_instanced_draw)
		{
			encoder->drawPrimitives(upload_info.primitive, 0, upload_info.vertex_draw_count, draw_call.pass_count());
		}
		else if (draw_call.is_single_draw())
		{
			encoder->drawPrimitives(upload_info.primitive, 0, upload_info.vertex_draw_count);
		}
		else
		{
			u32 vertex_offset = 0;
			const auto subranges = draw_call.get_subranges();
			for (const auto &range : subranges)
			{
				encoder->drawPrimitives(upload_info.primitive, vertex_offset, range.count);
				vertex_offset += range.count;
			}
		}
	}
	else
	{
		const MTL::IndexType index_type = std::get<1>(*upload_info.index_info);
		const u64 offset = std::get<0>(*upload_info.index_info);
		const u64 index_size = (index_type == MTL::IndexTypeUInt16) ? 2 : 4;
		const MTL::GPUAddress index_base = m_index_buffer_ring_info.gpu_address(offset);

		if (draw_call.is_trivial_instanced_draw)
		{
			encoder->drawIndexedPrimitives(upload_info.primitive, upload_info.vertex_draw_count, index_type,
				index_base, upload_info.vertex_draw_count * index_size, draw_call.pass_count(), 0, 0);
		}
		else if (rsx::method_registers.current_draw_clause.is_single_draw())
		{
			encoder->drawIndexedPrimitives(upload_info.primitive, upload_info.vertex_draw_count, index_type,
				index_base, upload_info.vertex_draw_count * index_size);
		}
		else
		{
			// Metal requires 4-byte aligned index buffer addresses. 16-bit sub-ranges following an odd index count start on
			// a 2-byte boundary: those are copied to a fresh (aligned) ring allocation.
			// NOTE: Read the source through the mapping taken before any allocation; a ring grow swaps the backing store
			// (the old buffer stays alive through the GC until this submission completes, so index_base remains valid).
			const u8* index_data = m_index_buffer_ring_info.map<u8>(offset, 0);

			u32 vertex_offset = 0;
			const auto subranges = draw_call.get_subranges();
			for (const auto &range : subranges)
			{
				const auto count = get_index_count(draw_call.primitive, range.count);
				const u64 range_offset = vertex_offset * index_size;
				const u64 range_length = count * index_size;
				vertex_offset += count;

				if (!count)
				{
					continue;
				}

				MTL::GPUAddress range_address = index_base + range_offset;
				if (range_address & 3)
				{
					const usz aligned_offset = m_index_buffer_ring_info.alloc<64>(range_length);
					std::memcpy(m_index_buffer_ring_info.map<u8>(aligned_offset, range_length), index_data + range_offset, range_length);
					range_address = m_index_buffer_ring_info.gpu_address(aligned_offset);
				}

				encoder->drawIndexedPrimitives(upload_info.primitive, count, index_type, range_address, range_length);
			}
		}
	}

	m_frame_stats.draw_exec_time += m_profiler.duration();
}

void MTLGSRender::begin()
{
	rsx::thread::begin();

	if (skip_current_frame ||
		swapchain_unavailable ||
		cond_render_ctrl.disable_rendering())
	{
		return;
	}

	mtl::autorelease_scope pool;
	init_buffers(rsx::framebuffer_creation_context::context_draw);

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		// Shaders need to be reloaded.
		m_prev_program = m_program;
		m_program = nullptr;
	}
}

void MTLGSRender::end()
{
	if (skip_current_frame || !m_graphics_state.test(rsx::rtt_config_valid) || swapchain_unavailable || cond_render_ctrl.disable_rendering())
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	mtl::autorelease_scope pool;

	m_profiler.start();

	// Check for frame resource status here because it is possible for an async flip to happen between begin/end
	if (m_current_frame->flags & frame_context_state::dirty) [[unlikely]]
	{
		check_present_status();

		if (m_current_frame->swap_command_buffer) [[unlikely]]
		{
			// Borrow time by using the auxilliary context
			m_aux_frame_context.grab_resources(*m_current_frame);
			m_current_frame = &m_aux_frame_context;
		}

		ensure(!m_current_frame->swap_command_buffer);

		m_current_frame->flags &= ~frame_context_state::dirty;
	}

	analyse_current_rsx_pipeline();

	m_frame_stats.setup_time += m_profiler.duration();

	load_texture_env();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	if (!load_program())
	{
		// Program is not ready, skip drawing this
		std::this_thread::yield();
		execute_nop_draw();
		// m_rtts.on_write(); - breaks games for obvious reasons
		rsx::thread::end();
		return;
	}

	// Load program execution environment
	load_program_env();
	m_frame_stats.setup_time += m_profiler.duration();

	// Apply write memory barriers
	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil))
	{
		ds->write_barrier(*m_current_command_buffer);

		if (m_graphics_state.test(rsx::zeta_address_cyclic_barrier))
		{
			// We actually need to end the pass as a minimum. Without this, early-Z optimiazations in following draws
			// will clobber reads from previous draws and cause flickering.
			ds->reset_surface_counters();
			invalidate_render_pass();
		}
	}

	for (auto &rtt : m_rtts.m_bound_render_targets)
	{
		if (auto surface = std::get<1>(rtt))
		{
			surface->write_barrier(*m_current_command_buffer);
		}
	}

	m_graphics_state.clear(rsx::zeta_address_cyclic_barrier);

	m_frame_stats.setup_time += m_profiler.duration();

	// Now bind the shader resources. It is important that this takes place after the barriers so that we don't end up with stale descriptors
	for (int retry = 0; retry < 3; ++retry)
	{
		if (retry > 0 && m_samplers_dirty) [[ unlikely ]]
		{
			// Reload texture env if referenced objects were invalidated during OOM handling.
			load_texture_env();

			// Do not trust fragment/vertex texture state after a texture state reset.
			// NOTE: We don't want to change the program - it's too late for that now. We just need to harmonize the state.
			m_graphics_state |= rsx::vertex_program_state_dirty | rsx::fragment_program_state_dirty;
			get_current_fragment_program(fs_sampler_state);
			get_current_vertex_program(vs_sampler_state);
			m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
		}

		const bool out_of_memory = bind_texture_env();
		if (!out_of_memory)
		{
			break;
		}

		// Handle OOM
		if (!on_vram_exhausted(rsx::problem_severity::fatal))
		{
			// It is not possible to free memory. Just use placeholder textures. Can cause graphics glitches but shouldn't crash otherwise
			break;
		}
	}

	m_texture_cache.release_uncached_temporary_subresources();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	u32 sub_index = 0;               // RSX subdraw ID
	m_current_draw.subdraw_id = 0;   // Host subdraw ID. Invalid RSX subdraws do not increment this value

	if (m_graphics_state & rsx::pipeline_state::invalidate_vk_dynamic_state)
	{
		m_current_command_buffer->flags |= mtl::command_list::cb_reload_dynamic_state;
	}

	auto& draw_call = rsx::method_registers.current_draw_clause;
	draw_call.begin();
	do
	{
		emit_geometry(sub_index++);

		if (draw_call.is_trivial_instanced_draw)
		{
			// We already completed. End the draw.
			draw_call.end();
		}
	}
	while (draw_call.next());

	m_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

	rsx::thread::end();
}

void MTLGSRender::clear_surface(u32 mask)
{
	if (skip_current_frame || swapchain_unavailable) return;

	// If stencil write mask is disabled, remove clear_stencil bit
	if (!rsx::method_registers.stencil_mask()) mask &= ~RSX_GCM_CLEAR_STENCIL_BIT;

	// Ignore invalid clear flags
	if (!(mask & RSX_GCM_CLEAR_ANY_MASK)) return;

	mtl::autorelease_scope pool;

	u8 ctx = rsx::framebuffer_creation_context::context_draw;
	if (mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_color;
	if (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_depth;
	init_buffers(rsx::framebuffer_creation_context{ctx});

	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	u32 depth_stencil_mask = 0;
	f32 depth_clear = 1.f;
	u8 stencil_clear = 0;

	u16 scissor_x = static_cast<u16>(m_scissor.x);
	u16 scissor_w = static_cast<u16>(m_scissor.width);
	u16 scissor_y = static_cast<u16>(m_scissor.y);
	u16 scissor_h = static_cast<u16>(m_scissor.height);

	const u16 fb_width = static_cast<u16>(m_draw_fbo.width);
	const u16 fb_height = static_cast<u16>(m_draw_fbo.height);

	//clip region
	std::tie(scissor_x, scissor_y, scissor_w, scissor_h) = rsx::clip_region<u16>(fb_width, fb_height, scissor_x, scissor_y, scissor_w, scissor_h, true);

	const bool full_frame = (scissor_w == fb_width && scissor_h == fb_height);
	bool update_color = false, update_z = false;
	auto surface_depth_format = rsx::method_registers.surface_depth_fmt();

	// Clears are either folded into the load action of a new pass (full frame, all channels) or drawn inside the pass
	mtl::attachment_clear_info load_clear{};
	mtl::inpass_clear::request inpass{};

	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil); mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK)
	{
		if (mask & RSX_GCM_CLEAR_DEPTH_BIT)
		{
			u32 max_depth_value = get_max_depth_value(surface_depth_format);

			u32 clear_depth = rsx::method_registers.z_clear_value(is_depth_stencil_format(surface_depth_format));
			depth_clear = static_cast<float>(clear_depth) / max_depth_value;

			depth_stencil_mask |= mtl::aspect_depth;
		}

		if (is_depth_stencil_format(surface_depth_format))
		{
			if (mask & RSX_GCM_CLEAR_STENCIL_BIT)
			{
				u8 clear_stencil = rsx::method_registers.stencil_clear_value();
				stencil_clear = clear_stencil;

				depth_stencil_mask |= mtl::aspect_stencil;

				if (ds->samples() > 1)
				{
					if (full_frame) ds->stencil_init_flags &= 0xFF;
					ds->stencil_init_flags |= clear_stencil;
				}
			}
		}

		if ((depth_stencil_mask && depth_stencil_mask != ds->aspect()) || !full_frame)
		{
			// At least one aspect is not being cleared or the clear does not cover the full frame
			// Steps to initialize memory are required

			if (ds->state_flags & rsx::surface_state_flags::erase_bkgnd &&  // Needs initialization
				ds->old_contents.empty() && !g_cfg.video.read_depth_buffer) // No way to load data from memory, so no initialization given
			{
				// Only one aspect was cleared. Make sure to memory initialize the other before removing dirty flag
				const auto ds_mask = (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK);
				if (ds_mask == RSX_GCM_CLEAR_DEPTH_BIT && (ds->aspect() & mtl::aspect_stencil))
				{
					// Depth was cleared, initialize stencil
					stencil_clear = 0xFF;
					depth_stencil_mask |= mtl::aspect_stencil;
				}
				else if (ds_mask == RSX_GCM_CLEAR_STENCIL_BIT)
				{
					// Stencil was cleared, initialize depth
					depth_clear = 1.f;
					depth_stencil_mask |= mtl::aspect_depth;
				}
			}
			else
			{
				// Barrier required before any writes
				ds->write_barrier(*m_current_command_buffer);
			}
		}
	}

	if (auto colormask = (mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK))
	{
		if (!m_draw_buffers.empty())
		{
			bool use_fast_clear = (colormask == RSX_GCM_CLEAR_COLOR_RGBA_MASK);
			u8 clear_a = rsx::method_registers.clear_color_a();
			u8 clear_r = rsx::method_registers.clear_color_r();
			u8 clear_g = rsx::method_registers.clear_color_g();
			u8 clear_b = rsx::method_registers.clear_color_b();

			switch (rsx::method_registers.surface_color())
			{
			case rsx::surface_color_format::x32:
			case rsx::surface_color_format::w16z16y16x16:
			case rsx::surface_color_format::w32z32y32x32:
			{
				//NOP
				colormask = 0;
				break;
			}
			case rsx::surface_color_format::b8:
			{
				rsx::get_b8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_b8_clearmask(colormask);
				use_fast_clear = (colormask & RSX_GCM_CLEAR_RED_BIT);
				break;
			}
			case rsx::surface_color_format::g8b8:
			{
				rsx::get_g8b8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_g8b8_r8g8_clearmask(colormask);
				use_fast_clear = ((colormask & RSX_GCM_CLEAR_COLOR_RG_MASK) == RSX_GCM_CLEAR_COLOR_RG_MASK);
				break;
			}
			case rsx::surface_color_format::r5g6b5:
			{
				rsx::get_rgb565_clear_color(clear_r, clear_g, clear_b, clear_a);
				use_fast_clear = ((colormask & RSX_GCM_CLEAR_COLOR_RGB_MASK) == RSX_GCM_CLEAR_COLOR_RGB_MASK);
				break;
			}
			case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			{
				rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 255);
				break;
			}
			case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			{
				rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 0);
				break;
			}
			case rsx::surface_color_format::a8b8g8r8:
			case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			{
				rsx::get_abgr8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_abgr8_clearmask(colormask);
				break;
			}
			default:
			{
				break;
			}
			}

			if (colormask)
			{
				if (!use_fast_clear || !full_frame)
				{
					// If we're not clobber all the memory, a barrier is required
					for (const auto& index : m_rtts.m_bound_render_target_ids)
					{
						m_rtts.m_bound_render_targets[index].second->write_barrier(*m_current_command_buffer);
					}
				}

				const color4f clear_color =
				{
					static_cast<float>(clear_r) / 255,
					static_cast<float>(clear_g) / 255,
					static_cast<float>(clear_b) / 255,
					static_cast<float>(clear_a) / 255
				};

				if (use_fast_clear && full_frame)
				{
					// Full-surface clear of every colour attachment: fold into the load action of a new pass
					load_clear.color_mask = (1u << m_draw_fbo.color_count) - 1;
					load_clear.color = MTL::ClearColor::Make(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
				}
				else
				{
					u8 write_mask = 0;
					if (use_fast_clear)
					{
						write_mask = static_cast<u8>(MTL::ColorWriteMaskAll);
					}
					else
					{
						if (colormask & RSX_GCM_CLEAR_RED_BIT) write_mask |= static_cast<u8>(MTL::ColorWriteMaskRed);
						if (colormask & RSX_GCM_CLEAR_GREEN_BIT) write_mask |= static_cast<u8>(MTL::ColorWriteMaskGreen);
						if (colormask & RSX_GCM_CLEAR_BLUE_BIT) write_mask |= static_cast<u8>(MTL::ColorWriteMaskBlue);
						if (colormask & RSX_GCM_CLEAR_ALPHA_BIT) write_mask |= static_cast<u8>(MTL::ColorWriteMaskAlpha);
					}

					inpass.color_write_mask = write_mask;
					inpass.color = clear_color;
				}

				update_color = true;
			}
		}
	}

	if (depth_stencil_mask)
	{
		const bool partial_stencil = (depth_stencil_mask & mtl::aspect_stencil) && rsx::method_registers.stencil_mask() != 0xff;

		if (full_frame)
		{
			if (depth_stencil_mask & mtl::aspect_depth)
			{
				load_clear.clear_depth = true;
				load_clear.depth = depth_clear;
			}

			if ((depth_stencil_mask & mtl::aspect_stencil) && !partial_stencil)
			{
				load_clear.clear_stencil = true;
				load_clear.stencil = stencil_clear;
			}
		}
		else
		{
			if (depth_stencil_mask & mtl::aspect_depth)
			{
				inpass.depth = true;
				inpass.depth_value = depth_clear;
			}

			if ((depth_stencil_mask & mtl::aspect_stencil) && !partial_stencil)
			{
				inpass.stencil = true;
				inpass.stencil_value = stencil_clear;
				inpass.stencil_write_mask = 0xFF;
			}
		}

		if (partial_stencil)
		{
			// Partial stencil clear. Disables fast stencil clear
			inpass.stencil = true;
			inpass.stencil_value = stencil_clear;
			inpass.stencil_write_mask = rsx::method_registers.stencil_mask();
		}

		update_z = true;
	}

	if (update_color || update_z)
	{
		m_rtts.on_write({ update_color, update_color, update_color, update_color }, update_z);
	}

	if (!load_clear.empty())
	{
		// Ends the current pass and reopens it with loadAction=Clear on the requested attachments
		begin_render_pass(&load_clear);
	}

	if (!inpass.color_write_mask && !inpass.depth && !inpass.stencil)
	{
		return;
	}

	if (!scissor_w || !scissor_h)
	{
		return;
	}

	// Scissored / masked clear: draw a quad inside the main pass
	begin_render_pass();
	auto encoder = ensure(get_render_encoder());

	auto program = mtl::inpass_clear::get_program(m_draw_fbo, m_current_renderpass_key, inpass.color_write_mask);
	if (!program)
	{
		return;
	}

	mtl::inpass_clear::push_constants_t push{};
	push.color[0] = inpass.color.r;
	push.color[1] = inpass.color.g;
	push.color[2] = inpass.color.b;
	push.color[3] = inpass.color.a;
	push.depth = inpass.depth_value;

	program->push_constants(mtl::glsl::binding_set_index_vertex, 0, sizeof(push), &push);
	program->bind(*m_current_command_buffer, mtl::get_scratch_heap());

	mtl::stencil_face_desc stencil_face{};
	stencil_face.fail = MTL::StencilOperationReplace;
	stencil_face.zfail = MTL::StencilOperationReplace;
	stencil_face.pass = MTL::StencilOperationReplace;
	stencil_face.func = MTL::CompareFunctionAlways;
	stencil_face.read_mask = 0xFF;
	stencil_face.write_mask = inpass.stencil_write_mask;

	const bool clear_depth = inpass.depth && m_draw_fbo.depth_stencil;
	const bool clear_stencil = inpass.stencil && m_draw_fbo.has_stencil();
	const u64 ds_key = mtl::make_depth_stencil_key(
		MTL::CompareFunctionAlways, clear_depth,
		clear_stencil ? &stencil_face : nullptr,
		clear_stencil ? &stencil_face : nullptr);

	auto ds_state = get_depth_stencil_state(ds_key);
	encoder->setDepthStencilState(ds_state);
	encoder->setStencilReferenceValue(inpass.stencil_value);
	encoder->setCullMode(MTL::CullModeNone);
	encoder->setTriangleFillMode(MTL::TriangleFillModeFill);
	encoder->setDepthClipMode(MTL::DepthClipModeClip);
	encoder->setDepthBias(0.f, 0.f, 0.f);

	MTL::Viewport viewport{};
	viewport.originX = 0.;
	viewport.originY = 0.;
	viewport.width = fb_width;
	viewport.height = fb_height;
	viewport.znear = 0.;
	viewport.zfar = 1.;
	encoder->setViewport(viewport);
	encoder->setScissorRect(MTL::ScissorRect{ scissor_x, scissor_y, scissor_w, scissor_h });

	if (m_device->caps().depth_bounds)
	{
		// The depth bounds test would discard the clear against the stored depth
		encoder->setDepthTestBounds(0.f, 1.f);
	}

	// Clears are never counted by occlusion queries (VK: vkCmdClearAttachments). Suspend the open query, if any.
	const bool query_open = (m_current_command_buffer->flags & mtl::command_list::cb_has_open_query) && m_active_query_info;
	if (query_open)
	{
		encoder->setVisibilityResultMode(MTL::VisibilityResultModeDisabled, 0);
	}

	encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));

	if (query_open)
	{
		const auto open_query = m_occlusion_map[m_active_query_info->driver_handle].indices.back();
		m_occlusion_query_manager->resume_query(encoder, open_query);
	}

	// The next draw must restore the full encoder state (viewport, scissor, depth bias, depth bounds, ...)
	m_encoder_state.pipeline = nullptr;
	m_encoder_state.depth_stencil = ds_state;
	m_current_command_buffer->flags |= mtl::command_list::cb_reload_dynamic_state;
}
