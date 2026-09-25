#include "stdafx.h"
#include "MTLOverlays.h"
#include "MTLResourceManager.h"

#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

#include "../Overlays/overlays.h"
#include "../Program/RSXOverlay.h"

#include "util/fnv_hash.hpp"
#include "Utilities/mutex.h"
#include "Utilities/stereo_config.h"

#include "Emu/Cell/timers.hpp"

namespace mtl
{
	std::unordered_map<u32, std::unique_ptr<mtl::overlay_pass>> g_overlay_passes;

	namespace
	{
		// ---- Depth-stencil state cache -----------------------------------------------------------------------------
		shared_mutex g_ds_state_lock;
		std::unordered_map<u64, MTL::DepthStencilState*> g_ds_state_cache;

		// ---- Fallback textures bound to unused sampler slots (VK: vk::null_image_view) -----------------------------
		struct null_texture_set
		{
			std::unique_ptr<mtl::viewable_image> tex_2d;
			std::unique_ptr<mtl::viewable_image> tex_2d_array;
		};

		null_texture_set g_null_textures;

		std::unique_ptr<mtl::viewable_image> create_null_texture(MTL::TextureType type)
		{
			image_create_info info{};
			info.type = type;
			info.format = MTL::PixelFormatRGBA8Unorm;
			info.width = 1;
			info.height = 1;
			info.layers = 1;
			info.usage = MTL::TextureUsageShaderRead;
			info.storage = memory_location::host_visible;
			info.format_class = RSX_FORMAT_CLASS_COLOR;

			auto result = std::make_unique<mtl::viewable_image>(*g_render_device, info);

			const u32 zero = 0;
			result->value->replaceRegion(MTL::Region(0, 0, 1, 1), 0, 0, &zero, 4, 4);
			result->set_debug_name(type == MTL::TextureType2D ? "overlay null 2D" : "overlay null 2D array");
			return result;
		}

		mtl::image_view* get_null_image_view(MTL::TextureType type)
		{
			auto& slot = (type == MTL::TextureType2DArray) ? g_null_textures.tex_2d_array : g_null_textures.tex_2d;
			if (!slot)
			{
				slot = create_null_texture(type);
			}

			return slot->get_identity_view();
		}

		u8 get_topology_class(MTL::PrimitiveType type)
		{
			switch (type)
			{
			case MTL::PrimitiveTypePoint:
				return static_cast<u8>(MTL::PrimitiveTopologyClassPoint);
			case MTL::PrimitiveTypeLine:
			case MTL::PrimitiveTypeLineStrip:
				return static_cast<u8>(MTL::PrimitiveTopologyClassLine);
			default:
				return static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
			}
		}

		// Clamp a scissor rectangle to the render target (Metal validation requires it to lie inside the attachments)
		MTL::ScissorRect clamp_scissor(const overlay_target& target, u32 x, u32 y, u32 w, u32 h)
		{
			const u32 rt_w = target.width();
			const u32 rt_h = target.height();

			const u32 x1 = std::min(x, rt_w);
			const u32 y1 = std::min(y, rt_h);
			const u32 x2 = std::min(x + w, rt_w);
			const u32 y2 = std::min(y + h, rt_h);

			return { x1, y1, x2 - x1, y2 - y1 };
		}
	}

	// Fragment resources live in set 1 (glsl::binding_set_index_fragment). Shared VK snippets declare them in set 0.
	std::string remap_fragment_set(const std::string& source)
	{
		// Rewrites `set = 0` layout qualifiers (any spacing) to `set = 1`. Only whole `set` tokens are matched, so
		// qualifiers such as `offset=0` are left alone.
		const auto is_ident_char = [](char c)
		{
			return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
		};

		std::string result;
		result.reserve(source.size());

		usz pos = 0;
		const usz size = source.size();

		while (pos < size)
		{
			const usz found = source.find("set", pos);
			if (found == std::string::npos)
			{
				result.append(source, pos, std::string::npos);
				break;
			}

			usz cursor = found + 3;
			bool matched = false;

			if ((found == 0 || !is_ident_char(source[found - 1])) && (cursor >= size || !is_ident_char(source[cursor])))
			{
				while (cursor < size && source[cursor] == ' ') cursor++;

				if (cursor < size && source[cursor] == '=')
				{
					cursor++;
					while (cursor < size && source[cursor] == ' ') cursor++;

					if (cursor < size && source[cursor] == '0' && (cursor + 1 >= size || !std::isdigit(static_cast<unsigned char>(source[cursor + 1]))))
					{
						result.append(source, pos, cursor - pos);
						result += '1';
						pos = cursor + 1;
						matched = true;
					}
				}
			}

			if (!matched)
			{
				result.append(source, pos, (found + 3) - pos);
				pos = found + 3;
			}
		}

		return result;
	}

	// ---- depth_stencil_config -------------------------------------------------------------------------------------

	u64 depth_stencil_config::key() const
	{
		u64 result = 0;
		result |= static_cast<u64>(depth_test ? 1 : 0);
		result |= static_cast<u64>(depth_write ? 1 : 0) << 1;
		result |= static_cast<u64>(depth_compare & 0x7) << 2;
		result |= static_cast<u64>(stencil_test ? 1 : 0) << 5;
		result |= static_cast<u64>(stencil_fail & 0x7) << 6;
		result |= static_cast<u64>(depth_fail & 0x7) << 9;
		result |= static_cast<u64>(stencil_pass & 0x7) << 12;
		result |= static_cast<u64>(stencil_compare & 0x7) << 15;
		result |= static_cast<u64>(stencil_read_mask & 0xFF) << 18;
		result |= static_cast<u64>(stencil_write_mask & 0xFF) << 26;
		return result;
	}

	MTL::DepthStencilState* get_depth_stencil_state(const depth_stencil_config& config)
	{
		const u64 key = config.key();

		{
			reader_lock lock(g_ds_state_lock);
			if (auto found = g_ds_state_cache.find(key); found != g_ds_state_cache.end())
			{
				return found->second;
			}
		}

		autorelease_scope pool;

		auto desc = ref(MTL::DepthStencilDescriptor::alloc()->init());
		desc->setDepthCompareFunction(config.depth_test ? config.depth_compare : MTL::CompareFunctionAlways);
		desc->setDepthWriteEnabled(config.depth_write);

		if (config.stencil_test)
		{
			auto stencil = ref(MTL::StencilDescriptor::alloc()->init());
			stencil->setStencilCompareFunction(config.stencil_compare);
			stencil->setStencilFailureOperation(config.stencil_fail);
			stencil->setDepthFailureOperation(config.depth_fail);
			stencil->setDepthStencilPassOperation(config.stencil_pass);
			stencil->setReadMask(config.stencil_read_mask & 0xFF);
			stencil->setWriteMask(config.stencil_write_mask & 0xFF);

			desc->setFrontFaceStencil(stencil.get());
			desc->setBackFaceStencil(stencil.get());
		}

		MTL::DepthStencilState* state = g_render_device->handle()->newDepthStencilState(desc.get());
		ensure(state, "Metal: failed to create depth-stencil state");

		std::lock_guard lock(g_ds_state_lock);
		auto [it, inserted] = g_ds_state_cache.emplace(key, state);
		if (!inserted)
		{
			// Raced with another thread
			state->release();
		}

		return it->second;
	}

	// ---- overlay_pipeline_config ------------------------------------------------------------------------------------

	overlay_pipeline_config::overlay_pipeline_config()
	{
		for (auto& color : state.color)
		{
			color.write_mask = static_cast<u8>(MTL::ColorWriteMaskAll);
			color.blend_enable = 0;
			color.src_rgb = static_cast<u8>(MTL::BlendFactorOne);
			color.dst_rgb = static_cast<u8>(MTL::BlendFactorZero);
			color.op_rgb = static_cast<u8>(MTL::BlendOperationAdd);
			color.src_a = static_cast<u8>(MTL::BlendFactorOne);
			color.dst_a = static_cast<u8>(MTL::BlendFactorZero);
			color.op_a = static_cast<u8>(MTL::BlendOperationAdd);
		}

		state.sample_count = 1;
		state.rasterization_enabled = 1;
		set_primitive_type(MTL::PrimitiveTypeTriangleStrip);
	}

	void overlay_pipeline_config::set_primitive_type(MTL::PrimitiveType type)
	{
		primitive = type;
		state.topology_class = get_topology_class(type);
	}

	void overlay_pipeline_config::set_attachment_count(u32 /*count*/)
	{
		// Attachment formats (and therefore the count) are taken from the render target at draw time
	}

	void overlay_pipeline_config::set_color_mask(u32 index, bool r, bool g, bool b, bool a)
	{
		u8 mask = 0;
		if (r) mask |= static_cast<u8>(MTL::ColorWriteMaskRed);
		if (g) mask |= static_cast<u8>(MTL::ColorWriteMaskGreen);
		if (b) mask |= static_cast<u8>(MTL::ColorWriteMaskBlue);
		if (a) mask |= static_cast<u8>(MTL::ColorWriteMaskAlpha);
		state.color[index].write_mask = mask;
	}

	void overlay_pipeline_config::enable_blend(u32 index,
		MTL::BlendFactor src_factor_rgb, MTL::BlendFactor src_factor_a,
		MTL::BlendFactor dst_factor_rgb, MTL::BlendFactor dst_factor_a,
		MTL::BlendOperation equation_rgb, MTL::BlendOperation equation_a)
	{
		auto& color = state.color[index];
		color.blend_enable = 1;
		color.src_rgb = static_cast<u8>(src_factor_rgb);
		color.src_a = static_cast<u8>(src_factor_a);
		color.dst_rgb = static_cast<u8>(dst_factor_rgb);
		color.dst_a = static_cast<u8>(dst_factor_a);
		color.op_rgb = static_cast<u8>(equation_rgb);
		color.op_a = static_cast<u8>(equation_a);
	}

	void overlay_pipeline_config::disable_blend(u32 index)
	{
		state.color[index].blend_enable = 0;
	}

	void overlay_pipeline_config::set_depth_mask(bool enable)
	{
		ds.depth_write = enable;
	}

	void overlay_pipeline_config::enable_depth_test(MTL::CompareFunction op)
	{
		ds.depth_test = true;
		ds.depth_compare = op;
	}

	void overlay_pipeline_config::disable_depth_test()
	{
		ds.depth_test = false;
		ds.depth_compare = MTL::CompareFunctionAlways;
	}

	void overlay_pipeline_config::enable_stencil_test(MTL::StencilOperation fail, MTL::StencilOperation zfail, MTL::StencilOperation pass,
		MTL::CompareFunction func, u32 func_mask, u32 ref)
	{
		ds.stencil_test = true;
		ds.stencil_fail = fail;
		ds.depth_fail = zfail;
		ds.stencil_pass = pass;
		ds.stencil_compare = func;
		ds.stencil_read_mask = func_mask;
		ds.stencil_ref = ref;
	}

	void overlay_pipeline_config::disable_stencil_test()
	{
		ds.stencil_test = false;
	}

	void overlay_pipeline_config::set_stencil_mask(u32 write_mask)
	{
		ds.stencil_write_mask = write_mask;
	}

	// ---- overlay_target ---------------------------------------------------------------------------------------------

	overlay_target::overlay_target(MTL::Texture* texture, u32 level_, u32 slice_)
		: level(level_), slice(slice_)
	{
		if (texture && (get_format_aspect(texture->pixelFormat()) & aspect_depth_stencil))
		{
			depth_stencil = texture;
		}
		else
		{
			color = texture;
		}
	}

	overlay_target::overlay_target(const mtl::image* image, u32 level_, u32 slice_)
		: level(level_), slice(slice_)
	{
		if (!image)
		{
			return;
		}

		if (image->aspect() & aspect_depth_stencil)
		{
			depth_stencil = image->value;
		}
		else
		{
			color = image->value;
		}
	}

	overlay_target::overlay_target(const mtl::image* color_image, const mtl::image* depth_stencil_image)
	{
		color = color_image ? color_image->value : nullptr;
		depth_stencil = depth_stencil_image ? depth_stencil_image->value : nullptr;
	}

	u32 overlay_target::width() const
	{
		const auto tex = primary();
		return tex ? std::max<u32>(static_cast<u32>(tex->width()) >> level, 1u) : 0u;
	}

	u32 overlay_target::height() const
	{
		const auto tex = primary();
		return tex ? std::max<u32>(static_cast<u32>(tex->height()) >> level, 1u) : 0u;
	}

	u32 overlay_target::samples() const
	{
		const auto tex = primary();
		return tex ? std::max<u32>(static_cast<u32>(tex->sampleCount()), 1u) : 1u;
	}

	bool overlay_target::has_depth() const
	{
		return depth_stencil && is_depth_format(depth_stencil->pixelFormat());
	}

	bool overlay_target::has_stencil() const
	{
		return depth_stencil && is_stencil_format(depth_stencil->pixelFormat());
	}

	// ---- overlay_pass -----------------------------------------------------------------------------------------------

	overlay_pass::overlay_pass()
	{
		// Override-able defaults
		renderpass_config.set_primitive_type(MTL::PrimitiveTypeTriangleStrip);
	}

	overlay_pass::~overlay_pass() = default;

	u64 overlay_pass::get_pipeline_key(const glsl::graphics_pipeline_state& state) const
	{
		return rpcs3::hash_struct(state);
	}

	std::vector<glsl::program_input> overlay_pass::get_vertex_inputs()
	{
		return {};
	}

	std::vector<glsl::program_input> overlay_pass::get_fragment_inputs()
	{
		using namespace mtl::glsl;

		std::vector<program_input> fs_inputs;
		u32 binding = 0;

		for (u32 n = 0; n < m_num_uniform_buffers; ++n, ++binding)
		{
			std::string name = std::string("static_data") + (n > 0 ? std::to_string(n) : "");
			fs_inputs.push_back(program_input::make(::glsl::program_domain::glsl_fragment_program, std::move(name), program_input_type::input_type_uniform_buffer, binding_set_index_fragment, binding));
		}

		for (u32 n = 0; n < m_num_usable_samplers; ++n, ++binding)
		{
			std::string name = "fs" + std::to_string(n);
			fs_inputs.push_back(program_input::make(::glsl::program_domain::glsl_fragment_program, std::move(name), program_input_type::input_type_texture, binding_set_index_fragment, binding));
		}

		for (u32 n = 0; n < m_num_input_attachments; ++n, ++binding)
		{
			std::string name = "sp" + std::to_string(n);
			fs_inputs.push_back(program_input::make(::glsl::program_domain::glsl_fragment_program, std::move(name), program_input_type::input_type_texture, binding_set_index_fragment, binding));
		}

		return fs_inputs;
	}

	void overlay_pass::upload_vertex_data_raw(const void* data, usz size)
	{
		auto& heap = get_scratch_heap();
		m_vao_offset = heap.alloc<16>(size);
		m_vao_length = size;

		if (size)
		{
			std::memcpy(heap.map(m_vao_offset, size), data, size);
		}
	}

	void overlay_pass::upload_uniform_data(const void* data, usz size)
	{
		auto& heap = get_scratch_heap();
		const usz alloc_size = std::max<usz>(size, m_ubo_length);
		m_ubo_offset = heap.alloc<256>(alloc_size);

		auto dst = heap.map<u8>(m_ubo_offset, alloc_size);
		std::memcpy(dst, data, size);
		if (alloc_size > size)
		{
			std::memset(dst + size, 0, alloc_size - size);
		}
	}

	glsl::program* overlay_pass::build_pipeline(u64 storage_key, const glsl::graphics_pipeline_state& state)
	{
		auto program = glsl::create_graphics_program(vs_src, get_vertex_inputs(), fs_src, get_fragment_inputs(), state);
		if (!program)
		{
			fmt::throw_exception("Metal: failed to build overlay pipeline.\nVS:\n%s\nFS:\n%s", vs_src, fs_src);
		}

		auto result = program.get();
		m_program_cache[storage_key] = std::move(program);
		return result;
	}

	mtl::sampler* overlay_pass::get_sampler(bool linear)
	{
		auto& slot = m_samplers[linear ? 1 : 0];
		if (!slot)
		{
			sampler_create_info info{};
			info.clamp_u = MTL::SamplerAddressModeClampToEdge;
			info.clamp_v = MTL::SamplerAddressModeClampToEdge;
			info.clamp_w = MTL::SamplerAddressModeClampToEdge;
			info.unnormalized_coordinates = false;
			info.mip_lod_bias = 0.f;
			info.max_anisotropy = 1.f;
			info.min_lod = 0.f;
			info.max_lod = 0.f;
			info.min_filter = linear ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest;
			info.mag_filter = info.min_filter;
			info.mip_filter = MTL::SamplerMipFilterNotMipmapped;
			info.border_color = border_color_t(MTL::SamplerBorderColorOpaqueBlack);

			slot = std::make_unique<mtl::sampler>(m_device ? *m_device : *g_render_device, info);
		}

		return slot.get();
	}

	glsl::program* overlay_pass::load_program(mtl::command_list& cmd, const overlay_target& target, const std::vector<mtl::image_view*>& src)
	{
		// Complete the pipeline description with the target attachments
		auto state = renderpass_config.state;

		for (u32 i = 1; i < state.color.size(); ++i)
		{
			state.color[i] = {};
		}

		if (target.color)
		{
			state.color_count = 1;
			state.color[0].pixel_format = static_cast<u32>(target.color->pixelFormat());
		}
		else
		{
			state.color_count = 0;
			state.color[0] = {};
		}

		state.depth_stencil_format = target.depth_stencil ? static_cast<u32>(target.depth_stencil->pixelFormat()) : 0u;
		state.sample_count = static_cast<u8>(target.samples());
		state.topology_class = get_topology_class(renderpass_config.primitive);

		glsl::program* program = nullptr;
		const auto key = get_pipeline_key(state);

		if (auto found = m_program_cache.find(key); found != m_program_cache.end())
		{
			program = found->second.get();
		}
		else
		{
			program = build_pipeline(key, state);
		}

		update_uniforms(cmd, program);

		if (m_num_uniform_buffers > 0)
		{
			auto& heap = get_scratch_heap();
			program->bind_uniform({ heap.heap.get(), m_ubo_offset, std::max(m_ubo_length, 4u) }, glsl::binding_set_index_fragment, 0);
		}

		if (m_vao_length > 0)
		{
			auto& heap = get_scratch_heap();
			program->bind_uniform({ heap.heap.get(), m_vao_offset, m_vao_length }, glsl::binding_set_index_vertex, 0);
		}

		if (!src.empty())
		{
			ensure(src.size() <= (m_num_usable_samplers + m_num_input_attachments));
			const auto sampler = get_sampler(m_sampler_filter == MTL::SamplerMinMagFilterLinear);

			for (u32 n = 0; n < src.size(); ++n)
			{
				program->bind_uniform(glsl::image_binding_info(src[n], sampler), glsl::binding_set_index_fragment, sampler_location(n));
			}
		}

		program->bind(cmd, get_scratch_heap());
		return program;
	}

	void overlay_pass::create(const mtl::render_device& dev)
	{
		if (!initialized)
		{
			m_device = &dev;
			initialized = true;
		}
	}

	void overlay_pass::destroy()
	{
		if (initialized)
		{
			m_program_cache.clear();
			m_samplers[0].reset();
			m_samplers[1].reset();

			initialized = false;
		}
	}

	void overlay_pass::free_resources()
	{
		// Vertex and uniform data live in the shared scratch heap which is recycled by the renderer
	}

	void overlay_pass::begin_pass(mtl::command_list& cmd, const overlay_target& target, const areau& viewport)
	{
		ensure(target.primary(), "Overlay pass has no render target");

		autorelease_scope pool;

		const bool covers_target =
			viewport.x1 == 0 && viewport.y1 == 0 &&
			viewport.x2 >= target.width() && viewport.y2 >= target.height();

		auto desc = ref(MTL4::RenderPassDescriptor::alloc()->init());

		if (target.color)
		{
			auto attachment = desc->colorAttachments()->object(0);
			attachment->setTexture(target.color);
			attachment->setLevel(target.level);
			attachment->setSlice(target.slice);
			attachment->setStoreAction(MTL::StoreActionStore);

			if (target.clear_color_on_load)
			{
				const auto& c = target.clear_color_value;
				attachment->setLoadAction(MTL::LoadActionClear);
				attachment->setClearColor(MTL::ClearColor::Make(c.r, c.g, c.b, c.a));
			}
			else
			{
				attachment->setLoadAction((m_overwrites_color && covers_target) ? MTL::LoadActionDontCare : MTL::LoadActionLoad);
			}
		}

		if (target.has_depth())
		{
			auto attachment = desc->depthAttachment();
			attachment->setTexture(target.depth_stencil);
			attachment->setLevel(target.level);
			attachment->setSlice(target.slice);
			attachment->setLoadAction((m_overwrites_depth && covers_target) ? MTL::LoadActionDontCare : MTL::LoadActionLoad);
			attachment->setStoreAction(MTL::StoreActionStore);
		}

		if (target.has_stencil())
		{
			auto attachment = desc->stencilAttachment();
			attachment->setTexture(target.depth_stencil);
			attachment->setLevel(target.level);
			attachment->setSlice(target.slice);
			attachment->setLoadAction((m_overwrites_stencil && covers_target) ? MTL::LoadActionDontCare : MTL::LoadActionLoad);
			attachment->setStoreAction(MTL::StoreActionStore);
		}

		configure_attachments(desc.get(), target, covers_target);

		cmd.begin_render_pass(desc.get());

		// This call clobbers dynamic state
		cmd.set_flag(mtl::command_list::cb_reload_dynamic_state);
	}

	void overlay_pass::end_pass(mtl::command_list& cmd)
	{
		cmd.end_render_pass();
	}

	void overlay_pass::draw(mtl::command_list& cmd, const areau& viewport, const overlay_target& target, const std::vector<mtl::image_view*>& src)
	{
		ensure(cmd.is_render_pass_open());

		if (!viewport.width() || !viewport.height() || viewport.x1 >= target.width() || viewport.y1 >= target.height())
		{
			// Fully outside the target; the clamped scissor would be empty, which Metal rejects
			return;
		}

		auto program = load_program(cmd, target, src);
		set_up_viewport(cmd, target, viewport.x1, viewport.y1, viewport.width(), viewport.height());

		auto encoder = cmd.render_encoder();
		encoder->setCullMode(renderpass_config.cull_mode);

		if (target.depth_stencil)
		{
			encoder->setDepthStencilState(get_depth_stencil_state(renderpass_config.ds));
			encoder->setStencilReferenceValue(renderpass_config.ds.stencil_ref & 0xFF);
		}

		emit_geometry(cmd, program);
	}

	void overlay_pass::emit_geometry(mtl::command_list& cmd, glsl::program* /*program*/)
	{
		if (!num_drawable_elements)
		{
			return;
		}

		cmd.render_encoder()->drawPrimitives(renderpass_config.primitive, first_vertex, num_drawable_elements);
	}

	void overlay_pass::set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h)
	{
		auto encoder = cmd.render_encoder();

		MTL::Viewport vp{};
		vp.originX = static_cast<f64>(x);
		vp.originY = static_cast<f64>(y);
		vp.width = static_cast<f64>(w);
		vp.height = static_cast<f64>(h);
		vp.znear = 0.;
		vp.zfar = 1.;
		encoder->setViewport(vp);

		encoder->setScissorRect(clamp_scissor(target, x, y, w, h));
	}

	void overlay_pass::run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target, const std::vector<mtl::image_view*>& src)
	{
		begin_pass(cmd, target, viewport);
		draw(cmd, viewport, target, src);
		end_pass(cmd);
	}

	void overlay_pass::run(mtl::command_list& cmd, const areau& viewport, mtl::image* target, const std::vector<mtl::image_view*>& src)
	{
		run(cmd, viewport, overlay_target(target), src);
	}

	void overlay_pass::run(mtl::command_list& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src)
	{
		std::vector<mtl::image_view*> views = { src };
		run(cmd, viewport, overlay_target(target), views);
	}

	// ---- ui_overlay_renderer ----------------------------------------------------------------------------------------

	ui_overlay_renderer::ui_overlay_renderer()
		: m_texture_type(rsx::overlays::texture_sampling_mode::none)
	{
		vs_src =
		#include "../Program/GLSLSnippets/OverlayRenderVS.glsl"
		;

		fs_src =
		#include "../Program/GLSLSnippets/OverlayRenderFS.glsl"
		;

		vs_src = fmt::replace_all(vs_src,
		{
			{ "%preprocessor", "// %preprocessor" },
			{ "%push_block", "push_constant" }
		});

		// No vertex descriptors: pull the vertex from a storage buffer in the scratch heap
		const std::string vs_before = vs_src;
		vs_src = fmt::replace_all(vs_src,
		{
			{ "layout(location=0) in vec4 in_pos;",
				"layout(set=0, binding=0, std430) readonly restrict buffer VertexData { vec4 vertex_data[]; };\n"
				"#define in_pos vertex_data[gl_VertexIndex]" }
		});
		ensure(vs_src != vs_before, "OverlayRenderVS.glsl changed; update the vertex pulling patch");

		// Push constants form one Vulkan-style address space shared by both stages (see glsl::program::push_constants):
		// the fragment block follows the vertex block, exactly like the VK backend.
		fs_src = fmt::replace_all(fs_src,
		{
			{ "%preprocessor", "// %preprocessor" },
			{ "%push_block_offset", "layout(offset=68)" },
			{ "%push_block", "push_constant" }
		});
		fs_src = remap_fragment_set(fs_src);

		// 2 input textures
		m_num_usable_samplers = 2;
		m_num_uniform_buffers = 0;

		renderpass_config.set_attachment_count(1);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_depth_mask(false);
		renderpass_config.enable_blend(0,
			MTL::BlendFactorSourceAlpha, MTL::BlendFactorZero,
			MTL::BlendFactorOneMinusSourceAlpha, MTL::BlendFactorOne,
			MTL::BlendOperationAdd, MTL::BlendOperationAdd);
	}

	void ui_overlay_renderer::upload_simple_texture(mtl::image* tex, mtl::command_list& cmd,
		mtl::data_heap& upload_heap, u32 w, u32 h, u32 layers, bool font, const void* pixel_src)
	{
		const u32 pitch = (font) ? w : w * 4;
		const u32 layer_size = pitch * h;
		const u32 data_size = layer_size * layers;

		if (!data_size)
		{
			return;
		}

		const auto offset = upload_heap.alloc<512>(data_size);
		const auto addr = upload_heap.map(offset, data_size);

		if (pixel_src)
			std::memcpy(addr, pixel_src, data_size);
		else
			std::memset(addr, 0, data_size);

		upload_heap.unmap();

		for (u32 layer = 0; layer < layers; ++layer)
		{
			// Layers are independent: only the first copy needs ordering against earlier compute/blit work
			auto encoder = (layer == 0) ? cmd.compute() : cmd.compute_unordered();
			encoder->copyFromBuffer(upload_heap.value(), offset + (layer * layer_size), pitch, layer_size,
				MTL::Size(w, h, 1), tex->value, layer, 0, MTL::Origin(0, 0, 0));
		}
	}

	mtl::image_view* ui_overlay_renderer::upload_simple_texture(const mtl::render_device& dev, mtl::command_list& cmd,
		mtl::data_heap& upload_heap, u64 key, u32 w, u32 h, u32 layers, bool font, bool temp, const void* pixel_src, u32 owner_uid)
	{
		image_create_info info{};
		info.type = (layers > 1) ? MTL::TextureType2DArray : MTL::TextureType2D;
		info.format = (font) ? MTL::PixelFormatR8Unorm : MTL::PixelFormatBGRA8Unorm;
		info.width = std::max(w, 1u);
		info.height = std::max(h, 1u);
		info.layers = std::max(layers, 1u);
		info.usage = MTL::TextureUsageShaderRead;
		info.storage = memory_location::device_local;
		info.format_class = RSX_FORMAT_CLASS_COLOR;

		auto tex = std::make_unique<mtl::image>(dev, info);

		upload_simple_texture(tex.get(), cmd, upload_heap, w, h, layers, font, pixel_src);

		auto view = std::make_unique<mtl::image_view>(tex.get());

		auto result = view.get();

		if (!temp || font)
			view_cache[key] = std::move(view);
		else
			temp_view_cache[key] = std::move(view);

		if (font)
			font_cache[key] = std::move(tex);
		else if (!temp)
			resources.push_back(std::move(tex));
		else
			temp_image_cache[key] = std::make_pair(owner_uid, std::move(tex));

		return result;
	}

	void ui_overlay_renderer::init(mtl::command_list& cmd, mtl::data_heap& upload_heap)
	{
		rsx::overlays::resource_config configuration;
		configuration.load_files();

		const auto& dev = m_device ? *m_device : *g_render_device;
		u64 storage_key = 1;

		for (const auto &res : configuration.texture_raw_data)
		{
			upload_simple_texture(dev, cmd, upload_heap, storage_key++, res->w, res->h, 1, false, false, res->get_data(), -1);
		}

		configuration.free_resources();
	}

	void ui_overlay_renderer::destroy()
	{
		temp_view_cache.clear();
		temp_image_cache.clear();

		view_cache.clear();
		resources.clear();
		font_cache.clear();

		overlay_pass::destroy();
	}

	void ui_overlay_renderer::remove_temp_resources(u32 key)
	{
		std::vector<u64> keys_to_remove;
		for (const auto& temp_image : temp_image_cache)
		{
			if (temp_image.second.first == key)
			{
				keys_to_remove.push_back(temp_image.first);
			}
		}

		for (const auto& _key : keys_to_remove)
		{
			auto& img_data = temp_image_cache[_key];
			auto& view_data = temp_view_cache[_key];

			// In-flight work may still sample these: defer destruction (view first; it references the image)
			auto gc = mtl::get_resource_manager();
			gc->dispose(view_data);
			gc->dispose(img_data.second);

			temp_image_cache.erase(_key);
			temp_view_cache.erase(_key);
		}
	}

	mtl::image_view* ui_overlay_renderer::find_font(const rsx::overlays::font* font, mtl::command_list& cmd, mtl::data_heap& upload_heap)
	{
		const auto image_size = font->get_glyph_data_dimensions();

		u64 key = reinterpret_cast<u64>(font);
		auto found = view_cache.find(key);
		if (found != view_cache.end())
		{
			if (const auto raw = found->second->image();
				image_size.width == raw->width() &&
				image_size.height == raw->height() &&
				image_size.depth == raw->layers())
			{
				return found->second.get();
			}

			auto gc = mtl::get_resource_manager();
			gc->dispose(view_cache[key]);
			gc->dispose(font_cache[key]);
		}

		// Create font resource
		const std::vector<u8>& bytes = font->get_glyph_data();

		const auto& dev = m_device ? *m_device : *g_render_device;
		return upload_simple_texture(dev, cmd, upload_heap, key, image_size.width, image_size.height, image_size.depth,
				true, false, bytes.data(), -1);
	}

	mtl::image_view* ui_overlay_renderer::find_temp_image(const rsx::overlays::image_info_base* desc, mtl::command_list& cmd, mtl::data_heap& upload_heap, u32 owner_uid)
	{
		const bool dirty = std::exchange(desc->dirty, false);
		const u64 key = reinterpret_cast<u64>(desc);

		auto cached = temp_view_cache.find(key);
		if (cached != temp_view_cache.end())
		{
			mtl::image_view* view = cached->second.get();

			if (dirty)
			{
				upload_simple_texture(view->image(), cmd, upload_heap, desc->w, desc->h, 1, false, desc->get_data());
			}

			return view;
		}

		const auto& dev = m_device ? *m_device : *g_render_device;
		return upload_simple_texture(dev, cmd, upload_heap, key, desc->w, desc->h, 1,
				false, true, desc->get_data(), owner_uid);
	}

	std::vector<glsl::program_input> ui_overlay_renderer::get_vertex_inputs()
	{
		auto result = overlay_pass::get_vertex_inputs();
		result.push_back(
			glsl::program_input::make(
				::glsl::glsl_vertex_program,
				"VertexData",
				glsl::input_type_storage_buffer,
				glsl::binding_set_index_vertex,
				0
			)
		);
		result.push_back(
			glsl::program_input::make(
				::glsl::glsl_vertex_program,
				"push_constants",
				glsl::input_type_push_constant,
				glsl::binding_set_index_vertex,
				umax,
				glsl::push_constant_ref { .offset = 0, .size = vertex_push_constants_size }
			)
		);
		return result;
	}

	std::vector<glsl::program_input> ui_overlay_renderer::get_fragment_inputs()
	{
		auto result = overlay_pass::get_fragment_inputs();
		result.push_back(
			glsl::program_input::make(
				::glsl::glsl_fragment_program,
				"push_constants",
				glsl::input_type_push_constant,
				glsl::binding_set_index_fragment,
				umax,
				glsl::push_constant_ref { .offset = vertex_push_constants_size, .size = fragment_push_constants_size }
			)
		);
		return result;
	}

	void ui_overlay_renderer::update_uniforms(mtl::command_list& /*cmd*/, glsl::program* program)
	{
		// Byte Layout (one push constant space shared by both stages, VK layout)
		// 00: vec4 ui_scale;
		// 16: vec4 albedo;
		// 32: vec4 viewport;
		// 48: vec4 clip_bounds;
		// 64: uint vertex_config;
		// 68: uint fragment_config;
		// 72: float timestamp;
		// 76: float blur_intensity;
		// 80: vec4 sdf_params;
		// 96: vec2 sdf_origin;
		// 104: vec2 reserved;
		// 112: vec4 sdf_border_color;

		usz pos = 0;
		std::array<f32, std::max(vertex_push_constants_size, fragment_push_constants_size) / sizeof(f32)> push_buf{};

		// 1. Vertex config (00 - 67)
		write_to_ptr(push_buf, pos, m_scale_offset.rgba);
		pos += sizeof(m_scale_offset.rgba) / sizeof(f32);
		write_to_ptr(push_buf, pos, m_color.rgba);
		pos += sizeof(m_color.rgba) / sizeof(f32);

		push_buf[pos++] = m_viewport.width;
		push_buf[pos++] = m_viewport.height;
		push_buf[pos++] = m_viewport.x;
		push_buf[pos++] = m_viewport.y;

		push_buf[pos++] = m_clip_region.x1;
		push_buf[pos++] = m_clip_region.y1;
		push_buf[pos++] = m_clip_region.x2;
		push_buf[pos++] = m_clip_region.y2;

		rsx::overlays::vertex_options vert_opts {};
		const auto vert_config = vert_opts
			.disable_vertex_snap(m_disable_vertex_snap)
			.get();
		push_buf[pos++] = std::bit_cast<f32>(vert_config);

		ensure(pos <= push_buf.size());
		ensure(pos == (vertex_push_constants_size / sizeof(f32)));
		program->push_constants(glsl::binding_set_index_vertex, 0, vertex_push_constants_size, push_buf.data());

		// 2. Fragment stuff
		rsx::overlays::fragment_options frag_opts {};
		const auto frag_config = frag_opts
			.texture_mode(m_texture_type)
			.clip_fragments(m_clip_enabled)
			.pulse_glow(m_pulse_glow)
			.set_sdf(m_sdf_config.func)
			.get();

		pos = 0;
		push_buf[pos++] = std::bit_cast<f32>(frag_config);
		push_buf[pos++] = m_time;
		push_buf[pos++] = m_blur_strength;
		push_buf[pos++] = m_sdf_config.hx;
		push_buf[pos++] = m_sdf_config.hy;
		push_buf[pos++] = m_sdf_config.br;
		push_buf[pos++] = m_sdf_config.bw;
		push_buf[pos++] = m_sdf_config.cx;
		push_buf[pos++] = m_sdf_config.cy;
		push_buf[pos++] = 0.f;
		push_buf[pos++] = 0.f;

		write_to_ptr(push_buf, pos, m_sdf_config.border_color.rgba);
		pos += sizeof(m_sdf_config.border_color.rgba) / sizeof(f32);

		ensure(pos <= push_buf.size());
		ensure(pos == (fragment_push_constants_size / sizeof(f32)));
		program->push_constants(glsl::binding_set_index_fragment, vertex_push_constants_size, fragment_push_constants_size, push_buf.data());
	}

	void ui_overlay_renderer::set_primitive_type(rsx::overlays::primitive_type type)
	{
		m_current_primitive_type = type;

		switch (type)
		{
			case rsx::overlays::primitive_type::quad_list:
			case rsx::overlays::primitive_type::triangle_fan:
				// Expanded to triangle lists at upload time (Metal has neither quads nor fans)
				renderpass_config.set_primitive_type(MTL::PrimitiveTypeTriangle);
				break;
			case rsx::overlays::primitive_type::triangle_strip:
				renderpass_config.set_primitive_type(MTL::PrimitiveTypeTriangleStrip);
				break;
			case rsx::overlays::primitive_type::line_list:
				renderpass_config.set_primitive_type(MTL::PrimitiveTypeLine);
				break;
			case rsx::overlays::primitive_type::line_strip:
				renderpass_config.set_primitive_type(MTL::PrimitiveTypeLineStrip);
				break;
			default:
				fmt::throw_exception("Unexpected primitive type %d", static_cast<s32>(type));
		}
	}

	void ui_overlay_renderer::emit_geometry(mtl::command_list& cmd, glsl::program* program)
	{
		// Quad lists and fans were converted to triangle lists by run(); a single draw covers the whole command
		// (VK issues one 4-vertex strip draw per quad).
		overlay_pass::emit_geometry(cmd, program);
	}

	void ui_overlay_renderer::run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target,
			mtl::data_heap& upload_heap, rsx::overlays::overlay& ui)
	{
		ui.set_render_viewport(
		    static_cast<u16>(std::min<u32>(viewport.width(), std::numeric_limits<u16>::max())),
		    static_cast<u16>(std::min<u32>(viewport.height(), std::numeric_limits<u16>::max()))
		);
		m_scale_offset = color4f(ui.get_virtual_width(), ui.get_virtual_height(), 1.f, 1.f);
		m_viewport = { { static_cast<f32>(viewport.x1), static_cast<f32>(viewport.y1) }, { static_cast<f32>(viewport.width()), static_cast<f32>(viewport.height()) } };

		if (ui.status_flags & rsx::overlays::status_bits::invalidate_image_cache)
		{
			remove_temp_resources(ui.uid);
			ui.status_flags.clear(rsx::overlays::status_bits::invalidate_image_cache);
		}

		const auto compiled = ui.get_compiled();
		const auto& draw_commands = compiled.draw_commands;

		// 1. Resolve (and upload) all textures first. Uploads are compute/blit work and cannot be recorded inside
		//    the render pass, which is shared by all the draw commands below.
		struct command_source
		{
			mtl::image_view* view = nullptr;
			rsx::overlays::texture_sampling_mode mode = rsx::overlays::texture_sampling_mode::none;
		};

		std::vector<command_source> sources(draw_commands.size());

		for (usz i = 0; i < draw_commands.size(); ++i)
		{
			const auto& command = draw_commands[i];
			auto& source = sources[i];
			source.mode = rsx::overlays::texture_sampling_mode::texture2D;

			switch (command.config.texture_ref)
			{
			case rsx::overlays::image_resource_id::game_icon:
			case rsx::overlays::image_resource_id::backbuffer:
				// TODO
			case rsx::overlays::image_resource_id::none:
				source.mode = rsx::overlays::texture_sampling_mode::none;
				break;
			case rsx::overlays::image_resource_id::font_file:
				source.view = find_font(command.config.font_ref, cmd, upload_heap);
				source.mode = source.view->image()->layers() == 1
					? rsx::overlays::texture_sampling_mode::font2D
					: rsx::overlays::texture_sampling_mode::font3D;
				break;
			case rsx::overlays::image_resource_id::raw_image:
				source.view = find_temp_image(static_cast<const rsx::overlays::image_info_base*>(command.config.external_data_ref), cmd, upload_heap, ui.uid);
				break;
			default:
				source.view = view_cache[command.config.texture_ref].get();
				break;
			}
		}

		const auto null_view_2d = get_null_image_view(MTL::TextureType2D);
		const auto null_view_2d_array = get_null_image_view(MTL::TextureType2DArray);

		// 2. Draw everything in one render pass
		begin_pass(cmd, target, viewport);

		for (usz i = 0; i < draw_commands.size(); ++i)
		{
			const auto& command = draw_commands[i];
			const auto& source = sources[i];

			set_primitive_type(command.config.primitives);

			const auto& verts = command.verts;
			const u32 vertex_count = static_cast<u32>(verts.size());

			switch (m_current_primitive_type)
			{
			case rsx::overlays::primitive_type::quad_list:
			{
				// Quad (strip order v0 v1 v2 v3) -> triangles (v0 v1 v2) (v2 v1 v3)
				const u32 num_quads = vertex_count / 4;
				m_expanded_verts.resize(num_quads * 6);

				for (u32 q = 0; q < num_quads; ++q)
				{
					const auto* in = &verts[q * 4];
					auto* out = &m_expanded_verts[q * 6];
					out[0] = in[0]; out[1] = in[1]; out[2] = in[2];
					out[3] = in[2]; out[4] = in[1]; out[5] = in[3];
				}

				num_drawable_elements = num_quads * 6;
				upload_vertex_data(m_expanded_verts.data(), num_drawable_elements);
				break;
			}
			case rsx::overlays::primitive_type::triangle_fan:
			{
				const u32 num_tris = vertex_count >= 3 ? (vertex_count - 2) : 0;
				m_expanded_verts.resize(num_tris * 3);

				for (u32 t = 0; t < num_tris; ++t)
				{
					m_expanded_verts[t * 3 + 0] = verts[0];
					m_expanded_verts[t * 3 + 1] = verts[t + 1];
					m_expanded_verts[t * 3 + 2] = verts[t + 2];
				}

				num_drawable_elements = num_tris * 3;
				upload_vertex_data(m_expanded_verts.data(), num_drawable_elements);
				break;
			}
			default:
				num_drawable_elements = vertex_count;
				upload_vertex_data(verts.data(), num_drawable_elements);
				break;
			}

			first_vertex = 0;

			if (!num_drawable_elements)
			{
				continue;
			}

			m_time = command.config.get_sinus_value();
			m_texture_type = source.mode;
			m_color = command.config.color;
			m_pulse_glow = command.config.pulse_glow;
			m_blur_strength = static_cast<f32>(command.config.blur_strength) * 0.01f;
			m_clip_enabled = command.config.clip_region;
			m_clip_region = command.config.clip_rect;
			m_disable_vertex_snap = command.config.disable_vertex_snap;

			m_sdf_config = command.config.sdf_config;
			m_sdf_config.transform(static_cast<areaf>(viewport), { m_scale_offset.x, m_scale_offset.y });

			std::vector<mtl::image_view*> image_views
			{
				null_view_2d,
				null_view_2d_array
			};

			if (source.view)
			{
				const int res_id = source.view->image()->layers() > 1 ? 1 : 0;
				image_views[res_id] = source.view;
			}

			overlay_pass::draw(cmd, viewport, target, image_views);
		}

		end_pass(cmd);

		// Do not keep a stale vertex binding around for the next user of this pass
		m_vao_length = 0;

		ui.update(get_system_time());
	}

	// ---- attachment_clear_pass --------------------------------------------------------------------------------------

	attachment_clear_pass::attachment_clear_pass()
	{
		vs_src =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(push_constant) uniform static_data{ vec4 regs[2]; };\n"
			"layout(location=0) out vec4 color;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	color = regs[0];\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], 0., 1.);\n"
			"}\n";

		fs_src =
			"#version 420\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(location=0) in vec4 color;\n"
			"layout(location=0) out vec4 out_color;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	out_color = color;\n"
			"}\n";

		// Disable samplers
		m_num_usable_samplers = 0;

		// Disable UBOs
		m_num_uniform_buffers = 0;

		renderpass_config.set_depth_mask(false);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_attachment_count(1);
	}

	std::vector<glsl::program_input> attachment_clear_pass::get_vertex_inputs()
	{
		return
		{
			glsl::program_input::make(
				::glsl::glsl_vertex_program,
				"push_constants",
				glsl::input_type_push_constant,
				glsl::binding_set_index_vertex,
				umax,
				glsl::push_constant_ref{ .offset = 0, .size = vertex_push_constants_size })
		};
	}

	void attachment_clear_pass::update_uniforms(mtl::command_list& /*cmd*/, glsl::program* program)
	{
		f32 data[8];
		data[0] = clear_color.r;
		data[1] = clear_color.g;
		data[2] = clear_color.b;
		data[3] = clear_color.a;
		data[4] = colormask.r;
		data[5] = colormask.g;
		data[6] = colormask.b;
		data[7] = colormask.a;

		static_assert(sizeof(data) == vertex_push_constants_size);
		program->push_constants(glsl::binding_set_index_vertex, 0, vertex_push_constants_size, data);
	}

	void attachment_clear_pass::set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h)
	{
		auto encoder = cmd.render_encoder();

		MTL::Viewport vp{};
		vp.originX = static_cast<f64>(x);
		vp.originY = static_cast<f64>(y);
		vp.width = static_cast<f64>(w);
		vp.height = static_cast<f64>(h);
		vp.znear = 0.;
		vp.zfar = 1.;
		encoder->setViewport(vp);

		encoder->setScissorRect(clamp_scissor(target, region.x, region.y, region.width, region.height));
	}

	void attachment_clear_pass::configure_attachments(MTL4::RenderPassDescriptor* desc, const overlay_target& target, bool /*covers_target*/)
	{
		const bool full_mask = colormask.r > 0.f && colormask.g > 0.f && colormask.b > 0.f && colormask.a > 0.f;
		const bool full_region = region.x == 0 && region.y == 0 && region.width >= target.width() && region.height >= target.height();

		if (target.color && full_mask && full_region)
		{
			// Whole attachment: use the hardware clear on load
			auto attachment = desc->colorAttachments()->object(0);
			attachment->setLoadAction(MTL::LoadActionClear);
			attachment->setClearColor(MTL::ClearColor::Make(clear_color.r, clear_color.g, clear_color.b, clear_color.a));
		}
	}

	void attachment_clear_pass::run(mtl::command_list& cmd, const overlay_target& target, const coordu& rect, u32 clearmask, color4f color)
	{
		region = rect;

		color4f mask = { 0.f, 0.f, 0.f, 0.f };
		if (clearmask & 0x10) mask.r = 1.f;
		if (clearmask & 0x20) mask.g = 1.f;
		if (clearmask & 0x40) mask.b = 1.f;
		if (clearmask & 0x80) mask.a = 1.f;

		if (mask != colormask || color != clear_color)
		{
			colormask = mask;
			clear_color = color;

			// Update color mask to match request
			renderpass_config.set_color_mask(0, colormask.r > 0.f, colormask.g > 0.f, colormask.b > 0.f, colormask.a > 0.f);
		}

		// Color attachment only; ignore any depth-stencil part of the target
		overlay_target color_target = target;
		color_target.depth_stencil = nullptr;

		const areau full_area = { 0, 0, color_target.width(), color_target.height() };

		if (!rect.width || !rect.height || rect.x >= full_area.x2 || rect.y >= full_area.y2)
		{
			// Nothing to clear (Metal rejects empty scissor rectangles)
			return;
		}

		const bool full_mask = colormask.r > 0.f && colormask.g > 0.f && colormask.b > 0.f && colormask.a > 0.f;
		const bool full_region = region.x == 0 && region.y == 0 && region.width >= full_area.x2 && region.height >= full_area.y2;

		begin_pass(cmd, color_target, full_area);

		if (!(full_mask && full_region))
		{
			// Partial clear: render the quad through the scissor with the requested write mask
			draw(cmd, full_area, color_target, {});
		}

		end_pass(cmd);
	}

	// ---- stencil_clear_pass -----------------------------------------------------------------------------------------

	stencil_clear_pass::stencil_clear_pass()
	{
		vs_src =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], 0., 1.);\n"
			"}\n";

		// Depth-stencil only target: no color output (the stencil value comes from the reference value)
		fs_src =
			"#version 420\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"void main()\n"
			"{\n"
			"}\n";

		m_num_uniform_buffers = 0;
		m_num_usable_samplers = 0;
	}

	void stencil_clear_pass::set_up_viewport(mtl::command_list& cmd, const overlay_target& target, u32 x, u32 y, u32 w, u32 h)
	{
		auto encoder = cmd.render_encoder();

		MTL::Viewport vp{};
		vp.originX = static_cast<f64>(x);
		vp.originY = static_cast<f64>(y);
		vp.width = static_cast<f64>(w);
		vp.height = static_cast<f64>(h);
		vp.znear = 0.;
		vp.zfar = 1.;
		encoder->setViewport(vp);

		encoder->setScissorRect(clamp_scissor(target, region.x, region.y, region.width, region.height));
	}

	void stencil_clear_pass::run(mtl::command_list& cmd, const overlay_target& target, const coordu& rect, u32 stencil_clear, u32 stencil_write_mask)
	{
		ensure(target.has_stencil(), "Stencil clear on a target without stencil");
		region = rect;

		if (!rect.width || !rect.height || rect.x >= target.width() || rect.y >= target.height())
		{
			// Nothing to clear (Metal rejects empty scissor rectangles)
			return;
		}

		// Stencil setup. Replace all pixels in the scissor region with stencil_clear with the correct write mask.
		renderpass_config.enable_stencil_test(
			MTL::StencilOperationReplace, MTL::StencilOperationReplace, MTL::StencilOperationReplace,  // Always replace
			MTL::CompareFunctionAlways,                                                                 // Always pass
			0xFF,                                                                                       // Full write-through
			stencil_clear);                                                                             // Write active bit

		renderpass_config.set_stencil_mask(stencil_write_mask);
		renderpass_config.set_depth_mask(false);

		overlay_target ds_target = target;
		ds_target.color = nullptr;

		overlay_pass::run(cmd, { 0, 0, ds_target.width(), ds_target.height() }, ds_target, std::vector<mtl::image_view*>{});
	}

	// ---- video_out_calibration_pass ---------------------------------------------------------------------------------

	video_out_calibration_pass::video_out_calibration_pass()
	{
		vs_src =
		#include "../Program/GLSLSnippets/GenericVSPassthrough.glsl"
		;

		fs_src =
		#include "../Program/GLSLSnippets/VideoOutCalibrationPass.glsl"
		;

		std::pair<std::string_view, std::string> repl_list[] =
		{
			{ "%sampler_binding", "x" },
			{ "%set_decorator", "set=1" },
		};
		fs_src = fmt::replace_all(fs_src, repl_list);

		renderpass_config.set_depth_mask(false);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_attachment_count(1);

		m_num_usable_samplers = 2;
		m_num_uniform_buffers = 0;
	}

	std::vector<glsl::program_input> video_out_calibration_pass::get_fragment_inputs()
	{
		auto result = overlay_pass::get_fragment_inputs();
		result.push_back(
			glsl::program_input::make(
				::glsl::glsl_fragment_program,
				"push_constants",
				glsl::input_type_push_constant,
				glsl::binding_set_index_fragment,
				umax,
				glsl::push_constant_ref{ .offset = 0, .size = fragment_push_constants_size }
			)
		);
		return result;
	}

	void video_out_calibration_pass::update_uniforms(mtl::command_list& /*cmd*/, glsl::program* program)
	{
		static_assert(sizeof(config.data) == fragment_push_constants_size);
		program->push_constants(glsl::binding_set_index_fragment, 0, fragment_push_constants_size, config.data);
	}

	void video_out_calibration_pass::run(mtl::command_list& cmd, const areau& viewport, const overlay_target& target,
		const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
		bool stereo_enabled)
	{
		static stereo_config stereo_cfg = stereo_config(true);
		stereo_cfg.update_from_config(stereo_enabled);
		const auto& matrices = stereo_cfg.matrices();

		config.gamma = gamma;
		config.limit_range = limited_rgb ? 1 : 0;
		config.stereo_display_mode = static_cast<u8>(stereo_cfg.stereo_mode());
		config.stereo_image_count = std::min(::size32(src), 2u);

		for (u32 i = 0; i < 3; i++)
		{
			std::memcpy(config.left_anaglyph_matrix[i].rgba, matrices.left[i].rgb, sizeof(matrices.left[i].rgb));
			std::memcpy(config.right_anaglyph_matrix[i].rgba, matrices.right[i].rgb, sizeof(matrices.right[i].rgb));
		}

		std::vector<mtl::image_view*> views;
		views.reserve(2);

		for (auto& img : src)
		{
			// Raw data, ignore the native component layout (VK_REMAP_IDENTITY)
			views.push_back(img->get_identity_view());
		}

		if (views.size() < 2)
		{
			views.push_back(get_null_image_view(MTL::TextureType2D));
		}

		overlay_pass::run(cmd, viewport, target, views);
	}

	// ---- Global management ------------------------------------------------------------------------------------------

	void reset_overlay_passes()
	{
		for (const auto& p : g_overlay_passes)
		{
			p.second->free_resources();
		}
	}

	void destroy_overlay_passes()
	{
		for (const auto& p : g_overlay_passes)
		{
			p.second->destroy();
		}
		g_overlay_passes.clear();

		{
			std::lock_guard lock(g_ds_state_lock);
			for (auto& [key, state] : g_ds_state_cache)
			{
				state->release();
			}
			g_ds_state_cache.clear();
		}

		g_null_textures.tex_2d.reset();
		g_null_textures.tex_2d_array.reset();

		destroy_blit_resources();
	}
}
