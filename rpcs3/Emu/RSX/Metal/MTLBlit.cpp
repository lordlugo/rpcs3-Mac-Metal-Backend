#include "stdafx.h"

// Scaled image copies. Metal has no vkCmdBlitImage: every scaled/format-converting copy is a sampled draw through
// mtl::blit_pass (color) or its depth/stencil variants, which write [[depth]] / [[stencil]] from the fragment shader.

#include "MTLOverlays.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLResourceManager.h"
#include "upscalers/upscaling.h"

#include "mtlutils/device.h"
#include "mtlutils/image.h"

namespace mtl
{
	namespace
	{
		const char* s_blit_vertex_shader =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(push_constant) uniform blit_parameters\n"
			"{\n"
			"	vec4 src_rect; // Normalized source origin (xy) and extent (zw). Negative extents mirror.\n"
			"};\n"
			"\n"
			"layout(location=0) out vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	vec2 coords[] = {vec2(0., 0.), vec2(1., 0.), vec2(0., 1.), vec2(1., 1.)};\n"
			"	tc0 = fma(coords[gl_VertexIndex % 4], src_rect.zw, src_rect.xy);\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], 0., 1.);\n"
			"}\n";

		const char* s_blit_color_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"layout(location=0) out vec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	ocol = textureLod(fs0, tc0, 0.);\n"
			"}\n";

		const char* s_blit_depth_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	gl_FragDepth = textureLod(fs0, tc0, 0.).x;\n"
			"}\n";

		// Integer and stencil sources are fetched (nearest texel of the mapped coordinate) instead of sampled
		#define BLIT_NEAREST_TEXEL(tex) \
			"	ivec2 size_" #tex " = textureSize(" #tex ", 0);\n" \
			"	ivec2 coord_" #tex " = clamp(ivec2(floor(tc0 * vec2(size_" #tex "))), ivec2(0), size_" #tex " - ivec2(1));\n"

		const char* s_blit_uint_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform usampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"layout(location=0) out uvec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			BLIT_NEAREST_TEXEL(fs0)
			"	ocol = texelFetch(fs0, coord_fs0, 0);\n"
			"}\n";

		const char* s_blit_sint_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform isampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"layout(location=0) out ivec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			BLIT_NEAREST_TEXEL(fs0)
			"	ocol = texelFetch(fs0, coord_fs0, 0);\n"
			"}\n";

		const char* s_blit_stencil_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"#extension GL_ARB_shader_stencil_export : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform usampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			BLIT_NEAREST_TEXEL(fs0)
			"	gl_FragStencilRefARB = int(texelFetch(fs0, coord_fs0, 0).x & 0xFF);\n"
			"}\n";

		const char* s_blit_depth_stencil_fs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"#extension GL_ARB_shader_stencil_export : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(set=1, binding=1) uniform usampler2D fs1;\n"
			"layout(location=0) in vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			BLIT_NEAREST_TEXEL(fs1)
			"	gl_FragDepth = textureLod(fs0, tc0, 0.).x;\n"
			"	gl_FragStencilRefARB = int(texelFetch(fs1, coord_fs1, 0).x & 0xFF);\n"
			"}\n";

		#undef BLIT_NEAREST_TEXEL

		// Clip a destination span to [0, limit) and adjust the (floating point) source span proportionally
		void clip_span(s32& d1, s32& d2, f32& s1, f32& s2, s32 limit)
		{
			// Precondition: d1 < d2
			const f32 scale = (s2 - s1) / static_cast<f32>(d2 - d1);

			if (d1 < 0)
			{
				s1 += scale * static_cast<f32>(-d1);
				d1 = 0;
			}

			if (d2 > limit)
			{
				s2 -= scale * static_cast<f32>(d2 - limit);
				d2 = limit;
			}
		}

		// Scratch images used when the source and destination subresource are the same, or when the destination
		// image lacks RenderTarget usage. Slot 0 = source copy (sampled only, any format), slot 1 = render target
		// stand-in (only requested for renderable formats, see copy_scaled_image).
		struct blit_scratch_key
		{
			u32 slot;
			MTL::PixelFormat format;

			bool operator==(const blit_scratch_key&) const = default;
		};

		struct blit_scratch_key_hash
		{
			usz operator()(const blit_scratch_key& key) const
			{
				return (static_cast<usz>(key.format) << 1) | key.slot;
			}
		};

		std::unordered_map<blit_scratch_key, std::unique_ptr<mtl::image>, blit_scratch_key_hash> g_blit_scratch;

		// `exact`: the scratch must have exactly width x height (whole-subresource copies)
		mtl::image* get_blit_scratch(u32 slot, const mtl::image* like, u32 width, u32 height, bool exact = false)
		{
			const blit_scratch_key key{ slot, like->format() };
			const bool render_target = (slot == 1);
			ensure(!render_target || format_is_renderable(like->format()));
			auto& entry = g_blit_scratch[key];

			if (entry && (exact
				? (entry->width() == width && entry->height() == height)
				: (entry->width() >= width && entry->height() >= height)))
			{
				return entry.get();
			}

			const u32 new_w = (entry && !exact) ? std::max(entry->width(), width) : width;
			const u32 new_h = (entry && !exact) ? std::max(entry->height(), height) : height;

			if (entry)
			{
				// May still be referenced by in-flight work
				mtl::get_resource_manager()->dispose(entry);
			}

			image_create_info info{};
			info.type = MTL::TextureType2D;
			info.format = like->format();
			info.width = new_w;
			info.height = new_h;
			info.usage = render_target ? (MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget) : MTL::TextureUsageShaderRead;
			info.storage = memory_location::device_local;
			info.format_class = like->format_class();

			entry = std::make_unique<mtl::image>(*g_render_device, info);
			entry->set_debug_name(fmt::format("blit scratch %u", slot));
			return entry.get();
		}

		// VK mag_max: max on the magnitude, preserving the sign
		int mag_max(int value, int clamp_val)
		{
			if (value >= 0)
			{
				return std::max(value, clamp_val);
			}

			return -std::max(-value, clamp_val);
		}

		// Next mip level of a (possibly mirrored) area, VK increment_mip_level() semantics
		areai next_mip_area(const areai& area)
		{
			const int mip_w = (area.x2 - area.x1) / 2;
			const int mip_h = (area.y2 - area.y1) / 2;

			areai result;
			result.x1 = area.x1 / 2;
			result.y1 = area.y1 / 2;
			result.x2 = result.x1 + mag_max(mip_w, 1);
			result.y2 = result.y1 + mag_max(mip_h, 1);
			return result;
		}

		areai to_area(const coord3i& rect)
		{
			return { rect.x, rect.y, rect.x + rect.width, rect.y + rect.height };
		}

		blit_pass* get_blit_pass_for(u32 dst_aspect, u32 src_aspect, MTL::PixelFormat dst_format)
		{
			if (dst_aspect & aspect_depth)
			{
				if ((dst_aspect & aspect_stencil) && (src_aspect & aspect_stencil))
				{
					return get_overlay_pass<depth_stencil_blit_pass>();
				}

				return get_overlay_pass<depth_blit_pass>();
			}

			if (dst_aspect & aspect_stencil)
			{
				return get_overlay_pass<stencil_blit_pass>();
			}

			switch (get_blit_output_type(dst_format))
			{
			case blit_output_type::uint_:
				return get_overlay_pass<uint_blit_pass>();
			case blit_output_type::sint_:
				return get_overlay_pass<sint_blit_pass>();
			default:
				return get_overlay_pass<blit_pass>();
			}
		}
	}

	// ---- blit_pass --------------------------------------------------------------------------------------------------

	blit_output_type get_blit_output_type(MTL::PixelFormat format)
	{
		switch (format)
		{
		case MTL::PixelFormatR8Uint:
		case MTL::PixelFormatR16Uint:
		case MTL::PixelFormatR32Uint:
		case MTL::PixelFormatRG8Uint:
		case MTL::PixelFormatRG16Uint:
		case MTL::PixelFormatRG32Uint:
		case MTL::PixelFormatRGBA8Uint:
		case MTL::PixelFormatRGB10A2Uint:
		case MTL::PixelFormatRGBA16Uint:
		case MTL::PixelFormatRGBA32Uint:
		case MTL::PixelFormatStencil8:
		case MTL::PixelFormatX32_Stencil8:
		case MTL::PixelFormatX24_Stencil8:
			return blit_output_type::uint_;
		case MTL::PixelFormatR8Sint:
		case MTL::PixelFormatR16Sint:
		case MTL::PixelFormatR32Sint:
		case MTL::PixelFormatRG8Sint:
		case MTL::PixelFormatRG16Sint:
		case MTL::PixelFormatRG32Sint:
		case MTL::PixelFormatRGBA8Sint:
		case MTL::PixelFormatRGBA16Sint:
		case MTL::PixelFormatRGBA32Sint:
			return blit_output_type::sint_;
		default:
			return blit_output_type::float_;
		}
	}

	blit_pass::blit_pass(u32 dst_aspect, blit_output_type output_type)
		: m_dst_aspect(dst_aspect)
		, m_output_type(output_type)
	{
		vs_src = s_blit_vertex_shader;

		m_num_uniform_buffers = 0;
		m_num_usable_samplers = 1;

		switch (dst_aspect)
		{
		case aspect_color:
			switch (output_type)
			{
			case blit_output_type::uint_:
				fs_src = s_blit_uint_fs;
				m_sampler_filter = MTL::SamplerMinMagFilterNearest;
				break;
			case blit_output_type::sint_:
				fs_src = s_blit_sint_fs;
				m_sampler_filter = MTL::SamplerMinMagFilterNearest;
				break;
			default:
				fs_src = s_blit_color_fs;
				break;
			}
			renderpass_config.set_color_mask(0, true, true, true, true);
			renderpass_config.set_depth_mask(false);
			m_overwrites_color = true;
			break;
		case aspect_depth:
			fs_src = s_blit_depth_fs;
			renderpass_config.set_depth_mask(true);
			renderpass_config.enable_depth_test(MTL::CompareFunctionAlways);
			m_sampler_filter = MTL::SamplerMinMagFilterNearest;
			m_overwrites_depth = true;
			break;
		case aspect_stencil:
			fs_src = s_blit_stencil_fs;
			renderpass_config.set_depth_mask(false);
			renderpass_config.enable_stencil_test(
				MTL::StencilOperationReplace, MTL::StencilOperationReplace, MTL::StencilOperationReplace,
				MTL::CompareFunctionAlways, 0xFF, 0);
			renderpass_config.set_stencil_mask(0xFF);
			m_sampler_filter = MTL::SamplerMinMagFilterNearest;
			m_overwrites_stencil = true;
			break;
		case aspect_depth_stencil:
			fs_src = s_blit_depth_stencil_fs;
			renderpass_config.set_depth_mask(true);
			renderpass_config.enable_depth_test(MTL::CompareFunctionAlways);
			renderpass_config.enable_stencil_test(
				MTL::StencilOperationReplace, MTL::StencilOperationReplace, MTL::StencilOperationReplace,
				MTL::CompareFunctionAlways, 0xFF, 0);
			renderpass_config.set_stencil_mask(0xFF);
			m_sampler_filter = MTL::SamplerMinMagFilterNearest;
			m_num_usable_samplers = 2;
			m_overwrites_depth = true;
			m_overwrites_stencil = true;
			break;
		default:
			fmt::throw_exception("Invalid blit aspect 0x%x", dst_aspect);
		}
	}

	std::vector<glsl::program_input> blit_pass::get_vertex_inputs()
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

	void blit_pass::update_uniforms(mtl::command_list& /*cmd*/, glsl::program* program)
	{
		static_assert(sizeof(m_src_rect) == vertex_push_constants_size);
		program->push_constants(glsl::binding_set_index_vertex, 0, vertex_push_constants_size, m_src_rect);
	}

	void blit_pass::run(mtl::command_list& cmd, const overlay_target& target, const areai& dst_area,
		const std::vector<mtl::image_view*>& src, const areai& src_area, bool linear_filter)
	{
		ensure(!src.empty() && src[0]);

		s32 dx1 = dst_area.x1, dx2 = dst_area.x2, dy1 = dst_area.y1, dy2 = dst_area.y2;
		f32 sx1 = static_cast<f32>(src_area.x1), sx2 = static_cast<f32>(src_area.x2);
		f32 sy1 = static_cast<f32>(src_area.y1), sy2 = static_cast<f32>(src_area.y2);

		// Mirrored destination == mirrored source on a positive destination rectangle
		if (dx1 > dx2)
		{
			std::swap(dx1, dx2);
			std::swap(sx1, sx2);
		}

		if (dy1 > dy2)
		{
			std::swap(dy1, dy2);
			std::swap(sy1, sy2);
		}

		if (dx1 == dx2 || dy1 == dy2 || sx1 == sx2 || sy1 == sy2)
		{
			return;
		}

		// Keep the viewport inside the render target
		clip_span(dx1, dx2, sx1, sx2, static_cast<s32>(target.width()));
		clip_span(dy1, dy2, sy1, sy2, static_cast<s32>(target.height()));

		if (dx1 >= dx2 || dy1 >= dy2)
		{
			return;
		}

		const auto view = src[0];
		const auto image = view->image();
		const f32 src_w = static_cast<f32>(std::max(image->width() >> view->info.base_level, 1u));
		const f32 src_h = static_cast<f32>(std::max(image->height() >> view->info.base_level, 1u));

		m_src_rect[0] = sx1 / src_w;
		m_src_rect[1] = sy1 / src_h;
		m_src_rect[2] = (sx2 - sx1) / src_w;
		m_src_rect[3] = (sy2 - sy1) / src_h;

		// Depth, stencil and integer formats are never filtered
		m_sampler_filter = (linear_filter && m_dst_aspect == aspect_color && m_output_type == blit_output_type::float_)
			? MTL::SamplerMinMagFilterLinear
			: MTL::SamplerMinMagFilterNearest;

		const areau viewport =
		{
			static_cast<u32>(dx1), static_cast<u32>(dy1),
			static_cast<u32>(dx2), static_cast<u32>(dy2)
		};

		overlay_pass::run(cmd, viewport, target, src);
	}

	// ---- Scaled copies ----------------------------------------------------------------------------------------------

	void upscale_blit(mtl::command_list& cmd, mtl::viewable_image* src, MTL::Texture* dst,
		const areai& src_area, const areai& dst_area, bool linear_filter)
	{
		ensure(src && dst);

		// Raw copy semantics (vkCmdBlitImage ignores the native component layout)
		std::vector<mtl::image_view*> views = { src->get_identity_view() };
		get_overlay_pass<blit_pass>()->run(cmd, overlay_target(dst), dst_area, views, src_area, linear_filter);
	}

	void copy_scaled_image(mtl::command_list& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		bool compatible_formats, bool linear_filter)
	{
		// Compatible formats is now optional (false), so we should check the formats again just in case.
		compatible_formats |= (src->format() == dst->format());

		if (compatible_formats && !src_rect.is_flipped() && !dst_rect.is_flipped() &&
			src_rect.width == dst_rect.width && src_rect.height == dst_rect.height && src_rect.depth == dst_rect.depth)
		{
			copy_image(cmd, src, dst, src_rect, dst_rect, mip_layers);
			return;
		}

		if (!src_rect.width || !src_rect.height || !dst_rect.width || !dst_rect.height)
		{
			return;
		}

		if (src->samples() > 1 || dst->samples() > 1)
		{
			// Same restriction as vkCmdBlitImage
			rsx_log.error("Metal: scaled copy of multisampled images is not supported (src samples=%u, dst samples=%u)", src->samples(), dst->samples());
			return;
		}

		if (src->type() == MTL::TextureType3D || dst->type() == MTL::TextureType3D ||
			std::abs(src_rect.depth) > 1 || std::abs(dst_rect.depth) > 1)
		{
			rsx_log.error("Metal: scaled copy of 3D image regions is not supported");
			return;
		}

		if (!format_is_renderable(dst->format()))
		{
			// Block-compressed (or otherwise non-renderable) destinations cannot be drawn into, and a stand-in of another
			// format would change the data encoding. vkCmdBlitImage does not support them either.
			rsx_log.error("Metal: scaled copy into non-renderable format %d is not supported (src fmt=%d)",
				static_cast<int>(dst->format()), static_cast<int>(src->format()));
			return;
		}

		const u32 src_aspect = src->aspect();
		const u32 dst_aspect = dst->aspect();

		if (dst_aspect == aspect_color)
		{
			// Integer and non-integer color formats cannot be blitted into each other (same rule as vkCmdBlitImage)
			const auto dst_type = get_blit_output_type(dst->format());
			const auto src_type = (src_aspect == aspect_color) ? get_blit_output_type(src->format()) : blit_output_type::float_;

			if (dst_type != src_type)
			{
				rsx_log.error("Metal: scaled copy between integer and non-integer formats is not supported (src fmt=%d, dst fmt=%d)",
					static_cast<int>(src->format()), static_cast<int>(dst->format()));
				return;
			}
		}

		auto pass = get_blit_pass_for(dst_aspect, src_aspect, dst->format());

		// Aspects sampled from the source for this pass
		const bool needs_stencil_view = (pass->m_dst_aspect & aspect_stencil) != 0;
		const bool needs_depth_view = (pass->m_dst_aspect & aspect_depth) != 0;

		if (needs_stencil_view && !(src_aspect & aspect_stencil))
		{
			rsx_log.error("Metal: stencil scaled copy from a source without stencil (src fmt=%d)", static_cast<int>(src->format()));
			return;
		}

		const bool dst_renderable = (dst->info.usage & MTL::TextureUsageRenderTarget) != 0;
		auto gc = mtl::get_resource_manager();

		areai src_area = to_area(src_rect);
		areai dst_area = to_area(dst_rect);

		for (u32 level = 0; level < mip_layers.mipmap_count; ++level)
		{
			const u32 src_level = mip_layers.src_mip_level + level;
			const u32 dst_level = mip_layers.dst_mip_level + level;

			for (u32 layer = 0; layer < mip_layers.layer_count; ++layer)
			{
				const u32 src_layer = mip_layers.src_layer + layer;
				const u32 dst_layer = mip_layers.dst_layer + layer;

				mtl::image* sample_image = src;
				u32 sample_level = src_level;
				u32 sample_layer = src_layer;
				areai sample_area = src_area;

				if (src == dst && src_level == dst_level && src_layer == dst_layer)
				{
					// Sampling a subresource that is bound as the render target is a feedback loop. Copy the source
					// region out first (the regions are expected not to overlap, but the pass would still read the
					// attachment it renders to).
					const s32 x = std::min(src_area.x1, src_area.x2);
					const s32 y = std::min(src_area.y1, src_area.y2);
					const u32 w = static_cast<u32>(std::abs(src_area.x2 - src_area.x1));
					const u32 h = static_cast<u32>(std::abs(src_area.y2 - src_area.y1));
					const u32 level_w = std::max(src->width() >> src_level, 1u);
					const u32 level_h = std::max(src->height() >> src_level, 1u);

					if (src_aspect & aspect_depth_stencil)
					{
						// Depth/stencil: whole-subresource copy (partial blit copies of depth/stencil textures are not
						// allowed on every Metal GPU). Sampling coordinates stay the same.
						auto scratch = get_blit_scratch(0, src, level_w, level_h, true);
						cmd.compute()->copyFromTexture(src->value, src_layer, src_level, scratch->value, 0, 0, 1, 1);

						sample_image = scratch;
						sample_level = 0;
						sample_layer = 0;
					}
					else
					{
						if (x < 0 || y < 0 || (static_cast<u32>(x) + w) > level_w || (static_cast<u32>(y) + h) > level_h)
						{
							rsx_log.error("Metal: scaled self-copy source region (%d,%d %ux%u) is outside the image (%ux%u)", x, y, w, h, level_w, level_h);
							continue;
						}

						auto scratch = get_blit_scratch(0, src, w, h);
						cmd.compute()->copyFromTexture(src->value, src_layer, src_level, MTL::Origin(x, y, 0), MTL::Size(w, h, 1),
							scratch->value, 0, 0, MTL::Origin(0, 0, 0));

						sample_image = scratch;
						sample_level = 0;
						sample_layer = 0;
						sample_area = { 0, 0, static_cast<s32>(w), static_cast<s32>(h) };
						if (src_area.x1 > src_area.x2) std::swap(sample_area.x1, sample_area.x2);
						if (src_area.y1 > src_area.y2) std::swap(sample_area.y1, sample_area.y2);
					}
				}

				// Source views (single level, single slice, 2D)
				auto make_view = [&](u32 aspect) -> std::unique_ptr<mtl::image_view>
				{
					image_view_info info{};
					info.type = MTL::TextureType2D;
					info.base_level = sample_level;
					info.level_count = 1;
					info.base_layer = sample_layer;
					info.layer_count = 1;
					info.aspect = aspect;
					return std::make_unique<mtl::image_view>(sample_image, info);
				};

				std::unique_ptr<mtl::image_view> view0, view1;
				std::vector<mtl::image_view*> views;

				if (pass->m_dst_aspect == aspect_depth_stencil)
				{
					view0 = make_view(aspect_depth);
					view1 = make_view(aspect_stencil);
					views = { view0.get(), view1.get() };
				}
				else if (needs_depth_view || pass->m_dst_aspect == aspect_color)
				{
					// Color and depth read the first plane (depth of a depth-stencil source, or color)
					view0 = make_view((src_aspect & aspect_depth) ? static_cast<u32>(aspect_depth) : static_cast<u32>(aspect_color));
					views = { view0.get() };
				}
				else
				{
					view0 = make_view(aspect_stencil);
					views = { view0.get() };
				}

				if (dst_renderable)
				{
					pass->run(cmd, overlay_target(dst, dst_level, dst_layer), dst_area, views, sample_area, linear_filter);
				}
				else
				{
					// Render into a stand-in with RenderTarget usage (same, renderable format), then copy into the destination
					const s32 x = std::min(dst_area.x1, dst_area.x2);
					const s32 y = std::min(dst_area.y1, dst_area.y2);
					const u32 w = static_cast<u32>(std::abs(dst_area.x2 - dst_area.x1));
					const u32 h = static_cast<u32>(std::abs(dst_area.y2 - dst_area.y1));
					const u32 level_w = std::max(dst->width() >> dst_level, 1u);
					const u32 level_h = std::max(dst->height() >> dst_level, 1u);

					if (dst_aspect & aspect_depth_stencil)
					{
						// Depth/stencil: only whole-subresource blit copies. Round-trip the full level through the
						// stand-in so the pixels outside dst_area are preserved.
						auto scratch = get_blit_scratch(1, dst, level_w, level_h, true);
						cmd.compute()->copyFromTexture(dst->value, dst_layer, dst_level, scratch->value, 0, 0, 1, 1);

						pass->run(cmd, overlay_target(scratch), dst_area, views, sample_area, linear_filter);

						cmd.compute()->copyFromTexture(scratch->value, 0, 0, dst->value, dst_layer, dst_level, 1, 1);
					}
					else if (x < 0 || y < 0 || (static_cast<u32>(x) + w) > level_w || (static_cast<u32>(y) + h) > level_h)
					{
						rsx_log.error("Metal: scaled copy destination region (%d,%d %ux%u) is outside the image (%ux%u)", x, y, w, h, level_w, level_h);
					}
					else
					{
						auto scratch = get_blit_scratch(1, dst, w, h);

						areai scratch_area = { 0, 0, static_cast<s32>(w), static_cast<s32>(h) };
						if (dst_area.x1 > dst_area.x2) std::swap(scratch_area.x1, scratch_area.x2);
						if (dst_area.y1 > dst_area.y2) std::swap(scratch_area.y1, scratch_area.y2);

						pass->run(cmd, overlay_target(scratch), scratch_area, views, sample_area, linear_filter);

						cmd.compute()->copyFromTexture(scratch->value, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(w, h, 1),
							dst->value, dst_layer, dst_level, MTL::Origin(x, y, 0));
					}
				}

				// The views are referenced by the recorded pass
				gc->dispose(view0);
				gc->dispose(view1);
			}

			if (level + 1 < mip_layers.mipmap_count)
			{
				src_area = next_mip_area(src_area);
				dst_area = next_mip_area(dst_area);
			}
		}
	}

	void destroy_blit_resources()
	{
		g_blit_scratch.clear();
	}
}
