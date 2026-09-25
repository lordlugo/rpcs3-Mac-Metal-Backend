#include "stdafx.h"

#include "MTLResolveHelper.h"
#include "mtlutils/device.h"

namespace mtl
{
	std::unique_ptr<mtl::color_resolve_pass> g_color_resolver;
	std::unique_ptr<mtl::color_unresolve_pass> g_color_unresolver;
	std::unique_ptr<mtl::depthonly_resolve> g_depth_resolver;
	std::unique_ptr<mtl::depthonly_unresolve> g_depth_unresolver;
	std::unique_ptr<mtl::depthstencil_resolve_EXT> g_depthstencil_resolver;
	std::unique_ptr<mtl::depthstencil_unresolve_EXT> g_depthstencil_unresolver;

	template <typename T, typename ...Args>
	void initialize_pass(std::unique_ptr<T>& ptr, const mtl::render_device& dev, Args&&... extras)
	{
		if (!ptr)
		{
			ptr = std::make_unique<T>(std::forward<Args>(extras)...);
			ptr->create(dev);
		}
	}

	// ---- Color passes -----------------------------------------------------------------------------------------------

	color_resolve_pass::color_resolve_pass()
	{
		vs_src =
			#include "Emu/RSX/Program/GLSLSnippets/GenericVSPassthrough.glsl"
			;

		fs_src =
			#include "Emu/RSX/Program/MSAA/ColorResolvePassFS.glsl"
			;

		fs_src = remap_fragment_set(fs_src);

		renderpass_config.set_depth_mask(false);
		renderpass_config.disable_depth_test();
		renderpass_config.set_color_mask(0, true, true, true, true);
		m_overwrites_color = true;
	}

	void color_resolve_pass::run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		ensure(msaa_image->samples() > 1);
		ensure(resolve_image->samples() == 1);

		update_sample_configuration(msaa_image);

		// Raw data (VK_REMAP_VIEW_MULTISAMPLED / VK_REMAP_IDENTITY): ignore the native component layout on both sides
		auto src_view = msaa_image->get_identity_view(aspect_color);

		overlay_pass::run(
			cmd,
			{ 0, 0, resolve_image->width(), resolve_image->height() },
			resolve_image, src_view);
	}

	color_unresolve_pass::color_unresolve_pass()
	{
		vs_src =
			#include "Emu/RSX/Program/GLSLSnippets/GenericVSPassthrough.glsl"
			;

		fs_src =
			#include "Emu/RSX/Program/MSAA/ColorUnresolvePassFS.glsl"
			;

		fs_src = remap_fragment_set(fs_src);

		renderpass_config.set_depth_mask(false);
		renderpass_config.disable_depth_test();
		renderpass_config.set_color_mask(0, true, true, true, true);
		m_overwrites_color = true;
	}

	void color_unresolve_pass::run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		ensure(msaa_image->samples() > 1);
		ensure(resolve_image->samples() == 1);

		// Per-sample shading is implied by gl_SampleID ([[sample_id]]) in the fragment shader
		update_sample_configuration(msaa_image);

		auto src_view = resolve_image->get_identity_view(aspect_color);

		overlay_pass::run(
			cmd,
			{ 0, 0, msaa_image->width(), msaa_image->height() },
			msaa_image, src_view);
	}

	// ---- Entry points -----------------------------------------------------------------------------------------------

	void resolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src)
	{
		const auto& dev = *g_render_device;
		const u32 aspect = src->aspect();

		if (aspect == aspect_color)
		{
			initialize_pass(g_color_resolver, dev);
			g_color_resolver->run(cmd, src, dst);
			return;
		}

		if ((aspect & aspect_depth) && (aspect & aspect_stencil) && (dst->aspect() & aspect_stencil))
		{
			initialize_pass(g_depthstencil_resolver, dev);
			g_depthstencil_resolver->run(cmd, src, dst);
		}
		else if (aspect & aspect_depth)
		{
			initialize_pass(g_depth_resolver, dev);
			g_depth_resolver->run(cmd, src, dst);
		}
		else
		{
			// RSX has no stencil-only surfaces
			rsx_log.error("Metal: MSAA resolve of stencil-only format %d is not supported", static_cast<int>(src->format()));
		}
	}

	void unresolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src)
	{
		const auto& dev = *g_render_device;
		const u32 aspect = src->aspect();

		if (aspect == aspect_color)
		{
			initialize_pass(g_color_unresolver, dev);
			g_color_unresolver->run(cmd, dst, src);
			return;
		}

		if ((aspect & aspect_depth) && (aspect & aspect_stencil) && (dst->aspect() & aspect_stencil))
		{
			initialize_pass(g_depthstencil_unresolver, dev);
			g_depthstencil_unresolver->run(cmd, dst, src);
		}
		else if (aspect & aspect_depth)
		{
			initialize_pass(g_depth_unresolver, dev);
			g_depth_unresolver->run(cmd, dst, src);
		}
		else
		{
			rsx_log.error("Metal: MSAA unresolve of stencil-only format %d is not supported", static_cast<int>(src->format()));
		}
	}

	void clear_resolve_helpers()
	{
		auto destroy_pass = [](auto& pass)
		{
			if (pass)
			{
				pass->destroy();
				pass.reset();
			}
		};

		destroy_pass(g_color_resolver);
		destroy_pass(g_color_unresolver);
		destroy_pass(g_depth_resolver);
		destroy_pass(g_depthstencil_resolver);
		destroy_pass(g_depth_unresolver);
		destroy_pass(g_depthstencil_unresolver);
	}

	void reset_resolve_resources()
	{
		if (g_color_resolver) g_color_resolver->free_resources();
		if (g_color_unresolver) g_color_unresolver->free_resources();
		if (g_depth_resolver) g_depth_resolver->free_resources();
		if (g_depth_unresolver) g_depth_unresolver->free_resources();
		if (g_depthstencil_resolver) g_depthstencil_resolver->free_resources();
		if (g_depthstencil_unresolver) g_depthstencil_unresolver->free_resources();
	}

	void depth_resolve_base::build(bool resolve_depth, bool resolve_stencil, bool is_unresolve)
	{
		vs_src =
			#include "Emu/RSX/Program/GLSLSnippets/GenericVSPassthrough.glsl"
			;

		static const char* depth_resolver =
			#include "Emu/RSX/Program/MSAA/DepthResolvePass.glsl"
			;

		static const char* depth_unresolver =
			#include "Emu/RSX/Program/MSAA/DepthUnresolvePass.glsl"
			;

		static const char* stencil_resolver =
			#include "Emu/RSX/Program/MSAA/StencilResolvePass.glsl"
			;

		static const char* stencil_unresolver =
			#include "Emu/RSX/Program/MSAA/StencilUnresolvePass.glsl"
			;

		static const char* depth_stencil_resolver =
			#include "Emu/RSX/Program/MSAA/DepthStencilResolvePass.glsl"
			;

		static const char* depth_stencil_unresolver =
			#include "Emu/RSX/Program/MSAA/DepthStencilUnresolvePass.glsl"
			;

		if (resolve_depth && resolve_stencil)
		{
			fs_src = is_unresolve ? depth_stencil_unresolver : depth_stencil_resolver;
		}
		else if (resolve_depth)
		{
			fs_src = is_unresolve ? depth_unresolver : depth_resolver;
		}
		else if (resolve_stencil)
		{
			fs_src = is_unresolve ? stencil_unresolver : stencil_resolver;
		}

		// Fragment resources live in set 1 on Metal
		fs_src = remap_fragment_set(fs_src);

		rsx_log.notice("Resolve shader:\n%s", fs_src);
	}
}
