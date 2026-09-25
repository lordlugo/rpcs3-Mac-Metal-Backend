#pragma once

// MSAA resolve / unresolve helpers (port of VK/VKResolveHelper.h).
//
// RSX "resolve" is not an averaging resolve: the resolved image is the sample-expanded image (width * samples_x,
// height * samples_y) where every sample becomes one pixel. All passes are fragment passes:
//  - color:         texelFetch(sampler2DMS) -> color output (resolve), per-sample shading from the expanded image
//                   (unresolve; Metal cannot write multisampled textures from compute)
//  - depth:         [[depth]] output (gl_FragDepth)
//  - depth+stencil: [[depth]] + [[stencil]] output (gl_FragStencilRefARB, GL_ARB_shader_stencil_export, supported by
//                   every Metal 4 GPU, so VK's 8-pass stencil write-mask fallback is not needed)

#include "MTLCompute.h"
#include "MTLOverlays.h"

#include "mtlutils/image.h"

namespace mtl
{
	struct depth_resolve_base : public overlay_pass
	{
		u8 samples_x = 1;
		u8 samples_y = 1;
		s32 static_parameters[4]{};
		s32 static_parameters_width = 2;

		static constexpr usz fragment_push_constants_size = sizeof(static_parameters);

		depth_resolve_base()
		{
			renderpass_config.set_depth_mask(true);
			renderpass_config.enable_depth_test(MTL::CompareFunctionAlways);

			// Depth-stencil buffers are almost never filterable, and we do not need it here (1:1 mapping)
			m_sampler_filter = MTL::SamplerMinMagFilterNearest;

			// Do not use UBOs
			m_num_uniform_buffers = 0;
		}

		void build(bool resolve_depth, bool resolve_stencil, bool unresolve);

		std::vector<glsl::program_input> get_fragment_inputs() override
		{
			auto result = overlay_pass::get_fragment_inputs();
			result.push_back(glsl::program_input::make(
				::glsl::glsl_fragment_program,
				"push_constants",
				glsl::input_type_push_constant,
				glsl::binding_set_index_fragment,
				umax,
				glsl::push_constant_ref{ .offset = 0, .size = static_cast<u32>(fragment_push_constants_size) }
			));
			return result;
		}

		void update_uniforms(mtl::command_list& /*cmd*/, glsl::program* program) override
		{
			const u32 size_to_push = static_parameters_width * sizeof(decltype(static_parameters[0]));
			ensure(size_to_push <= fragment_push_constants_size);
			program->push_constants(glsl::binding_set_index_fragment, 0, size_to_push, static_parameters);
		}

		void update_sample_configuration(mtl::image* msaa_image)
		{
			switch (msaa_image->samples())
			{
			case 1:
				fmt::throw_exception("MSAA input not multisampled!");
				break;
			case 2:
				samples_x = 2;
				samples_y = 1;
				break;
			case 4:
				samples_x = samples_y = 2;
				break;
			default:
				fmt::throw_exception("Unsupported sample count %d", msaa_image->samples());
			}

			static_parameters[0] = samples_x;
			static_parameters[1] = samples_y;
		}
	};

	// Replaces VK cs_resolve_task: renders the sample-expanded color image from the multisampled one
	struct color_resolve_pass : depth_resolve_base
	{
		color_resolve_pass();

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	// Replaces VK cs_unresolve_task: per-sample fragment pass into the multisampled color image
	struct color_unresolve_pass : depth_resolve_base
	{
		color_unresolve_pass();

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	struct depthonly_resolve : depth_resolve_base
	{
		depthonly_resolve()
		{
			m_overwrites_depth = true;
			build(true, false, false);
		}

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
		{
			update_sample_configuration(msaa_image);
			auto src_view = msaa_image->get_identity_view(aspect_depth);

			// Depth only: a stencil plane of the target (if any) is loaded and preserved
			overlay_pass::run(
				cmd,
				{ 0, 0, resolve_image->width(), resolve_image->height() },
				resolve_image, src_view);
		}
	};

	struct depthonly_unresolve : depth_resolve_base
	{
		depthonly_unresolve()
		{
			m_overwrites_depth = true;
			build(true, false, true);
		}

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
		{
			// Per-sample shading is implied by gl_SampleID ([[sample_id]]) in the fragment shader
			update_sample_configuration(msaa_image);

			auto src_view = resolve_image->get_identity_view(aspect_depth);

			overlay_pass::run(
				cmd,
				{ 0, 0, msaa_image->width(), msaa_image->height() },
				msaa_image, src_view);
		}
	};

	struct depthstencil_resolve_EXT : depth_resolve_base
	{
		depthstencil_resolve_EXT()
		{
			renderpass_config.enable_stencil_test(
				MTL::StencilOperationReplace, MTL::StencilOperationReplace, MTL::StencilOperationReplace,  // Always replace
				MTL::CompareFunctionAlways,                                                                 // Always pass
				0xFF,                                                                                       // Full write-through
				0);                                                                                         // Unused (exported)

			renderpass_config.set_stencil_mask(0xFF);
			m_num_usable_samplers = 2;
			m_overwrites_depth = true;
			m_overwrites_stencil = true;

			build(true, true, false);
		}

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
		{
			update_sample_configuration(msaa_image);
			auto depth_view = msaa_image->get_identity_view(aspect_depth);
			auto stencil_view = msaa_image->get_identity_view(aspect_stencil);

			overlay_pass::run(
				cmd,
				{ 0, 0, resolve_image->width(), resolve_image->height() },
				resolve_image, { depth_view, stencil_view });
		}
	};

	struct depthstencil_unresolve_EXT : depth_resolve_base
	{
		depthstencil_unresolve_EXT()
		{
			renderpass_config.enable_stencil_test(
				MTL::StencilOperationReplace, MTL::StencilOperationReplace, MTL::StencilOperationReplace,  // Always replace
				MTL::CompareFunctionAlways,                                                                 // Always pass
				0xFF,                                                                                       // Full write-through
				0);                                                                                         // Unused (exported)

			renderpass_config.set_stencil_mask(0xFF);
			m_num_usable_samplers = 2;
			m_overwrites_depth = true;
			m_overwrites_stencil = true;

			build(true, true, true);
		}

		void run(mtl::command_list& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
		{
			update_sample_configuration(msaa_image);

			auto depth_view = resolve_image->get_identity_view(aspect_depth);
			auto stencil_view = resolve_image->get_identity_view(aspect_stencil);

			overlay_pass::run(
				cmd,
				{ 0, 0, msaa_image->width(), msaa_image->height() },
				msaa_image, { depth_view, stencil_view });
		}
	};

	// dst/src follow the VK convention: resolve writes the single-sampled `dst` from the multisampled `src`,
	// unresolve writes the multisampled `dst` from the single-sampled `src`.
	void resolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);
	void unresolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);

	// Per-frame hook (VK parity)
	void reset_resolve_resources();
	// Destroys the resolve passes. Call on shutdown once the GPU is idle.
	void clear_resolve_helpers();
}
