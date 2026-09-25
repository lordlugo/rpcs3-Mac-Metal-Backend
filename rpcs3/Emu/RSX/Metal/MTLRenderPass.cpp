#include "stdafx.h"
#include "MTLRenderPass.h"
#include "mtlutils/commands.h"
#include "mtlutils/image.h"

namespace mtl
{
	bool framebuffer_info::has_stencil() const
	{
		return depth_stencil && is_stencil_format(depth_stencil->format());
	}

	std::array<const MTL::Texture*, 5> framebuffer_info::get_textures() const
	{
		std::array<const MTL::Texture*, 5> result{};
		for (u32 i = 0; i < color_count; ++i)
		{
			result[i] = color[i] ? color[i]->value : nullptr;
		}

		result[4] = depth_stencil ? depth_stencil->value : nullptr;
		return result;
	}

	u64 get_renderpass_key(MTL::PixelFormat color_format, MTL::PixelFormat depth_format, u8 color_attachment_count, u8 sample_count)
	{
		// [ 0..15: depth format | 16..19: sample count | 20..23: colour count | 24..: colour formats (10 bits each) ]
		u64 key = static_cast<u64>(depth_format) & 0xFFFF;
		key |= static_cast<u64>(sample_count & 0xF) << 16;
		key |= static_cast<u64>(color_attachment_count & 0xF) << 20;

		for (u8 i = 0; i < color_attachment_count && i < 4; ++i)
		{
			key |= (static_cast<u64>(color_format) & 0x3FF) << (24 + (i * 10));
		}

		return key;
	}

	u64 get_renderpass_key(const std::vector<mtl::image*>& images)
	{
		u64 key = 0;
		u8 color_count = 0;
		u8 samples = 1;

		for (const auto& surface : images)
		{
			if (!surface)
			{
				continue;
			}

			samples = std::max<u8>(samples, surface->samples());

			if (surface->aspect() & aspect_color)
			{
				if (color_count < 4)
				{
					key |= (static_cast<u64>(surface->format()) & 0x3FF) << (24 + (color_count * 10));
				}

				color_count++;
				continue;
			}

			key |= static_cast<u64>(surface->format()) & 0xFFFF;
		}

		key |= static_cast<u64>(samples & 0xF) << 16;
		key |= static_cast<u64>(color_count & 0xF) << 20;
		return key;
	}

	static void set_attachment_texture(MTL::RenderPassAttachmentDescriptor* attachment, const MTL::Texture* texture)
	{
		attachment->setTexture(texture);
		attachment->setLevel(0);
		attachment->setSlice(0);
		attachment->setDepthPlane(0);
		attachment->setLoadAction(MTL::LoadActionLoad);
		attachment->setStoreAction(MTL::StoreActionStore);
	}

	MTL4::RenderPassDescriptor* create_render_pass_descriptor(const framebuffer_info& fb, const MTL::Buffer* visibility_result_buffer)
	{
		autorelease_scope pool;

		auto desc = MTL4::RenderPassDescriptor::alloc()->init();
		ensure(desc, "Metal: failed to create render pass descriptor");

		for (u32 index = 0; index < fb.color_count; ++index)
		{
			const auto surface = ensure(fb.color[index]);
			set_attachment_texture(desc->colorAttachments()->object(index), surface->value);
		}

		if (fb.depth_stencil)
		{
			// Depth32Float_Stencil8 (the only depth-stencil format used) must be bound to both attachments
			const auto aspect = fb.depth_stencil->aspect();
			if (aspect & aspect_depth)
			{
				set_attachment_texture(desc->depthAttachment(), fb.depth_stencil->value);
			}

			if (aspect & aspect_stencil)
			{
				set_attachment_texture(desc->stencilAttachment(), fb.depth_stencil->value);
			}
		}

		desc->setRenderTargetWidth(fb.width);
		desc->setRenderTargetHeight(fb.height);
		desc->setDefaultRasterSampleCount(std::max<u8>(1, fb.samples));

		set_visibility_result_buffer(desc, visibility_result_buffer);
		return desc;
	}

	void set_visibility_result_buffer(MTL4::RenderPassDescriptor* desc, const MTL::Buffer* visibility_result_buffer)
	{
		desc->setVisibilityResultBuffer(visibility_result_buffer);

		// Queries may span several passes (pass splits for feedback loops, clears, copies); accumulate instead of reset
		desc->setVisibilityResultType(visibility_result_buffer ? MTL::VisibilityResultTypeAccumulate : MTL::VisibilityResultTypeReset);
	}

	void apply_clear_load_ops(MTL4::RenderPassDescriptor* desc, const framebuffer_info& fb, const attachment_clear_info& clear)
	{
		for (u32 index = 0; index < fb.color_count; ++index)
		{
			if (clear.color_mask & (1u << index))
			{
				auto attachment = desc->colorAttachments()->object(index);
				attachment->setLoadAction(MTL::LoadActionClear);
				attachment->setClearColor(clear.color);
			}
		}

		if (!fb.depth_stencil)
		{
			return;
		}

		const auto aspect = fb.depth_stencil->aspect();
		if (clear.clear_depth && (aspect & aspect_depth))
		{
			auto attachment = desc->depthAttachment();
			attachment->setLoadAction(MTL::LoadActionClear);
			attachment->setClearDepth(clear.depth);
		}

		if (clear.clear_stencil && (aspect & aspect_stencil))
		{
			auto attachment = desc->stencilAttachment();
			attachment->setLoadAction(MTL::LoadActionClear);
			attachment->setClearStencil(clear.stencil);
		}
	}

	void restore_load_ops(MTL4::RenderPassDescriptor* desc, const framebuffer_info& fb)
	{
		for (u32 index = 0; index < fb.color_count; ++index)
		{
			desc->colorAttachments()->object(index)->setLoadAction(MTL::LoadActionLoad);
		}

		if (!fb.depth_stencil)
		{
			return;
		}

		const auto aspect = fb.depth_stencil->aspect();
		if (aspect & aspect_depth)
		{
			desc->depthAttachment()->setLoadAction(MTL::LoadActionLoad);
		}

		if (aspect & aspect_stencil)
		{
			desc->stencilAttachment()->setLoadAction(MTL::LoadActionLoad);
		}
	}

	MTL4::RenderCommandEncoder* begin_single_target_pass(mtl::command_list& cmd, MTL::Texture* target, u32 width, u32 height, const MTL::ClearColor* clear_color)
	{
		auto desc = ref(MTL4::RenderPassDescriptor::alloc()->init());
		auto attachment = desc->colorAttachments()->object(0);
		attachment->setTexture(target);
		attachment->setStoreAction(MTL::StoreActionStore);

		if (clear_color)
		{
			attachment->setLoadAction(MTL::LoadActionClear);
			attachment->setClearColor(*clear_color);
		}
		else
		{
			attachment->setLoadAction(MTL::LoadActionLoad);
		}

		desc->setRenderTargetWidth(width);
		desc->setRenderTargetHeight(height);
		desc->setDefaultRasterSampleCount(1);

		return cmd.begin_render_pass(desc.get());
	}

	void clear_color_texture(mtl::command_list& cmd, MTL::Texture* target, u32 width, u32 height, const MTL::ClearColor& color)
	{
		begin_single_target_pass(cmd, target, width, height, &color);
		cmd.end_render_pass();
	}

	// -----------------------------------------------------------------------------------------------------------------

	MTL4::RenderCommandEncoder* render_pass_tracker::begin(mtl::command_list& cmd, const MTL4::RenderPassDescriptor* desc)
	{
		auto encoder = cmd.begin_render_pass(desc);

		m_cmd = &cmd;
		m_encoder = ref<MTL4::RenderCommandEncoder>::retain(encoder);
		m_pass_id++;
		return encoder;
	}

	void render_pass_tracker::end(mtl::command_list& cmd)
	{
		if (is_open(cmd))
		{
			cmd.end_render_pass();
		}

		reset();
	}

	void render_pass_tracker::reset()
	{
		m_encoder.reset();
		m_cmd = nullptr;
	}

	bool render_pass_tracker::is_open(const mtl::command_list& cmd) const
	{
		return m_encoder && m_cmd == &cmd && cmd.render_encoder() == m_encoder.get();
	}

	MTL4::RenderCommandEncoder* render_pass_tracker::encoder(const mtl::command_list& cmd) const
	{
		return is_open(cmd) ? m_encoder.get() : nullptr;
	}
}
