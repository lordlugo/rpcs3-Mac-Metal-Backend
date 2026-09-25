#include "stdafx.h"

#include "metalfx_pass.h"
#include "nearest_pass.hpp"
#include "bilinear_pass.hpp"

#include "../MTLProgramPipeline.h"
#include "../MTLResourceManager.h"
#include "../mtlutils/data_heap.h"
#include "../mtlutils/device.h"

namespace mtl
{
	namespace
	{
		// Deferred release of a metal-cpp object that in-flight GPU work may still reference
		struct ns_object_holder
		{
			NS::Object* object = nullptr;

			explicit ns_object_holder(NS::Object* obj) : object(obj) {}
			~ns_object_holder()
			{
				if (object)
				{
					object->release();
				}
			}

			ns_object_holder(const ns_object_holder&) = delete;
			ns_object_holder& operator=(const ns_object_holder&) = delete;
		};

		template <typename T>
		void dispose_ns_object(T*& object)
		{
			if (!object)
			{
				return;
			}

			auto holder = std::make_unique<ns_object_holder>(object);
			mtl::get_gc()->dispose(holder);
			object = nullptr;
		}
	}

	metalfx_upscale_pass::~metalfx_upscale_pass()
	{
		dispose_images();
		dispose_scaler();
		dispose_ns_object(m_fence);
	}

	MTL::PixelFormat metalfx_upscale_pass::get_output_format(MTL::PixelFormat input_format)
	{
		switch (input_format)
		{
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatRGBA8Unorm:
		case MTL::PixelFormatRGB10A2Unorm:
		case MTL::PixelFormatRGBA16Float:
			return input_format;
		default:
			return MTL::PixelFormatBGRA8Unorm;
		}
	}

	void metalfx_upscale_pass::dispose_images()
	{
		auto safe_delete = [](auto& data)
		{
			if (data && data->value)
			{
				mtl::get_resource_manager()->dispose(data);
			}
			else if (data)
			{
				data.reset();
			}
		};

		safe_delete(m_output_left);
		safe_delete(m_output_right);
	}

	void metalfx_upscale_pass::dispose_scaler()
	{
		dispose_ns_object(m_scaler);
	}

	bool metalfx_upscale_pass::initialize(mtl::viewable_image* src, u32 output_w, u32 output_h, rsx::flags32_t mode)
	{
		const scaler_config config
		{
			.input_format = src->format(),
			.output_format = get_output_format(src->format()),
			.input_width = src->width(),
			.input_height = src->height(),
			.output_width = output_w,
			.output_height = output_h
		};

		if (m_config != config)
		{
			// New scaling configuration: drop the old scaler and outputs
			dispose_images();
			dispose_scaler();
			m_config = config;

			const auto device = g_render_device->handle();

			if (!MTLFX::SpatialScalerDescriptor::supportsMetal4FX(device))
			{
				m_unsupported = true;
				rsx_log.error("MetalFX spatial upscaling is not supported on this system. Will fall back to bilinear upscaling.");
				return false;
			}

			autorelease_scope pool;

			auto desc = ref(MTLFX::SpatialScalerDescriptor::alloc()->init());
			desc->setColorTextureFormat(config.input_format);
			desc->setOutputTextureFormat(config.output_format);
			desc->setInputWidth(config.input_width);
			desc->setInputHeight(config.input_height);
			desc->setOutputWidth(config.output_width);
			desc->setOutputHeight(config.output_height);

			// RSX output is display-referred (gamma encoded)
			desc->setColorProcessingMode(MTLFX::SpatialScalerColorProcessingModePerceptual);

			m_scaler = desc->newSpatialScaler(device, g_render_device->compiler());
			if (!m_scaler)
			{
				rsx_log.warning("MetalFX: failed to create a spatial scaler (%ux%u fmt=%d -> %ux%u fmt=%d). Will fall back to bilinear upscaling.",
					config.input_width, config.input_height, static_cast<int>(config.input_format),
					config.output_width, config.output_height, static_cast<int>(config.output_format));
				return false;
			}

			if (!m_fence)
			{
				m_fence = device->newFence();
				ensure(m_fence, "Metal: failed to create MetalFX fence");
			}
		}

		if (!m_scaler)
		{
			// Creation failed for this configuration; do not retry every frame
			return false;
		}

		auto& output = (mode & UPSCALE_LEFT_VIEW) ? m_output_left : m_output_right;
		if (!output)
		{
			image_create_info info{};
			info.type = MTL::TextureType2D;
			info.format = config.output_format;
			info.width = config.output_width;
			info.height = config.output_height;
			info.usage = m_scaler->outputTextureUsage() | MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
			info.storage = memory_location::device_local;
			info.format_class = RSX_FORMAT_CLASS_COLOR;

			output = std::make_unique<mtl::viewable_image>(*g_render_device, info);
			output->set_debug_name((mode & UPSCALE_LEFT_VIEW) ? "MetalFX output (left)" : "MetalFX output (right)");
		}

		return true;
	}

	void metalfx_upscale_pass::signal_fence(mtl::command_list& cmd)
	{
		// MetalFX encodes its own passes. Update the fence it waits on from a command that is itself ordered after all
		// previously recorded work (the compute encoder opens with a full queue barrier, compute() orders it against
		// earlier commands of the same encoder), then close the encoder so the scaler can encode into the buffer.
		auto& heap = get_scratch_heap();
		const auto offset = heap.alloc<16>(16);

		auto encoder = cmd.compute();
		encoder->fillBuffer(heap.value(), NS::Range::Make(offset, 4), 0);
		encoder->updateFence(m_fence, MTL::StageBlit | MTL::StageDispatch);

		cmd.end_encoder();
	}

	mtl::viewable_image* metalfx_upscale_pass::scale_output(
		mtl::command_list& cmd,
		mtl::viewable_image* src,
		MTL::Texture* present_surface,
		const areai& src_area,
		const areai& dst_area,
		rsx::flags32_t mode)
	{
		size2u input_size, output_size;
		input_size.width = std::abs(src_area.x2 - src_area.x1);
		input_size.height = std::abs(src_area.y2 - src_area.y1);
		output_size.width = std::abs(dst_area.x2 - dst_area.x1);
		output_size.height = std::abs(dst_area.y2 - dst_area.y1);

		auto src_image = src;
		areai output_src_area = src_area;

		if (!m_unsupported && input_size.width < output_size.width && input_size.height < output_size.height)
		{
			// Cannot upscale both LEFT and RIGHT images at the same time.
			// Default maps to LEFT for simplicity
			ensure((mode & (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW)) != (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW));

			if (initialize(src, output_size.width, output_size.height, mode))
			{
				auto& output = (mode & UPSCALE_LEFT_VIEW) ? m_output_left : m_output_right;
				const MTL::TextureUsage required_usage = m_scaler->colorTextureUsage();

				if ((src->value->usage() & required_usage) != required_usage)
				{
					if (!m_logged_fallback)
					{
						rsx_log.error("MetalFX: input texture usage 0x%x lacks required usage 0x%x. Falling back to bilinear upscaling.",
							static_cast<u64>(src->value->usage()), static_cast<u64>(required_usage));
						m_logged_fallback = true;
					}
				}
				else
				{
					// MetalFX creates autoreleased objects while encoding
					autorelease_scope pool;

					m_scaler->setInputContentWidth(input_size.width);
					m_scaler->setInputContentHeight(input_size.height);
					m_scaler->setColorTexture(src->value);
					m_scaler->setOutputTexture(output->value);

					signal_fence(cmd);

					m_scaler->setFence(m_fence);
					m_scaler->encodeToCommandBuffer(cmd.handle());

					// Swap input for the MetalFX target
					src_image = output.get();

					// Update output parameters to match expected output
					output_src_area = { 0, 0, static_cast<s32>(output_size.width), static_cast<s32>(output_size.height) };

					// Preserve mirroring/flipping
					if (src_area.x1 > src_area.x2)
					{
						std::swap(output_src_area.x1, output_src_area.x2);
					}

					if (src_area.y1 > src_area.y2)
					{
						std::swap(output_src_area.y1, output_src_area.y2);
					}
				}
			}
		}

		if (mode & UPSCALE_AND_COMMIT)
		{
			ensure(present_surface);

			upscale_blit(cmd, src_image, present_surface, output_src_area, dst_area, true);
			return nullptr;
		}

		return src_image;
	}

	std::unique_ptr<upscaler> create_upscaler(output_scaling_mode mode)
	{
		switch (mode)
		{
		case output_scaling_mode::nearest:
			return std::make_unique<nearest_upscale_pass>();
		case output_scaling_mode::fsr:
			// FidelityFX FSR1 is replaced by MetalFX spatial upscaling on Metal
			return std::make_unique<metalfx_upscale_pass>();
		case output_scaling_mode::bilinear:
		default:
			return std::make_unique<bilinear_upscale_pass>();
		}
	}
}
