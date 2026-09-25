#include "stdafx.h"

#include "metalfx_pass.h"
#include "nearest_pass.hpp"
#include "bilinear_pass.hpp"
#include "rcas_pass.h"

#include "../MTLCompute.h"
#include "../MTLHelpers.h"
#include "../MTLProgramPipeline.h"
#include "../MTLResourceManager.h"
#include "../mtlutils/data_heap.h"
#include "../mtlutils/device.h"

#include "Emu/system_config.h"
#include "Utilities/Thread.h"

#include <algorithm>
#include <chrono>

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

		void dispose_image(std::unique_ptr<mtl::viewable_image>& image)
		{
			if (image && image->value)
			{
				mtl::get_resource_manager()->dispose(image);
			}
			else
			{
				image.reset();
			}
		}
	}

	metalfx_upscale_pass::~metalfx_upscale_pass()
	{
		// Join the builder (bounded by one scaler build) and take ownership of its result
		poll_builder(true);

		for (auto& entry : m_scalers)
		{
			dispose_ns_object(entry.scaler);
		}

		m_scalers.clear();

		dispose_images();
		dispose_ns_object(m_fence);
	}

	bool metalfx_upscale_pass::is_supported()
	{
		if (m_support == support_state::unknown)
		{
			autorelease_scope pool;

			if (MTLFX::SpatialScalerDescriptor::supportsMetal4FX(g_render_device->handle()))
			{
				m_support = support_state::supported;
				rsx_log.notice("MetalFX: spatial upscaling is available.");
			}
			else
			{
				m_support = support_state::unsupported;
				rsx_log.error("MetalFX spatial upscaling is not supported on this system. Will fall back to bilinear upscaling.");
			}
		}

		return m_support == support_state::supported;
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

	MTL4FX::SpatialScaler* metalfx_upscale_pass::create_scaler(const scaler_config& config)
	{
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

		// +1. MTL4Compiler is thread-safe (shared with the pipeline compiler workers).
		return desc->newSpatialScaler(g_render_device->handle(), g_render_device->compiler());
	}

	metalfx_upscale_pass::scaler_entry* metalfx_upscale_pass::find_scaler(const scaler_config& config)
	{
		for (auto& entry : m_scalers)
		{
			if (entry.config == config)
			{
				entry.last_used = ++m_use_counter;
				return &entry;
			}
		}

		return nullptr;
	}

	void metalfx_upscale_pass::add_scaler(const scaler_config& config, MTL4FX::SpatialScaler* scaler)
	{
		if (m_scalers.size() >= max_cached_scalers)
		{
			// Evict the least recently used scaler. In-flight work may still use it.
			auto lru = std::min_element(m_scalers.begin(), m_scalers.end(), [](const scaler_entry& a, const scaler_entry& b)
			{
				return a.last_used < b.last_used;
			});

			dispose_ns_object(lru->scaler);
			m_scalers.erase(lru);
		}

		scaler_entry entry{};
		entry.config = config;
		entry.scaler = scaler;
		entry.color_usage = scaler->colorTextureUsage();
		entry.output_usage = scaler->outputTextureUsage();
		entry.last_used = ++m_use_counter;

		// The scaler waits on the fence before reading its input and updates it once its output is written
		entry.scaler->setFence(m_fence);

		m_scalers.push_back(entry);
	}

	void metalfx_upscale_pass::poll_builder(bool wait)
	{
		if (!m_pending_job)
		{
			return;
		}

		if (!wait && !m_pending_job->done.load())
		{
			return;
		}

		// Finished (or joining on destruction): the thread object only has to be joined
		m_builder.reset();
		ensure(m_pending_job->done.load());

		auto job = std::move(m_pending_job);
		const auto& config = job->config;

		if (!job->scaler)
		{
			rsx_log.warning("MetalFX: failed to create a spatial scaler (%ux%u fmt=%d -> %ux%u fmt=%d). Will fall back to bilinear upscaling for this configuration.",
				config.input_width, config.input_height, static_cast<int>(config.input_format),
				config.output_width, config.output_height, static_cast<int>(config.output_format));

			if (m_failed_configs.size() >= max_failed_configs)
			{
				m_failed_configs.erase(m_failed_configs.begin());
			}

			m_failed_configs.push_back(config);
			return;
		}

		rsx_log.notice("MetalFX: spatial scaler ready (%ux%u -> %ux%u, built in %u ms)",
			config.input_width, config.input_height, config.output_width, config.output_height, job->duration_us / 1000);

		if (wait)
		{
			// Destruction: never encoded, release directly
			job->scaler->release();
			job->scaler = nullptr;
			return;
		}

		add_scaler(config, std::exchange(job->scaler, nullptr));
	}

	void metalfx_upscale_pass::request_scaler(const scaler_config& config)
	{
		if (std::find(m_failed_configs.begin(), m_failed_configs.end(), config) != m_failed_configs.end())
		{
			return;
		}

		if (m_requested_config == config)
		{
			m_requested_count++;
		}
		else
		{
			m_requested_config = config;
			m_requested_count = 1;
		}

		if (m_pending_job)
		{
			// One build at a time. A build for another (e.g. transient) size completes first and is cached.
			return;
		}

		if (m_requested_count < build_request_threshold)
		{
			// Wait for the configuration to settle (live window resizes change the output size every frame)
			return;
		}

		if (!m_fence)
		{
			m_fence = g_render_device->handle()->newFence();
			ensure(m_fence, "Metal: failed to create MetalFX fence");
		}

		auto job = std::make_shared<build_job>();
		job->config = config;
		m_pending_job = job;

		m_builder = std::make_unique<named_thread<std::function<void()>>>("MetalFX Builder", [job]()
		{
			const auto start = std::chrono::steady_clock::now();

			job->scaler = create_scaler(job->config);
			job->duration_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());

			job->done.store(true);
		});
	}

	void metalfx_upscale_pass::prepare_image(std::unique_ptr<mtl::viewable_image>& image, MTL::PixelFormat format, u32 width, u32 height,
		MTL::TextureUsage usage, const char* debug_name)
	{
		if (image &&
			image->format() == format &&
			image->width() == width &&
			image->height() == height &&
			(image->info.usage & usage) == usage)
		{
			return;
		}

		// In-flight work may still reference the old image
		dispose_image(image);

		image_create_info info{};
		info.type = MTL::TextureType2D;
		info.format = format;
		info.width = width;
		info.height = height;
		info.usage = usage;
		info.storage = memory_location::device_local;
		info.format_class = RSX_FORMAT_CLASS_COLOR;

		image = std::make_unique<mtl::viewable_image>(*g_render_device, info);
		image->set_debug_name(debug_name);
	}

	void metalfx_upscale_pass::dispose_images()
	{
		dispose_image(m_output_left);
		dispose_image(m_output_right);
		dispose_image(m_intermediate);
		dispose_image(m_input_copy);
	}

	void metalfx_upscale_pass::signal_fence(mtl::command_list& cmd)
	{
		// MetalFX encodes its own passes. Update the fence it waits on from a command that is itself ordered after all
		// previously recorded work (the compute encoder opens with a full queue barrier, compute() orders it against
		// earlier commands of the same encoder), then close the encoder so the scaler can encode into the buffer.
		auto& heap = get_scratch_heap();
		const auto offset = heap.alloc<16>(16);

		auto encoder = cmd.compute();
		encoder->fillBuffer(heap.value(), NS::Range::Make(offset, 4), 0); // 16-byte aligned offset, 4-byte length
		encoder->updateFence(m_fence, MTL::StageBlit | MTL::StageDispatch);

		cmd.end_encoder();
	}

	void metalfx_upscale_pass::wait_fence(mtl::command_list& cmd)
	{
		// The scaler updates m_fence once its output is written. Open a fresh encoder and make its commands wait on the
		// fence. The opening queue barrier most likely covers the scaler's passes already; the fence makes the order
		// explicit. The (blocked) anchor command keeps the wait meaningful when nothing else is recorded here: every
		// later encoder starts with a queue barrier on all prior work, i.e. after it. RCAS dispatches in this encoder
		// (after the intra-encoder barrier compute() inserts).
		auto& heap = get_scratch_heap();
		const auto offset = heap.alloc<16>(16);

		ensure(cmd.active_encoder() == mtl::command_list::encoder_type::none);
		auto encoder = cmd.compute();
		encoder->waitForFence(m_fence, MTL::StageBlit | MTL::StageDispatch);
		encoder->fillBuffer(heap.value(), NS::Range::Make(offset, 4), 0);
	}

	mtl::viewable_image* metalfx_upscale_pass::upscale(
		mtl::command_list& cmd,
		mtl::viewable_image* src,
		const areai& src_area,
		const size2u& input_size,
		const size2u& output_size,
		rsx::flags32_t mode)
	{
		if (!is_supported())
		{
			return nullptr;
		}

		// Collect a finished scaler build
		poll_builder(false);

		if (src->samples() > 1 || src->type() != MTL::TextureType2D)
		{
			if (!m_logged_input)
			{
				rsx_log.warning("MetalFX: unsupported input (samples=%u, type=%d). Falling back to bilinear upscaling.",
					src->samples(), static_cast<int>(src->type()));
				m_logged_input = true;
			}

			return nullptr;
		}

		// Content region, top-left origin
		const u32 content_x = static_cast<u32>(std::max(std::min(src_area.x1, src_area.x2), 0));
		const u32 content_y = static_cast<u32>(std::max(std::min(src_area.y1, src_area.y2), 0));

		if (content_x + input_size.width > src->width() || content_y + input_size.height > src->height())
		{
			// Region outside the texture (the sampled fallback clamps)
			if (!m_logged_input)
			{
				rsx_log.warning("MetalFX: source region %ux%u at (%u, %u) exceeds the %ux%u input. Falling back to bilinear upscaling.",
					input_size.width, input_size.height, content_x, content_y, src->width(), src->height());
				m_logged_input = true;
			}

			return nullptr;
		}

		// MetalFX reads the content from the texture origin, and the input texture may not be larger than the output.
		// get_present_source() can return a surface larger than the displayed region: copy the content out then.
		const bool copy_input =
			content_x != 0 || content_y != 0 ||
			src->width() > output_size.width || src->height() > output_size.height;

		const scaler_config config
		{
			.input_format = src->format(),
			.output_format = get_output_format(src->format()),
			.input_width = copy_input ? input_size.width : src->width(),
			.input_height = copy_input ? input_size.height : src->height(),
			.output_width = output_size.width,
			.output_height = output_size.height
		};

		const auto entry = find_scaler(config);
		if (!entry)
		{
			// Not built yet (or failed): bilinear for this frame, the scaler is built on a worker thread
			request_scaler(config);

			if (!m_logged_waiting)
			{
				rsx_log.notice("MetalFX: building a spatial scaler for %ux%u -> %ux%u. Bilinear upscaling is used until it is ready.",
					config.input_width, config.input_height, config.output_width, config.output_height);
				m_logged_waiting = true;
			}

			return nullptr;
		}

		if (!copy_input && (src->value->usage() & entry->color_usage) != entry->color_usage)
		{
			if (!m_logged_usage)
			{
				rsx_log.error("MetalFX: input texture usage 0x%x lacks required usage 0x%x. Falling back to bilinear upscaling.",
					static_cast<u64>(src->value->usage()), static_cast<u64>(entry->color_usage));
				m_logged_usage = true;
			}

			return nullptr;
		}

		// RCAS sharpening (FSR1's second pass): 0 = off
		const u32 sharpening_intensity = g_cfg.video.rcas_sharpening_intensity;
		FidelityFX::rcas_pass* rcas = nullptr;

		if (sharpening_intensity > 0 && !m_rcas_unavailable)
		{
			rcas = mtl::get_compute_task<FidelityFX::rcas_pass>();

			// Normally already built by prepare(); failures only disable sharpening
			if (!rcas->prepare())
			{
				m_rcas_unavailable = true;
				rcas = nullptr;
			}
		}

		// Output images. The view output can also be the RCAS target, so both modes share it.
		auto& output = (mode & UPSCALE_LEFT_VIEW) ? m_output_left : m_output_right;
		prepare_image(output, config.output_format, output_size.width, output_size.height,
			entry->output_usage | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite,
			(mode & UPSCALE_LEFT_VIEW) ? "MetalFX output (left)" : "MetalFX output (right)");

		mtl::viewable_image* scaler_target = output.get();

		if (rcas)
		{
			prepare_image(m_intermediate, config.output_format, output_size.width, output_size.height,
				entry->output_usage | MTL::TextureUsageShaderRead, "MetalFX intermediate");

			scaler_target = m_intermediate.get();
		}

		mtl::viewable_image* input = src;

		if (copy_input)
		{
			prepare_image(m_input_copy, src->format(), input_size.width, input_size.height,
				entry->color_usage | MTL::TextureUsageShaderRead, "MetalFX input");

			const areai content = { static_cast<s32>(content_x), static_cast<s32>(content_y),
				static_cast<s32>(content_x + input_size.width), static_cast<s32>(content_y + input_size.height) };
			const areai dst_rect = { 0, 0, static_cast<s32>(input_size.width), static_cast<s32>(input_size.height) };

			mtl::copy_image(cmd, src, m_input_copy.get(), content, dst_rect);
			input = m_input_copy.get();
		}

		// 1. Everything recorded so far (the producer of the input, readers of the previous output) -> fence
		signal_fence(cmd);

		// 2. Scaler passes. Self-contained: MetalFX may create autoreleased objects while encoding
		{
			autorelease_scope pool;

			const auto scaler = entry->scaler;
			scaler->setInputContentWidth(input_size.width);
			scaler->setInputContentHeight(input_size.height);
			scaler->setColorTexture(input->value);
			scaler->setOutputTexture(scaler_target->value);
			scaler->setFence(m_fence);
			scaler->encodeToCommandBuffer(cmd.handle());

			// The command buffer holds what it needs; do not keep the textures alive through the scaler
			scaler->setColorTexture(nullptr);
			scaler->setOutputTexture(nullptr);
		}

		// 3. Scaler output -> everything recorded afterwards
		wait_fence(cmd);

		// 4. Sharpening into the view output (same compute encoder, ordered after the fence wait)
		if (rcas)
		{
			rcas->run(cmd, m_intermediate.get(), output.get(), sharpening_intensity);
		}

		return output.get();
	}

	void metalfx_upscale_pass::prepare()
	{
		poll_builder(false);

		if (g_cfg.video.rcas_sharpening_intensity > 0 && !m_rcas_unavailable && is_supported())
		{
			// One-time synchronous kernel build, done here rather than in the middle of a present
			if (!mtl::get_compute_task<FidelityFX::rcas_pass>()->prepare())
			{
				m_rcas_unavailable = true;
			}
		}
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

		// MetalFX (+ RCAS) runs for every upscale, however small: it is what the user selected, and even at ~1.1x (250% of
		// 720p on a 3456x1944 display) its edge-directed reconstruction and the sharpening look clearly better than bilinear.
		if (input_size.width < output_size.width && input_size.height < output_size.height)
		{
			// Cannot upscale both LEFT and RIGHT images at the same time.
			// Default maps to LEFT for simplicity
			ensure((mode & (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW)) != (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW));

			if (auto upscaled = upscale(cmd, src, src_area, input_size, output_size, mode))
			{
				// Swap input for the MetalFX target
				src_image = upscaled;

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
		else
		{
			// Not upscaling: still collect a finished build
			poll_builder(false);
		}

		if (mode & UPSCALE_AND_COMMIT)
		{
			ensure(present_surface);

			upscale_blit(cmd, src_image, present_surface, output_src_area, dst_area, true, !!(mode & UPSCALE_CLEAR_TARGET));
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
			// FidelityFX FSR1 is replaced by MetalFX spatial upscaling (+ RCAS sharpening) on Metal
			return std::make_unique<metalfx_upscale_pass>();
		case output_scaling_mode::bilinear:
		default:
			return std::make_unique<bilinear_upscale_pass>();
		}
	}
}
