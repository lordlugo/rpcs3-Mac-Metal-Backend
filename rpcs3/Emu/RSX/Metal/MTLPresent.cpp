#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLCommandStream.h"
#include "MTLFormats.h"
#include "MTLRenderPass.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/metal_layer.h"

#include "Emu/RSX/Overlays/overlay_manager.h"
#include "Emu/RSX/Overlays/overlay_debug_overlay.h"
#include "Emu/Cell/Modules/cellVideoOut.h"

#include "util/asm.hpp"
#include "util/video_provider.h"

#include <cstring>

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

namespace
{
	MTL::PixelFormat RSX_display_format_to_mtl_format(u8 format)
	{
		switch (format)
		{
		default:
			rsx_log.error("Unhandled video output format 0x%x", static_cast<s32>(format));
			[[fallthrough]];
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8R8G8B8:
			return MTL::PixelFormatBGRA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8B8G8R8:
			return MTL::PixelFormatRGBA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_R16G16B16X16_FLOAT:
			return MTL::PixelFormatRGBA16Float;
		}
	}
}

void MTLGSRender::configure_metal_layer()
{
	if (!m_metal_layer)
	{
		return;
	}

	// MetalFX (output_scaling_mode::fsr) may write the upscaled image straight into the drawable
	const bool framebuffer_only = (g_cfg.video.output_scaling != output_scaling_mode::fsr);
	if (framebuffer_only != m_layer_framebuffer_only || framebuffer_only != m_metal_layer->framebufferOnly())
	{
		m_metal_layer->setFramebufferOnly(framebuffer_only);
		m_layer_framebuffer_only = framebuffer_only;
	}

	m_metal_layer->setDisplaySyncEnabled(g_cfg.video.vsync != vsync_mode::off);
	m_vsync_mode = g_cfg.video.vsync;

	if (m_swapchain_dims.width && m_swapchain_dims.height)
	{
		m_metal_layer->setDrawableSize(CGSize{ static_cast<CGFloat>(m_swapchain_dims.width), static_cast<CGFloat>(m_swapchain_dims.height) });
	}
}

bool MTLGSRender::reinitialize_swapchain()
{
	m_swapchain_dims.width = m_frame->client_width();
	m_swapchain_dims.height = m_frame->client_height();

	// Reject requests to resize the surface if the window is minimized
	if (!m_metal_layer || m_swapchain_dims.width == 0 || m_swapchain_dims.height == 0)
	{
		swapchain_unavailable = true;
		return false;
	}

	// Unlike a Vulkan swapchain, a CAMetalLayer is never out of date: in-flight drawables stay valid and new ones are
	// created with the new drawableSize. No GPU sync is required.

	// Discard the current upscaling pipeline if any (output size changed)
	m_upscaler.reset();

	configure_metal_layer();
	mtl::sync_layer_scale(m_view, m_metal_layer);

	swapchain_unavailable = false;
	should_reinitialize_swapchain = false;
	return true;
}

void MTLGSRender::present(mtl::frame_context_t *ctx)
{
	ensure(ctx->drawable);

	// Partial CS flush: present() must follow the commit that signals the drawable (deferred submissions)
	ctx->swap_command_buffer->flush();
	ctx->swap_timeline_value = ctx->swap_command_buffer->get_fence().value;

	if (!swapchain_unavailable)
	{
		ctx->drawable->present();
	}

	// Presentation image released; the layer keeps the drawable alive until it has been displayed
	ctx->drawable->release();
	ctx->drawable = nullptr;
}

void MTLGSRender::advance_queued_frames()
{
	// Check all other frames for completion and clear resources
	check_present_status();

	// Run video memory balancer
	if (const auto load_severity = mtl::vmm_determine_memory_load_severity();
		load_severity >= rsx::problem_severity::moderate)
	{
		on_vram_exhausted(load_severity);
	}

	// m_rtts storage is double buffered and should be safe to tag on frame boundary
	m_rtts.trim(*m_current_command_buffer, mtl::vmm_determine_memory_load_severity());

	// Texture cache is also double buffered to prevent use-after-free
	m_texture_cache.on_frame_end();
	m_samplers_dirty.store(true);

	m_vertex_cache->purge();
	m_current_frame->tag_frame_end();

	m_queued_frames.push_back(m_current_frame);
	ensure(m_queued_frames.size() <= m_max_async_frames);

	m_current_queue_index = (m_current_queue_index + 1) % m_max_async_frames;
	m_current_frame = &m_frame_context_storage[m_current_queue_index];
	m_current_frame->flags |= frame_context_state::dirty;

	mtl::advance_frame_counter();
}

void MTLGSRender::queue_swap_request()
{
	ensure(!m_current_frame->swap_command_buffer);
	m_current_frame->swap_command_buffer = m_current_command_buffer;

	mtl::submit_info_t submit_info{};
	if (m_current_frame->drawable)
	{
		// The queue waits for the drawable before running the frame's work, then signals it for presentation
		submit_info.wait_drawable = m_current_frame->drawable;
		submit_info.signal_drawable = m_current_frame->drawable;
	}

	close_and_submit_command_buffer(submit_info);

	// Set up a present request for this frame as well
	if (m_current_frame->drawable)
	{
		present(m_current_frame);
	}

	// Grab next cb in line and make it usable
	m_current_command_buffer = m_primary_cb_list.next();
	m_current_command_buffer->reset();
	m_current_command_buffer->clear_flags();

	if (m_occlusion_query_active)
	{
		m_current_command_buffer->flags |= mtl::command_list::cb_load_occluson_task;
	}

	m_current_command_buffer->begin();

	// Set up new pointers for the next frame
	advance_queued_frames();
}

void MTLGSRender::frame_context_cleanup(mtl::frame_context_t *ctx)
{
	ensure(ctx->swap_command_buffer);

	// Perform hard swap here
	if (!ctx->swap_command_buffer->wait(FRAME_PRESENT_TIMEOUT))
	{
		// GPU hang, stop presenting
		swapchain_unavailable = true;
	}

	// Resource cleanup.
	{
		if (m_overlay_manager && m_overlay_manager->has_dirty())
		{
			auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
			m_overlay_manager->lock_shared();

			std::vector<u32> uids_to_dispose;
			uids_to_dispose.reserve(m_overlay_manager->get_dirty().size());

			for (const auto& view : m_overlay_manager->get_dirty())
			{
				ui_renderer->remove_temp_resources(view->uid);
				uids_to_dispose.push_back(view->uid);
			}

			m_overlay_manager->unlock_shared();
			m_overlay_manager->dispose(uids_to_dispose);
		}

		mtl::get_resource_manager()->trim();

		mtl::reset_global_resources();

		if (ctx->last_frame_sync_time > m_last_heap_sync_time)
		{
			m_last_heap_sync_time = ctx->last_frame_sync_time;

			// Heap cleanup; deallocates memory consumed by the frame if it is still held
			mtl::data_heap_manager::restore_snapshot(ctx->heap_snapshot);
		}
	}

	ctx->swap_command_buffer = nullptr;
	ctx->swap_timeline_value = 0;

	// Remove from queued list
	while (!m_queued_frames.empty())
	{
		auto frame = m_queued_frames.front();
		m_queued_frames.pop_front();

		if (frame == ctx)
		{
			break;
		}
	}

	mtl::advance_completed_frame_counter();
}

mtl::viewable_image* MTLGSRender::get_present_source(/* inout */ mtl::present_surface_info* info, const rsx::avconf& avconfig)
{
	mtl::viewable_image* image_to_flip = nullptr;

	// @FIXME: This entire function needs to be rewritten to go through the texture cache's "upload_texture" routine.
	// That method is not a 1:1 replacement due to handling of insets that is done differently here.

	// Check the surface store first
	const auto format_bpp = rsx::get_format_block_size_in_bytes(info->format);
	const auto overlap_info = m_rtts.get_merged_texture_memory_region(*m_current_command_buffer,
		info->address, info->width, info->height, info->pitch, format_bpp, rsx::surface_access::transfer_read);

	if (!overlap_info.empty())
	{
		const auto& section = overlap_info.back();
		auto surface = mtl::as_rtt(section.surface);
		bool viable = false;

		if (section.base_address >= info->address)
		{
			const auto surface_width = surface->template get_surface_width<rsx::surface_metrics::samples>();
			const auto surface_height = surface->template get_surface_height<rsx::surface_metrics::samples>();

			if (section.base_address == info->address)
			{
				// Check for fit or crop
				viable = (surface_width >= info->width && surface_height >= info->height);
			}
			else
			{
				// Check for borders and letterboxing
				const u32 inset_offset = section.base_address - info->address;
				const u32 inset_y = inset_offset / info->pitch;
				const u32 inset_x = (inset_offset % info->pitch) / format_bpp;

				const u32 full_width = surface_width + inset_x + inset_x;
				const u32 full_height = surface_height + inset_y + inset_y;

				viable = (full_width == info->width && full_height == info->height);
			}

			if (viable)
			{
				image_to_flip = section.surface->get_surface(rsx::surface_access::transfer_read);

				std::tie(info->width, info->height) = rsx::apply_resolution_scale<true>(
					resolution_scaling_config,
					std::min(surface_width, info->width),
					std::min(surface_height, info->height));
			}
		}
	}
	else if (auto surface = m_texture_cache.find_texture_from_dimensions<true>(info->address, info->format);
			 surface && surface->get_width() >= info->width && surface->get_height() >= info->height)
	{
		// Hack - this should be the first location to check for output
		// The render might have been done offscreen or in software and a blit used to display
		image_to_flip = dynamic_cast<mtl::viewable_image*>(surface->get_raw_texture());
	}

	// The correct output format is determined by the AV configuration set in CellVideoOutConfigure by the game.
	// 99.9% of the time, this will match the backbuffer fbo format used in rendering/compositing the output.
	// But in some cases, let's just say some devs are creative.
	const auto expected_format = RSX_display_format_to_mtl_format(avconfig.format);

	if (!image_to_flip) [[ unlikely ]]
	{
		// Read from cell
		const auto range = utils::address_range32::start_length(info->address, info->pitch * info->height);
		const u32  lookup_mask = rsx::texture_upload_context::blit_engine_dst | rsx::texture_upload_context::framebuffer_storage;
		const auto overlap = m_texture_cache.find_texture_from_range<true>(range, 0, lookup_mask);

		for (const auto & section : overlap)
		{
			if (!section->is_synchronized())
			{
				section->copy_texture(*m_current_command_buffer, true);
			}
		}

		if (m_current_command_buffer->flags & mtl::command_list::cb_has_dma_transfer)
		{
			// Submit for processing to lower hard fault penalty
			flush_command_queue();
		}

		m_texture_cache.invalidate_range(*m_current_command_buffer, range, rsx::invalidation_cause::read);
		image_to_flip = m_texture_cache.upload_image_simple(*m_current_command_buffer, expected_format, info->address, info->width, info->height, info->pitch);
	}
	else if (image_to_flip->format() != expected_format)
	{
		// Devs are being creative. Force-cast this to the proper pixel layout.
		auto dst_img = m_texture_cache.create_temporary_subresource_storage(
			RSX_FORMAT_CLASS_COLOR, expected_format, static_cast<u16>(info->width), static_cast<u16>(info->height), 1, 1, 1,
			MTL::TextureType2D, 0, MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);

		if (dst_img)
		{
			const areai src_rect = { 0, 0, static_cast<int>(info->width), static_cast<int>(info->height) };
			const areai dst_rect = src_rect;

			if (mtl::formats_are_bitcast_compatible(dst_img.get(), image_to_flip))
			{
				mtl::copy_image(*m_current_command_buffer, image_to_flip, dst_img.get(), src_rect, dst_rect);
			}
			else
			{
				mtl::copy_image_typeless(*m_current_command_buffer, image_to_flip, dst_img.get(), src_rect, dst_rect);
			}

			image_to_flip = dst_img.get();
			m_texture_cache.dispose_reusable_image(dst_img);
		}
	}

	return image_to_flip;
}

void MTLGSRender::flip(const rsx::display_flip_info_t& info)
{
	mtl::autorelease_scope pool;

	// Check surface condition/status. CAMetalLayer does not report resizes, poll the window size.
	if (m_swapchain_dims.width != m_frame->client_width() + 0u ||
		m_swapchain_dims.height != m_frame->client_height() + 0u)
	{
		should_reinitialize_swapchain = true;
	}

	if (m_vsync_mode != g_cfg.video.vsync)
	{
		should_reinitialize_swapchain = true;
	}

	if (swapchain_unavailable || should_reinitialize_swapchain)
	{
		// Reinitializing only fails for minimized windows (or a missing layer); the frame is then skipped below.
		reinitialize_swapchain();
	}

	m_profiler.start();

	ensure(m_current_frame, "Invalid frame context setup");

	if (m_current_frame == &m_aux_frame_context)
	{
		m_current_frame = &m_frame_context_storage[m_current_queue_index];
		if (m_current_frame->swap_command_buffer)
		{
			// Its possible this flip request is triggered by overlays and the flip queue is in undefined state
			frame_context_cleanup(m_current_frame);
		}

		// Swap aux storage and current frame; aux storage should always be ready for use at all times
		m_current_frame->grab_resources(m_aux_frame_context);
	}
	else if (m_current_frame->swap_command_buffer)
	{
		if (info.stats.draw_calls > 0)
		{
			// This can be 'legal' if the window was being resized and no polling happened because of swapchain_unavailable flag
			rsx_log.error("Possible data corruption on frame context storage detected");
		}

		// There were no draws and back-to-back flips happened
		frame_context_cleanup(m_current_frame);
	}

	// Frame without presentation (skipped frame, minimized window, drawable acquisition timeout)
	const auto mini_flip = [&](bool skip_frame)
	{
		if (!skip_frame)
		{
			// Perform a mini-flip here without invoking present code
			m_current_frame->swap_command_buffer = m_current_command_buffer;
			flush_command_queue(true);
			mtl::advance_frame_counter();
			frame_context_cleanup(m_current_frame);
		}

		m_frame->flip(m_context);
		rsx::thread::flip(info);
	};

	if (info.skip_frame || swapchain_unavailable)
	{
		mini_flip(info.skip_frame);
		return;
	}

	u32 buffer_width = display_buffers[info.buffer].width;
	u32 buffer_height = display_buffers[info.buffer].height;
	u32 buffer_pitch = display_buffers[info.buffer].pitch;

	u32 av_format;
	const auto& avconfig = g_fxo->get<rsx::avconf>();

	if (!buffer_width)
	{
		buffer_width = avconfig.resolution_x;
		buffer_height = avconfig.resolution_y;
	}

	if (avconfig.state)
	{
		av_format = avconfig.get_compatible_gcm_format();
		if (!buffer_pitch)
			buffer_pitch = buffer_width * avconfig.get_bpp();

		const size2u video_frame_size = avconfig.video_frame_size();
		buffer_width = std::min(buffer_width, video_frame_size.width);
		buffer_height = std::min(buffer_height, video_frame_size.height);
	}
	else
	{
		av_format = CELL_GCM_TEXTURE_A8R8G8B8;
		if (!buffer_pitch)
			buffer_pitch = buffer_width * 4;
	}

	// Scan memory for required data. This is done early to optimize waiting for the drawable below.
	mtl::viewable_image* image_to_flip = nullptr;
	mtl::viewable_image* image_to_flip2 = nullptr;

	if (info.buffer < display_buffers_count && buffer_width && buffer_height)
	{
		mtl::present_surface_info present_info
		{
			.address = rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL),
			.format = av_format,
			.width = buffer_width,
			.height = buffer_height,
			.pitch = buffer_pitch,
			.eye = 0
		};
		image_to_flip = get_present_source(&present_info, avconfig);

		if (avconfig.stereo_enabled) [[unlikely]]
		{
			const auto [unused, min_expected_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height + 30);
			if (image_to_flip->height() < min_expected_height)
			{
				// Get image for second eye
				const u32 image_offset = (buffer_height + 30) * buffer_pitch + display_buffers[info.buffer].offset;
				present_info.width = buffer_width;
				present_info.height = buffer_height;
				present_info.address = rsx::get_address(image_offset, CELL_GCM_LOCATION_LOCAL);
				present_info.eye = 1;

				image_to_flip2 = get_present_source(&present_info, avconfig);
			}
			else
			{
				// Account for possible insets
				const auto [unused2, scaled_buffer_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height);
				buffer_height = std::min<u32>(image_to_flip->height() - min_expected_height, scaled_buffer_height);
			}
		}

		buffer_width = present_info.width;
		buffer_height = present_info.height;
	}

	if (info.emu_flip)
	{
		evaluate_cpu_usage_reduction_limits();
	}

	const bool has_overlay = (m_overlay_manager && m_overlay_manager->has_visible());
	const bool user_asked_for_screenshot = g_user_asked_for_screenshot.exchange(false);
	const bool user_is_recording = (g_recording_mode != recording_mode::stopped && m_frame->can_consume_frame());
	const bool need_media_capture = user_asked_for_screenshot || user_is_recording;

	const auto render_overlays = [&](const mtl::overlay_target& target, const areau& area)
	{
		if (!has_overlay) return;

		// Lock to avoid modification during run-update chain
		auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
		std::lock_guard lock(*m_overlay_manager);

		const areau display_area = { 0, 0, target.width(), target.height() };
		for (const auto& view : m_overlay_manager->get_views())
		{
			const areau render_area = view->use_window_space ? display_area : area;
			ui_renderer->run(*m_current_command_buffer, render_area, target, m_texture_upload_buffer_ring_info, *view.get());
		}
	};

	// Screenshots / recording are captured before the drawable is acquired so that the hard sync does not delay it
	if (image_to_flip && need_media_capture)
	{
		const u32 bytes_per_pixel = (image_to_flip->format() == MTL::PixelFormatRGBA16Float) ? 8 : 4;

		if (bytes_per_pixel != 4)
		{
			rsx_log.error("Metal: screenshots and recording of 16-bit float display buffers are not supported");
		}
		else
		{
			const usz sshot_size = buffer_height * buffer_width * 4;
			mtl::buffer sshot_buf(*m_device, utils::align(sshot_size, 0x100000), mtl::memory_location::host_visible, "screenshot buffer");

			mtl::buffer_image_copy copy_info{};
			copy_info.buffer_offset = 0;
			copy_info.buffer_row_length = 0;
			copy_info.buffer_image_height = 0;
			copy_info.aspect = mtl::aspect_color;
			copy_info.base_layer = 0;
			copy_info.layer_count = 1;
			copy_info.mip_level = 0;
			copy_info.image_offset = MTL::Origin(0, 0, 0);
			copy_info.image_extent = MTL::Size(buffer_width, buffer_height, 1);

			mtl::image* image_to_copy = image_to_flip;

			if (g_cfg.video.record_with_overlays && has_overlay)
			{
				if (m_overlay_recording_img)
				{
					// Validate
					if (m_overlay_recording_img->format() != image_to_flip->format() ||
						m_overlay_recording_img->width() != image_to_flip->width() ||
						m_overlay_recording_img->height() != image_to_flip->height())
					{
						// Dispose correctly
						mtl::get_resource_manager()->dispose(m_overlay_recording_img);
					}
				}

				if (!m_overlay_recording_img)
				{
					mtl::image_create_info create_info{};
					create_info.type = MTL::TextureType2D;
					create_info.format = image_to_flip->format();
					create_info.width = image_to_flip->width();
					create_info.height = image_to_flip->height();
					create_info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
					create_info.format_class = RSX_FORMAT_CLASS_COLOR;

					m_overlay_recording_img = std::make_unique<mtl::viewable_image>(*m_device, create_info);
					m_overlay_recording_img->set_debug_name("overlay recording image");
				}

				const areai rect = areai(0, 0, buffer_width, buffer_height);
				mtl::copy_image(*m_current_command_buffer, image_to_flip, m_overlay_recording_img.get(), rect, rect);

				render_overlays(mtl::overlay_target(m_overlay_recording_img.get()), areau(rect));

				image_to_copy = m_overlay_recording_img.get();
			}

			mtl::copy_image_to_buffer(*m_current_command_buffer, image_to_copy, &sshot_buf, copy_info);

			flush_command_queue(true);
			const auto src = sshot_buf.map(0);
			std::vector<u8> sshot_frame(sshot_size);
			std::memcpy(sshot_frame.data(), src, sshot_size);

			const bool is_bgra = image_to_copy->format() == MTL::PixelFormatBGRA8Unorm;

			if (user_asked_for_screenshot)
			{
				m_frame->take_screenshot(std::move(sshot_frame), buffer_width, buffer_height, is_bgra);
			}
			else
			{
				m_frame->present_frame(std::move(sshot_frame), buffer_width * 4, buffer_width, buffer_height, is_bgra);
			}
		}
	}

	// Prepare surface for new frame. nextDrawable blocks until one is available (1 second timeout).
	ensure(!m_current_frame->drawable);
	ensure(m_current_frame->swap_command_buffer == nullptr);

	auto drawable = m_metal_layer->nextDrawable();
	if (!drawable)
	{
		rsx_log.warning("Metal: nextDrawable timed out. Frame skipped.");
		mini_flip(false);
		return;
	}

	drawable->retain();
	m_current_frame->drawable = drawable;

	// Drawables acquired while the layer is framebuffer-only can only be render targets
	const bool drawable_is_framebuffer_only = m_layer_framebuffer_only;

	MTL::Texture* target_texture = drawable->texture();
	const sizeu target_size = { static_cast<u32>(target_texture->width()), static_cast<u32>(target_texture->height()) };
	const mtl::overlay_target target(target_texture);

	// Calculate output dimensions. Done after acquisition since the layer may have been resized.
	areai aspect_ratio;
	if (!g_cfg.video.stretch_to_display_area)
	{
		const auto converted = avconfig.aspect_convert_region({ buffer_width, buffer_height }, target_size);
		aspect_ratio = static_cast<areai>(converted);
	}
	else
	{
		aspect_ratio = { 0, 0, s32(target_size.width), s32(target_size.height) };
	}

	if (!image_to_flip || aspect_ratio.x1 || aspect_ratio.y1 ||
		static_cast<u32>(aspect_ratio.x2) < target_size.width || static_cast<u32>(aspect_ratio.y2) < target_size.height)
	{
		// Clear the window background to black (drawable contents are undefined)
		mtl::clear_color_texture(*m_current_command_buffer, target_texture, target_size.width, target_size.height, MTL::ClearColor::Make(0., 0., 0., 1.));
	}

	const output_scaling_mode output_scaling = g_cfg.video.output_scaling.get();

	if (!m_upscaler || m_output_scaling != output_scaling)
	{
		m_output_scaling = output_scaling;
		m_upscaler = mtl::create_upscaler(m_output_scaling);

		// MetalFX writes into the drawable; the layer must not be framebuffer-only for the next drawables
		configure_metal_layer();
	}

	if (image_to_flip)
	{
		const bool use_full_rgb_range_output = g_cfg.video.full_rgb_range_output.get();
		const areai src_area = { 0, 0, s32(buffer_width), s32(buffer_height) };

		if (!use_full_rgb_range_output || !rsx::fcmp(avconfig.gamma, 1.f) || avconfig.stereo_enabled ||
			(m_output_scaling == output_scaling_mode::fsr && drawable_is_framebuffer_only)) [[unlikely]]
		{
			rsx::simple_array<mtl::viewable_image*> calibration_src;
			if (image_to_flip) calibration_src.push_back(image_to_flip);
			if (image_to_flip2) calibration_src.push_back(image_to_flip2);

			if (m_output_scaling == output_scaling_mode::fsr && !avconfig.stereo_enabled) // 3D will be implemented later
			{
				// Run upscaling pass before the rest of the output effects pipeline
				// This can be done with all upscalers but we already get bilinear upscaling for free if we just out the filters directly
				const areai dst_area = { 0, 0, aspect_ratio.width(), aspect_ratio.height() };

				for (unsigned i = 0; i < calibration_src.size(); ++i)
				{
					const rsx::flags32_t mode = (i == 0) ? UPSCALE_LEFT_VIEW : UPSCALE_RIGHT_VIEW;
					calibration_src[i] = m_upscaler->scale_output(*m_current_command_buffer, image_to_flip, nullptr, src_area, dst_area, mode);
				}
			}

			mtl::get_overlay_pass<mtl::video_out_calibration_pass>()->run(
				*m_current_command_buffer, areau(aspect_ratio), target, calibration_src,
				avconfig.gamma, !use_full_rgb_range_output, avconfig.stereo_enabled);
		}
		else
		{
			// Scaled draw straight into the drawable (Metal has no image blit)
			m_upscaler->scale_output(*m_current_command_buffer, image_to_flip, target_texture, src_area, aspect_ratio, UPSCALE_AND_COMMIT | UPSCALE_DEFAULT_VIEW);
		}
	}

	if (g_cfg.video.debug_overlay || has_overlay)
	{
		render_overlays(target, areau(aspect_ratio));

		if (g_cfg.video.debug_overlay)
		{
			m_render_pass_splits = mtl::g_feedback_loop_pass_splits.exchange(0);

			const auto num_dirty_textures = m_texture_cache.get_unreleased_textures_count();
			const auto texture_memory_size = m_texture_cache.get_texture_memory_in_use() / (1024 * 1024);
			const auto tmp_texture_memory_size = m_texture_cache.get_temporary_memory_in_use() / (1024 * 1024);
			const auto num_flushes = m_texture_cache.get_num_flush_requests();
			const auto num_mispredict = m_texture_cache.get_num_cache_mispredictions();
			const auto num_speculate = m_texture_cache.get_num_cache_speculative_writes();
			const auto num_misses = m_texture_cache.get_num_cache_misses();
			const auto num_unavoidable = m_texture_cache.get_num_unavoidable_hard_faults();
			const auto cache_miss_ratio = static_cast<u32>(ceil(m_texture_cache.get_cache_miss_ratio() * 100));
			const auto num_texture_upload = m_texture_cache.get_texture_upload_calls_this_frame();
			const auto num_texture_upload_miss = m_texture_cache.get_texture_upload_misses_this_frame();
			const auto texture_upload_miss_ratio = m_texture_cache.get_texture_upload_miss_percentage();
			const auto texture_copies_ellided = m_texture_cache.get_texture_copies_ellided_this_frame();
			const auto vertex_cache_hit_count = (info.stats.vertex_cache_request_count - info.stats.vertex_cache_miss_count);
			const auto vertex_cache_hit_ratio = info.stats.vertex_cache_request_count
				? (vertex_cache_hit_count * 100) / info.stats.vertex_cache_request_count
				: 0;
			const auto program_cache_lookups = info.stats.program_cache_lookups_total;
			const auto program_cache_ellided = info.stats.program_cache_lookups_ellided;
			const auto program_cache_ellision_rate = program_cache_lookups
				? (program_cache_ellided * 100) / program_cache_lookups
				: 0;

			rsx::overlays::set_debug_overlay_text(fmt::format(
				"Internal Resolution:      %s\n"
				"RSX Load:                 %3d%%\n"
				"draw calls: %17d\n"
				"submits: %20d\n"
				"render pass splits: %9d\n"
				"draw call setup: %12dus\n"
				"vertex upload time: %9dus\n"
				"texture upload time: %8dus\n"
				"draw call execution: %8dus\n"
				"submit and flip: %12dus\n"
				"Unreleased textures: %8d\n"
				"Texture cache memory: %7dM\n"
				"Temporary texture memory: %3dM\n"
				"Flush requests: %13d  = %2d (%3d%%) hard faults, %2d unavoidable, %2d misprediction(s), %2d speculation(s)\n"
				"Texture uploads: %12u (%u from CPU - %02u%%, %u copies avoided)\n"
				"Vertex cache hits: %10u/%u (%u%%)\n"
				"Program cache lookup ellision: %u/%u (%u%%)",
				info.stats.framebuffer_stats.to_string(resolution_scaling_config, !backend_config.supports_hw_msaa),
				get_load(), info.stats.draw_calls, info.stats.submit_count, m_render_pass_splits, info.stats.setup_time, info.stats.vertex_upload_time,
				info.stats.textures_upload_time, info.stats.draw_exec_time, info.stats.flip_time,
				num_dirty_textures, texture_memory_size, tmp_texture_memory_size,
				num_flushes, num_misses, cache_miss_ratio, num_unavoidable, num_mispredict, num_speculate,
				num_texture_upload, num_texture_upload_miss, texture_upload_miss_ratio, texture_copies_ellided,
				vertex_cache_hit_count, info.stats.vertex_cache_request_count, vertex_cache_hit_ratio,
				program_cache_ellided, program_cache_lookups, program_cache_ellision_rate)
			);
		}
	}

	queue_swap_request();

	m_frame_stats.flip_time = m_profiler.duration();

	m_frame->flip(m_context);
	rsx::thread::flip(info);

	// Data sync
	const rsx::surface_scaling_config_t active_res_scaling_config =
	{
		.scale_percent = static_cast<u16>(g_cfg.video.resolution_scale_percent),
		.min_scalable_dimension = static_cast<u16>(g_cfg.video.min_scalable_dimension),
	};

	if (active_res_scaling_config != this->resolution_scaling_config)
	{
		// First, try to reclaim any memory since the res scale upgrade is so memory intensive
		if (const auto severity = mtl::vmm_determine_memory_load_severity();
			severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(*m_current_command_buffer, severity))
		{
			flush_command_queue(true);
		}

		// Then apply the change
		m_rtts.sync_scaling_config(*m_current_command_buffer, active_res_scaling_config);
		this->resolution_scaling_config = active_res_scaling_config;

		// Finally reclaim any unused resources
		if (const auto severity = mtl::vmm_determine_memory_load_severity();
			severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(*m_current_command_buffer, severity))
		{
			flush_command_queue(true);
		}
	}
}
