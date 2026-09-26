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

#include <algorithm>
#include <cmath>
#include <cstring>

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

// Presentation model
// ------------------
// Metal 4 requires the queue that renders into a drawable to wait for it (waitForDrawable). That wait blocks every
// later command buffer of the queue, up to a display refresh. So the renderer keeps the drawable off the main queue:
//  1. flip() records the present passes (letterbox clear, calibration, upscaling, overlays) into the frame's last
//     main-queue list, rendering into the frame context's present_image (same size/format as the drawable).
//  2. present() records a present list (one texture copy) on the present queue: wait for the RSX timeline value of the
//     frame's work, wait for the drawable, commit, signal the drawable, then present with pacing (present_drawable()).
// The present list only touches the frame context's present_image and the drawable, both owned by the frame context,
// so the event-id GC (which assumes a single in-order queue) never frees anything it uses. The frame context (and
// its heap snapshot) is recycled once the present list has completed: frame_context_t::swap_command_buffer points to
// it, and its completion implies the completion of the frame's main-queue work.
//
// Pacing (present_drawable()): with R = the screen's minimum refresh interval and G = the guest frame interval
// (get_guest_frame_interval()), every frame stays on screen for k = max(1, ~G / R) refreshes
// (presentAfterMinimumDuration(k * R - R / 2)), e.g. 60 fps on a 120 Hz ProMotion panel = every other refresh, 30 fps =
// every 4th, instead of jittering between 1, 2 and 3 refreshes. Fullscreen on an Adaptive-Sync screen:
// presentAfterMinimumDuration(G - 0.5 ms). VSync off: plain present().

namespace
{
	constexpr u64 guest_interval_reset_us = 250'000;         // Longer gaps (loading, pause) restart the guest interval history
	constexpr u64 surface_poll_interval_us = 1'000'000;      // The window can move to another screen without a resize
	constexpr u64 present_stats_interval_us = 30'000'000;    // Telemetry log rate limit

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

	// The present list copies the frame's present image into the drawable (compute encoder copy), so drawables must be
	// copy destinations: framebuffer-only drawables can only be render pass attachments.
	const bool framebuffer_only = false;
	if (framebuffer_only != m_layer_framebuffer_only || framebuffer_only != m_metal_layer->framebufferOnly())
	{
		m_metal_layer->setFramebufferOnly(framebuffer_only);
		m_layer_framebuffer_only = framebuffer_only;
	}

	// VSync off: no display sync (tears in direct-to-display fullscreen; the compositor still syncs windows).
	// Adaptive/full: display synced and paced by present_drawable().
	const bool display_sync = g_cfg.video.vsync != vsync_mode::off;
	if (display_sync != m_metal_layer->displaySyncEnabled())
	{
		m_metal_layer->setDisplaySyncEnabled(display_sync);
	}
	m_vsync_mode = g_cfg.video.vsync;

	if (m_metal_layer->maximumDrawableCount() != MTL_MAX_DRAWABLE_COUNT)
	{
		m_metal_layer->setMaximumDrawableCount(MTL_MAX_DRAWABLE_COUNT);
	}

	// Only touch drawableSize when it really changes: every change replaces the layer's drawables and the next
	// nextDrawable() can stall up to its 1 s timeout (skipped frame, visible as a hitch or flicker). Qt keeps
	// drawableSize at bounds * contentsScale itself, which is the same value as the client size used here
	// (QWindow size * devicePixelRatio), so normally there is nothing to do.
	if (m_swapchain_dims.width && m_swapchain_dims.height)
	{
		const CGSize current = m_metal_layer->drawableSize();
		const CGSize wanted = { static_cast<CGFloat>(m_swapchain_dims.width), static_cast<CGFloat>(m_swapchain_dims.height) };

		if (current.width != wanted.width || current.height != wanted.height)
		{
			m_metal_layer->setDrawableSize(wanted);
		}
	}

	// contentsScale (the window's backing scale factor, the same source as the client size) and opaque are applied on
	// the main thread, together with sampling the screen's refresh properties. Never blocks.
	mtl::request_surface_update(m_view, m_metal_layer);
	m_present_pacing.surface_request_time = get_system_time();
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
	// created with the new drawableSize. No GPU sync is required. Frame contexts recreate their present image when the
	// drawable size changes, and the upscaler handles output size changes itself.
	configure_metal_layer();

	swapchain_unavailable = false;
	should_reinitialize_swapchain = false;
	return true;
}

void MTLGSRender::present_feedback_t::on_presented(f64 presented_time)
{
	if (presented_time <= 0.)
	{
		// Never displayed (replaced by a newer drawable before it reached the screen)
		dropped++;
		return;
	}

	const f64 previous = last_presented_time.exchange(presented_time);
	const f64 unit = refresh_interval.load();

	if (previous <= 0. || presented_time <= previous || unit <= 0.)
	{
		return;
	}

	// On-screen time of the previous frame, in refreshes
	const s64 refreshes = std::clamp<s64>(std::llround((presented_time - previous) / unit), 1, 6);
	intervals[refreshes - 1]++;
}

void MTLGSRender::update_present_pacing(bool emu_flip)
{
	const u64 now = get_system_time();
	auto& pacing = m_present_pacing;

	// Screen properties are sampled on the main thread. Poll: moving the window to another screen is not a resize.
	if (now - pacing.surface_request_time >= surface_poll_interval_us)
	{
		mtl::request_surface_update(m_view, m_metal_layer);
		pacing.surface_request_time = now;
	}

	if (const mtl::surface_properties props = mtl::get_surface_properties(); props.serial != pacing.surface_serial)
	{
		const bool first_update = !pacing.surface_serial;
		pacing.surface_serial = props.serial;

		// NSScreen.minimumRefreshInterval is the fastest refresh (1/120 s on ProMotion). Fall back to 60 Hz.
		const f64 refresh_interval = (props.min_refresh_interval >= 0.001 && props.min_refresh_interval <= 0.1)
			? props.min_refresh_interval
			: 1. / 60.;

		// Adaptive-Sync/ProMotion: minimum and maximum refresh intervals differ (WWDC21 "Optimize for variable refresh rate displays")
		const bool variable_refresh = props.max_refresh_interval > refresh_interval * 1.01;

		if (first_update || refresh_interval != pacing.refresh_interval || variable_refresh != pacing.variable_refresh || props.fullscreen != pacing.fullscreen)
		{
			rsx_log.notice("Metal: display refresh %.2f Hz (%s refresh, interval %.2f-%.2f ms, granularity %.2f ms), %s window, backing scale %.1f",
				1. / refresh_interval, variable_refresh ? "variable" : "fixed", props.min_refresh_interval * 1000., props.max_refresh_interval * 1000.,
				props.update_granularity * 1000., props.fullscreen ? "fullscreen" : "windowed", props.backing_scale);
		}

		pacing.refresh_interval = refresh_interval;
		pacing.variable_refresh = variable_refresh;
		pacing.fullscreen = props.fullscreen;
	}

	if (!emu_flip)
	{
		// Overlay-only flips (native UI while the game is paused, ...) say nothing about the guest's frame rate
		return;
	}

	if (pacing.last_emu_flip_time)
	{
		const u64 delta = now - pacing.last_emu_flip_time;

		if (delta >= guest_interval_reset_us)
		{
			// Loading, pause or a stall: start over
			pacing.guest_interval_count = 0;
			pacing.guest_interval_next = 0;
		}
		else
		{
			// Time the RSX thread spent waiting for drawables/frame contexts is back-pressure from the display, not the
			// guest's pace. Counting it would let a pacing target sustain itself even after the game got faster.
			const f64 sample = static_cast<f64>(delta - std::min(delta, pacing.blocked_time)) / 1'000'000.;

			pacing.guest_intervals[pacing.guest_interval_next] = sample;
			pacing.guest_interval_next = (pacing.guest_interval_next + 1) % pacing.guest_intervals.size();
			pacing.guest_interval_count = std::min<u32>(pacing.guest_interval_count + 1, ::size32(pacing.guest_intervals));
		}
	}

	pacing.last_emu_flip_time = now;
	pacing.blocked_time = 0;
}

f64 MTLGSRender::get_guest_frame_interval() const
{
	// Frame limit, as applied by rsx::thread::handle_emu_flip
	f64 limit = 0.;
	switch (g_disable_frame_limit ? frame_limit_type::none : g_cfg.video.frame_limit.get())
	{
	case frame_limit_type::none: limit = g_cfg.core.max_cpu_preempt_count_per_frame ? static_cast<f64>(g_cfg.video.vblank_rate) : 0.; break;
	case frame_limit_type::_30: limit = 30.; break;
	case frame_limit_type::_50: limit = 50.; break;
	case frame_limit_type::_60: limit = 60.; break;
	case frame_limit_type::_120: limit = 120.; break;
	case frame_limit_type::display_rate: limit = 1. / m_present_pacing.refresh_interval; break;
	case frame_limit_type::_auto: limit = static_cast<f64>(g_cfg.video.vblank_rate); break;
	case frame_limit_type::_ps3:
	case frame_limit_type::infinite:
		break;
	}

	if (const f64 limit2 = g_cfg.video.second_frame_limit; limit2 >= 0.1 && (limit2 < limit || !limit))
	{
		limit = limit2;
	}

	// Measured interval: the 25th percentile of the recent intervals. Overestimating the interval throttles the game
	// (every frame is held on screen too long) while underestimating it only degrades to plain FIFO, so low values win:
	// hitches (long frames) never raise the target, and a game alternating between two cadences is paced at the faster.
	const auto& pacing = m_present_pacing;
	f64 measured = 0.;

	if (const u32 count = pacing.guest_interval_count; count >= 4)
	{
		std::array<f64, std::tuple_size_v<decltype(pacing.guest_intervals)>> sorted = pacing.guest_intervals;
		const auto nth = sorted.begin() + count / 4;
		std::nth_element(sorted.begin(), nth, sorted.begin() + count);
		measured = *nth;

		// Guest frames follow the emulated vblank (1/60 s by default): snap to a whole number of vblanks (60, 30, 20,
		// 15 fps) so that jitter does not move the pacing target
		const f64 vblank = 1. / static_cast<f64>(std::max<s64>(1, g_cfg.video.vblank_rate));
		const f64 vblanks = std::round(measured / vblank);

		if (vblanks >= 1. && vblanks <= 4. && std::abs(measured - vblanks * vblank) <= vblank * 0.2)
		{
			measured = vblanks * vblank;
		}
	}

	// The limit is a lower bound: games often run below it (a 30 fps game with the default 60 fps limit)
	return std::max(measured, limit > 0. ? 1. / limit : 0.);
}

void MTLGSRender::present_drawable(mtl::frame_context_t* ctx)
{
	auto drawable = ctx->drawable;
	auto& pacing = m_present_pacing;

	if (!m_present_feedback)
	{
		m_present_feedback = std::make_shared<present_feedback_t>();
	}

	// Telemetry: actual on-screen times. Handlers must be added before presenting. The handler owns a reference to the
	// feedback block, so it may run after the renderer is gone.
	m_present_feedback->refresh_interval.store(pacing.refresh_interval);
	drawable->addPresentedHandler(MTL::DrawablePresentedHandlerFunction([feedback = m_present_feedback](MTL::Drawable* presented)
	{
		feedback->on_presented(presented->presentedTime());
	}));

	const f64 now = mtl::get_media_time();
	const f64 refresh = pacing.refresh_interval;
	f64 min_duration = 0.;
	u32 slot_refreshes = 0;

	if (m_vsync_mode != vsync_mode::off)
	{
		if (const f64 guest_interval = get_guest_frame_interval(); guest_interval > 0.)
		{
			f64 slot = 0.;

			if (pacing.fullscreen && pacing.variable_refresh)
			{
				// Adaptive-Sync (fullscreen only): the display refreshes when the frame is due
				slot = std::max(guest_interval, refresh);
				min_duration = slot - 0.0005;
			}
			else
			{
				// Fixed refresh grid: keep every frame on screen for the same whole number of refreshes. Intervals are
				// only rounded up when they are close to the next multiple: rounding a game running between two
				// cadences up would throttle it.
				slot_refreshes = std::max(1u, static_cast<u32>(guest_interval / refresh + 0.25));
				slot = slot_refreshes * refresh;

				// Any time between the previous refresh and the target one: half a refresh of margin absorbs jitter
				min_duration = slot - refresh * 0.5;
			}

			if (m_vsync_mode == vsync_mode::adaptive && pacing.last_present_time > 0. &&
				now - pacing.last_present_time > slot + refresh * 0.5)
			{
				// Adaptive: the frame is already late for its slot. Show it as soon as possible instead of holding the cadence.
				min_duration = 0.;
				slot_refreshes = 0;
			}
		}
	}

	if (min_duration > 0.)
	{
		drawable->presentAfterMinimumDuration(min_duration);
	}
	else
	{
		drawable->present();
	}

	pacing.last_present_time = now;
	pacing.min_duration = min_duration;
	pacing.slot_refreshes = slot_refreshes;

	// Rate-limited telemetry
	const u64 now_us = get_system_time();
	if (!pacing.stats_time)
	{
		pacing.stats_time = now_us;
	}
	else if (now_us - pacing.stats_time >= present_stats_interval_us)
	{
		auto& feedback = *m_present_feedback;
		std::array<u32, 6> counts{};
		for (usz i = 0; i < counts.size(); ++i)
		{
			counts[i] = feedback.intervals[i].exchange(0);
		}

		const u32 dropped = feedback.dropped.exchange(0);
		const std::string mode = (m_vsync_mode == vsync_mode::off) ? "vsync off" :
			(min_duration <= 0.) ? "unpaced" :
			slot_refreshes ? fmt::format("%u refresh(es) per frame", slot_refreshes) : std::string("variable refresh");

		rsx_log.notice("Metal: presentation over %us: frames on screen for 1/2/3/4/5/6+ refreshes (%.2f ms): %u/%u/%u/%u/%u/%u, not displayed: %u. "
			"Pacing: %s, guest frame interval %.2f ms, minimum duration %.2f ms",
			(now_us - pacing.stats_time) / 1'000'000, refresh * 1000., counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], dropped,
			mode, get_guest_frame_interval() * 1000., min_duration * 1000.);

		// GPU load. A rising GPU time per frame in the same scene means the GPU clock dropped (heat); a high pass count
		// means attachments are stored and reloaded often (feedback splits, clears, copies between draws).
		u32 frames = dropped;
		for (const u32 count : counts)
		{
			frames += count;
		}

		const auto gpu = mtl::get_gpu_stats_and_reset();
		const f64 window_ms = (now_us - pacing.stats_time) / 1000.;
		if (frames && window_ms > 0.)
		{
			const f64 busy_ms = gpu.busy_ns / 1'000'000.;
			const auto per_frame = [frames](u64 count) { return static_cast<f64>(count) / frames; };
			const auto& reasons = gpu.splits_by_reason;
			// Skipped draws / pipeline waits: shader compilation stutter (no shader interpreter on Metal)
			rsx_log.notice("Metal: GPU busy %.2f ms per frame (%.0f%% of the time), %.1f render passes (%.1f for draws) and %.1f feedback splits per frame "
				"(read after write %.1f, through a copy %.1f, write after read %.1f, depth compare %.1f, vertex read %.1f; %.1f feedback reads kept in the pass). "
				"Pipelines: %.1f draws skipped and %.2f ms waited per frame",
				busy_ms / frames, 100. * busy_ms / window_ms, per_frame(gpu.render_passes), per_frame(gpu.draw_render_passes), per_frame(gpu.feedback_splits),
				per_frame(reasons[static_cast<u32>(mtl::pass_split_reason::read_after_write)]),
				per_frame(reasons[static_cast<u32>(mtl::pass_split_reason::read_through_copy)]),
				per_frame(reasons[static_cast<u32>(mtl::pass_split_reason::write_after_read)]),
				per_frame(reasons[static_cast<u32>(mtl::pass_split_reason::depth_compare)]),
				per_frame(reasons[static_cast<u32>(mtl::pass_split_reason::vertex_read)]),
				per_frame(gpu.feedback_reads_in_pass),
				per_frame(m_skipped_draws), per_frame(m_pipeline_wait_us) / 1000.);
		}

		m_skipped_draws = 0;
		m_pipeline_wait_us = 0;
		pacing.stats_time = now_us;
	}
}

void MTLGSRender::present(mtl::frame_context_t *ctx)
{
	ensure(ctx->drawable && ctx->present_image && ctx->swap_command_buffer);
	ensure(m_present_queue);

	// The frame's main-queue work (including the present passes) is complete once the RSX timeline reaches the value of
	// its last list. Deferred (MTRSX) submissions get that value once the offloader thread has committed the list.
	auto frame_cb = ctx->swap_command_buffer;
	frame_cb->flush();

	u64 frame_eid = 0;
	{
		reader_lock lock(frame_cb->guard_mutex);
		ctx->swap_timeline_value = frame_cb->get_fence().value; // 0: already retired (complete)
		frame_eid = frame_cb->eid_tag;
	}

	if (!ctx->present_command_buffer)
	{
		const auto index = static_cast<u32>(ctx - m_frame_context_storage.data());
		ctx->present_command_buffer = std::make_unique<mtl::command_buffer_chunk>();
		ctx->present_command_buffer->create(*m_device, m_present_queue, m_present_timeline, fmt::format("RSX present #%u", index));
		ctx->present_command_buffer->access_hint = mtl::command_list::access_type_hint::all;
	}

	auto cmd = ctx->present_command_buffer.get();
	cmd->reset(); // Idle: the frame context was cleaned up before it was reused
	cmd->begin();
	cmd->clear_flags();

	// Same size and format: flip() (re)creates present_image from this drawable's texture
	MTL::Texture* src = ctx->present_image->value;
	MTL::Texture* dst = ctx->drawable->texture();
	const NS::UInteger width = std::min(src->width(), dst->width());
	const NS::UInteger height = std::min(src->height(), dst->height());
	cmd->compute()->copyFromTexture(src, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(width, height, 1), dst, 0, 0, MTL::Origin(0, 0, 0));

	// Present queue: wait for the frame's work and for the drawable, copy, signal the drawable
	mtl::submit_info_t submit_info{};
	if (ctx->swap_timeline_value)
	{
		submit_info.wait_event = m_timeline.handle();
		submit_info.wait_value = ctx->swap_timeline_value;
	}
	submit_info.wait_drawable = ctx->drawable;
	submit_info.signal_drawable = ctx->drawable;

	// Completion of the present list implies completion of the frame's main-queue list: report that list's event id
	// (GC) when it completes, like the single-queue path did when retiring the swap list
	cmd->eid_tag = frame_eid;
	{
		std::lock_guard lock(cmd->guard_mutex);
		mtl::queue_submit_now(*cmd, submit_info);
	}

	// From now on the present list ends the frame (check_present_status / frame_context_cleanup poll it)
	ctx->swap_command_buffer = cmd;

	if (!swapchain_unavailable)
	{
		present_drawable(ctx);
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

	// The frame's last main-queue list. It does not wait for the drawable: the present passes rendered into the frame's
	// present image, which the present list copies into the drawable on the present queue.
	close_and_submit_command_buffer();

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

	// New frame, new budget for waiting on pipelines that are still compiling (see load_program)
	m_async_compile_wait_spent_us = 0;

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

	// Screen properties and the guest frame interval (pacing inputs)
	update_present_pacing(info.emu_flip);

	ensure(m_current_frame, "Invalid frame context setup");

	// Waiting for an older frame to leave the present queue is back-pressure from the display (see update_present_pacing)
	const auto wait_for_frame_context = [&](mtl::frame_context_t* ctx)
	{
		const u64 wait_start = get_system_time();
		frame_context_cleanup(ctx);
		m_present_pacing.blocked_time += get_system_time() - wait_start;
	};

	if (m_current_frame == &m_aux_frame_context)
	{
		m_current_frame = &m_frame_context_storage[m_current_queue_index];
		if (m_current_frame->swap_command_buffer)
		{
			// Its possible this flip request is triggered by overlays and the flip queue is in undefined state
			wait_for_frame_context(m_current_frame);
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
		wait_for_frame_context(m_current_frame);
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

	if (!image_to_flip && !has_overlay && !g_cfg.video.debug_overlay)
	{
		// Nothing to show (no valid display buffer, e.g. while booting). Keep the previous image on screen instead of
		// presenting a black frame.
		mini_flip(false);
		return;
	}

	// Output scaling, set up before the drawable is acquired so that slow work (MetalFX kernels, mode switches) never
	// runs while a drawable is held. MetalFX builds its scalers on a worker thread (bilinear until ready) and keeps them
	// across resizes; the upscaler handles output size changes itself.
	const output_scaling_mode output_scaling = g_cfg.video.output_scaling.get();

	if (!m_upscaler || m_output_scaling != output_scaling)
	{
		m_output_scaling = output_scaling;
		m_upscaler = mtl::create_upscaler(m_output_scaling);
	}

	if (image_to_flip)
	{
		m_upscaler->prepare();
	}

	// Prepare surface for new frame. nextDrawable blocks until one is available (1 second timeout).
	ensure(!m_current_frame->drawable);
	ensure(m_current_frame->swap_command_buffer == nullptr);

	// Submit the frame's work so far: the GPU works on it while nextDrawable may block. Nothing on the main queue waits
	// for the drawable (see present()).
	flush_command_queue();

	const u64 acquire_start = get_system_time();
	auto drawable = m_metal_layer->nextDrawable();
	m_present_pacing.blocked_time += get_system_time() - acquire_start;

	if (!drawable)
	{
		// The previous frame stays on screen
		rsx_log.warning("Metal: nextDrawable timed out. Frame skipped.");
		mini_flip(false);
		return;
	}

	drawable->retain();
	m_current_frame->drawable = drawable;

	MTL::Texture* drawable_texture = drawable->texture();
	const sizeu target_size = { static_cast<u32>(drawable_texture->width()), static_cast<u32>(drawable_texture->height()) };

	// The frame's output image. The present passes below render into it on the main queue; present() copies it into
	// the drawable on the present queue. Same size and format as the drawable. The frame context is idle here (its
	// previous present list has completed), so the image can be reused or replaced.
	auto& present_image = m_current_frame->present_image;
	if (!present_image ||
		present_image->width() != target_size.width ||
		present_image->height() != target_size.height ||
		present_image->format() != drawable_texture->pixelFormat())
	{
		if (present_image)
		{
			mtl::get_resource_manager()->dispose(present_image);
		}

		mtl::image_create_info create_info{};
		create_info.type = MTL::TextureType2D;
		create_info.format = drawable_texture->pixelFormat();
		create_info.width = target_size.width;
		create_info.height = target_size.height;
		create_info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
		create_info.format_class = RSX_FORMAT_CLASS_COLOR;

		present_image = std::make_unique<mtl::viewable_image>(*m_device, create_info);
		present_image->set_debug_name("present image");
	}

	MTL::Texture* target_texture = present_image->value;
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

	// The window background must be cleared to black (the present image holds an older frame, or nothing yet)
	const bool needs_letterbox_clear = !image_to_flip || aspect_ratio.x1 || aspect_ratio.y1 ||
		static_cast<u32>(aspect_ratio.x2) < target_size.width || static_cast<u32>(aspect_ratio.y2) < target_size.height;

	const bool use_full_rgb_range_output = g_cfg.video.full_rgb_range_output.get();
	const bool use_calibration_pass = image_to_flip &&
		(!use_full_rgb_range_output || !rsx::fcmp(avconfig.gamma, 1.f) || avconfig.stereo_enabled);

	if (needs_letterbox_clear && !image_to_flip)
	{
		// Nothing to draw: clear on its own. Otherwise the final pass (calibration pass or upscaler draw) clears on load.
		mtl::clear_color_texture(*m_current_command_buffer, target_texture, target_size.width, target_size.height, MTL::ClearColor::Make(0., 0., 0., 1.));
	}

	if (image_to_flip)
	{
		const areai src_area = { 0, 0, s32(buffer_width), s32(buffer_height) };

		if (use_calibration_pass) [[unlikely]]
		{
			rsx::simple_array<mtl::viewable_image*> calibration_src;
			if (image_to_flip) calibration_src.push_back(image_to_flip);
			if (image_to_flip2) calibration_src.push_back(image_to_flip2);

			if (m_output_scaling == output_scaling_mode::fsr && !avconfig.stereo_enabled) // MetalFX: no stereo 3D (bilinear)
			{
				// Run upscaling pass before the rest of the output effects pipeline
				// This can be done with all upscalers but we already get bilinear upscaling for free if we just out the filters directly
				const areai dst_area = { 0, 0, aspect_ratio.width(), aspect_ratio.height() };

				for (unsigned i = 0; i < calibration_src.size(); ++i)
				{
					const rsx::flags32_t mode = (i == 0) ? UPSCALE_LEFT_VIEW : UPSCALE_RIGHT_VIEW;
					calibration_src[i] = m_upscaler->scale_output(*m_current_command_buffer, calibration_src[i], nullptr, src_area, dst_area, mode);
				}
			}

			// Letterbox clear folded into the pass (loadAction Clear) instead of a separate clear pass
			mtl::overlay_target calibration_target = target;
			calibration_target.clear_color_on_load = needs_letterbox_clear;
			calibration_target.clear_color_value = { 0.f, 0.f, 0.f, 1.f };

			mtl::get_overlay_pass<mtl::video_out_calibration_pass>()->run(
				*m_current_command_buffer, areau(aspect_ratio), calibration_target, calibration_src,
				avconfig.gamma, !use_full_rgb_range_output, avconfig.stereo_enabled);
		}
		else
		{
			// Scaled draw into the present image (Metal has no image blit). MetalFX (+ RCAS) upscales first when active.
			m_upscaler->scale_output(*m_current_command_buffer, image_to_flip, target_texture, src_area, aspect_ratio,
				UPSCALE_AND_COMMIT | UPSCALE_DEFAULT_VIEW | (needs_letterbox_clear ? UPSCALE_CLEAR_TARGET : 0));
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
				"Program cache lookup ellision: %u/%u (%u%%)\n"
				"Present pacing: %.1f Hz %s %s, guest frame %.2f ms, %u refresh(es)/frame, min duration %.2f ms",
				info.stats.framebuffer_stats.to_string(resolution_scaling_config, !backend_config.supports_hw_msaa),
				get_load(), info.stats.draw_calls, info.stats.submit_count, m_render_pass_splits, info.stats.setup_time, info.stats.vertex_upload_time,
				info.stats.textures_upload_time, info.stats.draw_exec_time, info.stats.flip_time,
				num_dirty_textures, texture_memory_size, tmp_texture_memory_size,
				num_flushes, num_misses, cache_miss_ratio, num_unavoidable, num_mispredict, num_speculate,
				num_texture_upload, num_texture_upload_miss, texture_upload_miss_ratio, texture_copies_ellided,
				vertex_cache_hit_count, info.stats.vertex_cache_request_count, vertex_cache_hit_ratio,
				program_cache_ellided, program_cache_lookups, program_cache_ellision_rate,
				1. / m_present_pacing.refresh_interval, m_present_pacing.variable_refresh ? "variable" : "fixed",
				m_present_pacing.fullscreen ? "fullscreen" : "windowed", get_guest_frame_interval() * 1000.,
				m_present_pacing.slot_refreshes, m_present_pacing.min_duration * 1000.)
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
