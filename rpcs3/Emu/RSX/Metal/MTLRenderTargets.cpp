#include "stdafx.h"
#include "MTLRenderTargets.h"
#include "MTLResourceManager.h"
#include "mtlutils/data_heap.h"

#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Common/tiled_dma_copy.hpp"
#include "Emu/Memory/vm.h"

#include <array>

namespace mtl
{
	atomic_t<u32> g_feedback_loop_pass_splits{ 0 };
	u64 g_feedback_draw_key = 0;

	u64 new_surface_content_tag()
	{
		// Surfaces can be created off the RSX thread (e.g. during flushes); keep values unique regardless
		static atomic_t<u64> s_content_tag{ 0 };
		return ++s_content_tag;
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Memory accounting
	// ---------------------------------------------------------------------------------------------------------------

	static std::array<atomic_t<u64>, VMM_ALLOCATION_POOL_COUNT> g_pool_usage{};

	u64 vmm_get_application_pool_usage(vmm_allocation_pool pool)
	{
		return g_pool_usage[pool].load();
	}

	void vmm_notify_pool_usage(vmm_allocation_pool pool, s64 delta_bytes)
	{
		if (delta_bytes >= 0)
		{
			g_pool_usage[pool] += static_cast<u64>(delta_bytes);
		}
		else
		{
			g_pool_usage[pool] -= static_cast<u64>(-delta_bytes);
		}
	}

	rsx::problem_severity vmm_determine_memory_load_severity()
	{
		const auto dev = get_current_renderer();
		if (!dev)
		{
			return rsx::problem_severity::low;
		}

		const u64 budget = dev->caps().recommended_working_set;
		if (!budget)
		{
			return rsx::problem_severity::low;
		}

		const f32 vmm_load = (static_cast<f32>(dev->allocated_bytes()) * 100.f) / static_cast<f32>(budget);
		if (vmm_load > 95.f)
		{
			// Metal starts paging / failing allocations past the recommended working set
			return rsx::problem_severity::fatal;
		}

		if (vmm_load > 90.f)
		{
			return rsx::problem_severity::severe;
		}

		if (vmm_load > 75.f)
		{
			return rsx::problem_severity::moderate;
		}

		return rsx::problem_severity::low;
	}

	pooled_image::pooled_image(const render_device& dev, const image_create_info& info, vmm_allocation_pool pool)
		: viewable_image(dev, info)
	{
		account_memory(pool);
	}

	pooled_image::~pooled_image()
	{
		if (m_accounted_size)
		{
			vmm_notify_pool_usage(m_pool, -static_cast<s64>(m_accounted_size));
			m_accounted_size = 0;
		}
	}

	void pooled_image::account_memory(vmm_allocation_pool pool)
	{
		if (m_accounted_size)
		{
			vmm_notify_pool_usage(m_pool, -static_cast<s64>(m_accounted_size));
		}

		m_pool = pool;
		m_accounted_size = value ? value->allocatedSize() : 0;
		vmm_notify_pool_usage(m_pool, static_cast<s64>(m_accounted_size));
	}

	void pooled_image::transfer_accounting(pooled_image* target)
	{
		ensure(target && !target->m_accounted_size);
		target->m_pool = m_pool;
		target->m_accounted_size = std::exchange(m_accounted_size, 0);
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Surface cache
	// ---------------------------------------------------------------------------------------------------------------

	namespace surface_cache_utils
	{
		void dispose(mtl::buffer* buf)
		{
			auto obj = mtl::disposable_t::make(buf);
			mtl::get_resource_manager()->dispose(obj);
		}
	}

	MTL::TextureUsage surface_cache_traits::get_attachment_usage(MTL::PixelFormat /*format*/, u8 /*samples*/, bool /*depth*/)
	{
		// No pass writes surfaces from compute (resolve/unresolve are fragment passes, transfers are blits), so
		// ShaderWrite is not requested: it would only disable lossless framebuffer compression.
		return MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead;
	}

	std::unique_ptr<mtl::render_target> surface_cache_traits::create_new_surface(
		mtl::command_list& /*cmd*/,
		u32 address,
		rsx::surface_color_format format,
		usz width, usz height, usz pitch,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& resolution_scaling_config)
	{
		const auto fmt = mtl::get_compatible_surface_format(format);
		const MTL::PixelFormat requested_format = fmt.first;

		u8 samples;
		rsx::surface_sample_layout sample_layout;
		if (g_cfg.video.antialiasing_level == msaa_level::_auto)
		{
			samples = get_format_sample_count(antialias);
			sample_layout = rsx::surface_sample_layout::ps3;
		}
		else
		{
			samples = 1;
			sample_layout = rsx::surface_sample_layout::null;
		}

		std::unique_ptr<mtl::render_target> rtt;
		const auto [width_, height_] = rsx::apply_resolution_scale<true>(resolution_scaling_config, static_cast<u16>(width), static_cast<u16>(height));

		image_create_info info{};
		info.type = (samples > 1) ? MTL::TextureType2DMultisample : MTL::TextureType2D;
		info.format = requested_format;
		info.width = static_cast<u32>(width_);
		info.height = static_cast<u32>(height_);
		info.samples = samples;
		info.usage = get_attachment_usage(requested_format, samples, false);
		info.storage = memory_location::device_local;
		info.format_class = RSX_FORMAT_CLASS_COLOR;

		rtt = std::make_unique<mtl::render_target>(*mtl::g_render_device, info);
		rtt->set_debug_name(fmt::format("RTV @0x%x, fmt=0x%x", address, static_cast<int>(format)));

		rtt->set_format(format);
		rtt->set_aa_mode(antialias);
		rtt->set_resolution_scaling_config(resolution_scaling_config);
		rtt->sample_layout = sample_layout;
		rtt->memory_usage_flags = rsx::surface_usage_flags::attachment;
		rtt->state_flags = rsx::surface_state_flags::erase_bkgnd;
		rtt->native_component_map = fmt.second;
		rtt->rsx_pitch = static_cast<u32>(pitch);
		rtt->native_pitch = static_cast<u32>(width) * get_format_block_size_in_bytes(format) * rtt->samples_x;
		rtt->surface_width = static_cast<u16>(width);
		rtt->surface_height = static_cast<u16>(height);
		rtt->queue_tag(address);

		rtt->add_ref();
		return rtt;
	}

	std::unique_ptr<mtl::render_target> surface_cache_traits::create_new_surface(
		mtl::command_list& /*cmd*/,
		u32 address,
		rsx::surface_depth_format2 format,
		usz width, usz height, usz pitch,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& resolution_scaling_config)
	{
		const MTL::PixelFormat requested_format = mtl::get_compatible_depth_surface_format(format);

		u8 samples;
		rsx::surface_sample_layout sample_layout;
		if (g_cfg.video.antialiasing_level == msaa_level::_auto)
		{
			samples = get_format_sample_count(antialias);
			sample_layout = rsx::surface_sample_layout::ps3;
		}
		else
		{
			samples = 1;
			sample_layout = rsx::surface_sample_layout::null;
		}

		std::unique_ptr<mtl::render_target> ds;
		const auto [width_, height_] = rsx::apply_resolution_scale<true>(resolution_scaling_config, static_cast<u16>(width), static_cast<u16>(height));

		image_create_info info{};
		info.type = (samples > 1) ? MTL::TextureType2DMultisample : MTL::TextureType2D;
		info.format = requested_format;
		info.width = static_cast<u32>(width_);
		info.height = static_cast<u32>(height_);
		info.samples = samples;
		info.usage = get_attachment_usage(requested_format, samples, true);
		info.storage = memory_location::device_local;
		info.format_class = rsx::classify_format(format);

		ds = std::make_unique<mtl::render_target>(*mtl::g_render_device, info);
		ds->set_debug_name(fmt::format("DSV @0x%x", address));

		ds->set_format(format);
		ds->set_aa_mode(antialias);
		ds->set_resolution_scaling_config(resolution_scaling_config);
		ds->sample_layout = sample_layout;
		ds->memory_usage_flags = rsx::surface_usage_flags::attachment;
		ds->state_flags = rsx::surface_state_flags::erase_bkgnd;
		ds->native_component_map = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed);
		ds->native_pitch = static_cast<u32>(width) * get_format_block_size_in_bytes(format) * ds->samples_x;
		ds->rsx_pitch = static_cast<u32>(pitch);
		ds->surface_width = static_cast<u16>(width);
		ds->surface_height = static_cast<u16>(height);
		ds->queue_tag(address);

		ds->add_ref();
		return ds;
	}

	void surface_cache_traits::clone_surface(
		mtl::command_list& cmd,
		std::unique_ptr<mtl::render_target>& sink, mtl::render_target* ref,
		u32 address, barrier_descriptor_t& prev,
		const rsx::surface_scaling_config_t& scaling_config)
	{
		const bool initialize = !sink || !sink->has_refs();
		if (sink && initialize)
		{
			prepare_for_reuse(cmd, sink.get());
		}

		if (!sink)
		{
			const auto [new_w, new_h] = rsx::apply_resolution_scale<true>(
				scaling_config,
				prev.width, prev.height,
				ref->get_surface_width<rsx::surface_metrics::pixels>(), ref->get_surface_height<rsx::surface_metrics::pixels>());

			image_create_info info = ref->info;
			info.width = new_w;
			info.height = new_h;
			info.format_class = ref->format_class();

			sink = std::make_unique<mtl::render_target>(cmd.device(), info);
		}

		if (initialize)
		{
			sink->reset();
			sink->msaa_flags = rsx::surface_state_flags::ready;
			sink->add_ref();

			sink->sample_layout = ref->sample_layout;
			sink->resolution_scaling_config = scaling_config;

			sink->set_aa_mode(ref->get_aa_mode());
			sink->format_info = ref->format_info;
			sink->memory_usage_flags = rsx::surface_usage_flags::storage;
			sink->state_flags = rsx::surface_state_flags::erase_bkgnd;
			sink->set_native_component_layout(ref->native_component_map);
			if (sink->resolve_surface)
			{
				sink->resolve_surface->set_native_component_layout(ref->native_component_map);
			}
			sink->sample_layout = ref->sample_layout;
			sink->stencil_init_flags = ref->stencil_init_flags;
			sink->native_pitch = static_cast<u32>(prev.width) * ref->get_bpp() * ref->samples_x;
			sink->rsx_pitch = ref->get_rsx_pitch();
			sink->surface_width = prev.width;
			sink->surface_height = prev.height;
			sink->queue_tag(address);
		}

		sink->on_clone_from(ref);
		sink->on_contents_changed();

		if (!sink->old_contents.empty())
		{
			// Deal with this, likely only needs to clear
			if (sink->surface_width > prev.width || sink->surface_height > prev.height)
			{
				sink->write_barrier(cmd);
			}
			else
			{
				sink->clear_rw_barrier();
			}
		}

		prev.target = sink.get();
		sink->set_old_contents_region(prev, false);
	}

	void surface_cache_traits::write_render_target_to_memory(
		mtl::command_list& cmd,
		mtl::buffer* bo,
		mtl::render_target* surface,
		u64 dst_offset_in_buffer,
		u64 src_offset_in_buffer,
		u64 max_copy_length)
	{
		surface->read_barrier(cmd);
		mtl::image* source = surface->get_surface(rsx::surface_access::transfer_read);
		const bool is_scaled = surface->width() != surface->surface_width;
		if (is_scaled)
		{
			const areai src_rect = { 0, 0, static_cast<int>(source->width()), static_cast<int>(source->height()) };
			const areai dst_rect = { 0, 0, surface->get_surface_width<rsx::surface_metrics::samples, int>(), surface->get_surface_height<rsx::surface_metrics::samples, int>() };

			auto scratch = mtl::get_typeless_helper(source->format(), source->format_class(), dst_rect.x2, dst_rect.y2, "rtt-resolve");

			if (!scratch)
			{
				rsx_log.error("Metal: rtt-resolve typeless helper for %ux%u (fmt 0x%x) refused. Skipping scaled RTT writeback; guest memory keeps stale contents.",
					static_cast<u32>(dst_rect.x2), static_cast<u32>(dst_rect.y2), static_cast<u32>(source->format()));
				return;
			}

			mtl::copy_scaled_image(cmd, source, scratch, src_rect, dst_rect, {}, true, false);

			source = scratch;
		}

		auto dest = bo;
		const auto transfer_size = surface->get_memory_range().length();
		if (transfer_size > max_copy_length || src_offset_in_buffer || surface->is_depth_surface())
		{
			auto scratch = mtl::get_scratch_buffer(cmd, transfer_size * 4);
			dest = scratch;
		}

		buffer_image_copy region{};
		region.buffer_offset = (dest == bo) ? dst_offset_in_buffer : 0;
		region.buffer_row_length = surface->rsx_pitch / surface->get_bpp();
		region.buffer_image_height = 0;
		region.aspect = source->aspect();
		region.mip_level = 0;
		region.base_layer = 0;
		region.layer_count = 1;
		region.image_offset = MTL::Origin::Make(0, 0, 0);
		region.image_extent = MTL::Size::Make(source->width(), source->height(), 1);

		// VK injected a post-transfer barrier here; Metal transfers are serialized by cmd.compute()
		image_readback_options_t options{};
		options.sync_region =
		{
			.offset = src_offset_in_buffer,
			.length = max_copy_length
		};
		mtl::copy_image_to_buffer(cmd, source, dest, region, options);

		if (dest != bo)
		{
			// Offsets come from guest addresses: any alignment
			mtl::copy_buffer_to_buffer_aligned(cmd, dest, src_offset_in_buffer, bo, dst_offset_in_buffer, max_copy_length);
		}
	}

	void surface_cache::destroy()
	{
		invalidate_all();
		invalidated_resources.clear();
	}

	u64 surface_cache::get_surface_cache_memory_quota(u64 total_device_memory)
	{
		total_device_memory /= 0x100000;
		u64 quota = 0;

		if (total_device_memory >= 2048)
		{
			quota = std::min<u64>(6144, (total_device_memory * 40) / 100);
		}
		else if (total_device_memory >= 1024)
		{
			quota = std::max<u64>(512, (total_device_memory * 30) / 100);
		}
		else if (total_device_memory >= 768)
		{
			quota = 256;
		}
		else
		{
			// Remove upto 128MB but at least aim for half of available VRAM
			quota = std::min<u64>(128, total_device_memory / 2);
		}

		return quota * 0x100000;
	}

	bool surface_cache::can_collapse_surface(const std::unique_ptr<mtl::render_target>& surface, rsx::problem_severity severity)
	{
		if (severity < rsx::problem_severity::fatal &&
			mtl::vmm_determine_memory_load_severity() < rsx::problem_severity::fatal)
		{
			// We may be able to allocate what we need.
			return true;
		}

		// Check if we need to do any allocations. Do not collapse in such a situation otherwise
		if (surface->samples() > 1 && !surface->resolve_surface)
		{
			return false;
		}

		// Resolve target does exist. Scan through the entire collapse chain
		for (auto& region : surface->old_contents)
		{
			// FIXME: This is just lazy
			auto proxy = std::unique_ptr<mtl::render_target>(mtl::as_rtt(region.source));
			const bool collapsible = can_collapse_surface(proxy, severity);
			proxy.release();

			if (!collapsible)
			{
				return false;
			}
		}

		return true;
	}

	bool surface_cache::handle_memory_pressure(mtl::command_list& cmd, rsx::problem_severity severity)
	{
		bool any_released = rsx::surface_store<surface_cache_traits>::handle_memory_pressure(cmd, severity);

		if (severity >= rsx::problem_severity::fatal)
		{
			std::vector<std::unique_ptr<mtl::viewable_image>> resolve_target_cache;
			std::vector<mtl::render_target*> deferred_spills;
			auto gc = mtl::get_resource_manager();

			// Drop MSAA resolve/unresolve caches. Only trigger when a hard sync is guaranteed to follow else it will cause even more problems!
			// 2-pass to ensure resources are available where they are most needed
			auto relieve_memory_pressure = [&](auto& list, const utils::address_range32& range)
			{
				for (auto it = list.begin_range(range); it != list.end(); ++it)
				{
					auto& rtt = it->second;
					if (!rtt->spill_request_tag || rtt->spill_request_tag < rtt->last_rw_access_tag)
					{
						// We're not going to be spilling into system RAM. If a MSAA resolve target exists, remove it to save memory.
						if (rtt->resolve_surface)
						{
							resolve_target_cache.emplace_back(std::move(rtt->resolve_surface));
							rtt->msaa_flags |= rsx::surface_state_flags::require_resolve;
							any_released |= true;
						}

						rtt->spill_request_tag = 0;
						continue;
					}

					if (rtt->resolve_surface || rtt->samples() == 1)
					{
						// Can spill immediately. Do it.
						ensure(rtt->spill(cmd, resolve_target_cache));
						any_released |= true;
						continue;
					}

					deferred_spills.push_back(rtt.get());
				}
			};

			// 1. Spill an strip any 'invalidated resources'. At this point it doesn't matter and we donate to the resolve cache which is a plus.
			for (auto& surface : invalidated_resources)
			{
				if (!surface->value && !surface->resolve_surface)
				{
					// Unspilled resources can have no value but have a resolve surface used for read
					continue;
				}

				// Only spill anything with references. Other surfaces already marked for removal should be inevitably deleted when it is time to free_invalidated
				if (surface->has_refs() && (surface->resolve_surface || surface->samples() == 1))
				{
					ensure(surface->spill(cmd, resolve_target_cache));
					any_released |= true;
				}
				else if (surface->resolve_surface)
				{
					ensure(!surface->has_refs());
					resolve_target_cache.emplace_back(std::move(surface->resolve_surface));
					surface->msaa_flags |= rsx::surface_state_flags::require_resolve;
					any_released |= true;
				}
				else if (surface->has_refs())
				{
					deferred_spills.push_back(surface.get());
				}
			}

			// 2. Scan the list and spill resources that can be spilled immediately if requested. Also gather resources from those that don't need it.
			relieve_memory_pressure(m_render_targets_storage, m_render_targets_memory_range);
			relieve_memory_pressure(m_depth_stencil_storage, m_depth_stencil_memory_range);

			// 3. Write to system heap everything marked to spill
			for (auto& surface : deferred_spills)
			{
				any_released |= surface->spill(cmd, resolve_target_cache);
			}

			// 4. Cleanup; removes all the resources used up here that are no longer needed for the moment
			for (auto& data : resolve_target_cache)
			{
				gc->dispose(data);
			}
		}

		return any_released;
	}

	void surface_cache::trim(mtl::command_list& cmd, rsx::problem_severity memory_pressure)
	{
		run_cleanup_internal(cmd, rsx::problem_severity::moderate, 300, [](mtl::command_list& cmd)
		{
			if (!cmd.is_recording())
			{
				cmd.begin();
			}
		});

		const u64 last_finished_frame = mtl::get_last_completed_frame_id();
		for (auto& rtt : invalidated_resources)
		{
			ensure(rtt->frame_tag != 0);

			if (rtt->has_refs())
			{
				// Actively in use, likely for a reading pass.
				// Call handle_memory_pressure before calling this method.
				continue;
			}

			if (rtt->frame_tag >= last_finished_frame)
			{
				// RTT itself still in use by the frame.
				continue;
			}

			if (!rtt->old_contents.empty())
			{
				rtt->clear_rw_barrier();
			}

			if (rtt->resolve_surface && memory_pressure >= rsx::problem_severity::moderate)
			{
				// We do not need to keep resolve targets around.
				// TODO: We should surrender this to an image cache immediately for reuse.
				mtl::get_resource_manager()->dispose(rtt->resolve_surface);
			}

			int threshold = 8;
			switch (memory_pressure)
			{
			case rsx::problem_severity::low:
				threshold = 2;
				break;
			case rsx::problem_severity::moderate:
				threshold = 1;
				break;
			case rsx::problem_severity::severe:
			case rsx::problem_severity::fatal:
				// We're almost dead anyway. Remove forcefully.
				threshold = -1;
				break;
			default:
				fmt::throw_exception("Unreachable");
			}

			if (threshold < 0 || (rtt->unused_check_count() >= threshold))
			{
				mtl::get_resource_manager()->dispose(rtt);
				ensure(!rtt);
			}
		}

		invalidated_resources.remove_if(
			[](auto& rtt) { return !rtt; }
		);
	}

	bool surface_cache::is_overallocated()
	{
		const auto surface_cache_vram_load = vmm_get_application_pool_usage(VMM_ALLOCATION_POOL_SURFACE_CACHE);
		const auto surface_cache_allocation_quota = get_surface_cache_memory_quota(get_current_renderer()->caps().recommended_working_set);
		return (surface_cache_vram_load > surface_cache_allocation_quota);
	}

	bool surface_cache::spill_unused_memory()
	{
		// Determine how much memory we need to save to system RAM if any
		const u64 current_surface_cache_memory = vmm_get_application_pool_usage(VMM_ALLOCATION_POOL_SURFACE_CACHE);
		const u64 total_device_memory = mtl::get_current_renderer()->caps().recommended_working_set;
		const u64 target_memory = get_surface_cache_memory_quota(total_device_memory);

		rsx_log.warning("Surface cache memory usage is %lluM", current_surface_cache_memory / 0x100000);
		if (current_surface_cache_memory < target_memory)
		{
			rsx_log.warning("Surface cache memory usage is very low. Will not spill contents to RAM");
			return false;
		}

		// Very slow, but should only be called when the situation is dire
		std::vector<render_target*> sorted_list;
		sorted_list.reserve(1024);

		auto process_list_function = [&](auto& list, const utils::address_range32& range)
		{
			for (auto it = list.begin_range(range); it != list.end(); ++it)
			{
				// NOTE: Check if memory is available instead of value in case we ran out of memory during unspill
				auto& surface = it->second;
				if (surface->value && !surface->is_bound)
				{
					sorted_list.push_back(surface.get());
				}
			}
		};

		process_list_function(m_render_targets_storage, m_render_targets_memory_range);
		process_list_function(m_depth_stencil_storage, m_depth_stencil_memory_range);

		std::sort(sorted_list.begin(), sorted_list.end(), FN(x->last_rw_access_tag < y->last_rw_access_tag));

		// Remove upto target_memory bytes from VRAM
		u64 bytes_spilled = 0;
		const u64 bytes_to_remove = current_surface_cache_memory - target_memory;
		const u64 spill_time = rsx::get_shared_tag();

		for (auto& surface : sorted_list)
		{
			bytes_spilled += surface->value->allocatedSize();
			surface->spill_request_tag = spill_time;

			if (bytes_spilled >= bytes_to_remove)
			{
				break;
			}
		}

		rsx_log.warning("Surface cache will attempt to spill %llu bytes.", bytes_spilled);
		return (bytes_spilled > 0);
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Surfaces
	// ---------------------------------------------------------------------------------------------------------------

	drawable_surface_t::drawable_surface_t(const render_device& dev, const image_create_info& info)
		: pooled_image(dev, info, VMM_ALLOCATION_POOL_SURFACE_CACHE)
	{
	}

	drawable_surface_t::~drawable_surface_t()
	{
		// VK removed cached framebuffers referencing this image here. Metal render pass descriptors are built per pass,
		// so there is nothing to invalidate. Destruction of the texture itself must already be GC-deferred by the owner.
	}

	drawable_surface_t* drawable_surface_t::clone()
	{
		// Destructive cloning. The clone grabs the GPU objects owned by this instance.
		// This instance can be rebuilt in-place by calling create_impl() which will create a duplicate now owned by this.
		auto result = new drawable_surface_t();
		result->info = this->info;
		result->value = this->value;
		result->native_component_map = this->native_component_map;
		result->views = std::move(this->views);
		transfer_accounting(result);

		this->views.clear();
		this->value = nullptr;
		return result;
	}

	// Get the linear resolve target bound to this surface. Initialize if none exists
	mtl::viewable_image* render_target::get_resolve_target_safe(mtl::command_list& /*cmd*/)
	{
		if (!resolve_surface)
		{
			// Create a resolve surface
			const auto resolve_w = width() * samples_x;
			const auto resolve_h = height() * samples_y;

			image_create_info create_info{};
			create_info.type = MTL::TextureType2D;
			create_info.format = format();
			create_info.width = resolve_w;
			create_info.height = resolve_h;
			create_info.samples = 1;
			// Resolve/unresolve are fragment passes (MTLResolveHelper): no ShaderWrite needed
			create_info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
			create_info.storage = memory_location::device_local;
			create_info.format_class = format_class();

			resolve_surface.reset(new mtl::drawable_surface_t(*g_render_device, create_info));
			resolve_surface->native_component_map = native_component_map;
		}

		return resolve_surface.get();
	}

	// Resolve the planar MSAA data into a linear block
	void render_target::resolve(mtl::command_list& cmd)
	{
		// No layouts or barriers: the resolve pass/dispatch is ordered after every previous command by cmd.
		mtl::resolve_image(cmd, resolve_surface.get(), this);

		msaa_flags &= ~(rsx::surface_state_flags::require_resolve);
	}

	// Unresolve the linear data into planar MSAA data
	void render_target::unresolve(mtl::command_list& cmd)
	{
		ensure(!(msaa_flags & rsx::surface_state_flags::require_resolve));

		mtl::unresolve_image(cmd, this, resolve_surface.get());
		on_contents_changed();

		msaa_flags &= ~(rsx::surface_state_flags::require_unresolve);
	}

	// Default-initialize memory without loading
	void render_target::clear_memory(mtl::command_list& cmd, mtl::image* surface)
	{
		image_clear_value clear{};
		if (surface->aspect() & aspect_color)
		{
			clear.color = { 0.f, 0.f, 0.f, 1.f };
		}
		else
		{
			clear.depth = 1.f;
			clear.stencil = 255;
		}

		mtl::clear_image(cmd, surface, clear, 0, 1);
		on_contents_changed();

		if (surface == this)
		{
			state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
		}
	}

	std::vector<buffer_image_copy> render_target::build_spill_transfer_descriptors(mtl::image* target)
	{
		std::vector<buffer_image_copy> result;
		result.reserve(2);

		result.push_back({});
		auto& rgn = result.back();
		rgn.image_extent = MTL::Size::Make(target->width(), target->height(), 1);
		rgn.aspect = target->aspect();
		rgn.layer_count = 1;

		if (aspect() == aspect_depth_stencil)
		{
			// One plane per transfer: depth (4 bytes float) followed by stencil (1 byte)
			result.push_back(rgn);
			result.front().aspect = aspect_depth;
			result.back().aspect = aspect_stencil;
			// Metal: texture <-> buffer copy offsets must be 16-byte aligned
			result.back().buffer_offset = utils::align<u64>(u64{ target->width() } * target->height() * 4, 16);
		}

		return result;
	}

	bool render_target::spill(mtl::command_list& cmd, std::vector<std::unique_ptr<mtl::viewable_image>>& resolve_cache)
	{
		u64 element_size;
		switch (const auto fmt = format())
		{
		case MTL::PixelFormatDepth32Float:
			element_size = 4;
			break;
		case MTL::PixelFormatDepth32Float_Stencil8:
		case MTL::PixelFormatDepth24Unorm_Stencil8:
			element_size = 5;
			break;
		default:
			element_size = get_format_block_info(fmt).bytes_per_block;
			break;
		}

		mtl::viewable_image* src = nullptr;
		if (samples() == 1) [[likely]]
		{
			ensure(value);
			src = this;
		}
		else if (resolve_surface)
		{
			src = resolve_surface.get();
		}
		else
		{
			const auto transfer_w = width() * samples_x;
			const auto transfer_h = height() * samples_y;

			for (auto& surface : resolve_cache)
			{
				if (surface->format() == format() &&
					surface->width() == transfer_w &&
					surface->height() == transfer_h)
				{
					src = surface.get();
					break;
				}
			}

			if (!src)
			{
				if (vmm_determine_memory_load_severity() <= rsx::problem_severity::moderate)
				{
					// We have some freedom to allocate something. Add to the shared cache
					src = get_resolve_target_safe(cmd);
				}
				else
				{
					// TODO: Spill to DMA buf
					// For now, just skip this one if we don't have the capacity for it
					rsx_log.warning("Could not spill memory due to resolve failure. Will ignore spilling for the moment.");
					return false;
				}
			}

			msaa_flags |= rsx::surface_state_flags::require_resolve;
		}

		// If a resolve is requested, move data to the target
		if (msaa_flags & rsx::surface_state_flags::require_resolve)
		{
			ensure(samples() > 1);
			const bool borrowed = [&]()
			{
				if (src != resolve_surface.get())
				{
					ensure(!resolve_surface);
					resolve_surface.reset(src);
					return true;
				}

				return false;
			}();

			resolve(cmd);

			if (borrowed)
			{
				resolve_surface.release();
			}
		}

		const auto pdev = mtl::get_current_renderer();
		const auto alloc_size = element_size * src->width() * src->height() + 16; // + alignment padding of the stencil plane

		m_spilled_mem = std::make_unique<mtl::buffer>(*pdev, alloc_size, memory_location::host_visible, "spilled surface");

		const auto regions = build_spill_transfer_descriptors(src);
		for (const auto& region : regions)
		{
			mtl::copy_image_to_buffer_raw(cmd, src, m_spilled_mem.get(), region);
		}

		// Destroy this object through a cloned object
		auto obj = std::unique_ptr<viewable_image>(clone());
		mtl::get_resource_manager()->dispose(obj);

		if (resolve_surface)
		{
			// Just add to the resolve cache and move on
			resolve_cache.emplace_back(std::move(resolve_surface));
		}

		ensure(!value && views.empty() && !resolve_surface);
		spill_request_tag = 0ull;
		return true;
	}

	void render_target::unspill(mtl::command_list& cmd)
	{
		// Recreate the image
		const auto pdev = mtl::get_current_renderer();
		create_impl(*pdev, info);
		account_memory(VMM_ALLOCATION_POOL_SURFACE_CACHE);
		on_contents_changed();

		// Load image from host-visible buffer
		ensure(m_spilled_mem);

		// Data transfer can be skipped if an erase command is being served
		if (!(state_flags & rsx::surface_state_flags::erase_bkgnd))
		{
			// Warn. Ideally this should never happen if you have enough resources
			rsx_log.warning("[PERFORMANCE WARNING] Loading spilled memory back to the GPU. You may want to lower your resolution scaling.");

			mtl::image* dst = (samples() > 1) ? get_resolve_target_safe(cmd) : this;
			const auto regions = build_spill_transfer_descriptors(dst);

			for (const auto& region : regions)
			{
				mtl::copy_buffer_to_image_raw(cmd, m_spilled_mem.get(), dst, region);
			}

			if (samples() > 1)
			{
				msaa_flags &= ~rsx::surface_state_flags::require_resolve;
				msaa_flags |= rsx::surface_state_flags::require_unresolve;
			}
		}

		// Delete host-visible buffer
		mtl::get_resource_manager()->dispose(m_spilled_mem);
	}

	// Load memory from cell and use to initialize the surface
	void render_target::load_memory(mtl::command_list& cmd)
	{
		auto& upload_heap = *mtl::get_upload_heap();
		const bool is_swizzled = (raster_type == rsx::surface_raster_type::swizzle);

		rsx::subresource_layout subres{};
		subres.width_in_block = subres.width_in_texel = surface_width * samples_x;
		subres.height_in_block = subres.height_in_texel = surface_height * samples_y;
		subres.pitch_in_block = rsx_pitch / get_bpp();
		subres.depth = 1;
		subres.data = { vm::get_super_ptr<const std::byte>(base_addr), static_cast<std::span<const std::byte>::size_type>(rsx_pitch * surface_height * samples_y) };

		const auto range = get_memory_range();
		rsx::flags32_t upload_flags = upload_contents_inline;
		u32 heap_align = rsx_pitch;
		on_contents_changed();

#if DEBUG_DMA_TILING
		std::vector<u8> ext_data;
#endif

		if (auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(range))
		{
#if DEBUG_DMA_TILING
			auto real_data = vm::get_super_ptr<u8>(range.start);
			ext_data.resize(tiled_region.tile->size);
			auto detile_func = get_bpp() == 4
				? rsx::detile_texel_data32
				: rsx::detile_texel_data16;

			detile_func(
				ext_data.data(),
				real_data,
				tiled_region.base_address,
				range.start - tiled_region.base_address,
				tiled_region.tile->size,
				tiled_region.tile->bank,
				tiled_region.tile->pitch,
				subres.width_in_block,
				subres.height_in_block
			);
			subres.data = std::span(ext_data);
			upload_flags |= source_is_userptr;
#else
			const auto [scratch_buf, linear_data_scratch_offset] = mtl::detile_memory_block(cmd, tiled_region, range, subres.width_in_block, subres.height_in_block, get_bpp());

			// FIXME: !!EVIL!!
			subres.data = { scratch_buf, linear_data_scratch_offset };
			subres.pitch_in_block = subres.width_in_block;
			upload_flags |= source_is_gpu_resident;
			heap_align = subres.width_in_block * get_bpp();
#endif
		}

		if (resolution_scaling_config.scale_percent == 100 && spp == 1) [[likely]]
		{
			mtl::upload_image(cmd, this, { subres }, get_gcm_format(), is_swizzled, 1, aspect(), upload_heap, heap_align, upload_flags);
		}
		else
		{
			mtl::image* content = nullptr;
			mtl::image* final_dst = (samples() > 1) ? get_resolve_target_safe(cmd) : this;

			if (final_dst->width() == subres.width_in_block && final_dst->height() == subres.height_in_block)
			{
				// Possible if MSAA is enabled with 100% resolution scale or
				// surface dimensions are less than resolution scale threshold and no MSAA.
				// Writethrough.
				content = final_dst;
			}
			else
			{
				content = mtl::get_typeless_helper(format(), format_class(), subres.width_in_block, subres.height_in_block, "rtt-upload");
			}

			if (!content)
			{
				return;
			}

			// Load Cell data into temp buffer
			mtl::upload_image(cmd, content, { subres }, get_gcm_format(), is_swizzled, 1, aspect(), upload_heap, heap_align, upload_flags);

			// Write into final image
			if (content != final_dst)
			{
				mtl::copy_scaled_image(cmd, content, final_dst,
					areai{ 0, 0, subres.width_in_block, subres.height_in_block },
					areai{ 0, 0, static_cast<s32>(final_dst->width()), static_cast<s32>(final_dst->height()) },
					{}, true, aspect() == aspect_color);
			}

			if (samples() > 1)
			{
				// Trigger unresolve
				msaa_flags = rsx::surface_state_flags::require_unresolve;
			}
		}

		state_flags &= ~(rsx::surface_state_flags::erase_bkgnd | rsx::surface_state_flags::force_data_load);
	}

	void render_target::initialize_memory(mtl::command_list& cmd, rsx::surface_access access)
	{
		const bool read_buffers_config = is_depth_surface() ?
			!!g_cfg.video.read_depth_buffer :
			!!g_cfg.video.read_color_buffers;

		const bool should_read_buffers = (state_flags & rsx::surface_state_flags::force_data_load) || read_buffers_config;

		if (!should_read_buffers)
		{
			clear_memory(cmd, this);

			if (samples() > 1 && access.is_transfer_or_read())
			{
				// Only clear the resolve surface if reading from it, otherwise it's a waste
				clear_memory(cmd, get_resolve_target_safe(cmd));
			}

			msaa_flags = rsx::surface_state_flags::ready;
		}
		else
		{
			load_memory(cmd);
		}
	}

	mtl::viewable_image* render_target::get_surface(rsx::surface_access access_type)
	{
		last_rw_access_tag = rsx::get_shared_tag();

		if (samples() == 1 || !access_type.is_transfer())
		{
			return this;
		}

		// A read barrier should have been called before this!
		ensure(resolve_surface); // "Read access without explicit barrier"
		ensure(!(msaa_flags & rsx::surface_state_flags::require_resolve));
		return resolve_surface.get();
	}

	bool render_target::is_depth_surface() const
	{
		return !!(aspect() & aspect_depth);
	}

	bool render_target::matches_dimensions(u16 _width, u16 _height) const
	{
		// Use forward scaling to account for rounding and clamping errors
		const auto [scaled_w, scaled_h] = rsx::apply_resolution_scale<true>(resolution_scaling_config, _width, _height);
		return (scaled_w == width()) && (scaled_h == height());
	}

	void render_target::texture_barrier(mtl::command_list& cmd)
	{
		const auto is_framebuffer_read_only = is_depth_surface() && !rsx::method_registers.depth_write_enabled();

		if (m_cyclic_ref_tracker.can_skip() && is_framebuffer_read_only)
		{
			// If we have back-to-back depth-read barriers, skip subsequent ones
			// If an actual write is happening, this flag will be automatically reset
			return;
		}

		// Apple GPUs cannot wait on attachment writes inside a render pass. Split the pass instead: the next encoder
		// begins with a barrier on all previously encoded work (see mtl::command_list). Writes of passes that already
		// ended are in memory, so only writes by the open pass (marked by the renderer) need the split, unless they
		// come from the feedback streak the current draw belongs to.
		if (cmd.is_render_pass_open() && !feedback_read_in_pass_allowed(cmd.open_pass_serial(), g_feedback_draw_key))
		{
			cmd.end_render_pass();
			g_feedback_loop_pass_splits++;
			count_feedback_split(pass_split_reason::read_after_write);
		}

		m_cyclic_ref_tracker.on_insert_texture_barrier();

		if (is_framebuffer_read_only)
		{
			m_cyclic_ref_tracker.allow_skip();
		}
	}

	void render_target::post_texture_barrier(mtl::command_list& cmd)
	{
		// This is a fall-out barrier after a cyclic ref when the same surface is still bound.
		// In this case, we're just checking that the previous read completes before the next write.
		const bool is_framebuffer_read_only = is_depth_surface() && !rsx::method_registers.depth_write_enabled();
		if (m_cyclic_ref_tracker.can_skip() && is_framebuffer_read_only)
		{
			// Barrier ellided if triggered by a chain of cyclic references with no actual writes
			m_cyclic_ref_tracker.reset();
			return;
		}

		// Reads by previous draws of this pass must complete before the next attachment write. On a tile-based GPU they
		// do: the write reaches memory when the tile is stored, after every earlier draw of that tile has been shaded.
		// Strict Rendering Mode keeps the explicit split.
		if (cmd.is_render_pass_open() && g_cfg.video.strict_rendering_mode)
		{
			cmd.end_render_pass();
			g_feedback_loop_pass_splits++;
			count_feedback_split(pass_split_reason::write_after_read);
		}

		m_cyclic_ref_tracker.reset();
	}

	void render_target::reset_surface_counters()
	{
		frame_tag = 0;
		m_cyclic_ref_tracker.reset();
	}

	image_view* render_target::get_view(const rsx::texture_channel_remap_t& remap, u32 mask)
	{
		if (remap.encoded == MTL_REMAP_VIEW_MULTISAMPLED)
		{
			// Special remap flag, intercept here
			return mtl::viewable_image::get_view(remap.with_encoding(MTL_REMAP_IDENTITY), mask);
		}

		return mtl::viewable_image::get_view(remap, mask);
	}

	void render_target::memory_barrier(mtl::command_list& cmd, rsx::surface_access access)
	{
		if (access == rsx::surface_access::gpu_reference)
		{
			// This barrier only requires that an object is made available for GPU usage.
			if (!value)
			{
				unspill(cmd);
			}

			spill_request_tag = 0;
			return;
		}

		const bool is_depth = is_depth_surface();
		const bool read_buffers_config = is_depth ? !!g_cfg.video.read_depth_buffer : !!g_cfg.video.read_color_buffers;
		const bool should_read_buffers = (state_flags & rsx::surface_state_flags::force_data_load) || read_buffers_config;

		if (should_read_buffers)
		{
			// TODO: Decide what to do when memory loads are disabled but the underlying has memory changed
			// NOTE: Assume test() is expensive when in a pinch
			if (last_use_tag && state_flags == rsx::surface_state_flags::ready && !test())
			{
				// TODO: Figure out why merely returning and failing the test does not work when reading (TLoU)
				// The result should have been the same either way
				state_flags |= rsx::surface_state_flags::erase_bkgnd;
			}
		}

		// Unspill here, because erase flag may have been set above.
		if (!value)
		{
			unspill(cmd);
		}

		if (access == rsx::surface_access::shader_write && m_cyclic_ref_tracker.is_enabled())
		{
			// VK checked for the feedback-loop layout here; Metal has no layouts, the tracker alone carries the state.
			// Flag draw barrier observed
			m_cyclic_ref_tracker.on_insert_draw_barrier();

			// Check if we've had more draws than barriers so far (fall-out condition)
			if (m_cyclic_ref_tracker.requires_post_loop_barrier())
			{
				post_texture_barrier(cmd);
			}
		}

		if (old_contents.empty()) [[likely]]
		{
			if (state_flags & rsx::surface_state_flags::erase_bkgnd)
			{
				// NOTE: This step CAN introduce MSAA flags!
				initialize_memory(cmd, access);

				ensure(state_flags == rsx::surface_state_flags::ready);
				on_write(rsx::get_shared_tag(), static_cast<rsx::surface_state_flags>(msaa_flags));
			}

			if (msaa_flags & rsx::surface_state_flags::require_resolve)
			{
				if (access.is_transfer())
				{
					// Only do this step when read access is required
					get_resolve_target_safe(cmd);
					resolve(cmd);
				}
			}
			else if (msaa_flags & rsx::surface_state_flags::require_unresolve)
			{
				if (access == rsx::surface_access::shader_write)
				{
					// Only do this step when it is needed to start rendering
					ensure(resolve_surface);
					unresolve(cmd);
				}
			}

			return;
		}

		// Memory transfers
		mtl::image* target_image = (samples() > 1) ? get_resolve_target_safe(cmd) : this;
		mtl::blitter hw_blitter;
		const auto dst_bpp = get_bpp();

		unsigned first = prepare_rw_barrier_for_transfer(this);
		const bool accept_all = (last_use_tag && test());
		bool optimize_copy = true;
		u64  newest_tag = 0;

		for (auto i = first; i < old_contents.size(); ++i)
		{
			auto& section = old_contents[i];
			auto src_texture = static_cast<mtl::render_target*>(section.source);
			src_texture->memory_barrier(cmd, rsx::surface_access::transfer_read);

			if (!accept_all && !src_texture->test()) [[likely]]
			{
				// If this surface is intact, accept all incoming data as it is guaranteed to be safe
				// If this surface has not been initialized or is dirty, do not add more dirty data to it
				continue;
			}

			const auto src_bpp = src_texture->get_bpp();
			rsx::typeless_xfer typeless_info{};

			if (src_texture->aspect() != aspect() ||
				!formats_are_bitcast_compatible(this, src_texture))
			{
				typeless_info.src_is_typeless = true;
				typeless_info.src_context = rsx::texture_upload_context::framebuffer_storage;
				typeless_info.src_native_format_override = static_cast<u32>(info.format);
				typeless_info.src_gcm_format = src_texture->get_gcm_format();
				typeless_info.src_scaling_hint = f32(src_bpp) / dst_bpp;
			}

			section.init_transfer(this);
			auto src_area = section.src_rect();
			auto dst_area = section.dst_rect();

			if (g_cfg.video.antialiasing_level != msaa_level::none)
			{
				src_texture->transform_pixels_to_samples(src_area);
				this->transform_pixels_to_samples(dst_area);
			}

			bool memory_load = true;
			if (dst_area.x1 == 0 && dst_area.y1 == 0 &&
				unsigned(dst_area.x2) == target_image->width() && unsigned(dst_area.y2) == target_image->height())
			{
				// Skip a bunch of useless work
				state_flags &= ~(rsx::surface_state_flags::erase_bkgnd);
				msaa_flags = rsx::surface_state_flags::ready;

				memory_load = false;
				stencil_init_flags = src_texture->stencil_init_flags;
			}
			else if (state_flags & rsx::surface_state_flags::erase_bkgnd)
			{
				// Might introduce MSAA flags
				initialize_memory(cmd, rsx::surface_access::memory_write);
				ensure(state_flags == rsx::surface_state_flags::ready);
			}

			if (msaa_flags & rsx::surface_state_flags::require_resolve)
			{
				// Need to forward resolve this
				resolve(cmd);
			}

			if (samples() > 1)
			{
				// Ensure a writable surface exists for this surface
				get_resolve_target_safe(cmd);
			}

			if (src_texture->samples() > 1)
			{
				// Ensure a readable surface exists for the source
				src_texture->get_resolve_target_safe(cmd);
			}

			hw_blitter.scale_image(
				cmd,
				src_texture->get_surface(rsx::surface_access::transfer_read),
				this->get_surface(rsx::surface_access::transfer_write),
				src_area,
				dst_area,
				/*linear?*/false, typeless_info);

			optimize_copy = optimize_copy && !memory_load;
			newest_tag = src_texture->last_use_tag;
		}

		if (!newest_tag) [[unlikely]]
		{
			// Underlying memory has been modified and we could not find valid data to fill it
			clear_rw_barrier();

			state_flags |= rsx::surface_state_flags::erase_bkgnd;
			initialize_memory(cmd, access);
			ensure(state_flags == rsx::surface_state_flags::ready);
		}

		// The inherited data was written above. on_write_copy() may set last_use_tag to an older value (the newest
		// source's), so only the content tag tells copies of this surface that its contents changed.
		on_contents_changed();

		// NOTE: Optimize flag relates to stencil resolve/unresolve for NVIDIA.
		on_write_copy(newest_tag, optimize_copy);

		if (access == rsx::surface_access::shader_write && samples() > 1)
		{
			// Write barrier, must initialize
			unresolve(cmd);
		}
	}
}
