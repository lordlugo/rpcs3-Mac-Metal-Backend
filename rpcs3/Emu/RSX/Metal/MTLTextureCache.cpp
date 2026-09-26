#include "stdafx.h"
#include "MTLTextureCache.h"
#include "MTLCompute.h"
#include "MTLGSRender.h"
#include "MTLCommandStream.h"

#include "mtlutils/data_heap.h"
#include "Emu/Memory/vm.h"
#include "Emu/system_config.h"

#include "util/asm.hpp"

#include <unordered_set>

namespace mtl
{
	// ---------------------------------------------------------------------------------------------------------------
	// DMA fence
	// ---------------------------------------------------------------------------------------------------------------

	dma_fence_t::dma_fence_t(const mtl::command_list& cmd_)
		: cmd(&cmd_), timeline(cmd_.get_fence().owner)
	{
		ensure(timeline);
		recorded_after = timeline->last_signaled_value();
	}

	bool dma_fence_t::wait(u64 timeout_us) const
	{
		// Every submission of `cmd` after the transfer was recorded signals a timeline value > recorded_after.
		if (const u64 pending_value = cmd->get_fence().value; pending_value > recorded_after)
		{
			// Submitted and (possibly) still in flight
			return timeline->wait(pending_value, timeout_us);
		}

		// Not pending: either the carrying submission already retired, or the command list was never submitted.
		// A retired submission implies the timeline completed past the recording point.
		if (timeline->completed_value() <= recorded_after)
		{
			if (const u64 last = timeline->last_signaled_value(); last > recorded_after)
			{
				// Some submission after the recording point is still executing; conservatively wait for it
				return timeline->wait(last, timeout_us);
			}

			rsx_log.error("[Metal] DMA fence wait on a transfer whose command list was never submitted!");
			return false;
		}

		return true;
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Pool helpers
	// ---------------------------------------------------------------------------------------------------------------

	static u64 hash_image_properties(MTL::PixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTL::TextureType type)
	{
		/**
		* Key layout:
		* 00-10: Format  (Max 1023)  <- MTLPixelFormat values exceed 255
		* 10-26: Width   (Max 64K)
		* 26-42: Height  (Max 64K)
		* 42-52: Depth   (Max 1023)
		* 52-58: Mipmaps (Max 63)
		* 58-62: Type    (Max 15)
		*/
		ensure(static_cast<u64>(format) < 0x400);
		return (static_cast<u64>(format) & 0x3FF) |
			(static_cast<u64>(w) << 10) |
			(static_cast<u64>(h) << 26) |
			((static_cast<u64>(d) & 0x3FF) << 42) |
			((static_cast<u64>(mipmaps) & 0x3F) << 52) |
			((static_cast<u64>(type) & 0xF) << 58);
	}

	// `shader_read_section`: image of a texture_upload_context::shader_read section. MTLGSRender::load_texture_env may
	// sample those through an snorm/sRGB view (image_view::as, hardware texel remapping); no other image of the
	// texture cache is ever viewed with another format.
	static MTL::TextureUsage get_texture_cache_usage(MTL::PixelFormat format, bool shader_read_section)
	{
		// Render target usage lets the image be cleared (loadAction=Clear) and be the destination of scaled blits
		MTL::TextureUsage usage = MTL::TextureUsageShaderRead;
		if (format_is_renderable(format))
		{
			usage |= MTL::TextureUsageRenderTarget;
		}

		// An snorm view changes the component type and needs MTLTextureUsagePixelFormatView, which disables lossless
		// compression, so only formats with an snorm twin get it. sRGB views need no flag. Decided independently of
		// texture_create_flags::mutable_format: "Disable Hardware ColorSpace Remapping" can be toggled at runtime while
		// sections and pooled images live on.
		if (shader_read_section && get_compatible_snorm_format(format) != MTL::PixelFormatInvalid)
		{
			usage |= MTL::TextureUsagePixelFormatView;
		}

		return usage;
	}

	texture_cache::cached_image_reference_t::cached_image_reference_t(texture_cache* parent, std::unique_ptr<mtl::viewable_image>& previous)
	{
		ensure(previous);

		this->parent = parent;
		this->data = std::move(previous);
	}

	texture_cache::cached_image_reference_t::~cached_image_reference_t()
	{
		// No layout information to erase on Metal. Move this object to the cached image pool
		const auto key = hash_image_properties(data->format(), static_cast<u16>(data->width()), static_cast<u16>(data->height()), static_cast<u16>(data->depth()), static_cast<u16>(data->mipmaps()), data->type());
		std::lock_guard lock(parent->m_cached_pool_lock);

		if (!parent->m_cache_is_exiting)
		{
			parent->m_cached_memory_size += data->value->allocatedSize();
			parent->m_cached_images.emplace_front(key, data);
		}
		else
		{
			// Destroy if the cache is closed. The GPU is done with this resource anyway.
			data.reset();
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Sections
	// ---------------------------------------------------------------------------------------------------------------

	void cached_texture_section::dma_transfer(mtl::command_list& cmd, mtl::image* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch)
	{
		ensure(src->samples() == 1);

		if (!m_device)
		{
			m_device = &cmd.device();
		}

		if (dma_fence)
		{
			// NOTE: This can be reached if previously synchronized, or a special path happens.
			// If a hard flush occurred while this surface was flush_always the cache would have reset its protection afterwards.
			// DMA resource would still be present but already used to flush previously.
			dma_fence.reset();
		}

		if (cmd.is_render_pass_open())
		{
			cmd.end_render_pass();
		}

		const auto internal_bpp = mtl::get_format_texel_width(src->format());
		const auto transfer_width = static_cast<u32>(src_area.width());
		const auto transfer_height = static_cast<u32>(src_area.height());
		real_pitch = internal_bpp * transfer_width;
		rsx_pitch = pitch;

		const bool require_format_conversion = !!(src->aspect() & aspect_stencil) || src->format() == MTL::PixelFormatDepth32Float;
		const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(valid_range);
		const bool require_tiling = !!tiled_region;
		const bool require_gpu_transform = require_format_conversion || pack_unpack_swap_bytes || require_tiling;

		auto dma_sync_region = valid_range;
		dma_mapping_handle dma_mapping = { 0, nullptr };

		auto dma_sync = [&](bool load, bool force = false)
		{
			if (dma_mapping.second && !force)
			{
				return;
			}

			dma_mapping = mtl::map_dma(dma_sync_region.start, dma_sync_region.length());
			if (load)
			{
				mtl::load_dma(dma_sync_region.start, dma_sync_region.length());
			}
		};

		if (require_gpu_transform)
		{
			const auto transfer_pitch = real_pitch;
			const auto task_length = transfer_pitch * src_area.height();
			auto working_buffer_length = calculate_working_buffer_size(task_length, src->aspect());

			// Metal: the tiled output block (and the kernel binding at its start) is kept 256-byte aligned
			const u32 tiled_output_offset = utils::align<u32>(task_length, 256);

#if !DEBUG_DMA_TILING
			if (require_tiling)
			{
				// Safety padding (+ alignment of the tiled output block)
				working_buffer_length += tiled_region.tile->size + 256;

				// Calculate actual working section for the memory op
				dma_sync_region = tiled_region.tile_align(dma_sync_region);
			}
#endif
			u32 result_offset = 0;
			auto working_buffer = mtl::get_scratch_buffer(cmd, working_buffer_length);

			buffer_image_copy region{};
			region.aspect = src->aspect();
			region.mip_level = 0;
			region.base_layer = 0;
			region.layer_count = 1;
			region.image_offset = MTL::Origin::Make(static_cast<NS::UInteger>(src_area.x1), static_cast<NS::UInteger>(src_area.y1), 0);
			region.image_extent = MTL::Size::Make(transfer_width, transfer_height, 1);

			image_readback_options_t xfer_options{};
			xfer_options.swap_bytes = require_format_conversion && pack_unpack_swap_bytes;
			mtl::copy_image_to_buffer(cmd, src, working_buffer, region, xfer_options);

			// NOTE: For depth/stencil formats, copying to buffer and byteswap are combined into one step above
			if (pack_unpack_swap_bytes && !require_format_conversion)
			{
				const auto texel_layout = mtl::get_format_element_size(src->format());
				const auto elem_size = texel_layout.first;
				mtl::cs_shuffle_base* shuffle_kernel;

				if (elem_size == 2)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_16>();
				}
				else if (elem_size == 4)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32>();
				}
				else
				{
					ensure(get_context() == rsx::texture_upload_context::dma);
					shuffle_kernel = nullptr;
				}

				if (shuffle_kernel)
				{
					shuffle_kernel->run(cmd, working_buffer, task_length);
				}
			}

			if (require_tiling)
			{
#if !DEBUG_DMA_TILING
				// We don't need to calibrate write if two conditions are met:
				// 1. The start offset of our 2D region is a multiple of 64 lines
				// 2. We use the whole pitch.
				// If these conditions are not met, we need to upload the entire tile (or at least the affected tiles wholly)

				// FIXME: There is a 3rd condition - write onto already-persisted range. e.g One transfer copies half the image then the other half is copied later.
				// We don't need to load again for the second copy in that scenario.

				if (valid_range.start != dma_sync_region.start || real_pitch != tiled_region.tile->pitch)
				{
					// Tile indices run to the end of the row (full pitch).
					// Tiles address outside their 64x64 area too, so we need to actually load the whole thing and "fill in" missing blocks.
					// Visualizing "hot" pixels when doing a partial copy is very revealing, there's lots of data from the padding areas to be filled in.

					dma_sync(true);
					ensure(dma_mapping.second);

					// Upload memory to the working buffer. The guest address has arbitrary alignment.
					const auto dst_offset = tiled_output_offset; // Append to the end of the input
					mtl::copy_buffer_to_buffer_aligned(cmd, dma_mapping.second, dma_mapping.first, working_buffer, dst_offset, dma_sync_region.length());
				}

				// Prepare payload
				const RSX_detiler_config config =
				{
					.tile_base_address = tiled_region.base_address,
					.tile_base_offset = valid_range.start - tiled_region.base_address,
					.tile_rw_offset = dma_sync_region.start - tiled_region.base_address,
					.tile_size = tiled_region.tile->size,
					.tile_pitch = tiled_region.tile->pitch,
					.bank = tiled_region.tile->bank,

					.dst = working_buffer,
					.dst_offset = tiled_output_offset,
					.src = working_buffer,
					.src_offset = 0,

					// TODO: Check interaction with anti-aliasing
					.image_width = static_cast<u16>(transfer_width),
					.image_height = static_cast<u16>(transfer_height),
					.image_pitch = real_pitch,
					.image_bpp = context == rsx::texture_upload_context::dma ? internal_bpp : rsx::get_format_block_size_in_bytes(gcm_format)
				};

				// Execute
				const auto job = mtl::get_compute_task<mtl::cs_tile_memcpy<RSX_detiler_op::encode>>();
				job->run(cmd, config);

				// Update internal variables
				result_offset = tiled_output_offset;
				real_pitch = tiled_region.tile->pitch; // We're always copying the full image. In case of partials we're "filling in" blocks, not doing partial 2D copies.
#endif
			}

			if (rsx_pitch == real_pitch) [[likely]]
			{
				dma_sync(false);

				// Guest destination: any alignment (blit when 4-byte aligned, compute byte copy otherwise)
				mtl::copy_buffer_to_buffer_aligned(cmd, working_buffer, result_offset, dma_mapping.second, dma_mapping.first, dma_sync_region.length());
			}
			else
			{
				dma_sync(true);

				// One strided copy: transfer_height rows of transfer_pitch bytes, the guest bytes between rows are preserved
				mtl::copy_buffer_rows(cmd, working_buffer, result_offset, real_pitch, dma_mapping.second, dma_mapping.first, rsx_pitch, transfer_pitch, transfer_height);
			}
		}
		else
		{
			dma_sync(false);

			buffer_image_copy region{};
			region.buffer_row_length = (rsx_pitch / internal_bpp);
			region.aspect = src->aspect();
			region.mip_level = 0;
			region.base_layer = 0;
			region.layer_count = 1;
			region.image_offset = MTL::Origin::Make(static_cast<NS::UInteger>(src_area.x1), static_cast<NS::UInteger>(src_area.y1), 0);
			region.image_extent = MTL::Size::Make(transfer_width, transfer_height, 1);

			// A misaligned guest destination is staged through an aligned buffer by the copy helper (Metal requires
			// 16-byte aligned buffer offsets for texture -> buffer copies)
			region.buffer_offset = dma_mapping.first;
			mtl::copy_image_to_buffer_raw(cmd, src, dma_mapping.second, region);
		}

		// Create the completion marker for this transfer. It retires with the submission of `cmd`.
		dma_fence.emplace(cmd);

		rsx_log.trace("[Metal] DMA readback: range 0x%x+0x%x context %u flags %u fmt 0x%x %ux%u.",
			valid_range.start, valid_range.length(), static_cast<u32>(get_context()),
			static_cast<u32>(get_memory_read_flags()), static_cast<u32>(src->format()),
			static_cast<u32>(src_area.width()), static_cast<u32>(src_area.height()));

		// Set cb flag for queued dma operations
		cmd.set_flag(mtl::command_list::cb_has_dma_transfer);

		if (get_context() == rsx::texture_upload_context::dma)
		{
			// Save readback hint in case transformation is required later
			switch (internal_bpp)
			{
			case 2:
				gcm_format = CELL_GCM_TEXTURE_R5G6B5;
				break;
			case 4:
			default:
				gcm_format = CELL_GCM_TEXTURE_A8R8G8B8;
				break;
			}
		}

		synchronized = true;
		sync_timestamp = rsx::get_shared_tag();
	}

	void cached_texture_section::imp_flush()
	{
		AUDIT(synchronized);

		// Synchronize, reset dma_fence after waiting
		if (!dma_fence->wait(GENERAL_WAIT_TIMEOUT))
		{
			rsx_log.error("[Metal] DMA fence wait has timed out!");
		}

		// Calculate smallest range to flush - for framebuffers, the raster region is enough
		const auto range = (context == rsx::texture_upload_context::framebuffer_storage) ? get_section_range() : get_confirmed_range();
		auto flush_length = range.length();

		const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(range);
		if (tiled_region)
		{
			const auto available_tile_size = tiled_region.tile->size - (range.start - tiled_region.base_address);
			const auto max_content_size = tiled_region.tile->pitch * utils::align(height, 64);
			flush_length = std::min(max_content_size, available_tile_size);
		}

		mtl::flush_dma(range.start, flush_length);

#if DEBUG_DMA_TILING
		// Are we a tiled region?
		if (const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(range))
		{
			auto real_data = vm::get_super_ptr<u8>(range.start);
			auto out_data = rsx::simple_array<u8>(tiled_region.tile->size);
			rsx::tile_texel_data<u32>(
				out_data.data(),
				real_data,
				tiled_region.base_address,
				range.start - tiled_region.base_address,
				tiled_region.tile->size,
				tiled_region.tile->bank,
				tiled_region.tile->pitch,
				width,
				height
			);
			std::memcpy(real_data, out_data.data(), flush_length);
		}
#endif

		if (is_swizzled())
		{
			// This format is completely worthless to CPU processing algorithms where cache lines on die are linear.
			// If this is happening, usually it means it was not a planned readback (e.g shared pages situation)
			rsx_log.trace("[Performance warning] CPU readback of swizzled data");

			// Read-modify-write to avoid corrupting already resident memory outside texture region
			void* data = get_ptr(range.start);
			rsx::simple_array<u8> tmp_data(rsx_pitch * height);
			std::memcpy(tmp_data.data(), data, tmp_data.size());

			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_A8R8G8B8:
			case CELL_GCM_TEXTURE_DEPTH24_D8:
				rsx::convert_linear_swizzle<u32, false>(tmp_data.data(), data, width, height, rsx_pitch);
				break;
			case CELL_GCM_TEXTURE_R5G6B5:
			case CELL_GCM_TEXTURE_DEPTH16:
				rsx::convert_linear_swizzle<u16, false>(tmp_data.data(), data, width, height, rsx_pitch);
				break;
			default:
				rsx_log.error("Unexpected swizzled texture format 0x%x", gcm_format);
			}
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Texture cache
	// ---------------------------------------------------------------------------------------------------------------

	void texture_cache::on_section_destroyed(cached_texture_section& tex)
	{
		if (tex.is_managed() && tex.exists())
		{
			auto disposable = mtl::disposable_t::make(new cached_image_reference_t(this, tex.get_texture()));
			mtl::get_resource_manager()->dispose(disposable);
		}
	}

	void texture_cache::clear()
	{
		{
			std::lock_guard lock(m_cached_pool_lock);
			m_cache_is_exiting = true;
		}

		// Views handed out for a draw that never finished (emulation stopped mid-draw); the common clear() skips them
		release_uncached_temporary_subresources();
		baseclass::clear();

		m_cached_images.clear();
		m_cached_memory_size = 0;

		// The common cache released its references above; nothing can use these images any more
		for (auto& entry : m_gather_cache)
		{
			if (entry->image)
			{
				dispose_gather_entry(*entry);
			}
		}

		m_gather_cache.clear();
	}

	// Channel layout of a temporary image copied or gathered from `source`
	static MTL::TextureSwizzleChannels get_temporary_component_layout(const mtl::image* source, u32 gcm_format)
	{
		// This method is almost exclusively used to work on framebuffer resources
		// Keep the original swizzle layout unless there is data format conversion
		if (!source || mtl::get_compatible_sampler_format(gcm_format) != source->info.format)
		{
			// This is a data cast operation
			// Use native mapping for the new type
			// TODO: Also simulate the readback+reupload step (very tricky)
			const auto remap = get_component_mapping(gcm_format);
			return MTL::TextureSwizzleChannels::Make(remap[1], remap[2], remap[3], remap[0]);
		}

		return source->native_component_map;
	}

	// Clears the subresources (mip level x cube face or 3D slice) of `level_mask` that the copies of `sections` do not
	// write completely; the copies overwrite the others anyway, and a clear is a render pass per subresource on Metal.
	// Strict Rendering Mode clears every selected level like before (a copy that fails, e.g. an unsupported scaled copy,
	// then leaves black texels instead of whatever the pooled image held).
	static void clear_gather_image(mtl::command_list& cmd, mtl::image* image,
		const rsx::simple_array<texture_cache::copy_region_descriptor>& sections, u64 level_mask)
	{
		image_clear_value clear{};
		if (image->aspect() & aspect_depth)
		{
			clear.depth = 1.f;
			clear.stencil = 0;
		}

		const u32 mipmaps = image->mipmaps();
		const u64 all_levels = (mipmaps >= 64) ? ~0ull : ((1ull << mipmaps) - 1);
		level_mask &= all_levels;

		if (g_cfg.video.strict_rendering_mode)
		{
			if (level_mask == all_levels)
			{
				mtl::clear_image(cmd, image, clear);
				return;
			}

			for (u32 level = 0; level < mipmaps; ++level)
			{
				if (level_mask & (1ull << level))
				{
					mtl::clear_image(cmd, image, clear, level, 1);
				}
			}

			return;
		}

		const bool is_3d = (image->type() == MTL::TextureType3D);
		std::vector<u8> covered;

		for (u32 level = 0; level < mipmaps; ++level)
		{
			if (!(level_mask & (1ull << level)))
			{
				continue;
			}

			const u32 level_w = std::max(image->width() >> level, 1u);
			const u32 level_h = std::max(image->height() >> level, 1u);
			const u32 slices = is_3d ? std::max(image->depth() >> level, 1u) : image->layers();
			covered.assign(slices, 0);

			for (const auto& section : sections)
			{
				// dst_z is the cube face (array layer) or the 3D slice. Each copy writes (dst_x, dst_y, dst_w, dst_h).
				if (section.src && section.level == level && section.dst_z < slices &&
					section.dst_x == 0 && section.dst_y == 0 && section.dst_w >= level_w && section.dst_h >= level_h)
				{
					covered[section.dst_z] = 1;
				}
			}

			for (u32 slice = 0; slice < slices;)
			{
				if (covered[slice])
				{
					++slice;
					continue;
				}

				u32 end = slice + 1;
				while (end < slices && !covered[end])
				{
					++end;
				}

				mtl::clear_image(cmd, image, clear, level, 1, slice, end - slice);
				slice = end;
			}
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Reusable mip-chain gathers
	// ---------------------------------------------------------------------------------------------------------------
	//
	// A gathered level is copied again only when what its copy read may have changed. For each section (in request
	// order) an entry keeps the full copy descriptor (source pointer, compared only, never dereferenced: requests
	// carry live pointers, so a matching pointer designates a live object) and, for single-sampled render targets, the
	// surface's content_tag and last_use_tag when it was copied. content_tag is unique per surface object and replaced
	// whenever GPU work that may write the surface is recorded (MTLRenderTargets.h), so an equal tag on the same live
	// object proves that no write was recorded since the copy: the copy's contents are what a new copy would read now.
	// Sources without such a tag (texture cache images, MSAA surfaces) are copied on every request.
	//
	// Ordering needs no extra barrier: copies into a reused image are recorded after the draws that sampled it, and
	// every compute encoder or render pass waits for all earlier work of the queue (mtl::command_list), so they do not
	// overtake those reads; a draw sampling a reused image runs in a pass that began after its last copy. The image is
	// never an attachment of the renderer's passes.

	bool texture_cache::gather_reuse_allowed(const deferred_subresource& desc) const
	{
		// Strict Rendering Mode: rebuild on every request, like upstream. A memory load (force_bg_load) reads guest
		// memory, which has no write tracking here. Section i of a mip gather is level i (levels are tracked in a u64).
		return !g_cfg.video.strict_rendering_mode &&
			desc.op == rsx::deferred_request_command::mipmap_gather &&
			!desc.force_bg_load &&
			!desc.sections_to_copy.empty() && desc.sections_to_copy.size() <= 64;
	}

	texture_cache::gather_entry_t* texture_cache::find_gather_entry(const deferred_subresource& desc) const
	{
		const u64 properties = desc.encoded_properties();
		for (const auto& entry : m_gather_cache)
		{
			if (!entry->orphaned && entry->address == desc.address && entry->properties == properties &&
				entry->remap == desc.remap.encoded)
			{
				return entry.get();
			}
		}

		return nullptr;
	}

	static bool same_copy_region(const texture_cache::copy_region_descriptor& a, const texture_cache::copy_region_descriptor& b)
	{
		return a.src == b.src && a.xform == b.xform && a.base_addr == b.base_addr && a.level == b.level &&
			a.src_x == b.src_x && a.src_y == b.src_y && a.src_w == b.src_w && a.src_h == b.src_h &&
			a.dst_x == b.dst_x && a.dst_y == b.dst_y && a.dst_z == b.dst_z && a.dst_w == b.dst_w && a.dst_h == b.dst_h;
	}

	u64 texture_cache::get_stale_gather_levels(const gather_entry_t& entry, const deferred_subresource& desc) const
	{
		constexpr u64 unusable = ~0ull;
		const auto& sections = desc.sections_to_copy;

		if (!entry.image || sections.size() != entry.sources.size())
		{
			return unusable;
		}

		// The view keeps the component layout it was created with
		if (!(get_temporary_component_layout(get_template_from_collection_impl(sections), desc.gcm_format) == entry.component_layout))
		{
			return unusable;
		}

		u64 stale = 0;
		for (usz i = 0; i < sections.size(); ++i)
		{
			const auto& section = sections[i];
			const auto& source = entry.sources[i];

			if (!same_copy_region(section, source.region) || section.level >= 64)
			{
				return unusable;
			}

			if (!section.src)
			{
				// Nothing copied
				continue;
			}

			// `section.src` is the live pointer of this request (equal to the recorded one)
			const auto surface = mtl::try_as_rtt(section.src);
			const bool current = source.content_tag && surface &&
				surface->content_tag == source.content_tag &&
				// Redundant with content_tag, which changes on every write; it costs nothing to also require it
				surface->last_use_tag == source.last_use_tag &&
				section.src->format() == source.format && section.src->samples() == source.samples;

			if (!current)
			{
				stale |= (1ull << section.level);
			}
		}

		return stale;
	}

	void texture_cache::record_gather_sources(gather_entry_t& entry, const deferred_subresource& desc, u64 levels) const
	{
		const auto& sections = desc.sections_to_copy;
		ensure(entry.sources.size() == sections.size());

		for (usz i = 0; i < sections.size(); ++i)
		{
			const auto& section = sections[i];
			if (section.level < 64 && !(levels & (1ull << section.level)))
			{
				continue;
			}

			auto& source = entry.sources[i];
			source = {};
			source.region = section;

			if (!section.src)
			{
				continue;
			}

			source.format = section.src->format();
			source.samples = section.src->samples();

			// Single-sampled render targets only: MSAA surfaces are read through resolve/unresolve steps whose
			// images this tag does not cover
			if (const auto surface = mtl::try_as_rtt(section.src); surface && section.src->samples() == 1)
			{
				source.content_tag = surface->content_tag;
				source.last_use_tag = surface->last_use_tag;
			}
		}
	}

	mtl::image_view* texture_cache::reuse_gather(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		const auto entry = find_gather_entry(desc);
		if (!entry)
		{
			return nullptr;
		}

		const u64 stale_levels = get_stale_gather_levels(*entry, desc);
		if (stale_levels == ~0ull || (stale_levels && entry->refs))
		{
			// Another copy plan, or out of date while the common cache still holds the view (its snapshot must not
			// change under it): build a new image
			const auto it = std::find_if(m_gather_cache.begin(), m_gather_cache.end(),
				[entry](const std::unique_ptr<gather_entry_t>& e) { return e.get() == entry; });
			evict_gather_entry(static_cast<usz>(it - m_gather_cache.begin()));
			return nullptr;
		}

		if (stale_levels)
		{
			// Copy the levels whose sources changed, all their sections in request order (overlaps resolve like a
			// full build). The copies end the open render pass, like a full build.
			rsx::simple_array<copy_region_descriptor> sections;
			for (const auto& section : desc.sections_to_copy)
			{
				if (section.src && (stale_levels & (1ull << section.level)))
				{
					sections.push_back(section);
				}
			}

			clear_gather_image(cmd, entry->image, desc.sections_to_copy, stale_levels);
			copy_transfer_regions_impl(cmd, entry->image, sections);
			record_gather_sources(*entry, desc, stale_levels);
		}

		entry->refs++;
		entry->last_use = ++m_gather_use_serial;
		return entry->view;
	}

	void texture_cache::remember_gather(const deferred_subresource& desc, mtl::image_view* view)
	{
		auto image = dynamic_cast<mtl::viewable_image*>(view->image());
		if (!image)
		{
			return;
		}

		auto entry = std::make_unique<gather_entry_t>();
		entry->address = desc.address;
		entry->properties = desc.encoded_properties();
		entry->remap = desc.remap.encoded;
		entry->component_layout = get_temporary_component_layout(get_template_from_collection_impl(desc.sections_to_copy), desc.gcm_format);
		entry->sources.resize(desc.sections_to_copy.size());
		record_gather_sources(*entry, desc, ~0ull);

		if (std::none_of(entry->sources.begin(), entry->sources.end(), [](const gather_source_t& source) { return source.content_tag != 0; }))
		{
			// Every level would be copied again on each request; leave the image to the usual pool
			return;
		}

		// A previous entry for this request could not serve it (reuse_gather() evicted it); drop any leftover
		for (usz i = m_gather_cache.size(); i-- > 0;)
		{
			const auto& other = *m_gather_cache[i];
			if (!other.orphaned && other.address == entry->address && other.properties == entry->properties && other.remap == entry->remap)
			{
				evict_gather_entry(i);
			}
		}

		entry->image = image;
		entry->view = view;
		entry->memory_size = image->value->allocatedSize();
		entry->refs = 1; // The view being returned
		entry->last_use = ++m_gather_use_serial;

		m_gather_cache_memory += entry->memory_size;
		m_gather_cache.push_back(std::move(entry));

		// Capacity: least recently used entries go first
		while (true)
		{
			usz live = 0;
			u64 live_memory = 0;
			usz lru = umax;

			for (usz i = 0; i < m_gather_cache.size(); ++i)
			{
				const auto& e = *m_gather_cache[i];
				if (e.orphaned)
				{
					continue;
				}

				live++;
				live_memory += e.memory_size;

				if (lru == umax || e.last_use < m_gather_cache[lru]->last_use)
				{
					lru = i;
				}
			}

			if (live <= max_gather_cache_entries && live_memory <= max_gather_cache_memory)
			{
				break;
			}

			evict_gather_entry(lru);
		}
	}

	void texture_cache::evict_gather_entry(usz index)
	{
		auto& entry = *m_gather_cache[index];
		if (entry.refs)
		{
			// Still referenced by the common cache or by the draw being prepared: dispose on the last release
			entry.orphaned = true;
			return;
		}

		dispose_gather_entry(entry);
		m_gather_cache.erase(m_gather_cache.begin() + index);
	}

	void texture_cache::dispose_gather_entry(gather_entry_t& entry)
	{
		ensure(entry.image);
		m_gather_cache_memory -= entry.memory_size;

		// Back to the image pool once the GPU is done with it (in-flight work may still sample it)
		auto image = std::unique_ptr<mtl::viewable_image>(entry.image);
		auto disposable = mtl::disposable_t::make(new cached_image_reference_t(this, image));
		mtl::get_resource_manager()->dispose(disposable);

		entry.image = nullptr;
		entry.view = nullptr;
	}

	bool texture_cache::release_gather_reference(mtl::image_view* view)
	{
		for (usz i = 0; i < m_gather_cache.size(); ++i)
		{
			auto& entry = *m_gather_cache[i];
			if (entry.view != view)
			{
				continue;
			}

			ensure(entry.refs > 0);
			if (--entry.refs == 0 && entry.orphaned)
			{
				dispose_gather_entry(entry);
				m_gather_cache.erase(m_gather_cache.begin() + i);
			}

			return true;
		}

		return false;
	}

	void texture_cache::trim_gather_cache(bool evict_all)
	{
		// Frame end: keep what the frame that ended used (validated on the next use like any other reuse)
		for (usz i = m_gather_cache.size(); i-- > 0;)
		{
			const auto& entry = *m_gather_cache[i];
			if (!entry.orphaned && (evict_all || entry.last_use <= m_gather_frame_start))
			{
				evict_gather_entry(i);
			}
		}

		m_gather_frame_start = m_gather_use_serial;
	}

	bool texture_cache::temporary_subresource_is_current(const deferred_subresource& desc) const
	{
		if (!gather_reuse_allowed(desc))
		{
			return false;
		}

		if (!desc.do_not_cache)
		{
			// The common cache serves the request if it holds it (no staleness check there): not a reused gather
			const auto properties = desc.encoded_properties();
			const auto found = m_temporary_subresource_cache.equal_range(desc.address);
			for (auto it = found.first; it != found.second; ++it)
			{
				if (it->second.first.encoded_properties() == properties)
				{
					return false;
				}
			}
		}

		const auto entry = find_gather_entry(desc);
		return entry && get_stale_gather_levels(*entry, desc) == 0;
	}

	void texture_cache::copy_transfer_regions_impl(mtl::command_list& cmd, mtl::image* dst, const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const
	{
		const auto dst_aspect = dst->aspect();
		const auto dst_bpp = mtl::get_format_texel_width(dst->format());

		// Writes a region to the final destination (VK: vkCmdCopyImage with get_output_region)
		const auto copy_output_region = [&](const copy_region_descriptor& section, s32 in_x, s32 in_y, u32 w, u32 h, mtl::image* data_src)
		{
			const coord3i src_rect = { { in_x, in_y, 0 }, { static_cast<s32>(w), static_cast<s32>(h), 1 } };
			coord3i dst_rect = { { section.dst_x, section.dst_y, 0 }, { static_cast<s32>(w), static_cast<s32>(h), 1 } };
			rsx::image_copy_subresource_layers mip_layers{};
			mip_layers.dst_mip_level = section.level;

			if (dst->type() == MTL::TextureType3D)
			{
				dst_rect.position.z = section.dst_z;
			}
			else
			{
				mip_layers.dst_layer = static_cast<u8>(section.dst_z);
			}

			mtl::copy_image(cmd, data_src, dst, src_rect, dst_rect, mip_layers);
		};

		const auto configure_subresource = [&](const copy_region_descriptor& section, coord3i& coord, rsx::image_copy_subresource_layers& mip_layers)
		{
			mip_layers.dst_mip_level = section.level;
			coord.position = { section.dst_x, section.dst_y, 0 };
			coord.size = { section.dst_w, section.dst_h, 1 };

			if (dst->type() == MTL::TextureType3D)
			{
				coord.position.z = section.dst_z;
			}
			else
			{
				mip_layers.dst_layer = static_cast<u8>(section.dst_z);
			}
		};

		// Source window of a section, in source texels
		struct source_window_t
		{
			u16 x, y, w, h;
			rsx::flags32_t transform;
			bool typeless;
		};

		const auto get_source_window = [&](const copy_region_descriptor& section)
		{
			source_window_t window{ section.src_x, section.src_y, section.src_w, section.src_h, section.xform, false };
			window.typeless = section.src->aspect() != dst_aspect ||
				!formats_are_bitcast_compatible(dst, section.src);

			if (section.xform == rsx::surface_transform::coordinate_transform)
			{
				// Dimensions were given in 'dst' space. Work out the real source coordinates
				const auto src_bpp = mtl::get_format_texel_width(section.src->format());
				window.x = (window.x * dst_bpp) / src_bpp;
				window.w = utils::aligned_div<u16>(window.w * dst_bpp, src_bpp);

				window.transform &= ~(rsx::surface_transform::coordinate_transform);
			}

			if (auto surface = dynamic_cast<mtl::render_target*>(section.src))
			{
				surface->transform_samples_to_pixels(window.x, window.w, window.y, window.h);
			}

			return window;
		};

		const auto transfer_section = [&](const copy_region_descriptor& section)
		{
			if (!section.src || !section.src_w || !section.src_h || !section.dst_w || !section.dst_h)
			{
				// Degenerate sections copy nothing. Recording them would build invalid
				// (zero-size) descriptors and abort in the Metal driver.
				return;
			}

			const auto window = get_source_window(section);
			const bool typeless = window.typeless;

			auto src_image = section.src;
			auto src_x = window.x;
			auto src_y = window.y;
			auto src_w = window.w;
			auto src_h = window.h;

			rsx::flags32_t transform = window.transform;

			if (typeless) [[unlikely]]
			{
				const auto src_bpp = mtl::get_format_texel_width(section.src->format());
				// NOTE: u16(src_w * src_bpp) narrows BEFORE the division (functional cast
				// binds tighter than /), wrapping mod 65536 once the source byte-width
				// reaches 64 KiB. Keep full width until the helper request.
				const u32 convert_w = (static_cast<u32>(src_w) * src_bpp) / dst_bpp;
				const u32 convert_x = (static_cast<u32>(src_x) * src_bpp) / dst_bpp;

				if (convert_w == section.dst_w && src_h == section.dst_h &&
					transform == rsx::surface_transform::identity)
				{
					// Optimization to avoid double transfer
					coord3i dst_rect;
					rsx::image_copy_subresource_layers mip_layers;
					configure_subresource(section, dst_rect, mip_layers);

					const auto src_rect = coord3i{{ src_x, src_y, 0 }, { src_w, src_h, 1 }};
					mtl::copy_image_typeless(cmd, section.src, dst, src_rect, dst_rect, mip_layers);
					return;
				}

				// The staged bytes are copied to the helper origin and read back from the
				// origin, so convert_w x src_h holds the whole section. The old
				// (convert_x + convert_w) x (src_y + src_h) bounding extent only served
				// cross-section cache reuse and can exceed the 16384 Metal 2D limit on the
				// offset alone while the content fits.
				src_image = mtl::get_typeless_helper(dst->format(), dst->format_class(), convert_w, src_h, "xfer-surfaces");

				if (!src_image || convert_w > 0xFFFFu)
				{
					// Diagnostic: split the refused extent into offset vs content. If the
					// next run shows convert_x/src_y dominating, the content-sized helper
					// above already fixed it and this means genuine oversize (tiling needed).
					rsx_log.error("Metal: xfer-surfaces typeless skipped: src fmt 0x%x (bpp %u) window (%u,%u %ux%u) -> dst fmt 0x%x (bpp %u %ux%u), section dst (%u,%u %ux%u), convert_x %u convert_w %u.",
						static_cast<u32>(section.src->format()), static_cast<u32>(src_bpp),
						static_cast<u32>(src_x), static_cast<u32>(src_y), static_cast<u32>(src_w), static_cast<u32>(src_h),
						static_cast<u32>(dst->format()), static_cast<u32>(dst_bpp), dst->width(), dst->height(),
						static_cast<u32>(section.dst_x), static_cast<u32>(section.dst_y), static_cast<u32>(section.dst_w), static_cast<u32>(section.dst_h),
						convert_x, convert_w);
					return;
				}

				const areai src_rect = coordi{{ src_x, src_y }, { src_w, src_h }};
				const areai dst_rect = coordi{{ 0, 0 }, { static_cast<s32>(convert_w), src_h }};
				mtl::copy_image_typeless(cmd, section.src, src_image, src_rect, dst_rect);

				src_x = 0;
				src_y = 0;
				src_w = static_cast<u16>(convert_w);
			}

			ensure(transform == rsx::surface_transform::identity);

			if (src_w == section.dst_w && src_h == section.dst_h) [[likely]]
			{
				copy_output_region(section, src_x, src_y, src_w, src_h, src_image);
			}
			else
			{
				mtl::image* _dst = dst;

				if (src_image->info.format != dst->info.format || dst->type() == MTL::TextureType3D) [[ unlikely ]]
				{
					// Either a bitcast is required or a scale+copy to mipmap level / layer
					// Metal: scaled blits (sampled draws) cannot target a 3D texture; scale into a 2D helper, then copy the slice
					const u32 requested_width = dst->width();
					const u32 requested_height = src_y + src_h + section.dst_h; // Accounts for possible typeless ref on the same helper on src
					_dst = mtl::get_typeless_helper(src_image->format(), src_image->format_class(), requested_width, requested_height, "xfer-bitcast");
				}

				if (!_dst)
				{
					return;
				}

				auto dst_rect = coord3i{ { section.dst_x, section.dst_y, 0 }, { section.dst_w, section.dst_h, 1 } };
				rsx::image_copy_subresource_layers mip_layers{ .dst_mip_level = section.level };

				if (_dst != dst)
				{
					// We place the output after the source to account for the initial typeless-xfer if applicable
					// If src_image == _dst then this is just a write-to-self. Either way, use best-fit placement.
					dst_rect.position.x = 0;
					dst_rect.position.y = src_y + src_h;
					mip_layers = {};
				}
				else if (dst->type() == MTL::TextureType3D)
				{
					dst_rect.position.z = section.dst_z;
				}
				else
				{
					mip_layers.dst_layer = static_cast<u8>(section.dst_z);
				}

				mtl::copy_scaled_image(cmd, src_image, _dst,
					coord3i{ { src_x, src_y, 0 }, { src_w, src_h, 1 } },
					dst_rect,
					mip_layers,
					src_image->format() == _dst->format(),
					false);

				if (_dst != dst) [[unlikely]]
				{
					// Casting comes after the scaling!
					copy_output_region(section, dst_rect.position.x, dst_rect.position.y, section.dst_w, section.dst_h, _dst);
				}
			}
		};

		// Plain copy: same format, no conversion, no scaling. mtl::copy_image() records it as one blit command.
		const auto is_plain_copy = [&](const copy_region_descriptor& section)
		{
			if (section.src->format() != dst->format() || section.src->samples() != 1 || dst->samples() != 1)
			{
				return false;
			}

			const auto window = get_source_window(section);
			return !window.typeless && window.transform == rsx::surface_transform::identity &&
				window.w == section.dst_w && window.h == section.dst_h;
		};

		// Same command as copy_output_region() -> mtl::copy_image() for a plain copy, with the ordering chosen here
		const auto record_plain_copy = [&](const copy_region_descriptor& section, bool ordered)
		{
			const auto window = get_source_window(section);
			const bool dst_3d = (dst->type() == MTL::TextureType3D);
			auto encoder = ordered ? cmd.compute() : cmd.compute_unordered();

			encoder->copyFromTexture(
				section.src->value, 0, 0, MTL::Origin::Make(window.x, window.y, 0), MTL::Size::Make(window.w, window.h, 1),
				dst->value, dst_3d ? 0 : static_cast<u8>(section.dst_z), section.level,
				MTL::Origin::Make(section.dst_x, section.dst_y, dst_3d ? section.dst_z : 0));
		};

		// Every section writes the rectangle (dst_x, dst_y, dst_w, dst_h) of subresource (level, dst_z) and nothing else
		// of `dst` (conversions and scaling go through scratch images)
		const auto destinations_are_disjoint = [&]()
		{
			for (usz i = 0; i < sections_to_transfer.size(); ++i)
			{
				const auto& a = sections_to_transfer[i];
				if (!a.src)
				{
					continue;
				}

				for (usz j = i + 1; j < sections_to_transfer.size(); ++j)
				{
					const auto& b = sections_to_transfer[j];
					if (!b.src || a.level != b.level || a.dst_z != b.dst_z)
					{
						continue;
					}

					if (a.dst_x < b.dst_x + b.dst_w && b.dst_x < a.dst_x + a.dst_w &&
						a.dst_y < b.dst_y + b.dst_h && b.dst_y < a.dst_y + a.dst_h)
					{
						return false;
					}
				}
			}

			return true;
		};

		if (sections_to_transfer.size() > 1 && !g_cfg.video.strict_rendering_mode)
		{
			// Gathers mix plain copies (blit commands) with scaled copies (a render pass each) and conversions. Recorded
			// in order, every plain copy after a scaled one opens a new compute encoder and each one waits for all
			// earlier work. When no two sections write the same texels the order does not matter: record the plain
			// copies first as one run, where only the first waits (for the clears and other earlier work) and the
			// others, which write disjoint regions and only read their sources, run concurrently. Scaled copies and
			// conversions follow in their original order and order themselves after all earlier work.
			std::vector<u8> plain(sections_to_transfer.size(), 0);
			usz plain_count = 0;

			for (usz i = 0; i < sections_to_transfer.size(); ++i)
			{
				if (sections_to_transfer[i].src && is_plain_copy(sections_to_transfer[i]))
				{
					plain[i] = 1;
					plain_count++;
				}
			}

			if (plain_count && destinations_are_disjoint())
			{
				bool first = true;
				for (usz i = 0; i < sections_to_transfer.size(); ++i)
				{
					if (plain[i])
					{
						record_plain_copy(sections_to_transfer[i], first);
						first = false;
					}
				}

				for (usz i = 0; i < sections_to_transfer.size(); ++i)
				{
					if (sections_to_transfer[i].src && !plain[i])
					{
						transfer_section(sections_to_transfer[i]);
					}
				}

				return;
			}
		}

		for (const auto& section : sections_to_transfer)
		{
			if (!section.src)
			{
				continue;
			}

			transfer_section(section);
		}
	}

	MTL::TextureSwizzleChannels texture_cache::apply_component_mapping_flags(u32 gcm_format, rsx::component_order flags, const rsx::texture_channel_remap_t& remap_vector) const
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			// Dont bother letting this propagate
			return MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed, MTL::TextureSwizzleRed);
		default:
			break;
		}

		MTL::TextureSwizzleChannels mapping = swizzle_identity;
		switch (flags)
		{
		case rsx::component_order::default_:
		{
			mapping = mtl::apply_swizzle_remap(mtl::get_component_mapping(gcm_format), remap_vector);
			break;
		}
		case rsx::component_order::native:
		{
			mapping = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleBlue, MTL::TextureSwizzleAlpha);
			break;
		}
		case rsx::component_order::swapped_native:
		{
			// VK: { r = A, g = R, b = G, a = B }
			mapping = MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzleAlpha, MTL::TextureSwizzleRed, MTL::TextureSwizzleGreen, MTL::TextureSwizzleBlue);
			break;
		}
		default:
			break;
		}

		return mapping;
	}

	mtl::image* texture_cache::get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const
	{
		if (sections_to_transfer.size() == 1) [[likely]]
		{
			return sections_to_transfer.front().src;
		}

		mtl::image* result = nullptr;
		for (const auto& section : sections_to_transfer)
		{
			if (!section.src)
				continue;

			if (!result)
			{
				result = section.src;
			}
			else
			{
				if (!(section.src->native_component_map == result->native_component_map))
				{
					// TODO
					// This requires a far more complex setup as its not always possible to mix and match without compute assistance
					return nullptr;
				}
			}
		}

		return result;
	}

	std::unique_ptr<mtl::viewable_image> texture_cache::find_cached_image(MTL::PixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTL::TextureType type, MTL::TextureUsage usage)
	{
		reader_lock lock(m_cached_pool_lock);

		if (!m_cached_images.empty())
		{
			const u64 desired_key = hash_image_properties(format, w, h, d, mipmaps, type);
			lock.upgrade();

			for (auto it = m_cached_images.begin(); it != m_cached_images.end(); ++it)
			{
				if (it->key == desired_key && (it->data->info.usage & usage) == usage)
				{
					auto ret = std::move(it->data);
					m_cached_images.erase(it);
					m_cached_memory_size -= ret->value->allocatedSize();
					return ret;
				}
			}
		}

		return {};
	}

	std::unique_ptr<mtl::viewable_image> texture_cache::create_temporary_subresource_storage(
		rsx::format_class format_class, MTL::PixelFormat format,
		u16 width, u16 height, u16 depth, u16 layers, u8 mips,
		MTL::TextureType image_type, u32 /*image_flags*/, MTL::TextureUsage usage_flags)
	{
		auto image = find_cached_image(format, width, height, depth, mips, image_type, usage_flags);

		if (!image)
		{
			image_create_info info{};
			info.type = image_type;
			info.format = format;
			info.width = width;
			info.height = height;
			info.depth = depth;
			info.mipmaps = mips;
			info.layers = layers;
			info.samples = 1;
			info.usage = usage_flags | get_texture_cache_usage(format, false);
			info.storage = memory_location::device_local;
			info.format_class = format_class;

			image = std::make_unique<mtl::texture_cache_image>(*mtl::get_current_renderer(), info);
		}

		return image;
	}

	void texture_cache::dispose_reusable_image(std::unique_ptr<mtl::viewable_image>& image)
	{
		auto disposable = mtl::disposable_t::make(new cached_image_reference_t(this, image));
		mtl::get_resource_manager()->dispose(disposable);
	}

	mtl::image_view* texture_cache::create_temporary_subresource_view_impl(
		mtl::command_list& cmd, mtl::image* source, MTL::TextureType image_type,
		u32 gcm_format, u16 w, u16 h, u16 d, u8 mips,
		const rsx::texture_channel_remap_t& remap_vector,
		const copy_region_descriptor* copy)
	{
		const MTL::PixelFormat dst_format = mtl::get_compatible_sampler_format(gcm_format);
		const MTL::TextureUsage usage_flags = get_texture_cache_usage(dst_format, false);
		const u16 layers = (image_type == MTL::TextureTypeCube) ? 6 : 1;

		// Provision
		auto image = create_temporary_subresource_storage(rsx::classify_format(gcm_format), dst_format, w, h, d, layers, mips, image_type, 0, usage_flags);

		// OOM?
		if (!image)
		{
			return nullptr;
		}

		const MTL::TextureSwizzleChannels view_swizzle = get_temporary_component_layout(source, gcm_format);

		if (mtl::debug_labels_enabled())
		{
			image->set_debug_name(fmt::format("Temp view, fmt=0x%x", gcm_format));
		}

		image->set_native_component_layout(view_swizzle);
		auto view = image->get_view(remap_vector);

		if (copy)
		{
			rsx::simple_array<copy_region_descriptor> region = { *copy };
			copy_transfer_regions_impl(cmd, image.get(), region);
		}

		// TODO: Floating reference. We can do better with some restructuring.
		image.release();
		return view;
	}

	mtl::image_view* texture_cache::create_temporary_subresource_view(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		ensure(desc.sections_to_copy.size() == 1);
		const auto& section = desc.sections_to_copy.front();

		// VK used the source image type (always 2D for these sources). Multisampled sources still produce a 2D copy.
		return create_temporary_subresource_view_impl(cmd, section.src, MTL::TextureType2D,
			desc.gcm_format, desc.width, desc.height, 1, 1, desc.remap, &section);
	}

	mtl::image_view* texture_cache::generate_cubemap_from_images(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		const u8 mip_count = desc.exact_mip_count();
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTL::TextureTypeCube,
			desc.gcm_format, desc.width, desc.height, 1, mip_count, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_cubemap);
		}
		else
		{
			// Only the (level, face) subresources the copies do not write completely
			clear_gather_image(cmd, image, sections_to_copy, ~0ull);
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_3d_from_2d_images(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		const u8 mip_count = desc.exact_mip_count();
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTL::TextureType3D,
			desc.gcm_format, desc.width, desc.height, desc.depth, mip_count, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_3d);
		}
		else
		{
			// Only the (level, slice) subresources the copies do not write completely
			clear_gather_image(cmd, image, sections_to_copy, ~0ull);
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_atlas_from_images(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTL::TextureType2D,
			desc.gcm_format, desc.width, desc.height, 1, 1, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();
		const u32 dst_aspect = mtl::get_format_aspect(image->format());

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}
		else if (sections_to_copy[0].dst_w != desc.width || sections_to_copy[0].dst_h != desc.height)
		{
			if (!(dst_aspect & aspect_depth))
			{
				mtl::clear_image(cmd, image, {});
			}
			else
			{
				image_clear_value clear{};
				clear.depth = 1.f;
				clear.stencil = 0;
				mtl::clear_image(cmd, image, clear);
			}
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_2d_mipmaps_from_images(mtl::command_list& cmd, const deferred_subresource& desc)
	{
		// Games that sample a mipmapped render target while drawing into one of its levels (bloom, luminance and
		// reflection chains) request this gather on every draw. Reuse the previous image when possible and copy again
		// only the levels whose sources were written since (see "Reusable mip-chain gathers").
		const bool reusable = gather_reuse_allowed(desc);
		if (reusable)
		{
			if (auto view = reuse_gather(cmd, desc))
			{
				return view;
			}
		}

		const auto& sections_to_copy = desc.sections_to_copy;
		const auto mipmaps = ::narrow<u8>(sections_to_copy.size());
		auto _template = get_template_from_collection_impl(sections_to_copy);
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTL::TextureType2D,
			desc.gcm_format, desc.width, desc.height, 1, mipmaps, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}
		else
		{
			// Only the levels the copies below do not write completely
			clear_gather_image(cmd, image, sections_to_copy, ~0ull);
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);

		if (reusable)
		{
			remember_gather(desc, result);
		}

		return result;
	}

	void texture_cache::release_temporary_subresource(mtl::image_view* view)
	{
		if (release_gather_reference(view))
		{
			// Kept by the gather cache
			return;
		}

		auto resource = dynamic_cast<mtl::viewable_image*>(view->image());
		ensure(resource);

		auto image = std::unique_ptr<mtl::viewable_image>(resource);
		auto disposable = mtl::disposable_t::make(new cached_image_reference_t(this, image));
		mtl::get_resource_manager()->dispose(disposable);
	}

	void texture_cache::initialize_subresource_from_memory(mtl::command_list& cmd, mtl::image* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const
	{
		const auto subresources_layout = rsx::get_subresources_layout(desc, type);
		const u16 layer_count = (type == rsx::texture_dimension_extended::texture_dimension_cubemap) ? 6 : 1;
		mtl::upload_image(cmd, dst, subresources_layout, desc.gcm_format, desc.swizzled, layer_count,
			dst->aspect(), *mtl::get_upload_heap(), desc.pitch, mtl::upload_contents_inline);
	}

	void texture_cache::update_image_contents(mtl::command_list& cmd, mtl::image_view* dst_view, const deferred_subresource& desc)
	{
		auto dst = dst_view->image();
		copy_transfer_regions_impl(cmd, dst, desc.sections_to_copy);
	}

	cached_texture_section* texture_cache::create_new_texture(mtl::command_list& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
		u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t flags)
	{
		const auto section_depth = depth;

		// Define desirable attributes based on type
		MTL::TextureType image_type;
		u8 layer = 0;

		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			// 1D samplers are emitted as 2D (height 1) by the decompiler
			image_type = MTL::TextureType2D;
			height = 1;
			depth = 1;
			layer = 1;
			break;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			image_type = MTL::TextureType2D;
			depth = 1;
			layer = 1;
			break;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			image_type = MTL::TextureTypeCube;
			depth = 1;
			layer = 6;
			break;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			image_type = MTL::TextureType3D;
			layer = 1;
			break;
		default:
			fmt::throw_exception("Unreachable");
		}

		// Check what actually exists at that address
		const rsx::image_section_attributes_t search_desc = { .gcm_format = gcm_format, .width = width, .height = height, .depth = section_depth, .mipmaps = mipmaps };
		const bool allow_dirty = (context != rsx::texture_upload_context::framebuffer_storage);
		cached_texture_section& region = *find_cached_texture(rsx_range, search_desc, true, true, allow_dirty);
		ensure(!region.is_locked());

		mtl::viewable_image* image = nullptr;
		if (region.exists())
		{
			image = dynamic_cast<mtl::viewable_image*>(region.get_raw_texture());
			bool reusable = true;

			if (flags & texture_create_flags::do_not_reuse)
			{
				reusable = false;
			}
			// NOTE: texture_create_flags::shareable (VK concurrent sharing for the async queue) has no Metal equivalent.

			if (image && !(image->info.usage & MTL::TextureUsagePixelFormatView) &&
				(get_texture_cache_usage(image->format(), context == rsx::texture_upload_context::shader_read) & MTL::TextureUsagePixelFormatView))
			{
				// E.g. a blit_engine_dst section re-uploaded as shader_read: may now need an snorm view
				reusable = false;
			}

			if (!reusable || !image || region.get_image_type() != type || image->depth() != depth) // TODO
			{
				// Incompatible view/type
				region.destroy();
				image = nullptr;
			}
			else
			{
				ensure(region.is_managed());

				// Reuse
				region.set_rsx_pitch(pitch);

				if (flags & texture_create_flags::initialize_image_contents)
				{
					// Wipe memory
					image_clear_value clear{};
					if (image->aspect() & aspect_color)
					{
						clear.color = { 0.f, 0.f, 0.f, 1.f };
					}
					else
					{
						clear.depth = 1.f;
						clear.stencil = 255;
					}

					mtl::clear_image(cmd, image, clear);
				}
			}
		}

		if (!image)
		{
			const MTL::PixelFormat mtl_format = get_compatible_sampler_format(gcm_format);
			// NOTE: texture_create_flags::mutable_format is not consulted, see get_texture_cache_usage. Pooled images
			// are only reused if they have every requested usage bit, including MTLTextureUsagePixelFormatView.
			const MTL::TextureUsage usage_flags = get_texture_cache_usage(mtl_format, context == rsx::texture_upload_context::shader_read);

			if (auto found = find_cached_image(mtl_format, width, height, depth, mipmaps, image_type, usage_flags))
			{
				image = found.release();
			}
			else
			{
				image_create_info info{};
				info.type = image_type;
				info.format = mtl_format;
				info.width = width;
				info.height = height;
				info.depth = depth;
				info.mipmaps = mipmaps;
				info.layers = layer;
				info.samples = 1;
				info.usage = usage_flags;
				info.storage = memory_location::device_local;
				info.format_class = rsx::classify_format(gcm_format);

				image = new mtl::texture_cache_image(*m_device, info);
			}

			// New section, we must prepare it
			region.reset(rsx_range);
			region.set_gcm_format(gcm_format);
			region.set_image_type(type);
			region.create(width, height, section_depth, mipmaps, image, pitch, true, gcm_format);
		}

		region.set_view_flags(swizzle_flags);
		region.set_context(context);
		region.set_swizzled(swizzled);
		region.set_dirty(false);

		// Pooled images may carry views built for another component layout; set_native_component_layout drops them.
		image->set_native_component_layout(apply_component_mapping_flags(gcm_format, swizzle_flags, rsx::default_remap_vector));

		// Its not necessary to lock blit dst textures as they are just reused as necessary
		switch (context)
		{
		case rsx::texture_upload_context::shader_read:
		case rsx::texture_upload_context::blit_engine_src:
			region.protect(utils::protection::ro);
			read_only_range = region.get_min_max(read_only_range, rsx::section_bounds::locked_range);
			break;
		case rsx::texture_upload_context::blit_engine_dst:
			region.set_unpack_swap_bytes(true);
			no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
			break;
		case rsx::texture_upload_context::dma:
		case rsx::texture_upload_context::framebuffer_storage:
			// Should not be initialized with this method
		default:
			fmt::throw_exception("Unexpected upload context 0x%x", u32(context));
		}

		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::create_nul_section(
		mtl::command_list& /*cmd*/,
		const utils::address_range32& rsx_range,
		const rsx::image_section_attributes_t& attrs,
		const rsx::GCM_tile_reference& tile,
		bool memory_load)
	{
		auto& region = *find_cached_texture(rsx_range, { .gcm_format = RSX_GCM_FORMAT_IGNORED }, true, false, false);
		ensure(!region.is_locked());

		// Prepare section
		region.reset(rsx_range);
		region.create_dma_only(attrs.width, attrs.height, attrs.pitch);
		region.set_dirty(false);
		region.set_unpack_swap_bytes(true);

		if (memory_load && !tile) // Memory load on DMA tiles will always happen during the actual copy command
		{
			mtl::map_dma(rsx_range.start, rsx_range.length());
			mtl::load_dma(rsx_range.start, rsx_range.length());
		}

		no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::upload_image_from_cpu(mtl::command_list& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
		rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool swizzled)
	{
		if (context != rsx::texture_upload_context::shader_read)
		{
			if (cmd.is_render_pass_open())
			{
				cmd.end_render_pass();
			}
		}

		// No async transfer queue on Metal (supports_asynchronous_compute is false). Instead, "Asynchronous Texture
		// Streaming" records an upload that would end the open render pass into the command list's prologue, which
		// executes before everything recorded in the list (mtl::command_list::prologue()). On a TBDR GPU every early pass
		// end stores and reloads all attachments. Reordering the upload ahead of the list is only valid if:
		//  - its sources are written by the CPU: shader_read uploads read the upload heap (and scratch data derived from
		//    it); blit engine sources may be detiled from GPU-resident memory, and they end the pass above anyway;
		//  - no work recorded so far in the list references the destination image: the section's image is never reused
		//    in place (earlier draws may sample its old contents), like VK async uploads. A new image, or one from the
		//    reusable pool, which only takes images back after the GPU finished with them (through the GC).
		// With no pass open the inline upload splits nothing and may reuse the image, so it stays inline.
		const bool upload_ahead = context == rsx::texture_upload_context::shader_read &&
			g_cfg.video.vk.asynchronous_texture_streaming &&
			cmd.is_render_pass_open() &&
			cmd.can_record_prologue();

		rsx::flags32_t create_flags = 0;

		if (upload_ahead)
		{
			create_flags |= texture_create_flags::do_not_reuse;
		}

		if (context == rsx::texture_upload_context::shader_read &&
			!g_cfg.video.disable_hardware_texel_remapping)
		{
			create_flags |= texture_create_flags::mutable_format;
		}

		auto section = create_new_texture(cmd, rsx_range, width, height, depth, mipmaps, pitch, gcm_format, context, type, swizzled,
			rsx::component_order::default_, create_flags);

		auto image = section->get_raw_texture();
		if (mtl::debug_labels_enabled())
		{
			image->set_debug_name(fmt::format("Raw Texture @0x%x", rsx_range.start));
		}

		mtl::enter_uninterruptible();

		bool input_swizzled = swizzled;
		if (context == rsx::texture_upload_context::blit_engine_src)
		{
			// Swizzling is ignored for blit engine copy and emulated using remapping
			input_swizzled = false;
		}

		rsx::flags32_t upload_command_flags = initialize_image_layout | (upload_ahead ? upload_contents_async : upload_contents_inline);

		auto p_subresource_layout = &subresource_layout;
		u32 heap_align = upload_heap_align_default;

		if (auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(rsx_range);
			context == rsx::texture_upload_context::blit_engine_src && tiled_region)
		{
			if (mipmaps > 1)
			{
				// This really shouldn't happen on framebuffer tiled memory
				rsx_log.error("Tiled decode of mipmapped textures is not supported.");
			}
			else
			{
				const auto bpp = rsx::get_format_block_size_in_bytes(gcm_format);
				const auto [scratch_buf, linear_data_scratch_offset] = mtl::detile_memory_block(cmd, tiled_region, rsx_range, width, height, bpp);

				std::vector<rsx::subresource_layout> tmp;
				tmp.reserve(1);
				auto subres = subresource_layout.front();
				// FIXME: !!EVIL!!
				subres.data = { scratch_buf, linear_data_scratch_offset };
				subres.pitch_in_block = width;
				upload_command_flags |= source_is_gpu_resident;
				heap_align = width * bpp;

				tmp.push_back(std::move(subres));
				p_subresource_layout = &tmp;
			}
		}

		const u16 layer_count = (type == rsx::texture_dimension_extended::texture_dimension_cubemap) ? 6 : 1;
		mtl::upload_image(cmd, image, *p_subresource_layout, gcm_format, input_swizzled, layer_count, image->aspect(),
			*m_texture_upload_heap, heap_align, upload_command_flags);

		mtl::leave_uninterruptible();

		// VK transitioned the image to its preferred layout here. Metal has no layouts; the next consumer is ordered
		// after the upload by the command list's barriers (after the whole prologue for an upload recorded there).

		section->last_write_tag = rsx::get_shared_tag();
		return section;
	}

	void texture_cache::set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order expected_flags)
	{
		if (expected_flags == section.get_view_flags())
			return;

		const MTL::TextureSwizzleChannels mapping = apply_component_mapping_flags(gcm_format, expected_flags, rsx::default_remap_vector);
		auto image = static_cast<mtl::viewable_image*>(section.get_raw_texture());

		ensure(image);
		image->set_native_component_layout(mapping);

		section.set_view_flags(expected_flags);
	}

	void texture_cache::insert_texture_barrier(mtl::command_list& cmd, mtl::image* tex, bool strong_ordering)
	{
		if (!strong_ordering && !cmd.is_render_pass_open())
		{
			// No pass to split: the next render pass begins with a barrier on all previous work already
			return;
		}

		mtl::as_rtt(tex)->texture_barrier(cmd);
	}

	bool texture_cache::render_target_format_is_compatible(mtl::image* tex, u32 gcm_format)
	{
		auto mtl_format = tex->info.format;
		switch (gcm_format)
		{
		default:
			//TODO
			warn_once("Format incompatibility detected, reporting failure to force data copy (MTL_FORMAT=0x%X, GCM_FORMAT=0x%X)", static_cast<u32>(mtl_format), gcm_format);
			return false;
		case CELL_GCM_TEXTURE_R5G6B5:
			return (mtl_format == MTL::PixelFormatB5G6R5Unorm);
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
			return (mtl_format == MTL::PixelFormatRGBA16Float);
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			return (mtl_format == MTL::PixelFormatRGBA32Float);
		case CELL_GCM_TEXTURE_X32_FLOAT:
			return (mtl_format == MTL::PixelFormatR32Float);
		case CELL_GCM_TEXTURE_A8R8G8B8:
		case CELL_GCM_TEXTURE_D8R8G8B8:
			return (mtl_format == MTL::PixelFormatBGRA8Unorm || mtl_format == MTL::PixelFormatDepth24Unorm_Stencil8 || mtl_format == MTL::PixelFormatDepth32Float_Stencil8);
		case CELL_GCM_TEXTURE_B8:
			return (mtl_format == MTL::PixelFormatR8Unorm);
		case CELL_GCM_TEXTURE_G8B8:
			return (mtl_format == MTL::PixelFormatRG8Unorm);
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
			return (mtl_format == MTL::PixelFormatDepth24Unorm_Stencil8 || mtl_format == MTL::PixelFormatDepth32Float_Stencil8);
		case CELL_GCM_TEXTURE_X16:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			return (mtl_format == MTL::PixelFormatDepth16Unorm || mtl_format == MTL::PixelFormatDepth32Float);
		}
	}

	void texture_cache::prepare_for_dma_transfers(mtl::command_list& cmd)
	{
		if (!cmd.is_recording())
		{
			cmd.begin();
		}
	}

	void texture_cache::cleanup_after_dma_transfers(mtl::command_list& cmd)
	{
		if (cmd.flags & mtl::command_list::cb_has_dma_transfer)
		{
			rsx_log.trace("[Metal] DMA batch: submit flags=0x%x access_hint=%d open_query=%d",
				cmd.flags, static_cast<int>(cmd.access_hint),
				!!(cmd.flags & mtl::command_list::cb_has_open_query));
		}

		bool occlusion_query_active = !!(cmd.flags & mtl::command_list::cb_has_open_query);
		if (occlusion_query_active)
		{
			// We really stepped in it
			auto renderer = dynamic_cast<MTLGSRender*>(rsx::get_current_renderer());
			ensure(renderer);
			renderer->emergency_query_cleanup(&cmd);
		}

		// End recording
		cmd.end();

		if (cmd.access_hint != mtl::command_list::access_type_hint::all)
		{
			// Primary access command queue, must restart it after.
			// (No async compute scheduler to flush on Metal.) Commit through the renderer's submit path so ordering and
			// the global submit lock match every other submission.
			mtl::queue_submit_now(cmd);
			if (!cmd.wait(GENERAL_WAIT_TIMEOUT))
			{
				rsx_log.error("[Metal] Timed out waiting for the DMA flush submission");
				cmd.wait();
			}

			cmd.clear_flags();
			cmd.begin();
		}
		else
		{
			// Auxilliary command queue with auto-restart capability
			mtl::queue_submit_now(cmd);
			cmd.clear_flags();
		}

		ensure(cmd.flags == 0);

		if (occlusion_query_active)
		{
			ensure(cmd.is_recording());
			cmd.flags |= mtl::command_list::cb_load_occluson_task;
		}
	}

	void texture_cache::initialize(mtl::render_device& device, mtl::data_heap& upload_heap)
	{
		m_device = &device;
		m_texture_upload_heap = &upload_heap;
	}

	void texture_cache::destroy()
	{
		clear();
	}

	bool texture_cache::is_depth_texture(u32 rsx_address, u32 rsx_size)
	{
		reader_lock lock(m_cache_mutex);

		auto& block = m_storage.block_for(rsx_address);

		if (block.get_locked_count() == 0)
			return false;

		for (auto& tex : block)
		{
			if (tex.is_dirty())
				continue;

			if (!tex.overlaps(rsx_address, rsx::section_bounds::full_range))
				continue;

			if ((rsx_address + rsx_size - tex.get_section_base()) <= tex.get_section_size())
			{
				switch (tex.get_format())
				{
				case MTL::PixelFormatDepth16Unorm:
				case MTL::PixelFormatDepth32Float:
				case MTL::PixelFormatDepth32Float_Stencil8:
				case MTL::PixelFormatDepth24Unorm_Stencil8:
					return true;
				default:
					return false;
				}
			}
		}

		//Unreachable; silence compiler warning anyway
		return false;
	}

	bool texture_cache::handle_memory_pressure(rsx::problem_severity severity)
	{
		auto any_released = baseclass::handle_memory_pressure(severity);

		if (severity > rsx::problem_severity::low && !m_gather_cache.empty())
		{
			// Reusable gathers are a convenience: drop them (images still held by the common cache or by the draw
			// being prepared go when released). Not from inside the cache (gathers are built under this lock).
			std::unique_lock lock(m_cache_mutex, std::defer_lock);
			if (lock.try_lock())
			{
				const auto memory_before = m_gather_cache_memory;
				trim_gather_cache(true);
				any_released |= (m_gather_cache_memory != memory_before);
			}
		}

		// TODO: This can cause invalidation of in-flight resources
		if (severity <= rsx::problem_severity::low || !m_cached_memory_size)
		{
			// Nothing left to do
			return any_released;
		}

		constexpr u64 _1M = 0x100000;
		if (severity <= rsx::problem_severity::moderate && m_cached_memory_size < (64 * _1M))
		{
			// Some memory is consumed by the temporary resources, but no need to panic just yet
			return any_released;
		}

		std::unique_lock lock(m_cache_mutex, std::defer_lock);
		if (!lock.try_lock())
		{
			rsx_log.warning("Unable to remove temporary resources because we're already in the texture cache!");
			return any_released;
		}

		// Nuke temporary resources. They will still be visible to the GPU.
		auto gc = mtl::get_resource_manager();
		any_released |= !m_cached_images.empty();
		for (auto& img : m_cached_images)
		{
			gc->dispose(img.data);
		}
		m_cached_images.clear();
		m_cached_memory_size = 0;

		any_released |= !m_temporary_subresource_cache.empty();
		for (auto& e : m_temporary_subresource_cache)
		{
			ensure(e.second.second);
			release_temporary_subresource(e.second.second);
		}
		m_temporary_subresource_cache.clear();

		return any_released;
	}

	void texture_cache::on_frame_end()
	{
		trim_sections();

		if (m_storage.m_unreleased_texture_objects >= m_max_zombie_objects)
		{
			purge_unreleased_sections();
		}

		if (m_cached_images.size() > max_cached_image_pool_size ||
			m_cached_memory_size > 256 * 0x100000)
		{
			std::lock_guard lock(m_cached_pool_lock);

			const auto new_size = m_cached_images.size() / 2;
			for (usz i = new_size; i < m_cached_images.size(); ++i)
			{
				m_cached_memory_size -= m_cached_images[i].data->value->allocatedSize();
			}

			// The trimmed images went through the GC before entering the pool (cached_image_reference_t), the GPU is
			// done with them.
			m_cached_images.resize(new_size);
		}

		baseclass::on_frame_end();
		reset_frame_statistics();

		// After the common cache released its temporary subresources: keep the gathers this frame used
		trim_gather_cache(false);
	}

	mtl::viewable_image* texture_cache::upload_image_simple(mtl::command_list& cmd, MTL::PixelFormat format, u32 address, u32 width, u32 height, u32 pitch)
	{
		switch (format)
		{
		case MTL::PixelFormatBGRA8Unorm:
		case MTL::PixelFormatRGBA8Unorm:
			break;
		default:
			rsx_log.error("Unsupported MTLPixelFormat 0x%x", static_cast<u32>(format));
			return nullptr;
		}

		// Uploads a linear memory range as a BGRA8/RGBA8 texture. VK mapped a linear host-visible image; Metal textures
		// are never written by the CPU here: stage through the upload heap and copy on the GPU.
		const u32 row_pitch = width * 4;
		const usz upload_size = usz{ row_pitch } * height;
		const auto heap_offset = m_texture_upload_heap->alloc<512>(upload_size);
		auto dst = static_cast<char*>(m_texture_upload_heap->map(heap_offset, upload_size));
		auto src = vm::_ptr<const char>(address);

		// TODO: SSE optimization
		for (u32 row = 0; row < height; ++row)
		{
			auto casted_src = reinterpret_cast<const be_t<u32>*>(src);
			auto casted_dst = reinterpret_cast<u32*>(dst);

			for (u32 col = 0; col < width; ++col)
				casted_dst[col] = casted_src[col];

			src += pitch;
			dst += row_pitch;
		}

		m_texture_upload_heap->unmap();

		image_create_info info{};
		info.type = MTL::TextureType2D;
		info.format = format;
		info.width = width;
		info.height = height;
		info.usage = MTL::TextureUsageShaderRead;
		info.storage = memory_location::device_local;
		info.format_class = RSX_FORMAT_CLASS_COLOR;

		auto image = std::make_unique<mtl::viewable_image>(*m_device, info);

		buffer_image_copy region{};
		region.buffer_offset = heap_offset;
		region.buffer_row_length = width;
		region.aspect = aspect_color;
		region.image_extent = MTL::Size::Make(width, height, 1);
		mtl::copy_buffer_to_image_raw(cmd, m_texture_upload_heap->heap.get(), image.get(), region);

		// Fully dispose immediately. These immages aren't really reusable right now.
		auto result = image.get();
		mtl::get_resource_manager()->dispose(image);

		return result;
	}

	bool texture_cache::blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate, mtl::surface_cache& m_rtts, mtl::command_list& cmd)
	{
		// Transfers into a render target change its contents: copies of it (reusable mip-chain gathers) must see that
		struct content_tracking_blitter : blitter
		{
			void scale_image(mtl::command_list& cmd, mtl::image* src, mtl::image* dst, areai src_area, areai dst_area, bool interpolate, const rsx::typeless_xfer& xfer_info)
			{
				blitter::scale_image(cmd, src, dst, src_area, dst_area, interpolate, xfer_info);

				if (auto surface = mtl::try_as_rtt(dst))
				{
					surface->on_contents_changed();
				}
			}
		};

		content_tracking_blitter helper;
		auto reply = upload_scaled_image(src, dst, interpolate, cmd, m_rtts, helper);

		if (reply.succeeded)
		{
			if (reply.dst_range.valid())
			{
				flush_if_cache_miss_likely(cmd, reply.dst_range);
			}

			return true;
		}

		return false;
	}

	u32 texture_cache::get_unreleased_textures_count() const
	{
		return baseclass::get_unreleased_textures_count() + ::size32(m_cached_images);
	}

	u64 texture_cache::get_temporary_memory_in_use() const
	{
		// TODO: Technically incorrect, we should have separate metrics for cached evictable resources (this value) and temporary active resources.
		return m_cached_memory_size + m_gather_cache_memory;
	}

	bool texture_cache::is_overallocated() const
	{
		// Unified memory: use the device's recommended working set as the "VRAM" size
		const auto total_device_memory = m_device->caps().recommended_working_set / 0x100000;
		u64 quota = 0;

		if (total_device_memory >= 2048)
		{
			quota = std::min<u64>(3072, (total_device_memory * 40) / 100);
		}
		else if (total_device_memory >= 1024)
		{
			quota = std::max<u64>(204, (total_device_memory * 30) / 100);
		}
		else if (total_device_memory >= 768)
		{
			quota = 192;
		}
		else
		{
			quota = std::min<u64>(128, total_device_memory / 2);
		}

		quota *= 0x100000;

		if (const u64 texture_cache_pool_usage = vmm_get_application_pool_usage(VMM_ALLOCATION_POOL_TEXTURE_CACHE);
			texture_cache_pool_usage > quota)
		{
			rsx_log.warning("Texture cache is using %lluM of memory which exceeds the allocation quota of %lluM",
				texture_cache_pool_usage, quota);
			return true;
		}

		return false;
	}
}
