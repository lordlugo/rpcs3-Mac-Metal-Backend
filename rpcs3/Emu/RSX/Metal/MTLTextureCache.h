#pragma once

// Port of VK/VKTextureCache.h.
//
// Metal specifics:
//  - No layouts, no queue ownership transfers, no async upload queue (uploads are always inline).
//  - Texture cache images are Private textures with usage ShaderRead | RenderTarget (when the format is renderable,
//    so they can be cleared and be destinations of scaled blits) | PixelFormatView (always, see mtl::image). This
//    replaces VK's mutable-format lists.
//  - DMA readbacks (dma_transfer) are completed by the submission of the command list that recorded them: Metal 4
//    cannot signal an event from inside a command buffer. dma_fence_t tracks that submission (VK: vk::event).

#include "MTLDMA.h"
#include "MTLRenderTargets.h"
#include "MTLResourceManager.h"
#include "MTLHelpers.h"
#include "MTLFormats.h"

#include "mtlutils/commands.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/image.h"
#include "mtlutils/sync.h"

#include "Emu/RSX/Common/texture_cache.h"
#include "Emu/RSX/Common/tiled_dma_copy.hpp"
#include "Emu/RSX/Utils/image_utils.hpp"

#include <deque>
#include <memory>
#include <optional>
#include <vector>

#ifndef GENERAL_WAIT_TIMEOUT
#define GENERAL_WAIT_TIMEOUT  2000000ull  // 2 seconds
#endif

namespace mtl
{
	class cached_texture_section;
	class texture_cache;

	// Host-side completion marker of a DMA readback (VK: vk::event with sync_domain::host).
	// The transfer retires with the first submission of `cmd` after it was recorded. Like VK, the renderer must have
	// submitted command lists flagged with cb_has_dma_transfer before a section is flushed (flush_command_queue).
	struct dma_fence_t
	{
		const mtl::command_list* cmd = nullptr;
		const mtl::timeline* timeline = nullptr;
		u64 recorded_after = 0;   // Last timeline value signaled when the transfer was recorded

		dma_fence_t(const mtl::command_list& cmd_);

		// Returns false on timeout. timeout_us == 0 waits forever.
		bool wait(u64 timeout_us) const;
	};

	// Texture cache owned image; its allocation is accounted to VMM_ALLOCATION_POOL_TEXTURE_CACHE
	class texture_cache_image : public pooled_image
	{
	public:
		texture_cache_image(const render_device& dev, const image_create_info& info)
			: pooled_image(dev, info, VMM_ALLOCATION_POOL_TEXTURE_CACHE)
		{}
	};

	struct texture_cache_traits
	{
		using commandbuffer_type      = mtl::command_list;
		using section_storage_type    = mtl::cached_texture_section;
		using texture_cache_type      = mtl::texture_cache;
		using texture_cache_base_type = rsx::texture_cache<texture_cache_type, texture_cache_traits>;
		using image_resource_type     = mtl::image*;
		using image_view_type         = mtl::image_view*;
		using image_storage_type      = mtl::image;
		using texture_format          = MTL::PixelFormat;
		using viewable_image_type     = mtl::viewable_image*;
	};

	class cached_texture_section : public rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>
	{
		using baseclass = typename rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>;
		friend baseclass;

		std::unique_ptr<mtl::viewable_image> managed_texture = nullptr;

		// DMA relevant data (optional: avoids a heap allocation per DMA readback; 2 pointers + u64 inline)
		std::optional<mtl::dma_fence_t> dma_fence;
		const mtl::render_device* m_device = nullptr;
		mtl::viewable_image* vram_texture = nullptr;

	public:
		using baseclass::cached_texture_section;

		void create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, u32 gcm_format, bool pack_swap_bytes = false)
		{
			auto new_texture = static_cast<mtl::viewable_image*>(image);
			ensure(!exists() || !is_managed() || vram_texture == new_texture);

			if (vram_texture != new_texture && !managed_texture && get_protection() == utils::protection::no)
			{
				// In-place image swap, still locked. Likely a color buffer that got rebound as depth buffer or vice-versa.
				mtl::as_rtt(vram_texture)->on_swap_out();

				if (!managed)
				{
					// Incoming is also an external resource, reference it immediately
					mtl::as_rtt(image)->on_swap_in(is_locked());
				}
			}

			vram_texture = new_texture;

			ensure(rsx_pitch);

			width = w;
			height = h;
			this->depth = depth;
			this->mipmaps = mipmaps;
			this->rsx_pitch = rsx_pitch;

			this->gcm_format = gcm_format;
			this->pack_unpack_swap_bytes = pack_swap_bytes;

			if (managed)
			{
				managed_texture.reset(vram_texture);
			}

			if (auto rtt = dynamic_cast<mtl::render_target*>(image))
			{
				swizzled = (rtt->raster_type != rsx::surface_raster_type::linear);
			}

			if (synchronized)
			{
				// Even if we are managing the same vram section, we cannot guarantee contents are static
				// The create method is only invoked when a new managed session is required
				release_dma_resources();
				synchronized = false;
				flushed = false;
				sync_timestamp = 0ull;
			}

			// Notify baseclass
			baseclass::on_section_resources_created();
		}

		void create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, const mtl::render_target* surface)
		{
			u32 gcm_format;
			bool swap_bytes;

			if (surface->is_depth_surface())
			{
				gcm_format = (surface->get_surface_depth_format() != rsx::surface_depth_format::z16) ? CELL_GCM_TEXTURE_DEPTH16 : CELL_GCM_TEXTURE_DEPTH24_D8;
				swap_bytes = true;
			}
			else
			{
				auto info = get_compatible_gcm_format(surface->get_surface_color_format());
				gcm_format = info.first;
				swap_bytes = info.second;
			}

			create(w, h, depth, mipmaps, image, rsx_pitch, managed, gcm_format, swap_bytes);
		}

		void release_dma_resources()
		{
			// Pure host-side bookkeeping on Metal, nothing for the GC to defer
			dma_fence.reset();
		}

		void dma_abort() override
		{
			// Called if a reset occurs, usually via reprotect path after a bad prediction.
			// Discard the sync event, the next sync, if any, will properly recreate this.
			ensure(synchronized);
			ensure(!flushed);
			ensure(dma_fence);
			dma_fence.reset();
		}

		void destroy()
		{
			if (!exists() && context != rsx::texture_upload_context::dma)
				return;

			m_tex_cache->on_section_destroyed(*this);

			vram_texture = nullptr;
			ensure(!managed_texture);
			release_dma_resources();

			baseclass::on_section_resources_destroyed();
		}

		bool exists() const
		{
			return (vram_texture != nullptr);
		}

		bool is_managed() const
		{
			return !exists() || managed_texture;
		}

		mtl::image_view* get_view(const rsx::texture_channel_remap_t& remap)
		{
			ensure(vram_texture != nullptr);
			return vram_texture->get_view(remap);
		}

		mtl::image_view* get_raw_view()
		{
			ensure(vram_texture != nullptr);
			return vram_texture->get_view(rsx::default_remap_vector);
		}

		mtl::viewable_image* get_raw_texture()
		{
			return managed_texture.get();
		}

		std::unique_ptr<mtl::viewable_image>& get_texture()
		{
			return managed_texture;
		}

		mtl::render_target* get_render_target() const
		{
			return mtl::as_rtt(vram_texture);
		}

		MTL::PixelFormat get_format() const
		{
			if (context == rsx::texture_upload_context::dma)
			{
				return MTL::PixelFormatR32Uint;
			}

			ensure(vram_texture != nullptr);
			return vram_texture->format();
		}

		bool is_flushed() const
		{
			//This memory section was flushable, but a flush has already removed protection
			return flushed;
		}

		void dma_transfer(mtl::command_list& cmd, mtl::image* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch);

		void copy_texture(mtl::command_list& cmd, bool miss)
		{
			ensure(exists());

			if (!miss) [[likely]]
			{
				ensure(!synchronized);
				baseclass::on_speculative_flush();
			}
			else
			{
				baseclass::on_miss();
			}

			if (m_device == nullptr)
			{
				m_device = &cmd.device();
			}

			mtl::image* locked_resource = vram_texture;
			u32 transfer_width = width;
			u32 transfer_height = height;
			u32 transfer_x = 0, transfer_y = 0;

			if (context == rsx::texture_upload_context::framebuffer_storage)
			{
				auto surface = mtl::as_rtt(vram_texture);
				surface->memory_barrier(cmd, rsx::surface_access::transfer_read);
				locked_resource = surface->get_surface(rsx::surface_access::transfer_read);
				transfer_width *= surface->samples_x;
				transfer_height *= surface->samples_y;
			}

			mtl::image* target = locked_resource;
			if (transfer_width != locked_resource->width() || transfer_height != locked_resource->height())
			{
				// TODO: Synchronize access to typeles textures
				target = mtl::get_typeless_helper(vram_texture->format(), vram_texture->format_class(), transfer_width, transfer_height, "readback-scale");

				if (!target)
				{
					// Helper refused (dimensions logged at its call site); fall back to the
					// unscaled resource instead of dereferencing null below.
					rsx_log.warning("Metal: readback-scale helper for %ux%u refused, reading back unscaled %ux%u.",
						transfer_width, transfer_height, locked_resource->width(), locked_resource->height());
					target = locked_resource;
					transfer_width = locked_resource->width();
					transfer_height = locked_resource->height();
				}
				else
				{
					// Allow bilinear filtering on color textures where compatibility is likely
					const bool linear_filter = (target->aspect() == aspect_color);

					mtl::copy_scaled_image(cmd, locked_resource, target,
						areai{ 0, 0, static_cast<s32>(locked_resource->width()), static_cast<s32>(locked_resource->height()) },
						areai{ 0, 0, static_cast<s32>(transfer_width), static_cast<s32>(transfer_height) },
						{}, true, linear_filter);
				}
			}

			const auto internal_bpp = mtl::get_format_texel_width(vram_texture->format());
			const auto valid_range = get_confirmed_range();

			if (const auto section_range = get_section_range(); section_range != valid_range)
			{
				if (const auto offset = (valid_range.start - get_section_base()))
				{
					transfer_y = offset / rsx_pitch;
					transfer_x = (offset % rsx_pitch) / internal_bpp;

					ensure(transfer_width >= transfer_x);
					ensure(transfer_height >= transfer_y);
					transfer_width -= transfer_x;
					transfer_height -= transfer_y;
				}

				if (const auto tail = (section_range.end - valid_range.end))
				{
					const auto row_count = tail / rsx_pitch;

					ensure(transfer_height >= row_count);
					transfer_height -= row_count;
				}
			}

			areai src_area;
			src_area.x1 = static_cast<s32>(transfer_x);
			src_area.y1 = static_cast<s32>(transfer_y);
			src_area.x2 = s32(transfer_x + transfer_width);
			src_area.y2 = s32(transfer_y + transfer_height);
			dma_transfer(cmd, target, src_area, valid_range, rsx_pitch);
		}

		/**
		 * Flush
		 */
		void imp_flush() override;

		void* map_synchronized(u32, u32)
		{
			return nullptr;
		}

		void finish_flush()
		{}

		/**
		 * Misc
		 */
		void set_unpack_swap_bytes(bool swap_bytes)
		{
			pack_unpack_swap_bytes = swap_bytes;
		}

		void set_rsx_pitch(u32 pitch)
		{
			ensure(!is_locked());
			rsx_pitch = pitch;
		}

		void sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces)
		{
			auto rtt = mtl::as_rtt(vram_texture);
			rtt->sync_tag();

			for (auto& surface : surfaces)
			{
				rtt->inherit_surface_contents(mtl::as_rtt(surface->vram_texture));
			}
		}

		bool has_compatible_format(mtl::image* tex) const
		{
			return vram_texture->info.format == tex->info.format;
		}

		bool is_depth_texture() const
		{
			return !!(vram_texture->aspect() & aspect_depth);
		}
	};

	class texture_cache : public rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>
	{
	private:
		using baseclass = rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>;
		friend baseclass;

		struct cached_image_reference_t
		{
			std::unique_ptr<mtl::viewable_image> data;
			texture_cache* parent;

			cached_image_reference_t(texture_cache* parent, std::unique_ptr<mtl::viewable_image>& previous);
			~cached_image_reference_t();
		};

		struct cached_image_t
		{
			u64 key;
			std::unique_ptr<mtl::viewable_image> data;

			cached_image_t() = default;
			cached_image_t(u64 key_, std::unique_ptr<mtl::viewable_image>& data_) :
				key(key_), data(std::move(data_)) {}
		};

		// Reusable mip-chain gathers (generate_2d_mipmaps_from_images). Games that sample a mipmapped render target
		// while drawing into one of its levels (bloom, luminance and reflection chains) get the gather rebuilt on every
		// draw: the common cache does not keep it while a gathered surface is bound (do_not_cache) and drops it when
		// one is bound again (notify_surface_changed). Entries keep the gathered image and, per copied section, what
		// the copy read; a later request with the same copy plan copies again only the levels whose sources changed.
		struct gather_source_t
		{
			copy_region_descriptor region{};
			MTL::PixelFormat format = MTL::PixelFormatInvalid;
			u8 samples = 0;
			u64 content_tag = 0;    // render_target::content_tag when copied; 0: untracked source, copied every time
			u64 last_use_tag = 0;   // render_target::last_use_tag when copied (extra check, see get_stale_gather_levels)
		};

		struct gather_entry_t
		{
			// Request identity: the common cache's key (address, encoded properties) plus the view's channel remap and
			// component layout
			u32 address = 0;
			u64 properties = 0;
			u32 remap = 0;
			MTL::TextureSwizzleChannels component_layout = swizzle_identity;

			std::vector<gather_source_t> sources; // one per deferred_subresource::sections_to_copy entry, same order

			mtl::viewable_image* image = nullptr; // Owned. Released through the GC (dispose_gather_entry)
			mtl::image_view* view = nullptr;
			u64 memory_size = 0;
			u32 refs = 0;             // views handed to the common cache and not yet returned (release_temporary_subresource)
			bool orphaned = false;    // evicted while handed out: never reused, disposed when the last reference returns
			u64 last_use = 0;         // m_gather_use_serial of the last request it served
		};

		// Only touched on the RSX thread, like the common cache's temporary subresource containers
		std::vector<std::unique_ptr<gather_entry_t>> m_gather_cache;
		u64 m_gather_use_serial = 0;
		u64 m_gather_frame_start = 0;  // m_gather_use_serial when the current frame began
		u64 m_gather_cache_memory = 0;
		static constexpr usz max_gather_cache_entries = 24;
		static constexpr u64 max_gather_cache_memory = 256 * 0x100000;

	public:
		enum texture_create_flags : u32
		{
			initialize_image_contents = 1,
			do_not_reuse = 2,
			shareable = 4,
			mutable_format = 8
		};

		void on_section_destroyed(cached_texture_section& tex) override;

	private:

		// Metal internals
		const mtl::render_device* m_device = nullptr;
		mtl::data_heap* m_texture_upload_heap = nullptr;

		// Stuff that has been dereferenced by the GPU goes into these
		const u32 max_cached_image_pool_size = 256;
		std::deque<cached_image_t> m_cached_images;
		atomic_t<u64> m_cached_memory_size = { 0 };
		shared_mutex m_cached_pool_lock;

		// Blocks some operations when exiting
		atomic_t<bool> m_cache_is_exiting = false;

		void clear();

		MTL::TextureSwizzleChannels apply_component_mapping_flags(u32 gcm_format, rsx::component_order flags, const rsx::texture_channel_remap_t& remap_vector) const;

		void copy_transfer_regions_impl(mtl::command_list& cmd, mtl::image* dst, const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const;

		mtl::image* get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const;

		std::unique_ptr<mtl::viewable_image> find_cached_image(MTL::PixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTL::TextureType type, MTL::TextureUsage usage);

		// Reusable mip-chain gathers (see gather_entry_t)
		bool gather_reuse_allowed(const deferred_subresource& desc) const;
		gather_entry_t* find_gather_entry(const deferred_subresource& desc) const;
		// Levels (bit i: mip level i) whose sources changed since they were copied; ~0 if the entry cannot serve `desc`
		u64 get_stale_gather_levels(const gather_entry_t& entry, const deferred_subresource& desc) const;
		mtl::image_view* reuse_gather(mtl::command_list& cmd, const deferred_subresource& desc);
		void remember_gather(const deferred_subresource& desc, mtl::image_view* view);
		void record_gather_sources(gather_entry_t& entry, const deferred_subresource& desc, u64 levels) const;
		void evict_gather_entry(usz index);
		void dispose_gather_entry(gather_entry_t& entry);
		bool release_gather_reference(mtl::image_view* view);
		void trim_gather_cache(bool evict_all);

	protected:
		// VK: (image_type, view_type). Metal views always have the texture type of their image (Cube, 3D, 2D).
		mtl::image_view* create_temporary_subresource_view_impl(
			mtl::command_list& cmd, mtl::image* source, MTL::TextureType image_type,
			u32 gcm_format, u16 w, u16 h, u16 d, u8 mips, const rsx::texture_channel_remap_t& remap_vector,
			const copy_region_descriptor* copy = nullptr);

		mtl::image_view* create_temporary_subresource_view(mtl::command_list& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_cubemap_from_images(mtl::command_list& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_3d_from_2d_images(mtl::command_list& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_atlas_from_images(mtl::command_list& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_2d_mipmaps_from_images(mtl::command_list& cmd, const deferred_subresource& desc) override;

		void release_temporary_subresource(mtl::image_view* view) override;

		void initialize_subresource_from_memory(mtl::command_list& cmd, mtl::image* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const;

		void update_image_contents(mtl::command_list& cmd, mtl::image_view* dst_view, const deferred_subresource& desc) override;

		cached_texture_section* create_new_texture(mtl::command_list& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
			u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t flags) override;

		cached_texture_section* create_nul_section(mtl::command_list& cmd, const utils::address_range32& rsx_range, const rsx::image_section_attributes_t& attrs,
			const rsx::GCM_tile_reference& tile, bool memory_load) override;

		cached_texture_section* upload_image_from_cpu(mtl::command_list& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
			rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool swizzled) override;

		void set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order expected_flags) override;

		// Apple GPUs cannot barrier inside a render pass: ends the pass through render_target::texture_barrier().
		// For same-pixel feedback the renderer may prefer framebuffer fetch and skip this entirely.
		void insert_texture_barrier(mtl::command_list& cmd, mtl::image* tex, bool strong_ordering) override;

		bool render_target_format_is_compatible(mtl::image* tex, u32 gcm_format) override;

		void prepare_for_dma_transfers(mtl::command_list& cmd) override;

		void cleanup_after_dma_transfers(mtl::command_list& cmd) override;

	public:
		using baseclass::texture_cache;

		void initialize(mtl::render_device& device, mtl::data_heap& upload_heap);

		void destroy() override;

		// `image_flags` is accepted for signature parity with VK (cube compatibility is expressed by image_type == Cube).
		// `usage_flags` is the minimum usage the returned image must have; ShaderRead (+ RenderTarget when renderable)
		// are always added.
		std::unique_ptr<mtl::viewable_image> create_temporary_subresource_storage(
			rsx::format_class format_class, MTL::PixelFormat format,
			u16 width, u16 height, u16 depth, u16 layers, u8 mips,
			MTL::TextureType image_type, u32 image_flags, MTL::TextureUsage usage_flags);

		void dispose_reusable_image(std::unique_ptr<mtl::viewable_image>& tex);

		// True when create_temporary_subresource(desc) will return a reused mip-chain gather without recording any
		// command: none of its sources was written since it was copied. The renderer then needs no pass split for it.
		bool temporary_subresource_is_current(const deferred_subresource& desc) const;

		bool is_depth_texture(u32 rsx_address, u32 rsx_size) override;

		void on_frame_end() override;

		// Uploads a linear BGRA8/RGBA8 memory range as a texture (through the upload heap; no CPU writes into textures).
		mtl::viewable_image* upload_image_simple(mtl::command_list& cmd, MTL::PixelFormat format, u32 address, u32 width, u32 height, u32 pitch);

		bool blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate, mtl::surface_cache& m_rtts, mtl::command_list& cmd);

		u32 get_unreleased_textures_count() const override;

		bool handle_memory_pressure(rsx::problem_severity severity) override;

		u64 get_temporary_memory_in_use() const;

		bool is_overallocated() const;
	};
}
