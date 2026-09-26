#pragma once

// Port of VK/VKRenderTargets.h: render targets (surface cache entries) and the rsx::surface_store traits.
//
// Metal specifics:
//  - Render targets are Private textures with usage RenderTarget | ShaderRead | PixelFormatView (added by mtl::image).
//    No ShaderWrite: nothing writes surfaces from compute, and it would cost lossless compression.
//    MSAA targets are TextureType2DMultisample.
//  - No layouts. Feedback loops (sampling a bound attachment) cannot be solved with a barrier inside a pass on Apple
//    GPUs: texture_barrier() ends the current render pass (the next encoder waits for all prior work) when the surface
//    was written by that pass (written_in_pass, marked by the renderer after draws and clears), and counts the split
//    in mtl::g_feedback_loop_pass_splits. Writes of passes that already ended are in memory and need no split.
//    Same-pixel feedback can use framebuffer fetch instead (renderer's call).
//  - Feedback streaks: a run of draws of the same material (same programs, textures, blending, viewport and scissor,
//    no texture cache invalidation or wait-for-idle in between) that each sample and write the same attachment (water
//    surfaces, refraction, distortion particles) no longer ends the pass before every draw. Each draw of the run reads
//    the attachment without the writes of the other draws of the run (except where a read crosses into a tile that
//    was already stored, the same race a single feedback draw has with its own writes). The RSX gives no ordering for
//    such reads without a sync command either. Any other write, a clear, another material or a sync ends the run.
//  - Write-after-read inside a pass: depth reads of effects are same-pixel reads, which a tile-based GPU orders before
//    the writes of later draws (a tile is stored after all its draws are shaded), so they need no split. Colour
//    reads are often offset (refraction, distortion) and could reach an already stored tile: a colour write by a draw
//    that does not sample the surface, after reads in the open pass, still splits the pass. Strict Rendering Mode
//    keeps the exact ordering: a split for every read of a surface written in the open pass and every such write.
//  - Destructive cloning for spills moves the MTLTexture into a disposable drawable_surface_t (GC-deferred release).

#include "util/types.hpp"
#include "util/atomic.hpp"
#include "Emu/RSX/Common/surface_store.h"
#include "Emu/system_config.h"

#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLResourceManager.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/commands.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include <memory>
#include <vector>

namespace mtl
{
	// ---- Memory accounting (VK: vmm_* helpers of VKResourceManager / vkutils/memory) -------------------------------
	// Metal textures carry no allocation pool. Images owned by the surface and texture caches derive from pooled_image
	// so the caches can compare their footprint against a quota derived from recommendedMaxWorkingSetSize.
	enum vmm_allocation_pool
	{
		VMM_ALLOCATION_POOL_UNDEFINED = 0,
		VMM_ALLOCATION_POOL_SURFACE_CACHE,
		VMM_ALLOCATION_POOL_TEXTURE_CACHE,

		VMM_ALLOCATION_POOL_COUNT
	};

	u64 vmm_get_application_pool_usage(vmm_allocation_pool pool);
	void vmm_notify_pool_usage(vmm_allocation_pool pool, s64 delta_bytes);

	// Device memory load (currentAllocatedSize vs recommendedMaxWorkingSetSize), same thresholds as VK:
	// > 95% fatal, > 90% severe, > 75% moderate, otherwise low.
	rsx::problem_severity vmm_determine_memory_load_severity();

	// Number of render passes that had to be split to resolve texture feedback loops (Apple GPUs cannot barrier
	// inside a pass). The renderer resets/reads this once per frame for logging and the debug overlay.
	extern atomic_t<u32> g_feedback_loop_pass_splits;

	// Material key of the draw whose texture reads are being prepared (0 outside of that): see feedback streaks above.
	// Set by the renderer around load_texture_env(); texture barriers of the texture cache compare it.
	extern u64 g_feedback_draw_key;

	// Globally unique, never reused value for render_target::content_tag
	u64 new_surface_content_tag();

	namespace surface_cache_utils
	{
		void dispose(mtl::buffer* buf);
	}

	// Provided by the MSAA resolve helpers (MTLResolveHelper)
	void resolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);
	void unresolve_image(mtl::command_list& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);

	class image_reference_sync_barrier
	{
		u32 m_texture_barrier_count = 0;
		u32 m_draw_barrier_count = 0;
		bool m_allow_skip_barrier = true;

	public:
		void on_insert_texture_barrier()
		{
			m_texture_barrier_count++;
			m_allow_skip_barrier = false;
		}

		void on_insert_draw_barrier()
		{
			// Account for corner case where the same texture can be bound to more than 1 slot
			m_draw_barrier_count = std::max(m_draw_barrier_count + 1, m_texture_barrier_count);
		}

		void allow_skip()
		{
			m_allow_skip_barrier = true;
		}

		void reset()
		{
			m_texture_barrier_count = m_draw_barrier_count = 0u;
			m_allow_skip_barrier = false;
		}

		bool can_skip() const
		{
			return m_allow_skip_barrier;
		}

		bool is_enabled() const
		{
			return !!m_texture_barrier_count;
		}

		bool requires_post_loop_barrier() const
		{
			return is_enabled() && m_texture_barrier_count < m_draw_barrier_count;
		}
	};

	// viewable_image whose texture allocation is accounted to a vmm_allocation_pool
	class pooled_image : public viewable_image
	{
		vmm_allocation_pool m_pool = VMM_ALLOCATION_POOL_UNDEFINED;
		u64 m_accounted_size = 0;

	protected:
		pooled_image() = default;

		// (Re)start accounting of the current texture (after create_impl)
		void account_memory(vmm_allocation_pool pool);
		// Move the accounting of the current texture to another object (destructive clone)
		void transfer_accounting(pooled_image* target);

	public:
		pooled_image(const render_device& dev, const image_create_info& info, vmm_allocation_pool pool);
		~pooled_image() override;

		vmm_allocation_pool pool() const { return m_pool; }
	};

	struct drawable_surface_t : public pooled_image
	{
	protected:
		drawable_surface_t() = default;

	public:
		drawable_surface_t(const render_device& dev, const image_create_info& info);
		~drawable_surface_t() override;

		// Destructive cloning (VK viewable_image::clone): the returned object takes the texture, its views and its memory
		// accounting. This object is left without a texture and can be rebuilt with create_impl().
		drawable_surface_t* clone();
	};

	class render_target : public drawable_surface_t, public rsx::render_target_descriptor<mtl::viewable_image*>
	{
		// Cyclic reference hazard tracking
		image_reference_sync_barrier m_cyclic_ref_tracker;

		// Memory spilling support
		std::unique_ptr<mtl::buffer> m_spilled_mem;

		// MSAA support:
		// Get the linear resolve target bound to this surface. Initialize if none exists
		mtl::viewable_image* get_resolve_target_safe(mtl::command_list& cmd);
		// Resolve the planar MSAA data into a linear block
		void resolve(mtl::command_list& cmd);
		// Unresolve the linear data into planar MSAA data
		void unresolve(mtl::command_list& cmd);

		// Memory management:
		// Default-initialize memory without loading
		void clear_memory(mtl::command_list& cmd, mtl::image* surface);
		// Load memory from cell and use to initialize the surface
		void load_memory(mtl::command_list& cmd);
		// Generic - chooses whether to clear or load.
		void initialize_memory(mtl::command_list& cmd, rsx::surface_access access);

		// Spill helpers
		// Re-initialize using spilled memory
		void unspill(mtl::command_list& cmd);
		// Build spill transfer descriptors
		std::vector<buffer_image_copy> build_spill_transfer_descriptors(mtl::image* target);

	public:
		u64 frame_tag = 0;              // frame id when invalidated, 0 if not invalid
		u64 last_rw_access_tag = 0;     // timestamp when this object was last used
		u64 spill_request_tag = 0;      // timestamp when spilling was requested
		bool is_bound = false;          // set when the surface is bound for rendering
		u64 written_in_pass = 0;        // command_list::open_pass_serial() of the pass whose draws/clears last wrote it
		u64 feedback_streak_pass = 0;   // pass in which only a feedback streak wrote it so far (0: none)
		u64 feedback_streak_key = 0;    // material key of that streak
		u64 read_in_pass = 0;           // pass in which draws sampled it while it was bound (feedback reads)
		u64 read_in_pass_key = 0;       // material key of those readers if they all were one streak writing it, else 0

		// Contents version for consumers that keep copies of this surface (reusable mip-chain gathers of the texture
		// cache): unique per object from construction (a new surface at a recycled address never matches an old
		// value), and replaced whenever GPU work that may write the surface is recorded: draws and clears of the
		// renderer (mark_attachment_writes), memory initialization, inheritance, unspill, unresolve, blit engine
		// transfers, recycling. last_use_tag cannot serve: it changes once per bind epoch and inheritance may set it to
		// an older value.
		u64 content_tag = new_surface_content_tag();

		void on_contents_changed()
		{
			content_tag = new_surface_content_tag();
		}

		// A read by the current draw may use the memory contents (no pass split): not written by the open pass, or
		// written only by earlier draws of the same feedback streak
		bool feedback_read_in_pass_allowed(u64 open_pass, u64 draw_key) const
		{
			return written_in_pass != open_pass ||
				(draw_key && feedback_streak_pass == open_pass && feedback_streak_key == draw_key && !g_cfg.video.strict_rendering_mode);
		}

		using drawable_surface_t::drawable_surface_t;

		mtl::viewable_image* get_surface(rsx::surface_access access_type) override;
		bool is_depth_surface() const override;
		bool matches_dimensions(u16 _width, u16 _height) const;
		void reset_surface_counters();

		image_view* get_view(const rsx::texture_channel_remap_t& remap,
			u32 mask = aspect_color | aspect_depth) override;

		// Memory management
		bool spill(mtl::command_list& cmd, std::vector<std::unique_ptr<mtl::viewable_image>>& resolve_cache);

		// Synchronization
		// Feedback loop: Metal cannot wait inside a render pass. Ends the pass (if open) so the next pass is ordered
		// after every previous attachment write. Increments g_feedback_loop_pass_splits when a pass had to be split.
		void texture_barrier(mtl::command_list& cmd);
		void post_texture_barrier(mtl::command_list& cmd);
		void memory_barrier(mtl::command_list& cmd, rsx::surface_access access);
		void read_barrier(mtl::command_list& cmd) { memory_barrier(cmd, rsx::surface_access::shader_read); }
		void write_barrier(mtl::command_list& cmd) { memory_barrier(cmd, rsx::surface_access::shader_write); }
	};

	static inline mtl::render_target* as_rtt(mtl::image* t)
	{
		return ensure(dynamic_cast<mtl::render_target*>(t));
	}

	static inline const mtl::render_target* as_rtt(const mtl::image* t)
	{
		return ensure(dynamic_cast<const mtl::render_target*>(t));
	}

	static inline mtl::render_target* try_as_rtt(mtl::image* t)
	{
		return dynamic_cast<mtl::render_target*>(t);
	}

	static inline const mtl::render_target* try_as_rtt(const mtl::image* t)
	{
		return dynamic_cast<const mtl::render_target*>(t);
	}

	static inline bool is_rtt(const mtl::image* t)
	{
		return dynamic_cast<const mtl::render_target*>(t) != nullptr;
	}

	struct surface_cache_traits
	{
		using surface_storage_type = std::unique_ptr<mtl::render_target>;
		using surface_type = mtl::render_target*;
		using buffer_object_storage_type = std::unique_ptr<mtl::buffer>;
		using buffer_object_type = mtl::buffer*;
		using command_list_type = mtl::command_list&;
		using download_buffer_object = void*;
		using barrier_descriptor_t = rsx::deferred_clipped_region<mtl::render_target*>;

		// VK: get_attachment_create_flags. No FBO-loop/vendor workarounds on Metal; this only selects the usage mask.
		static MTL::TextureUsage get_attachment_usage(MTL::PixelFormat format, u8 samples, bool depth);

		static std::unique_ptr<mtl::render_target> create_new_surface(
			mtl::command_list& cmd,
			u32 address,
			rsx::surface_color_format format,
			usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling_config);

		static std::unique_ptr<mtl::render_target> create_new_surface(
			mtl::command_list& cmd,
			u32 address,
			rsx::surface_depth_format2 format,
			usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling_config);

		static bool is_reusable_surface(const mtl::render_target* surface, const mtl::render_target* ref)
		{
			return surface->value && surface->format() == ref->format() &&
				surface->info.usage == ref->info.usage && surface->info.samples == ref->info.samples;
		}

		static void prepare_for_reuse(mtl::command_list&, mtl::render_target* surface)
		{
			surface->reset_surface_counters();
			surface->last_rw_access_tag = 0;
			surface->on_contents_changed();
		}

		static void clone_surface(
			mtl::command_list& cmd,
			std::unique_ptr<mtl::render_target>& sink, mtl::render_target* ref,
			u32 address, barrier_descriptor_t& prev,
			const rsx::surface_scaling_config_t& scaling_config);

		static std::unique_ptr<mtl::render_target> convert_pitch(
			mtl::command_list& /*cmd*/,
			std::unique_ptr<mtl::render_target>& src,
			usz /*out_pitch*/)
		{
			// TODO
			src->state_flags = rsx::surface_state_flags::erase_bkgnd;
			return {};
		}

		static bool is_compatible_surface(const mtl::render_target* surface, const mtl::render_target* ref, u16 width, u16 height, u8 sample_count)
		{
			return (surface->format() == ref->format() &&
				surface->get_spp() == sample_count &&
				surface->get_surface_width() == width &&
				surface->get_surface_height() == height);
		}

		static void prepare_surface_for_drawing(mtl::command_list& cmd, mtl::render_target* surface)
		{
			// Special case barrier
			surface->memory_barrier(cmd, rsx::surface_access::gpu_reference);

			surface->reset_surface_counters();
			surface->memory_usage_flags |= rsx::surface_usage_flags::attachment;
			surface->is_bound = true;
		}

		static void prepare_surface_for_sampling(mtl::command_list& /*cmd*/, mtl::render_target* surface)
		{
			surface->is_bound = false;
		}

		static bool surface_is_pitch_compatible(const std::unique_ptr<mtl::render_target>& surface, usz pitch)
		{
			return surface->rsx_pitch == pitch;
		}

		static void int_invalidate_surface_contents(
			mtl::command_list& /*cmd*/,
			mtl::render_target* surface,
			u32 address,
			usz pitch)
		{
			surface->rsx_pitch = static_cast<u32>(pitch);
			surface->queue_tag(address);
			surface->last_use_tag = 0;
			surface->stencil_init_flags = 0;
			surface->memory_usage_flags = rsx::surface_usage_flags::unknown;
			surface->raster_type = rsx::surface_raster_type::linear;
			surface->on_contents_changed();
		}

		static void invalidate_surface_contents(
			mtl::command_list& cmd,
			mtl::render_target* surface,
			rsx::surface_color_format format,
			u32 address,
			usz pitch)
		{
			const auto fmt = mtl::get_compatible_surface_format(format);
			surface->set_format(format);
			surface->set_native_component_layout(fmt.second);
			surface->set_debug_name(fmt::format("RTV @0x%x, fmt=0x%x", address, static_cast<int>(format)));

			int_invalidate_surface_contents(cmd, surface, address, pitch);
		}

		static void invalidate_surface_contents(
			mtl::command_list& cmd,
			mtl::render_target* surface,
			rsx::surface_depth_format2 format,
			u32 address,
			usz pitch)
		{
			surface->set_format(format);
			surface->set_debug_name(fmt::format("DSV @0x%x", address));

			int_invalidate_surface_contents(cmd, surface, address, pitch);
		}

		static void notify_surface_invalidated(const std::unique_ptr<mtl::render_target>& surface)
		{
			surface->frame_tag = mtl::get_current_frame_id();
			if (!surface->frame_tag) surface->frame_tag = 1;

			if (!surface->old_contents.empty())
			{
				// TODO: Retire the deferred writes
				surface->clear_rw_barrier();
			}

			surface->release();
			surface->on_contents_changed();
		}

		static void notify_surface_persist(const std::unique_ptr<mtl::render_target>& /*surface*/)
		{}

		static void notify_surface_reused(const std::unique_ptr<mtl::render_target>& surface)
		{
			surface->state_flags |= rsx::surface_state_flags::erase_bkgnd;
			surface->add_ref();
			surface->on_contents_changed();
		}

		static bool int_surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			MTL::PixelFormat format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs)
		{
			if (check_refs && surface->has_refs())
			{
				// Surface may still have read refs from data 'copy'
				return false;
			}

			return (surface->info.format == format &&
				surface->get_spp() == get_format_sample_count(antialias) &&
				surface->matches_dimensions(static_cast<u16>(width), static_cast<u16>(height))) &&
				surface->resolution_scaling_config == scaling_config;
		}

		static bool surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			rsx::surface_color_format format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs = false)
		{
			const MTL::PixelFormat mtl_format = mtl::get_compatible_surface_format(format).first;
			return int_surface_matches_properties(surface, mtl_format, width, height, antialias, scaling_config, check_refs);
		}

		static bool surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			rsx::surface_depth_format2 format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs = false)
		{
			const MTL::PixelFormat mtl_format = mtl::get_compatible_depth_surface_format(format);
			return int_surface_matches_properties(surface, mtl_format, width, height, antialias, scaling_config, check_refs);
		}

		static void spill_buffer(std::unique_ptr<mtl::buffer>& /*bo*/)
		{
			// TODO
		}

		static void unspill_buffer(std::unique_ptr<mtl::buffer>& /*bo*/)
		{
			// TODO
		}

		static void write_render_target_to_memory(
			mtl::command_list& cmd,
			mtl::buffer* bo,
			mtl::render_target* surface,
			u64 dst_offset_in_buffer,
			u64 src_offset_in_buffer,
			u64 max_copy_length);

		template <int BlockSize>
		static mtl::buffer* merge_bo_list(mtl::command_list& cmd, std::vector<mtl::buffer*>& list)
		{
			u32 required_bo_size = 0;
			for (auto& bo : list)
			{
				required_bo_size += (bo ? ::size32(*bo) : BlockSize);
			}

			// Create dst (word multiple, required by the alignment-safe copy path)
			auto dst = new mtl::buffer(cmd.device(), utils::align(required_bo_size, 4u), memory_location::device_local, "merged surface buffer");

			// TODO: Initialize the buffer with system RAM contents

			// Copy all the data over from the sub-blocks
			u32 offset = 0;
			for (auto& bo : list)
			{
				if (!bo)
				{
					offset += BlockSize;
					continue;
				}

				const u32 length = ::size32(*bo);
				mtl::copy_buffer_to_buffer_aligned(cmd, bo, 0, dst, offset, length);
				offset += length;

				// Cleanup
				mtl::surface_cache_utils::dispose(bo);
			}

			return dst;
		}

		template <typename T>
		static T* get(const std::unique_ptr<T>& obj)
		{
			return obj.get();
		}
	};

	class surface_cache : public rsx::surface_store<mtl::surface_cache_traits>
	{
	private:
		u64 get_surface_cache_memory_quota(u64 total_device_memory);

	public:
		void destroy();
		bool spill_unused_memory();
		bool is_overallocated();
		bool can_collapse_surface(const std::unique_ptr<mtl::render_target>& surface, rsx::problem_severity severity) override;
		bool handle_memory_pressure(mtl::command_list& cmd, rsx::problem_severity severity) override;
		void trim(mtl::command_list& cmd, rsx::problem_severity memory_pressure);
	};
}
