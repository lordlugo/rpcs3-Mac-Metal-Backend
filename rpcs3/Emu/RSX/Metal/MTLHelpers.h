#pragma once

#include "util/types.hpp"
#include "Utilities/geometry.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "mtlutils/mtl_api.h"

#include <vector>

namespace rsx
{
	struct GCM_tile_reference;
}

namespace mtl
{
	class buffer;
	class command_list;
	class data_heap;
	class image;
	class render_device;

	enum runtime_state
	{
		uninterruptible = 1,
		heap_dirty      = 2,
		heap_changed    = 4,
	};

	struct image_readback_options_t
	{
		bool swap_bytes = false;
		struct
		{
			u64 offset = 0;
			u64 length = 0;

			operator bool() const { return length != 0; }
		} sync_region {};
	};

	// Buffer <-> image copy region (Vulkan VkBufferImageCopy equivalent)
	struct buffer_image_copy
	{
		u64 buffer_offset = 0;
		u32 buffer_row_length = 0;     // in texels, 0 = tightly packed
		u32 buffer_image_height = 0;   // in texels, 0 = tightly packed
		u32 aspect = 0;                // mtl::image_aspect bits (color, depth or stencil — one at a time)
		u32 mip_level = 0;
		u32 base_layer = 0;
		u32 layer_count = 1;
		MTL::Origin image_offset{ 0, 0, 0 };
		MTL::Size image_extent{ 1, 1, 1 };
	};

	const mtl::render_device* get_current_renderer();

	// Compatibility switches (mirrors the VK backend's workarounds interface)
	bool emulate_primitive_restart(rsx::primitive_type type);
	bool sanitize_fp_values();
	bool emulate_conditional_rendering();

	void destroy_global_resources();
	void reset_global_resources();

	template<class T>
	T* get_compute_task();

	enum image_upload_options
	{
		upload_contents_async   = 0x0001,
		initialize_image_layout = 0x0002,
		preserve_image_layout   = 0x0004,
		source_is_gpu_resident  = 0x0008,
		source_is_userptr       = 0x0010,

		// meta-flags
		upload_contents_inline    = 0,
		upload_heap_align_default = 0
	};

	// ---- Texture helpers (implemented in MTLTexture.cpp) -----------------------------------------------------------
	void upload_image(mtl::command_list& cmd, mtl::image* dst_image,
		const std::vector<rsx::subresource_layout>& subresource_layout, int format, bool is_swizzled, u16 layer_count,
		u32 aspect_flags, mtl::data_heap& upload_heap, u32 heap_align, rsx::flags32_t image_setup_flags);

	std::pair<mtl::buffer*, u32> detile_memory_block(
		mtl::command_list& cmd, const rsx::GCM_tile_reference& tiled_region, const utils::address_range32& range,
		u16 width, u16 height, u8 bpp);

	void copy_image_to_buffer(mtl::command_list& cmd, const mtl::image* src, const mtl::buffer* dst, const buffer_image_copy& region, const image_readback_options_t& options = {});
	void copy_buffer_to_image(mtl::command_list& cmd, const mtl::buffer* src, const mtl::image* dst, const buffer_image_copy& region);

	u64 calculate_working_buffer_size(u64 base_size, u32 aspect);

	void copy_image_typeless(mtl::command_list& cmd, mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		u32 src_transfer_mask = 0xFF, u32 dst_transfer_mask = 0xFF);

	void copy_image(mtl::command_list& cmd, mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		u32 src_transfer_mask = 0xFF, u32 dst_transfer_mask = 0xFF);

	// Scaled copy (Metal has no vkCmdBlitImage): implemented as a sampled draw
	void copy_scaled_image(mtl::command_list& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		bool compatible_formats = false, bool linear_filter = true);

	std::pair<MTL::PixelFormat, MTL::TextureSwizzleChannels> get_compatible_surface_format(rsx::surface_color_format color_format);

	// ---- Runtime state (implemented in MTLHelpers.cpp) --------------------------------------------------------------
	void raise_status_interrupt(runtime_state status);
	void clear_status_interrupt(runtime_state status);
	bool test_status_interrupt(runtime_state status);
	void enter_uninterruptible();
	void leave_uninterruptible();
	bool is_uninterruptible();

	void reset_runtime_state();

	void advance_completed_frame_counter();
	void advance_frame_counter();
	u64 get_current_frame_id();
	u64 get_last_completed_frame_id();

	struct blitter
	{
		void scale_image(mtl::command_list& cmd, mtl::image* src, mtl::image* dst, areai src_area, areai dst_area, bool interpolate, const rsx::typeless_xfer& xfer_info);
	};
}
