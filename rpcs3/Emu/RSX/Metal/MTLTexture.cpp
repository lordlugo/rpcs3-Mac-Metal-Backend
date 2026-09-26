#include "stdafx.h"
#include "MTLCompute.h"
#include "MTLDMA.h"
#include "MTLHelpers.h"
#include "MTLFormats.h"
#include "MTLResourceManager.h"
#include "MTLGSRender.h"

#include "mtlutils/commands.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

#include "../GCM.h"
#include "../Core/RSXContext.h"
#include "../Utils/rsx_utils.h"
#include "Emu/Memory/vm.h"

#include "util/asm.hpp"

// Port of VK/VKTexture.cpp (+ the scratch resource helpers of VK/vkutils/scratch.cpp and the auxiliary upload heap).
//
// Metal specifics:
//  - No image layouts, no explicit barriers: every transfer is recorded through cmd.compute(), which ends any open
//    render pass and serializes against the previous command (see mtl::command_list).
//  - One buffer <-> texture copy addresses a single array slice; multi-layer regions are split per layer.
//  - Combined depth-stencil textures (Depth32Float_Stencil8) are copied one plane at a time with
//    MTLBlitOptionDepthFromDepthStencil (4 bytes/texel, float) or MTLBlitOptionStencilFromDepthStencil (1 byte/texel).
//  - Texture -> texture copies (copyFromTexture) require identical pixel formats and copy all planes, so format
//    mismatches or single-aspect copies are routed through a scratch buffer.

namespace mtl
{
	// ---------------------------------------------------------------------------------------------------------------
	// Low level transfer helpers
	// ---------------------------------------------------------------------------------------------------------------

	struct buffer_copy_t
	{
		u64 src_offset = 0;
		u64 dst_offset = 0;
		u64 size = 0;
	};

	struct image_copy_t
	{
		u32 src_level = 0;
		u32 dst_level = 0;
		u32 src_layer = 0;
		u32 dst_layer = 0;
		u32 layer_count = 1;
		MTL::Origin src_offset{ 0, 0, 0 };
		MTL::Origin dst_offset{ 0, 0, 0 };
		MTL::Size extent{ 1, 1, 1 };
	};

	static MTL::BlitOption get_blit_option(const mtl::image* img, u32 aspect)
	{
		if ((img->aspect() & aspect_depth_stencil) != aspect_depth_stencil)
		{
			// Single-plane (color or depth-only) formats do not take a plane selector
			return MTL::BlitOptionNone;
		}

		switch (aspect & aspect_depth_stencil)
		{
		case aspect_depth:
			return MTL::BlitOptionDepthFromDepthStencil;
		case aspect_stencil:
			return MTL::BlitOptionStencilFromDepthStencil;
		default:
			fmt::throw_exception("Metal cannot transfer both planes of a depth-stencil texture in one buffer copy (aspect=0x%x)", aspect);
		}
	}

	static u32 get_transfer_aspect(const mtl::image* img, const buffer_image_copy& region)
	{
		return region.aspect ? (region.aspect & img->aspect()) : img->aspect();
	}

	// Buffer memory layout of a region (bytes per row / bytes per image) for the selected plane
	static std::pair<u64, u64> get_buffer_layout(const mtl::image* img, const buffer_image_copy& region, u32 aspect)
	{
		const auto block = get_format_block_info(img->format(), aspect);
		const u32 row_length = region.buffer_row_length ? region.buffer_row_length : static_cast<u32>(region.image_extent.width);
		const u32 image_height = region.buffer_image_height ? region.buffer_image_height : static_cast<u32>(region.image_extent.height);
		const u64 bytes_per_row = u64{ utils::aligned_div(row_length, u32{ block.block_width }) } * block.bytes_per_block;
		const u64 bytes_per_image = bytes_per_row * utils::aligned_div(image_height, u32{ block.block_height });
		return { bytes_per_row, bytes_per_image };
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Alignment-safe transfers
	// ---------------------------------------------------------------------------------------------------------------
	// Apple GPU / macOS blit rules:
	//  - buffer <-> texture copies: the buffer offset must be a multiple of 16 bytes (covers every texel and block size
	//    used here) and bytesPerRow a multiple of the texel (block) size.
	//  - buffer -> buffer copies: source offset, destination offset and size must be multiples of 4.
	// Guest memory (DMA blocks, zero-copy uploads) has arbitrary alignment. Misaligned transfers go through an aligned
	// staging buffer and/or a byte-granular compute copy, so correctness never depends on guest alignment.
	static constexpr u64 s_texture_copy_buffer_alignment = 16;
	static constexpr u64 s_buffer_copy_alignment = 4;

	// Byte-granular strided buffer copy: `rows` rows of `row_length` bytes, source/destination row pitches independent.
	// Each invocation owns one destination word; bytes of that word outside the copied rows (edges, gaps between rows)
	// are preserved with a read-modify-write, exactly like a blit that only touches the copied bytes.
	struct cs_byte_copy_task : compute_task
	{
		const mtl::buffer* m_src_buffer = nullptr;
		const mtl::buffer* m_dst_buffer = nullptr;
		u64 m_src_base = 0;
		u64 m_src_range = 0;
		u64 m_dst_base = 0;
		u64 m_dst_range = 0;
		std::array<u32, 8> m_params{};

		cs_byte_copy_task()
		{
			ssbo_count = 2;
			use_push_constants = true;
			push_constants_size = 32;

			create();

			const std::pair<std::string_view, std::string> syntax_replace[] =
			{
				{ "%ws", std::to_string(optimal_group_size) },
			};

			m_src = fmt::replace_all(std::string(
				"#version 430\n"
				"layout(local_size_x=%ws, local_size_y=1, local_size_z=1) in;\n"
				"layout(set=0, binding=0, std430) readonly buffer ssbo0{ uint src_data[]; };\n"
				"layout(set=0, binding=1, std430) buffer ssbo1{ uint dst_data[]; };\n"
				"layout(push_constant) uniform ubo{ uvec4 params[2]; };\n"
				"\n"
				"// params[0] = { src byte offset, dst byte offset (both relative to the bound word), row length, rows }\n"
				"// params[1] = { src pitch, dst pitch, destination words to process, unused }\n"
				"\n"
				"uint linear_invocation_id()\n"
				"{\n"
				"	uint size_in_x = (gl_NumWorkGroups.x * gl_WorkGroupSize.x);\n"
				"	return (gl_GlobalInvocationID.y * size_in_x) + gl_GlobalInvocationID.x;\n"
				"}\n"
				"\n"
				"void main()\n"
				"{\n"
				"	uint word = linear_invocation_id();\n"
				"	if (word >= params[1].z)\n"
				"		return;\n"
				"\n"
				"	uint src_rel = params[0].x;\n"
				"	uint dst_rel = params[0].y;\n"
				"	uint row_length = params[0].z;\n"
				"	uint rows = params[0].w;\n"
				"	uint src_pitch = params[1].x;\n"
				"	uint dst_pitch = params[1].y;\n"
				"\n"
				"	uint value = 0u;\n"
				"	uint mask = 0u;\n"
				"\n"
				"	for (uint i = 0u; i < 4u; ++i)\n"
				"	{\n"
				"		uint dst_byte = (word << 2) + i;\n"
				"		if (dst_byte < dst_rel)\n"
				"			continue;\n"
				"\n"
				"		uint rel = dst_byte - dst_rel;\n"
				"		uint row = rel / dst_pitch;\n"
				"		uint col = rel - (row * dst_pitch);\n"
				"		if (row >= rows || col >= row_length)\n"
				"			continue;\n"
				"\n"
				"		uint src_byte = src_rel + (row * src_pitch) + col;\n"
				"		uint data = (src_data[src_byte >> 2] >> ((src_byte & 3u) << 3)) & 0xFFu;\n"
				"		value |= data << (i << 3);\n"
				"		mask |= 0xFFu << (i << 3);\n"
				"	}\n"
				"\n"
				"	if (mask == 0u)\n"
				"		return;\n"
				"\n"
				"	if (mask != 0xFFFFFFFFu)\n"
				"	{\n"
				"		value |= dst_data[word] & ~mask;\n"
				"	}\n"
				"\n"
				"	dst_data[word] = value;\n"
				"}\n"), syntax_replace);
		}

		void bind_resources(mtl::command_list& /*cmd*/) override
		{
			push_constants(0, ::size32(m_params) * 4, m_params.data());
			m_program->bind_uniform({ m_src_buffer, m_src_base, m_src_range }, glsl::binding_set_index_compute, 0);
			m_program->bind_uniform({ m_dst_buffer, m_dst_base, m_dst_range }, glsl::binding_set_index_compute, 1);
		}

		void run(mtl::command_list& cmd,
			const mtl::buffer* src, u64 src_offset, u64 src_pitch,
			const mtl::buffer* dst, u64 dst_offset, u64 dst_pitch,
			u64 row_length, u32 rows)
		{
			ensure(row_length && rows && src_pitch >= row_length && dst_pitch >= row_length);

			const u64 src_span = u64{ rows - 1 } * src_pitch + row_length;
			const u64 dst_span = u64{ rows - 1 } * dst_pitch + row_length;
			ensure((src_offset + src_span) <= src->size() && (dst_offset + dst_span) <= dst->size());

			// Bind 4-byte aligned windows; the kernel addresses bytes relative to them
			m_src_buffer = src;
			m_dst_buffer = dst;
			m_src_base = src_offset & ~(s_buffer_copy_alignment - 1);
			m_dst_base = dst_offset & ~(s_buffer_copy_alignment - 1);

			const u64 src_rel = src_offset - m_src_base;
			const u64 dst_rel = dst_offset - m_dst_base;
			const u64 dst_words = utils::align<u64>(dst_rel + dst_span, 4) / 4;

			// The kernel touches whole words; the partial edge words must still lie inside the buffers
			// (always true for our MB/64K granular allocations)
			m_src_range = utils::align<u64>(src_rel + src_span, 4);
			m_dst_range = dst_words * 4;
			ensure((m_src_base + m_src_range) <= src->size() && (m_dst_base + m_dst_range) <= dst->size());

			ensure(dst_words <= u32{ umax } && src_pitch <= u32{ umax } && dst_pitch <= u32{ umax } && (src_rel + src_span) <= u32{ umax });
			m_params =
			{
				static_cast<u32>(src_rel), static_cast<u32>(dst_rel), static_cast<u32>(row_length), rows,
				static_cast<u32>(src_pitch), static_cast<u32>(dst_pitch), static_cast<u32>(dst_words), 0
			};

			compute_task::run(cmd, utils::aligned_div(static_cast<u32>(dst_words), optimal_group_size));
		}
	};

	// Private aligned staging area for misaligned buffer <-> texture transfers. Separate from the double-buffered
	// scratch pool so it can never alias a scratch buffer the caller is currently using.
	static std::unique_ptr<mtl::buffer> g_transfer_staging_buffer;

	static mtl::buffer* get_transfer_staging_buffer(u64 min_required_size)
	{
		if (g_transfer_staging_buffer && g_transfer_staging_buffer->size() < min_required_size)
		{
			// In-flight commands may still reference it
			mtl::get_resource_manager()->dispose(g_transfer_staging_buffer);
		}

		if (!g_transfer_staging_buffer)
		{
			g_transfer_staging_buffer = std::make_unique<mtl::buffer>(*g_render_device, utils::align<u64>(min_required_size, 0x100000),
				memory_location::device_local, "transfer staging buffer");
		}

		return g_transfer_staging_buffer.get();
	}

	// Strided buffer -> buffer copy with any alignment. Rows of one call are independent of each other.
	static void copy_buffer_rows_impl(mtl::command_list& cmd,
		const mtl::buffer* src, u64 src_offset, u64 src_pitch,
		const mtl::buffer* dst, u64 dst_offset, u64 dst_pitch,
		u64 row_length, u32 rows, bool ordered)
	{
		if (!row_length || !rows)
		{
			return;
		}

		if (rows == 1 || (src_pitch == row_length && dst_pitch == row_length))
		{
			// Contiguous: collapse into a single row
			row_length *= rows;
			rows = 1;
			src_pitch = dst_pitch = row_length;
		}

		const u64 alignment_bits = src_offset | dst_offset | row_length | (rows > 1 ? (src_pitch | dst_pitch) : 0);
		if ((alignment_bits & (s_buffer_copy_alignment - 1)) == 0)
		{
			for (u32 row = 0; row < rows; ++row)
			{
				auto encoder = (ordered && row == 0) ? cmd.compute() : cmd.compute_unordered();
				encoder->copyFromBuffer(src->value(), src_offset + row * src_pitch, dst->value(), dst_offset + row * dst_pitch, row_length);
			}
			return;
		}

		// Always ordered (program::bind records a barrier), so edge-word read-modify-writes never race
		mtl::get_compute_task<cs_byte_copy_task>()->run(cmd, src, src_offset, src_pitch, dst, dst_offset, dst_pitch, row_length, rows);
	}

	void copy_buffer_rows(mtl::command_list& cmd,
		const mtl::buffer* src, u64 src_offset, u64 src_pitch,
		const mtl::buffer* dst, u64 dst_offset, u64 dst_pitch,
		u64 row_length, u32 rows)
	{
		copy_buffer_rows_impl(cmd, src, src_offset, src_pitch, dst, dst_offset, dst_pitch, row_length, rows, true);
	}

	void copy_buffer_to_buffer_aligned(mtl::command_list& cmd, const mtl::buffer* src, u64 src_offset, const mtl::buffer* dst, u64 dst_offset, u64 length)
	{
		copy_buffer_rows_impl(cmd, src, src_offset, length, dst, dst_offset, length, length, 1, true);
	}

	// Bytes of texel data per row, block rows and slices touched by a buffer <-> texture copy of `region`
	struct region_rows_t
	{
		u64 row_bytes;
		u32 rows;
		u32 slices;
	};

	static region_rows_t get_region_rows(const mtl::image* img, const buffer_image_copy& region, u32 aspect)
	{
		const auto block = get_format_block_info(img->format(), aspect);
		return
		{
			u64{ utils::aligned_div(static_cast<u32>(region.image_extent.width), u32{ block.block_width }) } * block.bytes_per_block,
			utils::aligned_div(static_cast<u32>(region.image_extent.height), u32{ block.block_height }),
			static_cast<u32>(region.image_extent.depth)
		};
	}

	// vkCmdCopyBufferToImage equivalent (one region, any number of layers)
	static void copy_buffer_to_image_impl(mtl::command_list& cmd, const mtl::buffer* src, const mtl::image* dst, const buffer_image_copy& region, bool ordered = true)
	{
		const u32 aspect = get_transfer_aspect(dst, region);
		const auto options = get_blit_option(dst, aspect);
		const auto [bytes_per_row, bytes_per_image] = get_buffer_layout(dst, region, aspect);
		const bool is_3d = dst->type() == MTL::TextureType3D;

		// Metal: bytesPerImage is the stride between 3D slices; it must be 0 when a single 2D image is copied
		const u64 bytes_per_image_arg = (is_3d && region.image_extent.depth > 1) ? bytes_per_image : 0;
		const u64 layer_stride = bytes_per_image * region.image_extent.depth;

		for (u32 layer = 0; layer < region.layer_count; ++layer)
		{
			const u64 offset = region.buffer_offset + layer * layer_stride;
			const u32 slice = is_3d ? 0 : (region.base_layer + layer);

			if ((offset % s_texture_copy_buffer_alignment) == 0) [[likely]]
			{
				auto encoder = (ordered && layer == 0) ? cmd.compute() : cmd.compute_unordered();
				encoder->copyFromBuffer(src->value(), offset, bytes_per_row, bytes_per_image_arg,
					region.image_extent, dst->value, slice, region.mip_level, region.image_offset, options);
				continue;
			}

			// Misaligned source (e.g. zero-copy guest memory): move the data into the aligned staging area first
			const auto extent = get_region_rows(dst, region, aspect);
			const u64 span = u64{ extent.slices - 1 } * bytes_per_image + u64{ extent.rows - 1 } * bytes_per_row + extent.row_bytes;
			const auto staging = get_transfer_staging_buffer(span);

			copy_buffer_rows_impl(cmd, src, offset, span, staging, 0, span, span, 1, true);
			cmd.compute()->copyFromBuffer(staging->value(), 0, bytes_per_row, bytes_per_image_arg,
				region.image_extent, dst->value, slice, region.mip_level, region.image_offset, options);
		}
	}

	// vkCmdCopyImageToBuffer equivalent (one region, any number of layers)
	static void copy_image_to_buffer_impl(mtl::command_list& cmd, const mtl::image* src, const mtl::buffer* dst, const buffer_image_copy& region, bool ordered = true)
	{
		ensure(src->samples() == 1, "Metal cannot copy multisampled textures to buffers");

		const u32 aspect = get_transfer_aspect(src, region);
		const auto options = get_blit_option(src, aspect);
		const auto [bytes_per_row, bytes_per_image] = get_buffer_layout(src, region, aspect);
		const bool is_3d = src->type() == MTL::TextureType3D;

		const u64 bytes_per_image_arg = (is_3d && region.image_extent.depth > 1) ? bytes_per_image : 0;
		const u64 layer_stride = bytes_per_image * region.image_extent.depth;

		for (u32 layer = 0; layer < region.layer_count; ++layer)
		{
			const u64 offset = region.buffer_offset + layer * layer_stride;
			const u32 slice = is_3d ? 0 : (region.base_layer + layer);

			if ((offset % s_texture_copy_buffer_alignment) == 0) [[likely]]
			{
				auto encoder = (ordered && layer == 0) ? cmd.compute() : cmd.compute_unordered();
				encoder->copyFromTexture(src->value, slice, region.mip_level, region.image_offset, region.image_extent,
					dst->value(), offset, bytes_per_row, bytes_per_image_arg, options);
				continue;
			}

			// Misaligned destination (e.g. a DMA block at an arbitrary guest address): copy into the aligned staging area,
			// then move only the texel rows so the bytes between rows (pitch padding) stay untouched, like a direct copy.
			const auto extent = get_region_rows(src, region, aspect);
			const u64 span = u64{ extent.slices - 1 } * bytes_per_image + u64{ extent.rows - 1 } * bytes_per_row + extent.row_bytes;
			const auto staging = get_transfer_staging_buffer(span);

			cmd.compute()->copyFromTexture(src->value, slice, region.mip_level, region.image_offset, region.image_extent,
				staging->value(), 0, bytes_per_row, bytes_per_image_arg, options);

			for (u32 z = 0; z < extent.slices; ++z)
			{
				copy_buffer_rows_impl(cmd, staging, z * bytes_per_image, bytes_per_row,
					dst, offset + z * bytes_per_image, bytes_per_row, extent.row_bytes, extent.rows, true);
			}
		}
	}

	// vkCmdCopyBuffer equivalent. Regions of one call are independent of each other (disjoint destinations).
	// Runs of equally sized regions with constant strides (row-by-row copies) are merged into one strided copy.
	static void copy_buffer_regions(mtl::command_list& cmd, const mtl::buffer* src, const mtl::buffer* dst, const buffer_copy_t* regions, usz count)
	{
		usz i = 0;
		while (i < count)
		{
			const auto& first = regions[i];
			const u64 size = first.size;
			usz j = i + 1;
			u64 src_pitch = size;
			u64 dst_pitch = size;

			if (size && j < count && regions[j].size == size &&
				regions[j].src_offset >= first.src_offset + size && regions[j].dst_offset >= first.dst_offset + size)
			{
				src_pitch = regions[j].src_offset - first.src_offset;
				dst_pitch = regions[j].dst_offset - first.dst_offset;

				while (j < count && regions[j].size == size &&
					regions[j].src_offset == regions[j - 1].src_offset + src_pitch &&
					regions[j].dst_offset == regions[j - 1].dst_offset + dst_pitch)
				{
					++j;
				}
			}

			copy_buffer_rows_impl(cmd, src, first.src_offset, src_pitch, dst, first.dst_offset, dst_pitch, size, static_cast<u32>(j - i), i == 0);
			i = j;
		}
	}

	// vkCmdCopyImage equivalent (identical formats only; copies all planes)
	static void copy_image_regions(mtl::command_list& cmd, const mtl::image* src, const mtl::image* dst, const image_copy_t* regions, usz count)
	{
		const bool src_3d = src->type() == MTL::TextureType3D;
		const bool dst_3d = dst->type() == MTL::TextureType3D;
		bool first = true;

		for (usz i = 0; i < count; ++i)
		{
			const auto& rgn = regions[i];
			for (u32 layer = 0; layer < rgn.layer_count; ++layer)
			{
				auto encoder = first ? cmd.compute() : cmd.compute_unordered();
				first = false;

				encoder->copyFromTexture(
					src->value, src_3d ? 0 : (rgn.src_layer + layer), rgn.src_level, rgn.src_offset, rgn.extent,
					dst->value, dst_3d ? 0 : (rgn.dst_layer + layer), rgn.dst_level, rgn.dst_offset);
			}
		}
	}

	static void gpu_swap_bytes_impl(mtl::command_list& cmd, mtl::buffer* buf, u32 element_size, u32 data_offset, u32 data_length)
	{
		if (element_size == 4)
		{
			mtl::get_compute_task<mtl::cs_shuffle_32>()->run(cmd, buf, data_length, data_offset);
		}
		else if (element_size == 2)
		{
			mtl::get_compute_task<mtl::cs_shuffle_16>()->run(cmd, buf, data_length, data_offset);
		}
		else
		{
			fmt::throw_exception("Unreachable");
		}
	}

	static void increment_mip_level(image_copy_t& copy)
	{
		copy.extent.width = std::max<NS::UInteger>(copy.extent.width / 2u, 1u);
		copy.extent.height = std::max<NS::UInteger>(copy.extent.height / 2u, 1u);
		copy.extent.depth = std::max<NS::UInteger>(copy.extent.depth / 2u, 1u);

		copy.src_offset.x /= 2;
		copy.src_offset.y /= 2;
		copy.src_offset.z /= 2;

		copy.dst_offset.x /= 2;
		copy.dst_offset.y /= 2;
		copy.dst_offset.z /= 2;

		copy.src_level++;
		copy.dst_level++;
	}

	static void increment_mip_level(buffer_image_copy& copy)
	{
		copy.image_offset.x /= 2;
		copy.image_offset.y /= 2;
		copy.image_offset.z /= 2;
		copy.image_extent.width = std::max<NS::UInteger>(copy.image_extent.width / 2u, 1u);
		copy.image_extent.height = std::max<NS::UInteger>(copy.image_extent.height / 2u, 1u);
		copy.image_extent.depth = std::max<NS::UInteger>(copy.image_extent.depth / 2u, 1u);
		copy.mip_level++;
	}

	static MTL::Origin to_origin(const coord3i& rect)
	{
		return MTL::Origin::Make(static_cast<NS::UInteger>(rect.x), static_cast<NS::UInteger>(rect.y), static_cast<NS::UInteger>(rect.z));
	}

	static MTL::Size to_size(const coord3i& rect)
	{
		return MTL::Size::Make(static_cast<NS::UInteger>(rect.width), static_cast<NS::UInteger>(rect.height), static_cast<NS::UInteger>(rect.depth));
	}

	void copy_image_to_buffer_raw(mtl::command_list& cmd, const mtl::image* src, const mtl::buffer* dst, const buffer_image_copy& region)
	{
		copy_image_to_buffer_impl(cmd, src, dst, region);
	}

	void copy_buffer_to_image_raw(mtl::command_list& cmd, const mtl::buffer* src, const mtl::image* dst, const buffer_image_copy& region)
	{
		copy_buffer_to_image_impl(cmd, src, dst, region);
	}

	u64 calculate_working_buffer_size(u64 base_size, u32 aspect)
	{
		if (aspect & (aspect_stencil | aspect_depth))
		{
			return (base_size * 3);
		}
		else
		{
			return base_size;
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Buffer <-> image transfers with format conversion
	// ---------------------------------------------------------------------------------------------------------------

	void copy_image_to_buffer(
		mtl::command_list& cmd,
		const mtl::image* src,
		const mtl::buffer* dst,
		const buffer_image_copy& region,
		const image_readback_options_t& options)
	{
		if (cmd.is_render_pass_open())
		{
			cmd.end_render_pass();
		}

		ensure((region.image_extent.width + region.image_offset.x) <= src->width());
		ensure((region.image_extent.height + region.image_offset.y) <= src->height());

		// NOTE: options.sync_region only requested a post-transfer barrier in VK. Every Metal transfer is serialized.
		switch (src->format())
		{
		default:
		{
			ensure(!options.swap_bytes); // "Implicit byteswap option not supported for speficied format"
			copy_image_to_buffer_impl(cmd, src, dst, region);
			break;
		}
		case MTL::PixelFormatDepth32Float:
		{
			rsx_log.error("Unsupported transfer (D16_FLOAT)"); // Need real games to test this.
			ensure(get_transfer_aspect(src, region) == aspect_depth);

			const u32 out_w = region.buffer_row_length ? region.buffer_row_length : static_cast<u32>(region.image_extent.width);
			const u32 out_h = region.buffer_image_height ? region.buffer_image_height : static_cast<u32>(region.image_extent.height);
			const u32 packed32_length = out_w * out_h * 4;
			const u32 packed16_length = out_w * out_h * 2;

			const auto allocation_end = region.buffer_offset + packed32_length + packed16_length;
			ensure(dst->size() >= allocation_end);

			const auto data_offset = u32(region.buffer_offset);
			const auto z32_offset = utils::align<u32>(data_offset + packed16_length, 256);

			// 1. Copy the depth to buffer
			buffer_image_copy region2 = region;
			region2.buffer_offset = z32_offset;
			copy_image_to_buffer_impl(cmd, src, dst, region2);

			// 2. Do conversion with byteswap [D32->D16F]
			if (!options.swap_bytes) [[likely]]
			{
				auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f32, f16>>();
				job->run(cmd, dst, z32_offset, packed32_length, data_offset);
			}
			else
			{
				auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f32, f16, false, true>>();
				job->run(cmd, dst, z32_offset, packed32_length, data_offset);
			}
			break;
		}
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
		{
			ensure(get_transfer_aspect(src, region) == aspect_depth_stencil);

			const u32 out_w = region.buffer_row_length ? region.buffer_row_length : static_cast<u32>(region.image_extent.width);
			const u32 out_h = region.buffer_image_height ? region.buffer_image_height : static_cast<u32>(region.image_extent.height);
			const u32 packed_length = out_w * out_h * 4;
			const u32 in_depth_size = packed_length;
			const u32 in_stencil_size = out_w * out_h;

			const auto allocation_end = region.buffer_offset + packed_length + in_depth_size + in_stencil_size;
			ensure(dst->size() >= allocation_end);

			const auto data_offset = u32(region.buffer_offset);
			const auto z_offset = utils::align<u32>(data_offset + packed_length, 256);
			const auto s_offset = utils::align<u32>(z_offset + in_depth_size, 256);

			// 1. Copy the depth and stencil blocks to separate banks (one Metal copy per plane)
			buffer_image_copy sub_regions[2];
			sub_regions[0] = sub_regions[1] = region;
			sub_regions[0].buffer_offset = z_offset;
			sub_regions[0].aspect = aspect_depth;
			sub_regions[1].buffer_offset = s_offset;
			sub_regions[1].aspect = aspect_stencil;
			copy_image_to_buffer_impl(cmd, src, dst, sub_regions[0]);
			copy_image_to_buffer_impl(cmd, src, dst, sub_regions[1], false);

			// 2. Interleave the separated data blocks with a compute job
			mtl::cs_interleave_task *job;
			if (!options.swap_bytes) [[likely]]
			{
				if (src->format() == MTL::PixelFormatDepth24Unorm_Stencil8)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d24x8<false>>();
				}
				else if (src->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<false, true>>();
				}
				else
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<false>>();
				}
			}
			else
			{
				if (src->format() == MTL::PixelFormatDepth24Unorm_Stencil8)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d24x8<true>>();
				}
				else if (src->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<true, true>>();
				}
				else
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<true>>();
				}
			}

			job->run(cmd, dst, data_offset, packed_length, z_offset, s_offset);
			break;
		}
		}
	}

	void copy_buffer_to_image(mtl::command_list& cmd, const mtl::buffer* src, const mtl::image* dst, const buffer_image_copy& region)
	{
		if (cmd.is_render_pass_open())
		{
			cmd.end_render_pass();
		}

		switch (dst->format())
		{
		default:
		{
			copy_buffer_to_image_impl(cmd, src, dst, region);
			break;
		}
		case MTL::PixelFormatDepth32Float:
		{
			rsx_log.error("Unsupported transfer (D16_FLOAT)");
			ensure(get_transfer_aspect(dst, region) == aspect_depth);

			const u32 out_w = region.buffer_row_length ? region.buffer_row_length : static_cast<u32>(region.image_extent.width);
			const u32 out_h = region.buffer_image_height ? region.buffer_image_height : static_cast<u32>(region.image_extent.height);
			const u32 packed32_length = out_w * out_h * 4;
			const u32 packed16_length = out_w * out_h * 2;

			const auto allocation_end = region.buffer_offset + packed32_length + packed16_length;
			ensure(src->size() >= allocation_end);

			const auto data_offset = u32(region.buffer_offset);
			const auto z32_offset = utils::align<u32>(data_offset + packed16_length, 256);

			// 1. Do conversion with byteswap [D16F->D32F]
			auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f16, f32>>();
			job->run(cmd, src, data_offset, packed16_length, z32_offset);

			// 2. Copy the depth data to image
			buffer_image_copy region2 = region;
			region2.buffer_offset = z32_offset;
			copy_buffer_to_image_impl(cmd, src, dst, region2);
			break;
		}
		case MTL::PixelFormatDepth24Unorm_Stencil8:
		case MTL::PixelFormatDepth32Float_Stencil8:
		{
			const u32 out_w = region.buffer_row_length ? region.buffer_row_length : static_cast<u32>(region.image_extent.width);
			const u32 out_h = region.buffer_image_height ? region.buffer_image_height : static_cast<u32>(region.image_extent.height);
			const u32 packed_length = out_w * out_h * 4;
			const u32 in_depth_size = packed_length;
			const u32 in_stencil_size = out_w * out_h;

			const auto allocation_end = region.buffer_offset + packed_length + in_depth_size + in_stencil_size;
			ensure(src->size() >= allocation_end); // "Out of memory (compute heap). Lower your resolution scale setting."

			const auto data_offset = u32(region.buffer_offset);
			const auto z_offset = utils::align<u32>(data_offset + packed_length, 256);
			const auto s_offset = utils::align<u32>(z_offset + in_depth_size, 256);

			// Zero out the stencil block
			cmd.compute()->fillBuffer(src->value(), NS::Range::Make(s_offset, utils::align(in_stencil_size, 4)), 0);

			// 1. Scatter the interleaved data into separate depth and stencil blocks
			mtl::cs_interleave_task *job;
			if (dst->format() == MTL::PixelFormatDepth24Unorm_Stencil8)
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d24x8>();
			}
			else if (dst->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d32x8<true>>();
			}
			else
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d32x8<false>>();
			}

			job->run(cmd, src, data_offset, packed_length, z_offset, s_offset);

			// 2. Copy the separated blocks into the target, one plane per copy
			buffer_image_copy sub_regions[2];
			sub_regions[0] = sub_regions[1] = region;
			sub_regions[0].buffer_offset = z_offset;
			sub_regions[0].aspect = aspect_depth;
			sub_regions[1].buffer_offset = s_offset;
			sub_regions[1].aspect = aspect_stencil;
			copy_buffer_to_image_impl(cmd, src, dst, sub_regions[0]);
			copy_buffer_to_image_impl(cmd, src, dst, sub_regions[1], false);
			break;
		}
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Image <-> image copies
	// ---------------------------------------------------------------------------------------------------------------

	// Bit-exact copy of one plane (or the color data) through a scratch buffer. Used where copyFromTexture cannot be
	// used: format mismatch between size-compatible formats, or a single plane of a depth-stencil texture.
	static void copy_image_via_buffer(
		mtl::command_list& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		u32 src_aspect, u32 dst_aspect)
	{
		ensure(src_aspect != aspect_depth_stencil && dst_aspect != aspect_depth_stencil);

		buffer_image_copy src_copy{}, dst_copy{};
		src_copy.image_extent = to_size(src_rect);
		src_copy.image_offset = to_origin(src_rect);
		src_copy.aspect = src_aspect;
		src_copy.mip_level = mip_layers.src_mip_level;
		src_copy.base_layer = mip_layers.src_layer;
		src_copy.layer_count = mip_layers.layer_count;

		// vkCmdCopyImage semantics: the extent is taken from the source rectangle
		dst_copy.image_extent = to_size(src_rect);
		dst_copy.image_offset = to_origin(dst_rect);
		dst_copy.aspect = dst_aspect;
		dst_copy.mip_level = mip_layers.dst_mip_level;
		dst_copy.base_layer = mip_layers.dst_layer;
		dst_copy.layer_count = mip_layers.layer_count;

		const auto src_block = get_format_block_info(src->format(), src_aspect);
		const auto dst_block = get_format_block_info(dst->format(), dst_aspect);
		ensure(src_block.bytes_per_block == dst_block.bytes_per_block, "Incompatible formats for a raw image copy");

		mtl::buffer* scratch_buf = nullptr;
		for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; remaining_levels--)
		{
			const auto [bytes_per_row, bytes_per_image] = get_buffer_layout(src, src_copy, src_aspect);
			const u64 length = bytes_per_image * src_copy.image_extent.depth * mip_layers.layer_count;

			if (!scratch_buf)
			{
				scratch_buf = get_scratch_buffer(cmd, length);
			}

			copy_image_to_buffer_impl(cmd, src, scratch_buf, src_copy);
			copy_buffer_to_image_impl(cmd, scratch_buf, dst, dst_copy);

			if (remaining_levels > 1)
			{
				increment_mip_level(src_copy);
				increment_mip_level(dst_copy);
			}
		}
	}

	void copy_image_typeless(
		mtl::command_list& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		u32 src_transfer_mask, u32 dst_transfer_mask)
	{
		ensure(!src_rect.is_flipped() && !dst_rect.is_flipped()); //<- Flip conversion must be handled by the caller.

		if (src->format() == dst->format())
		{
			if (src->format_class() == dst->format_class())
			{
				rsx_log.warning("[Performance warning] Image copy requested incorrectly for matching formats.");
				copy_image(cmd, src, dst, src_rect, dst_rect, mip_layers, src_transfer_mask, dst_transfer_mask);
				return;
			}
			else
			{
				// Should only happen for DEPTH_FLOAT <-> DEPTH_UINT at this time
				const u32 mask = src->format_class() | dst->format_class();
				if (mask != (RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32 | RSX_FORMAT_CLASS_DEPTH24_UNORM_X8_PACK32))
				{
					rsx_log.error("Unexpected (and possibly incorrect) typeless transfer setup.");
				}
			}
		}

		if (cmd.is_render_pass_open())
		{
			cmd.end_render_pass();
		}

		buffer_image_copy src_copy{}, dst_copy{};
		src_copy.image_extent = to_size(src_rect);
		src_copy.image_offset = to_origin(src_rect);
		src_copy.aspect = src->aspect() & src_transfer_mask;
		src_copy.mip_level = mip_layers.src_mip_level;
		src_copy.base_layer = mip_layers.src_layer;
		src_copy.layer_count = mip_layers.layer_count;

		dst_copy.image_extent = to_size(dst_rect);
		dst_copy.image_offset = to_origin(dst_rect);
		dst_copy.aspect = dst->aspect() & dst_transfer_mask;
		dst_copy.mip_level = mip_layers.dst_mip_level;
		dst_copy.base_layer = mip_layers.dst_layer;
		dst_copy.layer_count = mip_layers.layer_count;

		const auto src_texel_size = mtl::get_format_texel_width(src->info.format);
		mtl::buffer* scratch_buf = nullptr;

		for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; remaining_levels--)
		{
			const auto src_length = static_cast<u32>(src_texel_size * src_copy.image_extent.width * src_copy.image_extent.height * src_copy.image_extent.depth * mip_layers.layer_count);
			if (!scratch_buf)
			{
				// Initialize scratch memory
				const auto min_scratch_size = calculate_working_buffer_size(src_length, src->aspect() | dst->aspect());
				scratch_buf = mtl::get_scratch_buffer(cmd, min_scratch_size);
			}

			mtl::copy_image_to_buffer(cmd, src, scratch_buf, src_copy);

			auto src_convert = get_format_convert_flags(src->info.format);
			auto dst_convert = get_format_convert_flags(dst->info.format);

			const bool needs_shuffle = (src_convert.first || dst_convert.first) &&
				(src_convert.first != dst_convert.first || src_convert.second != dst_convert.second);

			if (needs_shuffle)
			{
				mtl::cs_shuffle_base* shuffle_kernel = nullptr;
				if (src_convert.first && dst_convert.first)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32_16>();
				}
				else
				{
					const auto block_size = src_convert.first ? src_convert.second : dst_convert.second;
					if (block_size == 4)
					{
						shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32>();
					}
					else if (block_size == 2)
					{
						shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_16>();
					}
					else
					{
						fmt::throw_exception("Unreachable");
					}
				}

				shuffle_kernel->run(cmd, scratch_buf, src_length);
			}

			mtl::copy_buffer_to_image(cmd, scratch_buf, dst, dst_copy);

			if (remaining_levels > 1)
			{
				increment_mip_level(src_copy);
				increment_mip_level(dst_copy);
			}
		}
	}

	void copy_image(mtl::command_list& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		u32 src_transfer_mask, u32 dst_transfer_mask)
	{
		if (const u32 aspect_bridge = (src->aspect() | dst->aspect());
			(aspect_bridge & aspect_color) == 0 &&
			src->format() != dst->format())
		{
			// Copying between two depth formats must match exactly or crashes will happen
			rsx_log.warning("[Performance warning] Image copy was requested incorrectly for mismatched depth formats");
			copy_image_typeless(cmd, src, dst, src_rect, dst_rect, mip_layers);
			return;
		}

		ensure(!src_rect.is_flipped() && !dst_rect.is_flipped()); //<- Flip conversion must be handled by the caller.

		if (cmd.is_render_pass_open())
		{
			cmd.end_render_pass();
		}

		const u32 src_aspect = src->aspect() & src_transfer_mask;
		const u32 dst_aspect = dst->aspect() & dst_transfer_mask;

		if (src->format() != dst->format() || src_aspect != src->aspect() || dst_aspect != dst->aspect())
		{
			// copyFromTexture needs identical formats and always moves every plane
			ensure(src->samples() == 1 && dst->samples() == 1);
			copy_image_via_buffer(cmd, src, dst, src_rect, dst_rect, mip_layers, src_aspect, dst_aspect);
			return;
		}

		image_copy_t rgn{};
		rgn.extent = to_size(src_rect);
		rgn.src_offset = to_origin(src_rect);
		rgn.dst_offset = to_origin(dst_rect);
		rgn.src_level = mip_layers.src_mip_level;
		rgn.dst_level = mip_layers.dst_mip_level;
		rgn.src_layer = mip_layers.src_layer;
		rgn.dst_layer = mip_layers.dst_layer;
		rgn.layer_count = mip_layers.layer_count;

		rsx::simple_array<image_copy_t> regions;
		for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; --remaining_levels)
		{
			regions.push_back(rgn);

			if (remaining_levels > 1)
			{
				increment_mip_level(rgn);
			}
		}

		ensure(!regions.empty());
		copy_image_regions(cmd, src, dst, regions.data(), regions.size());
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Uploads
	// ---------------------------------------------------------------------------------------------------------------

	template <typename BaseType, typename BlockType>
	cs_deswizzle_base* get_deswizzle_transformation_swapped(bool swap_bytes)
	{
		if (swap_bytes) [[ likely ]]
		{
			return mtl::get_compute_task<cs_deswizzle_3d<BaseType, BlockType, true>>();
		}

		return mtl::get_compute_task<cs_deswizzle_3d<BaseType, BlockType, false>>();
	}

	template <typename WordType>
	cs_deswizzle_base* get_deswizzle_transformation(u32 block_size, bool swap_bytes)
	{
		switch (block_size)
		{
		case 1:
			return get_deswizzle_transformation_swapped<u8, u8>(swap_bytes);
		case 2:
			return get_deswizzle_transformation_swapped<u16, WordType>(swap_bytes);
		case 4:
		case 8:
		case 16:
			// Maximum block size on RSX is 4 bytes. Wider blocks are stored as multiple texels.
			return get_deswizzle_transformation_swapped<u32, WordType>(swap_bytes);
		default:
			fmt::throw_exception("Unreachable");
		}
	}

	static void gpu_deswizzle_sections_impl(mtl::command_list& cmd, mtl::buffer* scratch_buf, u32 dst_offset, int word_size, int word_count, bool swap_bytes, std::vector<buffer_image_copy>& sections)
	{
		// NOTE: This has to be done individually for every LOD
		mtl::cs_deswizzle_base* job = nullptr;
		const u32 block_size = (word_size * word_count);
		const u32 scale_x = std::max(block_size / 4u, 1u); // Virtual width multiplier. RSX only does texel sizes upto 32 bits.

		switch (word_size)
		{
		case 1:
			job = get_deswizzle_transformation<u8>(block_size, swap_bytes);
			break;
		case 2:
			job = get_deswizzle_transformation<u16>(block_size, swap_bytes);
			break;
		case 4:
			job = get_deswizzle_transformation<u32>(block_size, swap_bytes);
			break;
		default:
			fmt::throw_exception("Unimplemented deswizzle for format.");
		}

		ensure(job);

		auto next_layer = sections.front().base_layer;
		auto next_level = sections.front().mip_level;
		unsigned base = 0;
		unsigned lods = 0;

		std::vector<std::pair<unsigned, unsigned>> packets;
		for (unsigned i = 0; i < sections.size(); ++i)
		{
			ensure(sections[i].buffer_row_length);

			const auto layer = sections[i].base_layer;
			const auto level = sections[i].mip_level;

			if (layer == next_layer &&
				level == next_level)
			{
				next_level++;
				lods++;
				continue;
			}

			packets.emplace_back(base, lods);
			next_layer = layer;
			next_level = 1;
			base = i;
			lods = 1;
		}

		if (packets.empty() ||
			(packets.back().first + packets.back().second) < sections.size())
		{
			packets.emplace_back(base, lods);
		}

		for (const auto &packet : packets)
		{
			const auto& section = sections[packet.first];
			const auto src_offset = section.buffer_offset;

			// Align output to 128-byte boundary to keep some drivers happy
			dst_offset = utils::align(dst_offset, 128);

			u32 data_length = 0;
			for (unsigned i = 0, j = packet.first; i < packet.second; ++i, ++j)
			{
				const u32 packed_size = static_cast<u32>(sections[j].image_extent.width * sections[j].image_extent.height * sections[j].image_extent.depth * block_size);
				sections[j].buffer_offset = dst_offset;
				dst_offset += packed_size;
				data_length += packed_size;
			}

			const u32 buf_off32 = static_cast<u32>(section.buffer_offset);
			const u32 src_off32 = static_cast<u32>(src_offset);

			job->run(cmd, scratch_buf, buf_off32, scratch_buf, src_off32, data_length,
				static_cast<u32>(section.image_extent.width) * scale_x, static_cast<u32>(section.image_extent.height), static_cast<u32>(section.image_extent.depth), packet.second);
		}

		ensure(dst_offset <= scratch_buf->size());
	}

	static mtl::command_list& prepare_for_transfer(mtl::command_list& primary_cb, mtl::image* /*dst_image*/, rsx::flags32_t& flags)
	{
		// Metal has no async transfer queue. "Async" uploads are recorded into the primary list's prologue instead, which
		// executes before everything recorded in the primary list (see mtl::command_list), so an open render pass stays
		// open. The caller guarantees that no work already recorded in primary_cb references dst_image. The sources must
		// be written by the CPU: data produced by GPU work of this submission (source_is_gpu_resident, e.g. detiled
		// memory in the scratch buffer) would be read before it is written.
		if ((flags & image_upload_options::upload_contents_async) && !(flags & image_upload_options::source_is_gpu_resident))
		{
			if (auto prologue = primary_cb.prologue())
			{
				count_image_upload(true, false);
				return *prologue;
			}
		}

		flags &= ~image_upload_options::upload_contents_async;

		const bool ends_pass = primary_cb.is_render_pass_open();
		count_image_upload(false, ends_pass);

		if (ends_pass)
		{
			primary_cb.end_render_pass();
		}

		return primary_cb;
	}

	static const std::pair<u32, u32> calculate_upload_pitch(int format, u32 heap_align, mtl::image* dst_image, const rsx::subresource_layout& layout, const rsx::texture_uploader_capabilities& caps)
	{
		u32 block_in_pixel = rsx::get_format_block_size_in_texel(format);
		u8  block_size_in_bytes = rsx::get_format_block_size_in_bytes(format);

		u32 row_pitch, upload_pitch_in_texel;

		if (!heap_align) [[likely]]
		{
			if (!layout.border) [[likely]]
			{
				row_pitch = (layout.pitch_in_block * block_size_in_bytes);
			}
			else
			{
				// Skip the border texels if possible. Padding is undesirable for GPU deswizzle
				row_pitch = (layout.width_in_block * block_size_in_bytes);
			}

			// We have row_pitch in source coordinates. But some formats have a software decode step which can affect this packing!
			// For such formats, the packed pitch on src does not match packed pitch on dst
			if (!rsx::is_compressed_host_format(caps, format))
			{
				const auto host_texel_width = mtl::get_format_texel_width(dst_image->format());
				const auto host_packed_pitch = host_texel_width * layout.width_in_texel;
				row_pitch = std::max<u32>(row_pitch, host_packed_pitch);
				upload_pitch_in_texel = row_pitch / host_texel_width;
			}
			else
			{
				upload_pitch_in_texel = std::max<u32>(block_in_pixel * row_pitch / block_size_in_bytes, layout.width_in_texel);
			}
		}
		else
		{
			row_pitch = rsx::align2(layout.width_in_block * block_size_in_bytes, heap_align);
			upload_pitch_in_texel = std::max<u32>(block_in_pixel * row_pitch / block_size_in_bytes, layout.width_in_texel);
			ensure(row_pitch == heap_align);
		}

		return { row_pitch, upload_pitch_in_texel };
	}

	void upload_image(mtl::command_list& cmd, mtl::image* dst_image,
		const std::vector<rsx::subresource_layout>& subresource_layout, int format, bool is_swizzled, u16 layer_count,
		u32 flags, mtl::data_heap& upload_heap, u32 heap_align, rsx::flags32_t image_setup_flags)
	{
		const bool requires_depth_processing = (dst_image->aspect() & aspect_stencil) || (format == CELL_GCM_TEXTURE_DEPTH16_FLOAT);
		auto pdev = mtl::get_current_renderer();
		rsx::texture_uploader_capabilities caps{ .supports_dxt = pdev->caps().bc_texture_compression, .alignment = heap_align };
		rsx::texture_memory_info opt{};
		bool check_hw_caps = !(image_setup_flags & source_is_userptr);

		mtl::buffer* scratch_buf = nullptr;
		u32 scratch_offset = 0;
		u32 image_linear_size;

		mtl::buffer* upload_buffer = nullptr;
		usz offset_in_upload_buffer = 0;

		std::vector<buffer_image_copy> copy_regions;
		std::vector<buffer_copy_t> buffer_copies;
		std::vector<std::pair<mtl::buffer*, u32>> upload_commands;
		copy_regions.reserve(subresource_layout.size());

		auto& cmd2 = prepare_for_transfer(cmd, dst_image, image_setup_flags);

		for (const rsx::subresource_layout &layout : subresource_layout)
		{
			if (layout.level >= dst_image->mipmaps())
			{
				rsx_log.error("Invalid subresource definition for the output texture. Mip level does not exist.");
				continue;
			}

			const auto [row_pitch, upload_pitch_in_texel] = calculate_upload_pitch(format, heap_align, dst_image, layout, caps);
			caps.alignment = row_pitch;

			// Calculate estimated memory utilization for this subresource
			image_linear_size = row_pitch * layout.depth * (rsx::is_compressed_host_format(caps, format) ? layout.height_in_block : layout.height_in_texel);

			// Only do GPU-side conversion if occupancy is good
			if (check_hw_caps)
			{
				caps.supports_byteswap = (image_linear_size >= 1024) || (image_setup_flags & source_is_gpu_resident);
				caps.supports_hw_deswizzle = caps.supports_byteswap;

				// No zero-copy uploads: with a passthrough DMA buffer the GPU would read the texture from guest memory
				// when the command list executes, possibly frames after the RSX reached the draw and after the game was
				// told (texture read semaphore) that it may overwrite the memory. Video players decode the next frame
				// into the same buffer (Bink on SPU, cellVdec): the GPU then copies a half-written frame, which shows as
				// tearing, blocky frames and flicker. The data is copied into the upload heap here instead, which is what
				// the RSX would have read at this point (a memcpy; Apple silicon copies tens of GB/s).
				caps.supports_zero_copy = false;
				caps.supports_vtc_decoding = false;
				check_hw_caps = false;
			}

			auto buf_allocator = [&](usz) -> std::tuple<void*, usz>
			{
				if (image_setup_flags & source_is_gpu_resident)
				{
					// We should never reach here, unless something is very wrong...
					fmt::throw_exception("Cannot allocate CPU memory for GPU-only data");
				}

				// Map with extra padding bytes in case of realignment
				offset_in_upload_buffer = upload_heap.alloc<512>(image_linear_size + 8);
				void* mapped_buffer = upload_heap.map(offset_in_upload_buffer, image_linear_size + 8);
				return { mapped_buffer, image_linear_size };
			};

			auto io_buf = rsx::io_buffer(buf_allocator);
			opt = upload_texture_subresource(io_buf, layout, format, is_swizzled, caps);
			upload_heap.unmap();

			if (image_setup_flags & source_is_gpu_resident)
			{
				// Read from GPU buf if the input is already uploaded.
				auto [iobuf, io_offset] = layout.data.raw();
				upload_buffer = static_cast<mtl::buffer*>(iobuf);
				offset_in_upload_buffer = io_offset;
				// Never upload. Data is already resident.
				opt.require_upload = false;
			}
			else
			{
				// Read from upload buffer
				upload_buffer = upload_heap.heap.get();
			}

			copy_regions.push_back({});
			auto& copy_info = copy_regions.back();
			copy_info.buffer_offset = offset_in_upload_buffer;
			copy_info.image_extent.height = layout.height_in_texel;
			copy_info.image_extent.width = layout.width_in_texel;
			copy_info.image_extent.depth = layout.depth;
			copy_info.aspect = flags;
			copy_info.layer_count = 1;
			copy_info.base_layer = layout.layer;
			copy_info.mip_level = layout.level;
			copy_info.buffer_row_length = upload_pitch_in_texel;

			if (opt.require_upload)
			{
				ensure(!opt.deferred_cmds.empty());

				// Zero-copy reads a DMA block, which GPU work of this list may write back to (surface flushes): never from
				// the prologue (upload_contents_async is still set only when recording there). Unreachable while
				// supports_zero_copy stays off, see above.
				ensure(!(image_setup_flags & upload_contents_async));

				auto base_addr = static_cast<const char*>(opt.deferred_cmds.front().src);
				auto end_addr = static_cast<const char*>(opt.deferred_cmds.back().src) + opt.deferred_cmds.back().length;
				auto data_length = static_cast<u32>(end_addr - base_addr);
				u64 src_address = 0;

				if (uptr(base_addr) > uptr(vm::g_sudo_addr))
				{
					src_address = uptr(base_addr) - uptr(vm::g_sudo_addr);
				}
				else
				{
					src_address = uptr(base_addr) - uptr(vm::g_base_addr);
				}

				auto dma_mapping = mtl::map_dma(static_cast<u32>(src_address), static_cast<u32>(data_length));

				ensure(dma_mapping.second->size() >= (dma_mapping.first + data_length));
				mtl::load_dma(::narrow<u32>(src_address), data_length);

				upload_buffer = dma_mapping.second;
				offset_in_upload_buffer = dma_mapping.first;
				copy_info.buffer_offset = offset_in_upload_buffer;
			}
			else if (!layout.layer && !layout.level)
			{
				// Do not allow mixed transfer modes.
				// This can happen in special cases, e.g mipN having different processing than mip0 as is the case with the last VTC mip
				caps.supports_zero_copy = false;
			}

			if (opt.require_swap || opt.require_deswizzle || requires_depth_processing)
			{
				if (!scratch_buf)
				{
					// Calculate enough scratch memory. We need 2x the size of layer 0 to fit all the mip levels and an extra 128 bytes per level as alignment overhead.
					const u64 layer_size = (image_linear_size + image_linear_size);
					u64 scratch_buf_size = 128u * ::size32(subresource_layout) + (layer_size * layer_count);
					if (opt.require_deswizzle)
					{
						// Double the memory if hw deswizzle is going to be used.
						// For GPU deswizzle, the memory is not transformed in-place, rather the decoded texture is placed at the end of the uploaded data.
						scratch_buf_size += scratch_buf_size;
					}

					if (requires_depth_processing)
					{
						// D-S aspect requires a load section that can fit a separated block => D(4) + S(1)
						// Due to reverse processing of inputs, only enough space to fit one layer is needed here.
						scratch_buf_size += (image_linear_size * 5) / 4;
					}

					scratch_buf = mtl::get_scratch_buffer(cmd2, scratch_buf_size);
					buffer_copies.reserve(subresource_layout.size());
				}

				if (layout.level == 0)
				{
					// Align mip0 on a 128-byte boundary
					scratch_offset = utils::align(scratch_offset, 128);
				}

				// Copy from upload heap to scratch mem
				if (opt.require_upload)
				{
					for (const auto& copy_cmd : opt.deferred_cmds)
					{
						buffer_copies.push_back({});
						auto& copy = buffer_copies.back();
						copy.src_offset = uptr(copy_cmd.dst) + offset_in_upload_buffer;
						copy.dst_offset = scratch_offset;
						copy.size = copy_cmd.length;
					}
				}
				else if (upload_buffer != scratch_buf || offset_in_upload_buffer != scratch_offset)
				{
					buffer_copies.push_back({});
					auto& copy = buffer_copies.back();
					copy.src_offset = offset_in_upload_buffer;
					copy.dst_offset = scratch_offset;
					// Round up to a word so the (4-byte aligned) copy stays a blit: the heap allocation has 8 bytes of
					// padding and the next scratch subresource starts 16-byte aligned.
					copy.size = utils::align<u64>(image_linear_size, 4);
				}

				// Point data source to scratch mem
				copy_info.buffer_offset = scratch_offset;

				// Metal: keep every subresource 16-byte aligned (buffer -> texture copy offsets, compute bindings).
				// The 128 bytes per level reserved above cover the padding.
				scratch_offset = utils::align(scratch_offset + image_linear_size, static_cast<u32>(s_texture_copy_buffer_alignment));
				ensure((scratch_offset + image_linear_size) <= scratch_buf->size()); // "Out of scratch memory"
			}

			if (opt.require_upload)
			{
				if (upload_commands.empty() || upload_buffer->value() != upload_commands.back().first->value())
				{
					upload_commands.emplace_back(upload_buffer, 1);
				}
				else
				{
					upload_commands.back().second++;
				}

				copy_info.buffer_row_length = upload_pitch_in_texel;
			}
		}

		ensure(upload_buffer);

		if (opt.require_swap || opt.require_deswizzle || requires_depth_processing)
		{
			ensure(scratch_buf);

			if (upload_commands.size() > 1)
			{
				auto range_ptr = buffer_copies.data();
				for (const auto& op : upload_commands)
				{
					copy_buffer_regions(cmd2, op.first, scratch_buf, range_ptr, op.second);
					range_ptr += op.second;
				}
			}
			else
			{
				ensure(!buffer_copies.empty());
				copy_buffer_regions(cmd2, upload_buffer, scratch_buf, buffer_copies.data(), buffer_copies.size());
			}
		}

		// Swap and deswizzle if requested
		if (opt.require_deswizzle)
		{
			gpu_deswizzle_sections_impl(cmd2, scratch_buf, scratch_offset, opt.element_size, opt.block_length, opt.require_swap, copy_regions);
		}
		else if (opt.require_swap)
		{
			gpu_swap_bytes_impl(cmd2, scratch_buf, opt.element_size, 0, scratch_offset);
		}

		// CopyBufferToImage routines
		if (requires_depth_processing)
		{
			// Upload in reverse to avoid polluting data in lower space
			for (auto rIt = copy_regions.crbegin(); rIt != copy_regions.crend(); ++rIt)
			{
				mtl::copy_buffer_to_image(cmd2, scratch_buf, dst_image, *rIt);
			}
		}
		else if (scratch_buf)
		{
			ensure(opt.require_deswizzle || opt.require_swap);

			for (usz i = 0; i < copy_regions.size(); ++i)
			{
				copy_buffer_to_image_impl(cmd2, scratch_buf, dst_image, copy_regions[i], i == 0);
			}
		}
		else if (upload_commands.size() > 1)
		{
			auto region_ptr = copy_regions.data();
			for (const auto& op : upload_commands)
			{
				for (u32 i = 0; i < op.second; ++i)
				{
					copy_buffer_to_image_impl(cmd2, op.first, dst_image, region_ptr[i], i == 0);
				}
				region_ptr += op.second;
			}
		}
		else
		{
			for (usz i = 0; i < copy_regions.size(); ++i)
			{
				copy_buffer_to_image_impl(cmd2, upload_buffer, dst_image, copy_regions[i], i == 0);
			}
		}

		if (auto rsxthr = static_cast<MTLGSRender*>(rsx::get_current_renderer()))
		{
			rsxthr->on_guest_texture_read(cmd2);
		}
	}

	std::pair<mtl::buffer*, u32> detile_memory_block(mtl::command_list& cmd, const rsx::GCM_tile_reference& tiled_region,
		const utils::address_range32& range, u16 width, u16 height, u8 bpp)
	{
		// Calculate the true length of the usable memory section
		const auto available_tile_size = tiled_region.tile->size - (range.start - tiled_region.base_address);
		const auto max_content_size = tiled_region.tile->pitch * utils::align<u32>(height, 64);
		const auto section_length = std::min(max_content_size, available_tile_size);

		// Sync the DMA layer
		const auto dma_mapping = mtl::map_dma(range.start, section_length);
		mtl::load_dma(range.start, section_length);

		// Allocate scratch and prepare for the GPU job
		// 0 = linear data, 1 = padding (deswz), 2 = tiled data. Metal: the tiled block is 256-byte aligned for the kernel binding.
		const auto tiled_data_scratch_offset = utils::align<u32>(section_length * 2, 256);
		const auto scratch_buf = mtl::get_scratch_buffer(cmd, tiled_data_scratch_offset + section_length);
		const auto linear_data_scratch_offset = 0u;

		// Schedule the job
		const RSX_detiler_config config =
		{
			.tile_base_address = tiled_region.base_address,
			.tile_base_offset = range.start - tiled_region.base_address,
			.tile_rw_offset = range.start - tiled_region.base_address,   // TODO
			.tile_size = tiled_region.tile->size,
			.tile_pitch = tiled_region.tile->pitch,
			.bank = tiled_region.tile->bank,

			.dst = scratch_buf,
			.dst_offset = linear_data_scratch_offset,
			.src = scratch_buf,
			.src_offset = tiled_data_scratch_offset,

			.image_width = width,
			.image_height = height,
			.image_pitch = static_cast<u32>(width) * bpp,
			.image_bpp = bpp
		};

		// Transfer
		const buffer_copy_t copy_rgn
		{
			.src_offset = dma_mapping.first,
			.dst_offset = tiled_data_scratch_offset,
			.size = section_length
		};
		copy_buffer_regions(cmd, dma_mapping.second, scratch_buf, &copy_rgn, 1);

		// Detile
		mtl::get_compute_task<mtl::cs_tile_memcpy<RSX_detiler_op::decode>>()->run(cmd, config);

		// Return a descriptor pointing to the decrypted data
		return { scratch_buf, linear_data_scratch_offset };
	}

	void blitter::scale_image(mtl::command_list& cmd, mtl::image* src, mtl::image* dst, areai src_area, areai dst_area, bool interpolate, const rsx::typeless_xfer& xfer_info)
	{
		mtl::image* real_src = src;
		mtl::image* real_dst = dst;

		// Optimization pass; check for pass-through data transfer
		if (!xfer_info.flip_horizontal && !xfer_info.flip_vertical && src_area.height() == dst_area.height())
		{
			auto src_w = src_area.width();
			auto dst_w = dst_area.width();

			if (xfer_info.src_is_typeless) src_w = static_cast<int>(src_w * xfer_info.src_scaling_hint);
			if (xfer_info.dst_is_typeless) dst_w = static_cast<int>(dst_w * xfer_info.dst_scaling_hint);

			if (src_w == dst_w)
			{
				// Final dimensions are a match
				if (xfer_info.src_is_typeless || xfer_info.dst_is_typeless)
				{
					mtl::copy_image_typeless(cmd, src, dst, src_area, dst_area);
				}
				else
				{
					copy_image(cmd, src, dst, src_area, dst_area);
				}

				return;
			}
		}

		if (xfer_info.src_is_typeless)
		{
			const auto format = xfer_info.src_native_format_override ?
				static_cast<MTL::PixelFormat>(xfer_info.src_native_format_override) :
				mtl::get_compatible_sampler_format(xfer_info.src_gcm_format);

			if (format != src->format())
			{
				// Normalize input region (memory optimization)
				const auto old_src_area = src_area;
				src_area.y2 -= src_area.y1;
				src_area.y1 = 0;
				src_area.x2 = static_cast<int>(src_area.width() * xfer_info.src_scaling_hint);
				src_area.x1 = 0;

				// Transfer bits from src to typeless src
				real_src = mtl::get_typeless_helper(format, rsx::classify_format(xfer_info.src_gcm_format), src_area.width(), src_area.height());
				mtl::copy_image_typeless(cmd, src, real_src, old_src_area, src_area);
			}
		}

		// Save output region descriptor
		const auto old_dst_area = dst_area;

		if (xfer_info.dst_is_typeless)
		{
			const auto format = xfer_info.dst_native_format_override ?
				static_cast<MTL::PixelFormat>(xfer_info.dst_native_format_override) :
				mtl::get_compatible_sampler_format(xfer_info.dst_gcm_format);

			if (format != dst->format())
			{
				// Normalize output region (memory optimization)
				dst_area.y2 -= dst_area.y1;
				dst_area.y1 = 0;
				dst_area.x2 = static_cast<int>(dst_area.width() * xfer_info.dst_scaling_hint);
				dst_area.x1 = 0;

				// Account for possibility where SRC is typeless and DST is typeless and both map to the same format
				auto required_height = dst_area.height();
				if (real_src != src && real_src->format() == format)
				{
					required_height += src_area.height();

					// Move the dst area just below the src area
					dst_area.y1 += src_area.y2;
					dst_area.y2 += src_area.y2;
				}

				real_dst = mtl::get_typeless_helper(format, rsx::classify_format(xfer_info.dst_gcm_format), dst_area.width(), required_height);
			}
		}

		// Checks
		if (src_area.x2 <= src_area.x1 || src_area.y2 <= src_area.y1 || dst_area.x2 <= dst_area.x1 || dst_area.y2 <= dst_area.y1)
		{
			rsx_log.error("Blit request consists of an empty region descriptor!");
			return;
		}

		if (src_area.x1 < 0 || src_area.x2 > static_cast<s32>(real_src->width()) || src_area.y1 < 0 || src_area.y2 > static_cast<s32>(real_src->height()))
		{
			rsx_log.error("Blit request denied because the source region does not fit!");
			return;
		}

		if (dst_area.x1 < 0 || dst_area.x2 > static_cast<s32>(real_dst->width()) || dst_area.y1 < 0 || dst_area.y2 > static_cast<s32>(real_dst->height()))
		{
			rsx_log.error("Blit request denied because the destination region does not fit!");
			return;
		}

		if (xfer_info.flip_horizontal)
		{
			src_area.flip_horizontal();
		}

		if (xfer_info.flip_vertical)
		{
			src_area.flip_vertical();
		}

		ensure(real_src->aspect() == real_dst->aspect()); // "Incompatible source and destination format!"

		mtl::copy_scaled_image(cmd, real_src, real_dst, src_area, dst_area, {},
			formats_are_bitcast_compatible(real_src, real_dst),
			interpolate);

		if (real_dst != dst)
		{
			mtl::copy_image_typeless(cmd, real_dst, dst, dst_area, old_dst_area);
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Clears
	// ---------------------------------------------------------------------------------------------------------------

	void clear_image(mtl::command_list& cmd, mtl::image* image, const image_clear_value& value, u32 base_level, u32 level_count)
	{
		const u32 mip_count = image->mipmaps();
		if (level_count == ~0u)
		{
			level_count = mip_count - base_level;
		}

		ensure(base_level + level_count <= mip_count);

		const bool is_3d = image->type() == MTL::TextureType3D;
		const u32 layer_count = is_3d ? 1 : image->layers();

		if (!format_is_renderable(image->format()) || !(image->value->usage() & MTL::TextureUsageRenderTarget))
		{
			// Zero fill from a cleared scratch buffer
			ensure(image->samples() == 1);

			const auto block = get_format_block_info(image->format(), image->aspect() == aspect_depth_stencil ? aspect_depth : image->aspect());
			ensure(image->aspect() != aspect_depth_stencil, "Depth-stencil textures must be renderable to be cleared");

			u64 max_level_size = 0;
			for (u32 level = base_level; level < base_level + level_count; ++level)
			{
				const u32 w = std::max(image->width() >> level, 1u);
				const u32 h = std::max(image->height() >> level, 1u);
				const u32 d = is_3d ? std::max(image->depth() >> level, 1u) : 1u;
				const u64 row = u64{ utils::aligned_div(w, u32{ block.block_width }) } * block.bytes_per_block;
				max_level_size = std::max<u64>(max_level_size, row * utils::aligned_div(h, u32{ block.block_height }) * d);
			}

			auto zero_buf = get_scratch_buffer(cmd, max_level_size, true);
			bool first = true;

			for (u32 level = base_level; level < base_level + level_count; ++level)
			{
				const u32 w = std::max(image->width() >> level, 1u);
				const u32 h = std::max(image->height() >> level, 1u);
				const u32 d = is_3d ? std::max(image->depth() >> level, 1u) : 1u;
				const u64 row = u64{ utils::aligned_div(w, u32{ block.block_width }) } * block.bytes_per_block;
				const u64 image_size = row * utils::aligned_div(h, u32{ block.block_height });

				for (u32 layer = 0; layer < layer_count; ++layer)
				{
					auto encoder = first ? cmd.compute() : cmd.compute_unordered();
					first = false;

					encoder->copyFromBuffer(zero_buf->value(), 0, row, d > 1 ? image_size : 0, MTL::Size::Make(w, h, d),
						image->value, layer, level, MTL::Origin::Make(0, 0, 0));
				}
			}

			return;
		}

		const u32 aspect = image->aspect();
		autorelease_scope pool;

		for (u32 level = base_level; level < base_level + level_count; ++level)
		{
			const u32 w = std::max(image->width() >> level, 1u);
			const u32 h = std::max(image->height() >> level, 1u);
			const u32 slices = is_3d ? std::max(image->depth() >> level, 1u) : layer_count;

			for (u32 slice = 0; slice < slices; ++slice)
			{
				auto desc = ref(MTL4::RenderPassDescriptor::alloc()->init());

				auto setup_attachment = [&](MTL::RenderPassAttachmentDescriptor* attachment)
				{
					attachment->setTexture(image->value);
					attachment->setLevel(level);
					if (is_3d)
					{
						attachment->setDepthPlane(slice);
					}
					else
					{
						attachment->setSlice(slice);
					}
					attachment->setLoadAction(MTL::LoadActionClear);
					attachment->setStoreAction(MTL::StoreActionStore);
				};

				if (aspect & aspect_color)
				{
					auto attachment = desc->colorAttachments()->object(0);
					setup_attachment(attachment);
					attachment->setClearColor(MTL::ClearColor::Make(value.color.r, value.color.g, value.color.b, value.color.a));
				}

				if (aspect & aspect_depth)
				{
					auto attachment = desc->depthAttachment();
					setup_attachment(attachment);
					attachment->setClearDepth(value.depth);
				}

				if (aspect & aspect_stencil)
				{
					auto attachment = desc->stencilAttachment();
					setup_attachment(attachment);
					attachment->setClearStencil(value.stencil);
				}

				desc->setRenderTargetWidth(w);
				desc->setRenderTargetHeight(h);

				cmd.begin_render_pass(desc.get());
				cmd.end_render_pass();
			}
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Scratch resources (port of vkutils/scratch.cpp)
	// ---------------------------------------------------------------------------------------------------------------

	static std::unordered_map<u32, std::unique_ptr<viewable_image>> g_null_image_views;
	static std::unordered_map<u64, std::unique_ptr<image>> g_typeless_textures;
	static std::unique_ptr<mtl::sampler> g_null_sampler;

	// Scratch memory handling. Use double-buffered resource to significantly cut down on GPU stalls.
	// Prologue uploads (command_list::prologue()) share these buffers with the main lists. That is safe: scratch data is
	// only read by work recorded right after its producer on the same list (upload: copy from the upload heap ->
	// swap/deswizzle -> copy to the image; detile -> upload), never across submissions, and prologue work never overlaps
	// other work (its encoders wait for all earlier queue work, all work of its main list waits for it). Running it
	// before main-list work that was recorded earlier cannot clobber scratch data that work still needs.
	struct scratch_buffer_pool_t
	{
		std::array<std::unique_ptr<buffer>, 2> scratch_buffers;
		u32 current_index = 0;

		std::unique_ptr<buffer>& get_buf()
		{
			auto& ret = scratch_buffers[current_index];
			current_index ^= 1;
			return ret;
		}
	};

	static scratch_buffer_pool_t g_scratch_buffers_pool;
	static mtl::data_heap g_upload_heap;

	mtl::sampler* null_sampler()
	{
		if (g_null_sampler)
			return g_null_sampler.get();

		sampler_create_info info{};
		info.clamp_u = MTL::SamplerAddressModeRepeat;
		info.clamp_v = MTL::SamplerAddressModeRepeat;
		info.clamp_w = MTL::SamplerAddressModeRepeat;
		info.min_filter = MTL::SamplerMinMagFilterNearest;
		info.mag_filter = MTL::SamplerMinMagFilterNearest;
		info.mip_filter = MTL::SamplerMipFilterNearest;
		info.max_anisotropy = 1.f;
		info.compare_function = MTL::CompareFunctionNever;
		info.border_color = border_color_t(MTL::SamplerBorderColorOpaqueWhite);

		g_null_sampler = std::make_unique<mtl::sampler>(*g_render_device, info);
		return g_null_sampler.get();
	}

	mtl::image_view* null_image_view(mtl::command_list& cmd, MTL::TextureType type)
	{
		if (type == MTL::TextureType1D)
		{
			// 1D samplers are emitted as 2D (height 1) by the decompiler
			type = MTL::TextureType2D;
		}

		const u32 key = static_cast<u32>(type);
		if (auto found = g_null_image_views.find(key);
			found != g_null_image_views.end())
		{
			return found->second->get_view(rsx::default_remap_vector.with_encoding(MTL_REMAP_IDENTITY));
		}

		image_create_info info{};
		info.format = MTL::PixelFormatBGRA8Unorm;
		info.width = 4;
		info.height = 4;
		info.usage = MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget;
		info.storage = memory_location::device_local;
		info.format_class = RSX_FORMAT_CLASS_COLOR;

		switch (type)
		{
		case MTL::TextureType2D:
			info.type = MTL::TextureType2D;
			break;
		case MTL::TextureType3D:
			info.type = MTL::TextureType3D;
			info.depth = 4;
			break;
		case MTL::TextureTypeCube:
			info.type = MTL::TextureTypeCube;
			info.layers = 6;
			break;
		case MTL::TextureType2DArray:
			info.type = MTL::TextureType2DArray;
			info.layers = 2;
			break;
		default:
			rsx_log.fatal("Unexpected image view type 0x%x", static_cast<u32>(type));
			return nullptr;
		}

		auto& tex = g_null_image_views[key];
		tex = std::make_unique<viewable_image>(*g_render_device, info);
		tex->set_debug_name("Null image");

		// Initialize memory to transparent black
		clear_image(cmd, tex.get(), {});

		// Return view
		return tex->get_view(rsx::default_remap_vector.with_encoding(MTL_REMAP_IDENTITY));
	}

	mtl::image* get_typeless_helper(MTL::PixelFormat format, rsx::format_class format_class, u32 requested_width, u32 requested_height)
	{
		auto create_texture = [&]()
		{
			u32 new_width = utils::align(requested_width, 256u);
			u32 new_height = utils::align(requested_height, 256u);

			image_create_info info{};
			info.type = MTL::TextureType2D;
			info.format = format;
			info.width = new_width;
			info.height = new_height;
			info.usage = MTL::TextureUsageShaderRead;
			if (format_is_renderable(format))
			{
				// Typeless helpers are also destinations of scaled blits (sampled draws)
				info.usage |= MTL::TextureUsageRenderTarget;
			}
			info.storage = memory_location::device_local;
			info.format_class = format_class;

			return new mtl::image(*g_render_device, info);
		};

		const u64 key = (static_cast<u64>(format_class) << 32u) | static_cast<u64>(format);
		auto& ptr = g_typeless_textures[key];

		if (!ptr || ptr->width() < requested_width || ptr->height() < requested_height)
		{
			if (ptr)
			{
				requested_width = std::max(requested_width, ptr->width());
				requested_height = std::max(requested_height, ptr->height());
				get_resource_manager()->dispose(ptr);
			}

			ptr.reset(create_texture());
			ptr->set_debug_name(fmt::format("Scratch: Format=0x%x", static_cast<u32>(format)));
		}

		return ptr.get();
	}

	static std::pair<mtl::buffer*, bool> get_scratch_buffer(u64 min_required_size)
	{
		auto& scratch_buffer = g_scratch_buffers_pool.get_buf();
		bool is_new = false;

		if (scratch_buffer && scratch_buffer->size() < min_required_size)
		{
			// Scratch heap cannot fit requirements. Discard it and allocate a new one.
			mtl::get_resource_manager()->dispose(scratch_buffer);
		}

		if (!scratch_buffer)
		{
			// Choose optimal size
			const u64 alloc_size = utils::align(min_required_size, 0x100000);

			scratch_buffer = std::make_unique<mtl::buffer>(*g_render_device, alloc_size, memory_location::device_local, "scratch buffer");
			is_new = true;
		}

		return { scratch_buffer.get(), is_new };
	}

	mtl::buffer* get_scratch_buffer(mtl::command_list& cmd, u64 min_required_size, bool zero_memory)
	{
		const auto [buf, init_mem] = get_scratch_buffer(min_required_size);

		if (init_mem || zero_memory)
		{
			// Zero-initialize the allocated VRAM
			const u64 zero_length = init_mem ? buf->size() : utils::align(min_required_size, 4);
			cmd.compute()->fillBuffer(buf->value(), NS::Range::Make(0, zero_length), 0);
		}

		return buf;
	}

	void clear_scratch_resources()
	{
		g_null_image_views.clear();
		g_scratch_buffers_pool.scratch_buffers[0].reset();
		g_scratch_buffers_pool.scratch_buffers[1].reset();
		g_scratch_buffers_pool.current_index = 0;

		g_typeless_textures.clear();
		g_transfer_staging_buffer.reset();
		g_null_sampler.reset();
	}

	mtl::data_heap* get_upload_heap()
	{
		if (!g_upload_heap.heap)
		{
			g_upload_heap.create(64 * 0x100000, "auxilliary upload heap", 0x100000);
		}

		return &g_upload_heap;
	}
}
