#include "stdafx.h"
#include "MTLGSRender.h"
#include "../Common/BufferUtils.h"
#include "../rsx_methods.h"
#include "mtlutils/buffer_object.h"

#include <span>

namespace mtl
{
	std::pair<MTL::PrimitiveType, bool> get_appropriate_topology(rsx::primitive_type mode)
	{
		switch (mode)
		{
		case rsx::primitive_type::lines:
			return { MTL::PrimitiveTypeLine, false };
		case rsx::primitive_type::line_loop:
			return { MTL::PrimitiveTypeLineStrip, true };
		case rsx::primitive_type::line_strip:
			return { MTL::PrimitiveTypeLineStrip, false };
		case rsx::primitive_type::points:
			return { MTL::PrimitiveTypePoint, false };
		case rsx::primitive_type::triangles:
			return { MTL::PrimitiveTypeTriangle, false };
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip:
			return { MTL::PrimitiveTypeTriangleStrip, false };
		case rsx::primitive_type::triangle_fan:
			// Metal has no triangle fans
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon:
			return { MTL::PrimitiveTypeTriangle, true };
		default:
			fmt::throw_exception("Unsupported primitive topology 0x%x", static_cast<u8>(mode));
		}
	}

	bool is_primitive_native(rsx::primitive_type mode)
	{
		return !get_appropriate_topology(mode).second;
	}

	MTL::IndexType get_index_type(rsx::index_array_type type)
	{
		switch (type)
		{
		case rsx::index_array_type::u32:
			return MTL::IndexTypeUInt32;
		case rsx::index_array_type::u16:
			return MTL::IndexTypeUInt16;
		}
		fmt::throw_exception("Invalid index array type (%u)", static_cast<u8>(type));
	}
}

namespace
{
	std::tuple<u32, std::tuple<u64, MTL::IndexType>> generate_emulating_index_buffer(
		const rsx::draw_clause& clause, u32 vertex_count,
		mtl::data_heap& m_index_buffer_ring_info)
	{
		u32 index_count = get_index_count(clause.primitive, vertex_count);
		u32 upload_size = index_count * sizeof(u16);

		u64 offset_in_index_buffer = m_index_buffer_ring_info.alloc<256>(upload_size);
		void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer, upload_size);

		g_fxo->get<rsx::dma_manager>().emulate_as_indexed(buf, clause.primitive, vertex_count);

		m_index_buffer_ring_info.unmap();
		return std::make_tuple(
			index_count, std::make_tuple(offset_in_index_buffer, MTL::IndexTypeUInt16));
	}

	struct vertex_input_state
	{
		MTL::PrimitiveType native_primitive_type;
		bool index_rebase;
		u32 min_index;
		u32 max_index;
		u32 vertex_draw_count;
		u32 vertex_index_offset;
		std::optional<std::tuple<u64, MTL::IndexType>> index_info;
	};

	struct draw_command_visitor
	{
		draw_command_visitor(mtl::data_heap& index_buffer_ring_info, rsx::vertex_input_layout& layout)
			: m_index_buffer_ring_info(index_buffer_ring_info)
			, m_vertex_layout(layout)
		{
		}

		vertex_input_state operator()(const rsx::draw_array_command& /*command*/)
		{
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(rsx::method_registers.current_draw_clause.primitive);
			const u32 vertex_count = rsx::method_registers.current_draw_clause.get_elements_count();
			const u32 min_index = rsx::method_registers.current_draw_clause.min_index();
			const u32 max_index = (min_index + vertex_count) - 1;

			if (primitives_emulated)
			{
				u32 index_count;
				std::optional<std::tuple<u64, MTL::IndexType>> index_info;

				std::tie(index_count, index_info) =
					generate_emulating_index_buffer(rsx::method_registers.current_draw_clause,
						vertex_count, m_index_buffer_ring_info);

				return{ prims, false, min_index, max_index, index_count, 0, index_info };
			}

			return{ prims, false, min_index, max_index, vertex_count, 0, {} };
		}

		vertex_input_state operator()(const rsx::draw_indexed_array_command& command)
		{
			auto primitive = rsx::method_registers.current_draw_clause.primitive;
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(primitive);
			const bool restart_index_enabled = rsx::method_registers.restart_index_enabled();
			const u32 restart_index = rsx::method_registers.restart_index();
			// Metal restarts strips natively (0xFFFF / 0xFFFFFFFF, always on). Lists never restart: remove the markers.
			const bool native_strip = (prims == MTL::PrimitiveTypeTriangleStrip || prims == MTL::PrimitiveTypeLineStrip);
			const bool emulate_restart = restart_index_enabled && !native_strip && mtl::emulate_primitive_restart(primitive);

			rsx::index_array_type index_type = rsx::method_registers.current_draw_clause.is_immediate_draw ?
				rsx::index_array_type::u32 :
				rsx::method_registers.index_type();

			const u32 source_index_count = rsx::method_registers.current_draw_clause.get_elements_count();
			std::span<const std::byte> index_source = command.raw_index_buffer;

			// Storage for 16-bit indices widened to 32-bit (big endian, like the guest data)
			std::vector<be_t<u32>> widened_indices;

			while (true)
			{
				const u32 type_size = get_index_type_size(index_type);

				u32 index_count = source_index_count;
				if (primitives_emulated)
					index_count = get_index_count(primitive, index_count);
				u32 upload_size = index_count * type_size;

				if (emulate_restart) upload_size *= 2;

				u64 offset_in_index_buffer = m_index_buffer_ring_info.alloc<64>(upload_size);
				void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer, upload_size);

				std::span<std::byte> dst;
				stx::single_ptr<std::byte[]> tmp;
				if (emulate_restart)
				{
					tmp = stx::make_single<std::byte[], false, 64>(upload_size);
					dst = std::span<std::byte>(tmp.get(), upload_size);
				}
				else
				{
					dst = std::span<std::byte>(static_cast<std::byte*>(buf), upload_size);
				}

				/**
				* Upload index (and expands it if primitive type is not natively supported).
				*/
				u32 min_index, max_index;
				std::tie(min_index, max_index, index_count) = write_index_array_data_to_buffer(
					dst,
					index_source, index_type,
					primitive,
					restart_index_enabled,
					restart_index,
					[](auto prim) { return !mtl::is_primitive_native(prim); });

				if (min_index >= max_index)
				{
					//empty set, do not draw
					m_index_buffer_ring_info.unmap();
					return{ prims, false, 0, 0, 0, 0, {} };
				}

				if (index_type == rsx::index_array_type::u16 && max_index == 0xFFFF)
				{
					// Metal always restarts strips on 0xFFFF (and cannot disable it). Restart markers are never counted
					// in min/max, so this is a literal vertex index: widen the source to 32-bit and upload again.
					// The restart index keeps its value; a 16-bit restart index > 0xFFFF still never matches.
					if (widened_indices.empty())
					{
						const auto src16 = std::span<const be_t<u16>>(reinterpret_cast<const be_t<u16>*>(index_source.data()), index_source.size() / sizeof(u16));
						widened_indices.resize(src16.size());

						for (usz i = 0; i < src16.size(); ++i)
						{
							widened_indices[i] = static_cast<u32>(src16[i]);
						}
					}

					m_index_buffer_ring_info.unmap();

					index_source = std::as_bytes(std::span<const be_t<u32>>(widened_indices));
					index_type = rsx::index_array_type::u32;
					continue;
				}

				if (emulate_restart)
				{
					if (index_type == rsx::index_array_type::u16)
					{
						index_count = rsx::remove_restart_index(static_cast<u16*>(buf), reinterpret_cast<u16*>(tmp.get()), index_count, u16{umax});
					}
					else
					{
						index_count = rsx::remove_restart_index(static_cast<u32*>(buf), reinterpret_cast<u32*>(tmp.get()), index_count, u32{umax});
					}
				}

				m_index_buffer_ring_info.unmap();

				std::optional<std::tuple<u64, MTL::IndexType>> index_info =
					std::make_tuple(offset_in_index_buffer, mtl::get_index_type(index_type));

				const auto index_offset = rsx::method_registers.vertex_data_base_index();
				return {prims, true, min_index, max_index, index_count, index_offset, index_info};
			}
		}

		vertex_input_state operator()(const rsx::draw_inlined_array& /*command*/)
		{
			auto &draw_clause = rsx::method_registers.current_draw_clause;
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(draw_clause.primitive);

			const auto stream_length = rsx::method_registers.current_draw_clause.inline_vertex_array.size();
			const u32 vertex_count = u32(stream_length * sizeof(u32)) / m_vertex_layout.interleaved_blocks[0]->attribute_stride;

			if (!primitives_emulated)
			{
				return{ prims, false, 0, vertex_count - 1, vertex_count, 0, {} };
			}

			u32 index_count;
			std::optional<std::tuple<u64, MTL::IndexType>> index_info;
			std::tie(index_count, index_info) = generate_emulating_index_buffer(draw_clause, vertex_count, m_index_buffer_ring_info);
			return{ prims, false, 0, vertex_count - 1, index_count, 0, index_info };
		}

	private:
		mtl::data_heap& m_index_buffer_ring_info;
		rsx::vertex_input_layout& m_vertex_layout;
	};
}

mtl::vertex_upload_info MTLGSRender::upload_vertex_data()
{
	draw_command_visitor visitor(m_index_buffer_ring_info, m_vertex_layout);
	auto result = std::visit(visitor, m_draw_processor.get_draw_command(rsx::method_registers));

	const u32 vertex_count = (result.max_index - result.min_index) + 1;
	u32 vertex_base = result.min_index;
	u32 index_base = 0;

	if (result.index_rebase)
	{
		vertex_base = rsx::get_index_from_base(vertex_base, rsx::method_registers.vertex_data_base_index());
		index_base = result.min_index;
	}

	//Do actual vertex upload
	auto required = calculate_memory_requirements(m_vertex_layout, vertex_base, vertex_count);
	u32 persistent_range_base = -1, volatile_range_base = -1;
	usz persistent_offset = -1, volatile_offset = -1;

	// Handle ring swaps (growth) triggered by previous allocations
	check_heap_status();

	for (u32 attempt = 0;; ++attempt)
	{
		persistent_range_base = -1;
		volatile_range_base = -1;
		persistent_offset = -1;
		volatile_offset = -1;

		if (required.first > 0)
		{
			//Check if cacheable
			//Only data in the 'persistent' block may be cached
			//TODO: make vertex cache keep local data beyond frame boundaries and hook notify command
			bool in_cache = false;
			bool to_store = false;
			u32  storage_address = -1;

			m_frame_stats.vertex_cache_request_count++;

			if (m_vertex_layout.interleaved_blocks.size() == 1 &&
				rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
			{
				const auto data_offset = (vertex_base * m_vertex_layout.interleaved_blocks[0]->attribute_stride);
				storage_address = m_vertex_layout.interleaved_blocks[0]->real_offset_address + data_offset;

				if (auto cached = m_vertex_cache->find_vertex_range(storage_address, required.first))
				{
					ensure(cached->local_address == storage_address);

					in_cache = true;
					persistent_range_base = cached->offset_in_heap;
				}
				else
				{
					to_store = true;
				}
			}

			if (!in_cache)
			{
				m_frame_stats.vertex_cache_miss_count++;

				persistent_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.first));
				persistent_range_base = static_cast<u32>(persistent_offset);

				if (to_store)
				{
					//store ref in vertex cache
					m_vertex_cache->store_range(storage_address, required.first, static_cast<u32>(persistent_offset));
				}
			}
		}

		if (required.second > 0)
		{
			volatile_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.second));
			volatile_range_base = static_cast<u32>(volatile_offset);
		}

		if (attempt || !mtl::test_status_interrupt(mtl::heap_changed))
		{
			break;
		}

		// The attribute ring swapped its backing buffer while allocating. Offsets obtained before the swap (including
		// cached ranges) refer to the old buffer; drop them and allocate again from the new one.
		check_heap_status();
	}

	//Write all the data once if possible
	if (required.first && required.second && volatile_offset > persistent_offset)
	{
		//Do this once for both to save time on map/unmap cycles
		const usz block_end = (volatile_offset + required.second);
		const usz block_size = block_end - persistent_offset;
		const usz volatile_offset_in_block = volatile_offset - persistent_offset;

		void *block_mapping = m_attrib_ring_info.map(persistent_offset, block_size);
		m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, block_mapping, static_cast<char*>(block_mapping) + volatile_offset_in_block);
		m_attrib_ring_info.unmap();
	}
	else
	{
		if (required.first > 0 && persistent_offset != umax)
		{
			void *persistent_mapping = m_attrib_ring_info.map(persistent_offset, required.first);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, persistent_mapping, nullptr);
			m_attrib_ring_info.unmap();
		}

		if (required.second > 0)
		{
			void *volatile_mapping = m_attrib_ring_info.map(volatile_offset, required.second);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, nullptr, volatile_mapping);
			m_attrib_ring_info.unmap();
		}
	}

	if (persistent_range_base != umax)
	{
		if (!m_persistent_attribute_storage || !m_persistent_attribute_storage->in_range(persistent_range_base, required.first, persistent_range_base))
		{
			ensure(m_texbuffer_view_size >= required.first);
			mtl::get_resource_manager()->dispose(m_persistent_attribute_storage);

			//View 64M blocks at a time
			const usz view_size = (persistent_range_base + m_texbuffer_view_size) > m_attrib_ring_info.size() ? m_attrib_ring_info.size() - persistent_range_base : m_texbuffer_view_size;
			m_persistent_attribute_storage = std::make_unique<mtl::buffer_view>(*m_device, *m_attrib_ring_info.heap, MTL::PixelFormatR8Uint, persistent_range_base, view_size);
			persistent_range_base = 0;
		}
	}

	if (volatile_range_base != umax)
	{
		if (!m_volatile_attribute_storage || !m_volatile_attribute_storage->in_range(volatile_range_base, required.second, volatile_range_base))
		{
			ensure(m_texbuffer_view_size >= required.second);
			mtl::get_resource_manager()->dispose(m_volatile_attribute_storage);

			const usz view_size = (volatile_range_base + m_texbuffer_view_size) > m_attrib_ring_info.size() ? m_attrib_ring_info.size() - volatile_range_base : m_texbuffer_view_size;
			m_volatile_attribute_storage = std::make_unique<mtl::buffer_view>(*m_device, *m_attrib_ring_info.heap, MTL::PixelFormatR8Uint, volatile_range_base, view_size);
			volatile_range_base = 0;
		}
	}

	return{ result.native_primitive_type,                 // Primitive
			result.vertex_draw_count,                     // Vertex count
			vertex_count,                                 // Allocated vertex count
			vertex_base,                                  // First vertex in stream
			index_base,                                   // Index of vertex at data location 0
			result.vertex_index_offset,                   // Index offset
			persistent_range_base, volatile_range_base,   // Binding range
			result.index_info };                          // Index buffer info
}
