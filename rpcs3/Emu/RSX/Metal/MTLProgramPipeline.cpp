#include "stdafx.h"
#include "MTLProgramPipeline.h"
#include "MTLPipelineCompiler.h"
#include "MTLResourceManager.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/device.h"
#include "mtlutils/sampler.h"

#include "util/asm.hpp"

#include <algorithm>

namespace mtl
{
	namespace glsl
	{
		namespace
		{
			constexpr u32 max_buffer_slots = gpu_capabilities::max_buffers_per_stage;    // 31
			constexpr u32 max_texture_slots = 64;                                         // Argument tables are sized to 64
			constexpr u32 max_sampler_slots = gpu_capabilities::max_samplers_per_stage;  // 16

			static_assert(max_texture_slots <= argument_table_shadow::max_textures);

			// program::m_uid source (programs are built on pipe-compiler threads)
			atomic_t<u64> g_next_program_uid = 0;

			bool is_buffer_type(program_input_type type)
			{
				return type == input_type_uniform_buffer || type == input_type_storage_buffer;
			}

			const char* to_string(::glsl::program_domain domain)
			{
				switch (domain)
				{
				case ::glsl::glsl_vertex_program: return "vertex";
				case ::glsl::glsl_fragment_program: return "fragment";
				case ::glsl::glsl_compute_program: return "compute";
				default: return "invalid";
				}
			}

			// Sampler bound to sampled-texture slots that were given no sampler (e.g. texelFetch-only textures).
			// Keeps every sampler slot the shader declares valid, since argument tables are shared between programs.
			class default_sampler_holder
			{
				std::mutex m_lock;
				std::unique_ptr<mtl::sampler> m_sampler;
				MTL::ResourceID m_id{};
				bool m_exit_callback_registered = false;

			public:
				MTL::ResourceID get()
				{
					std::lock_guard lock(m_lock);

					if (!m_sampler)
					{
						sampler_create_info info{};
						info.min_filter = MTL::SamplerMinMagFilterNearest;
						info.mag_filter = MTL::SamplerMinMagFilterNearest;
						info.mip_filter = MTL::SamplerMipFilterNotMipmapped;

						m_sampler = std::make_unique<mtl::sampler>(*ensure(g_render_device), info);
						m_id = m_sampler->resource_id();

						if (!m_exit_callback_registered)
						{
							// Released with the rest of the renderer resources (GPU idle at that point)
							m_exit_callback_registered = true;
							get_resource_manager()->add_exit_callback([this]()
							{
								std::lock_guard lock(m_lock);
								m_sampler.reset();
								m_id = {};
								m_exit_callback_registered = false;
							});
						}
					}

					return m_id;
				}
			};

			default_sampler_holder g_default_sampler;
		}

		binding_layout build_binding_layout(const std::vector<program_input>& inputs)
		{
			binding_layout layout{};
			u32 next_buffer = 0;
			u32 next_texture = 0;
			u32 next_sampler = 0;
			u32 push_constant_end = 0;

			auto claim = [&](const program_input& in) -> resource_slot&
			{
				if (in.location == umax)
				{
					fmt::throw_exception("Program input '%s' (set %u, %s) has no binding location", in.name, in.set, to_string(in.domain));
				}

				const auto [it, inserted] = layout.slots.try_emplace(in.location);
				if (!inserted)
				{
					fmt::throw_exception("Program input '%s' (set %u, %s) reuses binding %u which is already taken by another input of this stage",
						in.name, in.set, to_string(in.domain), in.location);
				}

				it->second.type = in.type;
				it->second.array_size = std::max(1u, in.array_size);
				return it->second;
			};

			// Slot exhaustion is data dependent (e.g. 16 RSX textures + stencil mirrors), so it never throws: the slot
			// keeps umax and the translation of a shader that really uses the resource fails (the draw is skipped).
			auto take_slots = [](u32& next, u32 count, u32 limit) -> u32
			{
				if (next + count > limit)
				{
					next = limit;
					return umax;
				}

				const u32 first = next;
				next += count;
				return first;
			};

			// Pass 1: buffers, in input order
			for (const auto& in : inputs)
			{
				if (!is_buffer_type(in.type))
				{
					continue;
				}

				auto& slot = claim(in);
				slot.buffer_index = take_slots(next_buffer, slot.array_size, max_buffer_slots);
			}

			// Pass 2: sampled textures first so that sampler index == texture index for the first 16 sampler slots.
			// Textures past that get no sampler: the translator gives them a constexpr nearest/clamp-to-border sampler
			// (see MTLShaderCompiler.cpp), which is exactly what stencil mirrors and texelFetch-only inputs use anyway.
			for (const auto& in : inputs)
			{
				if (in.type != input_type_texture)
				{
					continue;
				}

				auto& slot = claim(in);
				slot.texture_index = take_slots(next_texture, slot.array_size, max_texture_slots);

				if (slot.texture_index != umax && slot.texture_index == next_sampler &&
					next_sampler + slot.array_size <= max_sampler_slots)
				{
					slot.sampler_index = slot.texture_index;
					next_sampler += slot.array_size;
				}
			}

			// Pass 3: texel buffers and storage textures (no sampler)
			for (const auto& in : inputs)
			{
				if (in.type != input_type_texel_buffer && in.type != input_type_storage_texture)
				{
					continue;
				}

				auto& slot = claim(in);
				slot.texture_index = take_slots(next_texture, slot.array_size, max_texture_slots);
			}

			// Push constants: one block per stage, emulating Vulkan's shared push-constant space (absolute offsets)
			for (const auto& in : inputs)
			{
				switch (in.type)
				{
				case input_type_push_constant:
					push_constant_end = std::max(push_constant_end, in.push_constant.offset + in.push_constant.size);
					break;
				case input_type_attachment:
					// Framebuffer fetch; nothing to bind
					break;
				case input_type_uniform_buffer:
				case input_type_storage_buffer:
				case input_type_texture:
				case input_type_texel_buffer:
				case input_type_storage_texture:
					break;
				default:
					fmt::throw_exception("Program input '%s' has an invalid type %u", in.name, static_cast<u32>(in.type));
				}
			}

			if (push_constant_end)
			{
				// umax if all buffer slots are taken; the translation then reports it
				layout.push_constant_buffer_index = take_slots(next_buffer, 1, max_buffer_slots);
				layout.push_constant_size = utils::align(push_constant_end, 16u);
			}

			layout.buffer_count = next_buffer;
			layout.texture_count = next_texture;
			layout.sampler_count = next_sampler;
			return layout;
		}

		MTL::GPUAddress buffer_binding_info::gpu_address() const
		{
			if (buffer)
			{
				return buffer->gpu_address(offset);
			}

			if (raw_buffer)
			{
				return raw_buffer->gpuAddress() + offset;
			}

			return 0;
		}

		image_binding_info::image_binding_info(const mtl::image_view* view, const mtl::sampler* smp)
			: texture(view ? view->value : nullptr)
			, sampler(smp ? smp->value : nullptr)
			, texture_id(view ? view->resource_id : MTL::ResourceID{})
			, sampler_id(smp ? smp->resource_id() : MTL::ResourceID{})
		{
		}

		// ---- program --------------------------------------------------------------------------------------------

		program::program(MTL::RenderPipelineState* pipeline,
			const std::vector<program_input>& vertex_inputs,
			const std::vector<program_input>& fragment_inputs)
			: m_render_pipeline(ensure(pipeline))
			, m_uid(++g_next_program_uid)
		{
			m_inputs[binding_set_index_vertex] = vertex_inputs;
			m_inputs[binding_set_index_fragment] = fragment_inputs;
			init_layouts();
		}

		program::program(MTL::ComputePipelineState* pipeline,
			const std::vector<program_input>& compute_inputs)
			: m_compute_pipeline(ensure(pipeline))
			, m_uid(++g_next_program_uid)
		{
			m_inputs[binding_set_index_compute] = compute_inputs;
			init_layouts();

			m_compute_threads_per_group = static_cast<u32>(m_compute_pipeline->maxTotalThreadsPerThreadgroup());
		}

		program::~program()
		{
			// NOTE: Metal 4 command buffers do not retain pipeline states. Owners must hand programs that may still be
			// referenced by in-flight work to mtl::get_gc() (VK disposes programs the same way).
			if (m_render_pipeline)
			{
				m_render_pipeline->release();
				m_render_pipeline = nullptr;
			}

			if (m_compute_pipeline)
			{
				m_compute_pipeline->release();
				m_compute_pipeline = nullptr;
			}
		}

		void program::init_layouts()
		{
			for (u32 stage = 0; stage < binding_set_index_max_enum; ++stage)
			{
				m_layouts[stage] = build_binding_layout(m_inputs[stage]);

				// bind() visits every slot of the stage on each draw: iterate a contiguous copy, not the hash map
				auto& table_slots = m_table_slots[stage];
				table_slots.clear();
				table_slots.reserve(m_layouts[stage].slots.size());
				for (const auto& [location, slot] : m_layouts[stage].slots)
				{
					table_slots.push_back(slot);
				}

				auto& bindings = m_bindings[stage];
				bindings.push_constants.assign(m_layouts[stage].push_constant_size, 0);
				bindings.dirty = true;

				for (const auto& in : m_inputs[stage])
				{
					if (in.type == input_type_push_constant || in.type == input_type_attachment)
					{
						// No argument-table slot (push constants go through the stage's push block)
						continue;
					}

					ensure(in.set < binding_set_index_max_enum, "Invalid descriptor set index");
					ensure(in.location < max_binding_locations, "Binding location out of range");
					m_binding_stage_mask[in.set][in.location] |= static_cast<u8>(1u << stage);
				}
			}
		}

		template <typename F>
		void program::for_each_bound_slot(u32 set_id, u32 binding_point, F&& func)
		{
			if (set_id >= binding_set_index_max_enum || binding_point >= max_binding_locations) [[unlikely]]
			{
				rsx_log.error("Binding (set=%u, binding=%u) is out of range for this program", set_id, binding_point);
				return;
			}

			const u32 mask = m_binding_stage_mask[set_id][binding_point];
			for (u32 stage = 0; stage < binding_set_index_max_enum; ++stage)
			{
				if (!(mask & (1u << stage)))
				{
					continue;
				}

				const auto& slots = m_layouts[stage].slots;
				if (const auto found = slots.find(binding_point); found != slots.end())
				{
					func(stage, found->second);
				}
			}

			// Bindings the program does not declare (e.g. input attachments, which use framebuffer fetch) are ignored
		}

		u32 program::max_total_threads_per_threadgroup() const
		{
			return is_compute() ? m_compute_threads_per_group : 0;
		}

		bool program::has_uniform(program_input_type type, std::string_view uniform_name) const
		{
			for (const auto& stage_inputs : m_inputs)
			{
				if (std::any_of(stage_inputs.cbegin(), stage_inputs.cend(), [&](const program_input& in)
				{
					return in.type == type && in.name == uniform_name;
				}))
				{
					return true;
				}
			}

			return false;
		}

		std::pair<u32, u32> program::get_uniform_location(::glsl::program_domain domain, program_input_type type, std::string_view uniform_name) const
		{
			for (const auto& stage_inputs : m_inputs)
			{
				const auto result = std::find_if(stage_inputs.cbegin(), stage_inputs.cend(), [&](const program_input& in)
				{
					return in.domain == domain && in.type == type && in.name == uniform_name;
				});

				if (result != stage_inputs.cend())
				{
					return { result->set, result->location };
				}
			}

			return { umax, umax };
		}

		void program::bind_uniform(const buffer_binding_info& buffer, u32 set_id, u32 binding_point)
		{
			const MTL::GPUAddress address = buffer.gpu_address();
			const u32 range = static_cast<u32>(std::min<u64>(buffer.range, u32{umax}));

			for_each_bound_slot(set_id, binding_point, [&](u32 stage, const resource_slot& slot)
			{
				if (slot.buffer_index == umax) [[unlikely]]
				{
					rsx_log.error("Buffer bound to (set=%u, binding=%u), which is not a buffer input", set_id, binding_point);
					return;
				}

				auto& bindings = m_bindings[stage];
				if (bindings.buffers[slot.buffer_index] == address &&
					bindings.buffer_sizes[slot.buffer_index] == range)
				{
					return;
				}

				bindings.buffers[slot.buffer_index] = address;
				bindings.buffer_sizes[slot.buffer_index] = range;
				bindings.dirty = true;
			});
		}

		void program::bind_uniform(const image_binding_info& image, u32 set_id, u32 binding_point)
		{
			const MTL::ResourceID texture_id = image.texture_id;
			const MTL::ResourceID sampler_id = image.sampler_id;

			for_each_bound_slot(set_id, binding_point, [&](u32 stage, const resource_slot& slot)
			{
				if (slot.texture_index == umax) [[unlikely]]
				{
					rsx_log.error("Image bound to (set=%u, binding=%u), which is not an image input", set_id, binding_point);
					return;
				}

				auto& bindings = m_bindings[stage];
				if (bindings.textures[slot.texture_index]._impl == texture_id._impl &&
					(slot.sampler_index == umax || bindings.samplers[slot.sampler_index]._impl == sampler_id._impl))
				{
					return;
				}

				bindings.textures[slot.texture_index] = texture_id;

				if (slot.sampler_index != umax)
				{
					bindings.samplers[slot.sampler_index] = sampler_id;
				}

				bindings.dirty = true;
			});
		}

		void program::bind_uniform(const mtl::buffer_view* view, u32 set_id, u32 binding_point)
		{
			const MTL::ResourceID texture_id = view ? view->resource_id : MTL::ResourceID{};

			for_each_bound_slot(set_id, binding_point, [&](u32 stage, const resource_slot& slot)
			{
				if (slot.texture_index == umax) [[unlikely]]
				{
					rsx_log.error("Texel buffer bound to (set=%u, binding=%u), which is not a texture input", set_id, binding_point);
					return;
				}

				auto& bindings = m_bindings[stage];
				if (bindings.textures[slot.texture_index]._impl == texture_id._impl)
				{
					return;
				}

				bindings.textures[slot.texture_index] = texture_id;
				bindings.dirty = true;
			});
		}

		void program::bind_uniform_array(std::span<const image_binding_info> images, u32 set_id, u32 binding_point)
		{
			for_each_bound_slot(set_id, binding_point, [&](u32 stage, const resource_slot& slot)
			{
				if (slot.texture_index == umax) [[unlikely]]
				{
					rsx_log.error("Image array bound to (set=%u, binding=%u), which is not an image input", set_id, binding_point);
					return;
				}

				auto& bindings = m_bindings[stage];
				const u32 count = std::min<u32>(::size32(images), slot.array_size);

				for (u32 i = 0; i < count; ++i)
				{
					const auto& image = images[i];
					bindings.textures[slot.texture_index + i] = image.texture_id;

					if (slot.sampler_index != umax)
					{
						bindings.samplers[slot.sampler_index + i] = image.sampler_id;
					}
				}

				bindings.dirty = true;
			});
		}

		void program::push_constants(u32 /*set_id*/, u32 offset, u32 size, const void* data)
		{
			// Vulkan semantics: push constants form one address space shared by all stages; every stage sees the
			// bytes of its declared range(s). Each stage keeps a staging copy covering [0, its block size), so the
			// update is applied to every stage whose block overlaps it (set_id is irrelevant for routing).
			ensure(data || !size);

			for (auto& bindings : m_bindings)
			{
				auto& block = bindings.push_constants;
				if (offset >= block.size())
				{
					continue;
				}

				const u32 length = std::min<u32>(size, ::size32(block) - offset);
				std::memcpy(block.data() + offset, data, length);
				bindings.dirty = true;
			}
		}

		void program::enable_buffer_size_table(u32 stage_index)
		{
			ensure(stage_index < binding_set_index_max_enum);
			ensure(m_layouts[stage_index].buffer_count < max_buffer_slots, "No buffer slot left for the buffer-size table");
			m_bindings[stage_index].needs_buffer_sizes = true;
			// The size-table upload is gated on dirty (see write_table); make sure enabling it forces one upload
			m_bindings[stage_index].dirty = true;
		}

		void program::bind(mtl::command_list& cmd, mtl::data_heap& scratch)
		{
			auto write_table = [&](u32 stage, argument_table_slot table_slot)
			{
				MTL4::ArgumentTable* table = cmd.argument_table(table_slot);
				argument_table_shadow& contents = cmd.argument_table_contents(table_slot);
				const binding_layout& layout = m_layouts[stage];
				auto& bindings = m_bindings[stage];
				bool missing = false;

				// Argument tables are shared by every program recorded into this command list, and each draw/dispatch
				// snapshots the table when encoded. Every slot this stage uses must hold this program's resource, but only
				// the slots whose value differs from what the table holds are written (see argument_table_shadow).
				for (const resource_slot& slot : m_table_slots[stage])
				{
					if (slot.buffer_index != umax)
					{
						for (u32 i = 0; i < slot.array_size; ++i)
						{
							const MTL::GPUAddress address = bindings.buffers[slot.buffer_index + i];
							missing |= !address;

							if (contents.update_buffer(slot.buffer_index + i, address))
							{
								table->setAddress(address, slot.buffer_index + i);
							}
						}
						continue;
					}

					if (slot.texture_index != umax)
					{
						for (u32 i = 0; i < slot.array_size; ++i)
						{
							const MTL::ResourceID texture_id = bindings.textures[slot.texture_index + i];
							missing |= !texture_id._impl;

							if (contents.update_texture(slot.texture_index + i, texture_id))
							{
								table->setTexture(texture_id, slot.texture_index + i);
							}
						}
					}

					if (slot.sampler_index != umax)
					{
						for (u32 i = 0; i < slot.array_size; ++i)
						{
							MTL::ResourceID sampler_id = bindings.samplers[slot.sampler_index + i];
							if (!sampler_id._impl)
							{
								sampler_id = g_default_sampler.get();
							}

							if (contents.update_sampler(slot.sampler_index + i, sampler_id))
							{
								table->setSamplerState(sampler_id, slot.sampler_index + i);
							}
						}
					}
				}

				// Scratch uploads can be skipped when neither the bytes changed (bindings.dirty) nor the
				// shared table moved on: the table still holds our last upload then, and re-uploading an
				// identical block only burns scratch-ring space and a memcpy on every draw. The shadow peek
				// covers program interleaving (another program's bind overwrote the slot) and fresh command
				// lists (shadow invalidated on begin); scratch addresses stay valid while the list is open.
				MTL::GPUAddress table_addr = 0;
				const bool push_in_table = layout.push_constant_buffer_index != umax &&
					contents.peek_buffer(layout.push_constant_buffer_index, table_addr) &&
					table_addr == bindings.last_push_constants_addr && table_addr != 0;

				if (layout.push_constant_buffer_index != umax && (bindings.dirty || !push_in_table))
				{
					const usz length = bindings.push_constants.size();
					const auto heap_offset = scratch.alloc<256>(length);
					std::memcpy(scratch.map(heap_offset, length), bindings.push_constants.data(), length);

					const MTL::GPUAddress address = scratch.gpu_address(heap_offset);
					bindings.last_push_constants_addr = address;
					if (contents.update_buffer(layout.push_constant_buffer_index, address))
					{
						table->setAddress(address, layout.push_constant_buffer_index);
					}
				}

				table_addr = 0;
				const bool sizes_in_table = bindings.needs_buffer_sizes &&
					contents.peek_buffer(layout.buffer_count, table_addr) &&
					table_addr == bindings.last_buffer_sizes_addr && table_addr != 0;

				if (bindings.needs_buffer_sizes && (bindings.dirty || !sizes_in_table))
				{
					// spvBufferSizeConstants: one uint per Metal buffer index of this stage
					const usz length = sizeof(u32) * std::max(layout.buffer_count, 1u);
					const auto heap_offset = scratch.alloc<256>(length);
					std::memcpy(scratch.map(heap_offset, length), bindings.buffer_sizes.data(), length);

					const MTL::GPUAddress address = scratch.gpu_address(heap_offset);
					bindings.last_buffer_sizes_addr = address;
					if (contents.update_buffer(layout.buffer_count, address))
					{
						table->setAddress(address, layout.buffer_count);
					}
				}

				if (missing && !m_missing_binding_reported) [[unlikely]]
				{
					m_missing_binding_reported = true;
					rsx_log.error("Program bound with unassigned resources (stage %u). The GPU will read invalid descriptors.", stage);
				}

				bindings.dirty = false;
				return table;
			};

			if (is_compute())
			{
				// compute() orders this command after the previous one; the caller's dispatch can then use
				// compute_unordered() since it depends only on this bind.
				// NOTE: The pipeline state and the table are set for every dispatch. The compute encoder also records the
				// copies and fills, and dispatches are rare next to draws, so its state is not tracked.
				MTL4::ComputeCommandEncoder* encoder = cmd.compute();
				encoder->setComputePipelineState(m_compute_pipeline);
				encoder->setArgumentTable(write_table(binding_set_index_compute, table_compute));
				return;
			}

			MTL4::RenderCommandEncoder* encoder = cmd.render_encoder();
			ensure(encoder, "Graphics program bound outside of a render pass");

			// The pipeline state and the stage tables are render encoder state: set once per encoder (and pipeline).
			// Changes to a table's contents after it was set are seen by the draws encoded later.
			render_encoder_bindings& encoder_state = cmd.render_bindings();

			if (encoder_state.program_uid != m_uid)
			{
				encoder->setRenderPipelineState(m_render_pipeline);
				encoder_state.program_uid = m_uid;
			}

			MTL4::ArgumentTable* vertex_table = write_table(binding_set_index_vertex, table_vertex);
			if (!(encoder_state.tables_set & (1u << table_vertex)))
			{
				encoder->setArgumentTable(vertex_table, MTL::RenderStageVertex);
				encoder_state.tables_set |= (1u << table_vertex);
			}

			MTL4::ArgumentTable* fragment_table = write_table(binding_set_index_fragment, table_fragment);
			if (!(encoder_state.tables_set & (1u << table_fragment)))
			{
				encoder->setArgumentTable(fragment_table, MTL::RenderStageFragment);
				encoder_state.tables_set |= (1u << table_fragment);
			}
		}

		// ---- Static program helpers ---------------------------------------------------------------------------------

		std::unique_ptr<program> create_compute_program(const std::string& compute_glsl, const std::vector<program_input>& inputs)
		{
			shader cs;
			cs.create(::glsl::glsl_compute_program, compute_glsl);

			// The pipeline does not reference the library once built; the shader goes away with this scope
			return build_compute_program(cs, inputs);
		}

		std::unique_ptr<program> create_graphics_program(
			const std::string& vertex_glsl, const std::vector<program_input>& vertex_inputs,
			const std::string& fragment_glsl, const std::vector<program_input>& fragment_inputs,
			const graphics_pipeline_state& state)
		{
			shader vs;
			vs.create(::glsl::glsl_vertex_program, vertex_glsl);

			shader fs;
			fs.create(::glsl::glsl_fragment_program, fragment_glsl);

			return build_graphics_program(vs, fs, state, vertex_inputs, fragment_inputs);
		}
	}
}
