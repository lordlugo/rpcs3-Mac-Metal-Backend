#include "stdafx.h"
#include "../Overlays/overlay_compile_notification.h"
#include "../Overlays/Shaders/shader_loading_dialog_native.h"

#include "MTLCommandStream.h"
#include "MTLCompute.h"
#include "MTLDMA.h"
#include "MTLFormats.h"
#include "MTLGSRender.h"
#include "MTLHelpers.h"
#include "MTLRenderPass.h"
#include "MTLResolveHelper.h"
#include "MTLResourceManager.h"

#include "mtlutils/metal_layer.h"

#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/Host/MM.h"
#include "Emu/RSX/NV47/HW/context_accessors.define.h"
#include "Emu/Memory/vm_locking.h"

#include "../Program/SPIRVCommon.h"

#include "util/asm.hpp"

#include <cstring>
#include <numeric>
#include <set>
#include <thread>

#define MTL_OCCLUSION_MAX_POOL_SIZE 16384

namespace mtl
{
	// MTLVertexBuffers.cpp
	std::pair<MTL::PrimitiveType, bool> get_appropriate_topology(rsx::primitive_type mode);

	// MTLDraw.cpp
	void destroy_inpass_clear_programs();

	// ---- Global scratch heap ----------------------------------------------------------------------------------------
	// The RSX thread (and the UI thread while the renderer is being created/destroyed) uses the main ring, which is
	// registered with the frame heap snapshots. Other threads (access violation handlers flushing through the secondary
	// command list chain, etc.) get a private ring each: data_heap is not thread-safe. Private rings are recycled by
	// their owning thread after each synchronous secondary submission (recycle_thread_scratch_heap).
	static mtl::data_heap g_scratch_heap;

	// Push constants, overlay UBOs/VAOs: a few KiB per pass. Sized so that grow() is not reached in normal use.
	static constexpr usz s_thread_scratch_heap_size = 4 * 0x100000;
	static constexpr usz s_thread_scratch_heap_limit = 64 * 0x100000;

	class thread_scratch_ring final : public mtl::data_heap
	{
		// Backing stores replaced by grow(). The owning thread's in-flight work may still read them; they are released
		// on the next successful recycle (after that work has been waited for). Never goes through the frame GC, whose
		// event ids track RSX thread submissions only.
		std::vector<std::unique_ptr<mtl::buffer>> m_retired_buffers;

		// Secondary list left recording by a previous operation of this thread. It may contain commands referencing this
		// ring, so the ring cannot be recycled before that list has been submitted and has completed.
		mtl::command_buffer_chunk* m_pending_cmd = nullptr;

	protected:
		// NOTE: Unlike mtl::data_heap::grow, never syncs the DMA offloader (this runs on PPU/SPU threads).
		bool grow(usz size) override
		{
			usz new_size = std::max<usz>(m_size * 2, m_size + size);
			new_size = std::min<usz>(new_size, s_thread_scratch_heap_limit);
			new_size = utils::align<usz>(std::max<usz>(new_size, size + 0x100000), 0x100000);

			rsx_log.warning("[%s] Thread scratch ring exhausted, growing from 0x%llx to 0x%llx bytes", m_name, u64{ m_size }, u64{ new_size });

			const char* name = m_name;
			const usz guard = m_min_guard_size;

			m_retired_buffers.push_back(std::move(heap));
			create(new_size, name, guard);
			return true;
		}

	public:
		void recycle(mtl::command_buffer_chunk& cmd)
		{
			if (m_pending_cmd)
			{
				if (m_pending_cmd->is_recording())
				{
					// Still not submitted; try again next time
					return;
				}

				m_pending_cmd->wait();
				m_pending_cmd = nullptr;
			}

			if (cmd.is_recording())
			{
				m_pending_cmd = &cmd;
				return;
			}

			// All work recorded by this thread has completed
			cmd.wait();

			m_retired_buffers.clear();
			reset_allocation_stats();
		}

		void release()
		{
			m_pending_cmd = nullptr;
			m_retired_buffers.clear();
			destroy();
		}
	};

	struct thread_scratch_heap
	{
		thread_scratch_ring ring;
	};

	static shared_mutex g_thread_scratch_heaps_lock;
	static std::unordered_map<std::thread::id, std::unique_ptr<thread_scratch_heap>> g_thread_scratch_heaps;

	mtl::data_heap& get_scratch_heap()
	{
		const auto renderer = rsx::get_current_renderer();
		if (!renderer || !renderer->rsx_thread_running || renderer->is_current_thread() || !g_scratch_heap.heap)
		{
			return g_scratch_heap;
		}

		const auto thread_id = std::this_thread::get_id();

		{
			reader_lock lock(g_thread_scratch_heaps_lock);
			if (const auto found = g_thread_scratch_heaps.find(thread_id); found != g_thread_scratch_heaps.end())
			{
				return found->second->ring;
			}
		}

		std::lock_guard lock(g_thread_scratch_heaps_lock);
		auto& entry = g_thread_scratch_heaps[thread_id];
		if (!entry)
		{
			entry = std::make_unique<thread_scratch_heap>();
			entry->ring.create(s_thread_scratch_heap_size, "thread scratch buffer", 0x1000);
		}

		return entry->ring;
	}

	// Called by a non-RSX thread after its synchronous work on the secondary chain (cmd) has been recorded/submitted.
	// Waits for that work and recycles the calling thread's private scratch ring. No-op on the RSX thread.
	static void recycle_thread_scratch_heap(mtl::command_buffer_chunk& cmd)
	{
		const auto renderer = rsx::get_current_renderer();
		if (!renderer || renderer->is_current_thread())
		{
			return;
		}

		thread_scratch_heap* entry = nullptr;
		{
			reader_lock lock(g_thread_scratch_heaps_lock);
			if (const auto found = g_thread_scratch_heaps.find(std::this_thread::get_id()); found != g_thread_scratch_heaps.end())
			{
				entry = found->second.get();
			}
		}

		if (entry)
		{
			// Only the owning thread touches its ring (entries are removed at renderer teardown only)
			entry->ring.recycle(cmd);
		}
	}

	static void destroy_thread_scratch_heaps()
	{
		std::lock_guard lock(g_thread_scratch_heaps_lock);
		for (auto& [id, entry] : g_thread_scratch_heaps)
		{
			entry->ring.release();
		}

		g_thread_scratch_heaps.clear();
	}

	// ---- Global resources (VKHelpers.cpp) ---------------------------------------------------------------------------
	void reset_global_resources()
	{
		// FIXME: These two shouldn't exist
		mtl::reset_resolve_resources();
		mtl::reset_overlay_passes();

		get_upload_heap()->reset_allocation_stats();
	}

	void destroy_global_resources()
	{
		mtl::clear_resolve_helpers();
		mtl::clear_dma_resources();
		mtl::clear_scratch_resources();

		mtl::get_upload_heap()->destroy();

		mtl::destroy_compute_tasks();
		mtl::destroy_overlay_passes();

		// This must be the last item destroyed
		mtl::get_resource_manager()->destroy();
	}

	// ---- RSX -> Metal state translation -----------------------------------------------------------------------------
	u8 get_topology_class(MTL::PrimitiveType type)
	{
		switch (type)
		{
		case MTL::PrimitiveTypePoint:
			return static_cast<u8>(MTL::PrimitiveTopologyClassPoint);
		case MTL::PrimitiveTypeLine:
		case MTL::PrimitiveTypeLineStrip:
			return static_cast<u8>(MTL::PrimitiveTopologyClassLine);
		default:
			return static_cast<u8>(MTL::PrimitiveTopologyClassTriangle);
		}
	}

	MTL::BlendFactor get_blend_factor(rsx::blend_factor factor)
	{
		switch (factor)
		{
		case rsx::blend_factor::one: return MTL::BlendFactorOne;
		case rsx::blend_factor::zero: return MTL::BlendFactorZero;
		case rsx::blend_factor::src_alpha: return MTL::BlendFactorSourceAlpha;
		case rsx::blend_factor::dst_alpha: return MTL::BlendFactorDestinationAlpha;
		case rsx::blend_factor::src_color: return MTL::BlendFactorSourceColor;
		case rsx::blend_factor::dst_color: return MTL::BlendFactorDestinationColor;
		case rsx::blend_factor::constant_color: return MTL::BlendFactorBlendColor;
		case rsx::blend_factor::constant_alpha: return MTL::BlendFactorBlendAlpha;
		case rsx::blend_factor::one_minus_src_color: return MTL::BlendFactorOneMinusSourceColor;
		case rsx::blend_factor::one_minus_dst_color: return MTL::BlendFactorOneMinusDestinationColor;
		case rsx::blend_factor::one_minus_src_alpha: return MTL::BlendFactorOneMinusSourceAlpha;
		case rsx::blend_factor::one_minus_dst_alpha: return MTL::BlendFactorOneMinusDestinationAlpha;
		case rsx::blend_factor::one_minus_constant_alpha: return MTL::BlendFactorOneMinusBlendAlpha;
		case rsx::blend_factor::one_minus_constant_color: return MTL::BlendFactorOneMinusBlendColor;
		case rsx::blend_factor::src_alpha_saturate: return MTL::BlendFactorSourceAlphaSaturated;
		default:
			fmt::throw_exception("Unknown blend factor 0x%x", static_cast<u32>(factor));
		}
	}

	MTL::BlendOperation get_blend_op(rsx::blend_equation op)
	{
		switch (op)
		{
		case rsx::blend_equation::add_signed:
			rsx_log.error("blend equation add_signed used. Emulating using FUNC_ADD");
			[[fallthrough]];
		case rsx::blend_equation::add: return MTL::BlendOperationAdd;
		case rsx::blend_equation::reverse_add_signed:
			rsx_log.error("blend equation reverse_add_signed used. Emulating using FUNC_ADD");
			return MTL::BlendOperationAdd;
		case rsx::blend_equation::subtract: return MTL::BlendOperationSubtract;
		case rsx::blend_equation::reverse_subtract_signed:
			rsx_log.error("blend equation reverse_subtract_signed used. Emulating using FUNC_REVERSE_SUBTRACT");
			[[fallthrough]];
		case rsx::blend_equation::reverse_subtract: return MTL::BlendOperationReverseSubtract;
		case rsx::blend_equation::min: return MTL::BlendOperationMin;
		case rsx::blend_equation::max: return MTL::BlendOperationMax;
		default:
			fmt::throw_exception("Unknown blend op: 0x%x", static_cast<u32>(op));
		}
	}

	// Port of vk::decode_rsx_state. Only the state baked into an MTL4 render pipeline is produced here; depth/stencil,
	// culling, winding and depth clipping are encoder state (see decode_rasterizer_state).
	mtl::pipeline_props decode_rsx_state(
		const rsx::context* ctx,
		u8 topology_class,
		const std::vector<mtl::image*>& color_attachments,
		mtl::render_target* ds,
		const rsx::backend_configuration& backend_config,
		u8 num_draw_buffers,
		u8 num_rasterization_samples,
		bool force_disable_blending)
	{
		mtl::pipeline_props properties{};
		auto& state = properties.state;

		// Input assembly
		state.topology_class = topology_class;
		state.rasterization_enabled = 1;

		// Attachments
		state.color_count = num_draw_buffers;
		state.depth_stencil_format = ds ? static_cast<u32>(ds->format()) : 0u;
		state.sample_count = std::max<u8>(1, num_rasterization_samples);

		const auto host_write_mask = rsx::get_write_output_mask(REGS(ctx)->surface_color());
		for (uint index = 0; index < num_draw_buffers; ++index)
		{
			state.color[index].pixel_format = static_cast<u32>(ensure(color_attachments[index])->format());

			bool color_mask_b = REGS(ctx)->color_mask_b(index);
			bool color_mask_g = REGS(ctx)->color_mask_g(index);
			bool color_mask_r = REGS(ctx)->color_mask_r(index);
			bool color_mask_a = REGS(ctx)->color_mask_a(index);

			switch (REGS(ctx)->surface_color())
			{
			case rsx::surface_color_format::b8:
				rsx::get_b8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
				break;
			case rsx::surface_color_format::g8b8:
				rsx::get_g8b8_r8g8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
				break;
			default:
				break;
			}

			u8 write_mask = 0;
			if (color_mask_r && host_write_mask[0]) write_mask |= static_cast<u8>(MTL::ColorWriteMaskRed);
			if (color_mask_g && host_write_mask[1]) write_mask |= static_cast<u8>(MTL::ColorWriteMaskGreen);
			if (color_mask_b && host_write_mask[2]) write_mask |= static_cast<u8>(MTL::ColorWriteMaskBlue);
			if (color_mask_a && host_write_mask[3]) write_mask |= static_cast<u8>(MTL::ColorWriteMaskAlpha);
			state.color[index].write_mask = write_mask;
		}

		// LogicOp and Blend are mutually exclusive. If both are enabled, LogicOp takes precedence.
		// Metal has no logic op state and the shader emulation is not written yet: blending is off and the color is
		// written as is (warned once in MTLDraw.cpp).
		if (!REGS(ctx)->logic_op_enabled() && !force_disable_blending)
		{
			if (const auto blend_enabled = REGS(ctx)->blend_enabled_mask())
			{
				const auto sfactor_rgb = get_blend_factor(REGS(ctx)->blend_func_sfactor_rgb());
				const auto sfactor_a = get_blend_factor(REGS(ctx)->blend_func_sfactor_a());
				const auto dfactor_rgb = get_blend_factor(REGS(ctx)->blend_func_dfactor_rgb());
				const auto dfactor_a = get_blend_factor(REGS(ctx)->blend_func_dfactor_a());
				const auto equation_rgb = get_blend_op(REGS(ctx)->blend_equation_rgb());
				const auto equation_a = get_blend_op(REGS(ctx)->blend_equation_a());

				for (u8 idx = 0; idx < num_draw_buffers; ++idx)
				{
					if (blend_enabled & (1u << idx))
					{
						auto& att = state.color[idx];
						att.blend_enable = 1;
						att.src_rgb = static_cast<u8>(sfactor_rgb);
						att.dst_rgb = static_cast<u8>(dfactor_rgb);
						att.op_rgb = static_cast<u8>(equation_rgb);
						att.src_a = static_cast<u8>(sfactor_a);
						att.dst_a = static_cast<u8>(dfactor_a);
						att.op_a = static_cast<u8>(equation_a);
					}
				}
			}
		}

		if (REGS(ctx)->stencil_test_enabled() && ds && ds->samples() > 1 && !(ds->stencil_init_flags & 0xFF00))
		{
			const auto keeps = [](rsx::stencil_op op) { return op == rsx::stencil_op::keep; };
			const bool two_sided = REGS(ctx)->two_sided_stencil_test_enabled();

			if (!keeps(REGS(ctx)->stencil_op_fail()) ||
				!keeps(REGS(ctx)->stencil_op_zfail()) ||
				!keeps(REGS(ctx)->stencil_op_zpass()) ||
				(two_sided && (!keeps(REGS(ctx)->back_stencil_op_fail()) ||
					!keeps(REGS(ctx)->back_stencil_op_zfail()) ||
					!keeps(REGS(ctx)->back_stencil_op_zpass()))))
			{
				// Toggle bit 9 to signal require full bit-wise transfer
				ds->stencil_init_flags |= (1 << 8);
			}
		}

		if (backend_config.supports_hw_a2c || num_rasterization_samples > 1)
		{
			// Mirrors VK set_multisample_state. msaa_enabled is ignored there as well.
			// NOTE: msaa_sample_mask is not applied: Metal pipelines have no fixed-function sample mask (it would need a
			// [[sample_mask]] fragment output emitted by the decompiler).
			const bool alpha_to_one_enable = REGS(ctx)->msaa_alpha_to_one_enabled() && backend_config.supports_hw_a2one;

			// Hardware A2C only on multisampled pipelines; single-sample A2C is emulated in the fragment shader
			// (backend_config.supports_hw_a2c_1spp = false -> RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE).
			const bool alpha_to_coverage_enable = REGS(ctx)->msaa_alpha_to_coverage_enabled() && num_rasterization_samples > 1;

			state.alpha_to_coverage = alpha_to_coverage_enable ? 1 : 0;
			state.alpha_to_one = alpha_to_one_enable ? 1 : 0;
		}

		return properties;
	}
}

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* ar) noexcept : GSRender(ar)
{
	mtl::autorelease_scope pool;

	// Initialize dependencies
	g_fxo->need<rsx::dma_manager>();

	auto device = std::make_unique<mtl::render_device>();
	if (!device->create())
	{
		rsx_log.fatal("Could not initialize Metal. The Metal renderer requires an Apple silicon Mac (M1 or newer) running macOS 26 or later.");
		return;
	}

	m_device = std::move(device);
	mtl::g_render_device = m_device.get();
	mtl::reset_runtime_state();

	m_timeline.create(*m_device, "RSX timeline");

	// Presentation surface. The game window is a QWindow created with QSurface::MetalSurface.
#if defined(__APPLE__)
	m_view = m_frame->handle();
#endif

	if (m_view)
	{
		m_metal_layer = mtl::get_metal_layer_from_view(m_view);
		if (m_metal_layer)
		{
			// The view owns the layer; keep our own reference in case the window drops it before the renderer is gone
			m_metal_layer->retain();
		}
	}

	if (m_metal_layer)
	{
		m_metal_layer->setDevice(m_device->handle());
		m_metal_layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
		m_metal_layer->setMaximumDrawableCount(MTL_MAX_DRAWABLE_COUNT);
		m_metal_layer->setAllowsNextDrawableTimeout(true);

		// Drawables live in the layer's residency set; the queues must see it (attached to the main and async queues)
		m_device->attach_residency_set(m_metal_layer->residencySet());
	}
	else
	{
		rsx_log.error("Metal: the game window has no CAMetalLayer. Nothing will be presented.");
	}

	// Present lists run on the device's second queue, so that waiting for a drawable never holds back the main queue
	// (see MTLPresent.cpp). That queue has the global and the layer's residency sets attached.
	m_present_queue = m_device->async_queue();
	if (!m_present_queue)
	{
		rsx_log.warning("Metal: no second command queue. Drawable waits will delay the main queue.");
		m_present_queue = m_device->queue();
	}

	m_present_timeline.create(*m_device, "RSX present timeline");

	// Also applies contentsScale/opaque and samples the screen's refresh properties (main thread, asynchronously)
	m_swapchain_dims.width = m_frame->client_width();
	m_swapchain_dims.height = m_frame->client_height();
	configure_metal_layer();

	if (!m_metal_layer || !m_swapchain_dims.width || !m_swapchain_dims.height)
	{
		swapchain_unavailable = true;
	}

	// Command lists. Primary lists may record texture uploads into a prologue so that they do not end the open render
	// pass (texture_cache::upload_image_from_cpu, "Asynchronous Texture Streaming").
	m_primary_cb_list.create(*m_device, m_device->queue(), m_timeline, "RSX primary", mtl::command_list::access_type_hint::flush_only, true);
	m_current_command_buffer = m_primary_cb_list.get();
	m_current_command_buffer->begin();

	// Secondary command lists for parallel operations (access violation handlers, ...)
	m_secondary_cb_list.create(*m_device, m_device->queue(), m_timeline, "RSX secondary", mtl::command_list::access_type_hint::all);

	// Occlusion
	m_occlusion_query_manager = std::make_unique<mtl::query_pool_manager>(*m_device, MTL_OCCLUSION_MAX_POOL_SIZE);
	m_occlusion_map.resize(rsx::reports::occlusion_query_count);

	for (u32 n = 0; n < rsx::reports::occlusion_query_count; ++n)
		m_occlusion_query_data[n].driver_handle = n;

	m_occlusion_query_manager->set_control_flags(!!g_cfg.video.precise_zpass_count);

	// Ring buffers. All of them live in shared (unified) memory; no staging or flushes are required.
	// This first set is bound persistently, so grow notifications are enabled.
	m_attrib_ring_info.create(MTL_ATTRIB_RING_BUFFER_SIZE_M * 0x100000, "attrib buffer", 0x400000, true);
	m_fragment_env_ring_info.create(MTL_UBO_RING_BUFFER_SIZE_M * 0x100000, "fragment env buffer", 0x10000, true);
	m_vertex_env_ring_info.create(MTL_UBO_RING_BUFFER_SIZE_M * 0x100000, "vertex env buffer", 0x10000, true);
	m_fragment_texture_params_ring_info.create(MTL_UBO_RING_BUFFER_SIZE_M * 0x100000, "fragment texture params buffer", 0x10000, true);
	m_vertex_layout_ring_info.create(MTL_UBO_RING_BUFFER_SIZE_M * 0x100000, "vertex layout buffer", 0x10000, true);
	m_fragment_constants_ring_info.create(MTL_FRAGMENT_CONSTANTS_BUFFER_SIZE_M * 0x100000, "fragment constants buffer", 0x10000, true);
	m_transform_constants_ring_info.create(MTL_TRANSFORM_CONSTANTS_BUFFER_SIZE_M * 0x100000, "transform constants buffer", 0x10000, true);
	m_raster_env_ring_info.create(MTL_UBO_RING_BUFFER_SIZE_M * 0x100000, "raster env buffer", 0x10000, true);
	// Below here, we do not bind these persistently. Each draw call specifies the range manually so we do not need heap_grow notifications.
	m_instancing_buffer_ring_info.create(MTL_TRANSFORM_CONSTANTS_BUFFER_SIZE_M * 0x100000, "instancing data buffer");
	m_index_buffer_ring_info.create(MTL_INDEX_RING_BUFFER_SIZE_M * 0x100000, "index buffer");
	m_texture_upload_buffer_ring_info.create(MTL_TEXTURE_UPLOAD_RING_BUFFER_SIZE_M * 0x100000, "texture upload buffer", 32 * 0x100000);
	// Push constants and small per-draw uniforms of every pass (Metal 4 has no setBytes)
	mtl::g_scratch_heap.create(MTL_SCRATCH_RING_BUFFER_SIZE_M * 0x100000, "scratch buffer", 0x10000);

	mtl::data_heap_manager::register_ring_buffers
	({
		std::ref(m_attrib_ring_info),
		std::ref(m_fragment_env_ring_info),
		std::ref(m_vertex_env_ring_info),
		std::ref(m_fragment_texture_params_ring_info),
		std::ref(m_vertex_layout_ring_info),
		std::ref(m_fragment_constants_ring_info),
		std::ref(m_transform_constants_ring_info),
		std::ref(m_index_buffer_ring_info),
		std::ref(m_texture_upload_buffer_ring_info),
		std::ref(m_raster_env_ring_info),
		std::ref(m_instancing_buffer_ring_info),
		std::ref(mtl::g_scratch_heap)
	});

	// Texel buffer views of the attribute heap are windowed (64M elements is well within Metal limits)
	m_texbuffer_view_size = MTL_ATTRIB_RING_BUFFER_SIZE_M * 0x100000u;

	// Initialize bulk allocators. Ring buffers are bound whole, so batches only need to keep the element stride intact.
	m_vertex_env_allocator = std::make_unique<rsx::data_heap::bulk_allocator<256, 96>>(m_vertex_env_ring_info, 1024u);
	m_transform_constants_allocator = std::make_unique<rsx::data_heap::bulk_allocator<256, 16>>(m_transform_constants_ring_info, 8192u);
	m_fragment_constants_allocator = std::make_unique<rsx::data_heap::bulk_allocator<256, 16>>(m_fragment_constants_ring_info, 8192u);

	m_max_async_frames = MTL_MAX_DRAWABLE_COUNT;
	m_frame_context_storage.resize(m_max_async_frames);
	m_current_frame = &m_frame_context_storage[0];

	// Fallback bindables
	null_buffer = std::make_unique<mtl::buffer>(*m_device, 32, mtl::memory_location::host_visible, "null buffer");
	std::memset(null_buffer->map(), 0, 32);
	null_buffer_view = std::make_unique<mtl::buffer_view>(*m_device, *null_buffer, MTL::PixelFormatR8Uint, 0, 32);

	// Conditional rendering predicate: word 0 receives the evaluated result, word 1 is all ones ("always draw")
	m_cond_render_buffer = std::make_unique<mtl::buffer>(*m_device, 8, mtl::memory_location::host_visible, "conditional render predicate");
	{
		auto predicate = static_cast<u32*>(m_cond_render_buffer->map());
		predicate[0] = 0;
		predicate[1] = 0xFFFFFFFFu;
	}

	spirv::initialize_compiler_context();
	mtl::initialize_pipe_compiler(g_cfg.video.shader_compiler_threads_count);
	mtl::initialize_pipeline_archive(); // Persistent pipeline binaries (MTLPipelineArchive.h); before any pipeline is built

	m_prog_buffer = std::make_unique<mtl::program_cache>
	(
		[this](const mtl::pipeline_props& props, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			// Program was linked or queued for linking
			m_shaders_cache->store(props, vp, fp);
		}
	);

	if (g_cfg.video.disable_vertex_cache)
		m_vertex_cache = std::make_unique<mtl::null_vertex_cache>();
	else
		m_vertex_cache = std::make_unique<mtl::weak_vertex_cache>();

	m_shaders_cache = std::make_unique<mtl::shader_cache>(*m_prog_buffer, "metal", "v1.0");

	m_texture_cache.initialize(*m_device, m_texture_upload_buffer_ring_info);

	mtl::get_overlay_pass<mtl::ui_overlay_renderer>()->init(*m_current_command_buffer, m_texture_upload_buffer_ring_info);

	// Backend capabilities
	backend_config.supports_multidraw = false;               // No multi-draw on Metal; sub-draws are looped
	backend_config.supports_hw_instanced_rendering = true;
	backend_config.supports_programmable_blending = true;    // Framebuffer fetch ([[color(n)]])
	backend_config.supports_last_provoking_vertex = false;   // Metal always uses the first vertex (smooth fallback)
	backend_config.supports_normalized_barycentrics = true;

	// NOTE: We do not actually need multiple sample support for A2C to work
	// This is here for visual consistency - will be removed when AA problems due to mipmaps are fixed
	if (g_cfg.video.antialiasing_level != msaa_level::none)
	{
		backend_config.supports_hw_msaa = true;
		backend_config.supports_hw_a2c = true;
		backend_config.supports_hw_a2c_1spp = false;          // Metal A2C is only used on multisampled pipelines (see decode_rsx_state)
		backend_config.supports_hw_a2one = true;
	}

	backend_config.supports_hw_renormalization = false;      // Only NVIDIA hardware matches RSX renormalization
	backend_config.supports_hw_conditional_render = false;   // Shader predicate path (emulate_conditional_rendering)
	backend_config.supports_passthrough_dma = mtl::is_passthrough_dma_supported();
	backend_config.supports_host_gpu_labels = false;
	backend_config.supports_asynchronous_compute = false;

	if (g_cfg.video.shadermode == shader_mode::async_with_interpreter || g_cfg.video.shadermode == shader_mode::interpreter_only)
	{
		rsx_log.warning("Metal: the shader interpreter is not available. Draws are skipped until their shaders are compiled (async recompiler behaviour).");
	}
}

MTLGSRender::~MTLGSRender()
{
	// Force-release ZCULL-ctrl since it is a pointer to self.
	zcull_ctrl.release();

	if (!m_device)
	{
		//Initialization failed
		return;
	}

	mtl::autorelease_scope pool;

	// Flush DMA queue
	while (!g_fxo->get<rsx::dma_manager>().sync())
	{
		do_local_task(rsx::FIFO::state::lock_wait);
	}

	// Wait for the device to finish up with resources. Pending recordings are discarded like Vulkan does.
	m_render_pass.reset();
	if (m_current_command_buffer && m_current_command_buffer->is_recording())
	{
		m_current_command_buffer->end();
	}

	// The destructor runs on the UI thread. After a fatal error on the RSX thread a submission can be left half-done
	// (submit_queued never cleared) or GPU work may never complete, so every wait here is bounded.
	m_primary_cb_list.abandon_queued_submits();
	m_secondary_cb_list.abandon_queued_submits();

	if (!m_primary_cb_list.wait_all(TEARDOWN_WAIT_TIMEOUT) ||
		!m_secondary_cb_list.wait_all(TEARDOWN_WAIT_TIMEOUT) ||
		!m_timeline.wait(m_timeline.last_signaled_value(), TEARDOWN_WAIT_TIMEOUT))
	{
		rsx_log.error("Metal: the GPU did not finish outstanding work during shutdown; continuing anyway");
	}

	// GC cleanup
	mtl::get_resource_manager()->flush();

	// Clear flush requests
	m_flush_requests.clear_pending_flag();

	// Shaders
	mtl::destroy_pipe_compiler();       // Ensure no pending shaders being compiled
	spirv::finalize_compiler_context(); // Shut down the glslang compiler
	m_prog_buffer->clear();             // Delete shader objects
	mtl::destroy_inpass_clear_programs();
	m_program = nullptr;
	m_prev_program = nullptr;

	// Passes and compute tasks (reference caches, heaps and samplers)
	m_upscaler.reset();
	mtl::destroy_overlay_passes();
	mtl::destroy_compute_tasks();

	// Frame contexts (drawables, present images, present lists)
	if (m_current_frame == &m_aux_frame_context)
	{
		// Return resources back to the owner
		m_current_frame = &m_frame_context_storage[m_current_queue_index];
		m_current_frame->grab_resources(m_aux_frame_context);
	}

	// The present queue is not covered by the waits above (bounded: this runs on the UI thread)
	const bool present_queue_idle = m_present_timeline.wait(m_present_timeline.last_signaled_value(), TEARDOWN_WAIT_TIMEOUT);
	if (!present_queue_idle)
	{
		rsx_log.error("Metal: the present queue did not finish outstanding work during shutdown; continuing anyway");
	}

	for (auto& ctx : m_frame_context_storage)
	{
		if (ctx.drawable)
		{
			ctx.drawable->release();
			ctx.drawable = nullptr;
		}

		if (ctx.present_command_buffer)
		{
			if (present_queue_idle)
			{
				// Retire the list (no GPU wait left). Otherwise destroy() performs its own bounded wait.
				ctx.present_command_buffer->wait(TEARDOWN_WAIT_TIMEOUT);
			}

			ctx.present_command_buffer->destroy();
			ctx.present_command_buffer.reset();
		}

		ctx.present_image.reset();
	}

	m_current_frame = nullptr;
	m_queued_frames.clear();
	m_frame_context_storage.clear();
	m_present_timeline.destroy();
	m_present_queue = nullptr;

	// Caches
	m_rtts.destroy();
	m_texture_cache.destroy();
	m_vertex_cache.reset();

	m_persistent_attribute_storage.reset();
	m_volatile_attribute_storage.reset();

	m_overlay_recording_img.reset();
	m_stencil_mirror_sampler.reset();
	for (auto& null_texture : m_null_depth_textures)
	{
		null_texture.reset();
	}
	fs_sampler_handles.fill(nullptr);
	vs_sampler_handles.fill(nullptr);

	// Render pass state
	m_draw_pass_desc.reset();
	m_depth_stencil_states.clear();
	m_fbo_images.clear();
	m_draw_fbo.clear();

	// Queries
	m_occlusion_query_manager.reset();
	m_cond_render_buffer.reset();

	// Fallback bindables
	null_buffer_view.reset();
	null_buffer.reset();

	// Heaps
	mtl::data_heap_manager::reset();
	mtl::destroy_thread_scratch_heaps();

	// Command lists
	m_primary_cb_list.destroy();
	m_secondary_cb_list.destroy();
	m_current_command_buffer = nullptr;

	// Global resources (scratch, DMA, resolve helpers, upload heap) and the final GC flush
	mtl::destroy_global_resources();

	// Device handles/contexts
	m_timeline.destroy();

	// The queue references the layer's residency set; drop our layer reference only once the device is gone
	const auto metal_layer = std::exchange(m_metal_layer, nullptr);

	mtl::g_render_device = nullptr;
	m_device->destroy();
	m_device.reset();

	if (metal_layer)
	{
		metal_layer->release();
	}
}

bool MTLGSRender::on_access_violation(u32 address, bool is_writing)
{
	// Runs on PPU/SPU threads, which have no autorelease pool of their own
	mtl::autorelease_scope pool;

	rsx::mm_flush(address);

	mtl::texture_cache::thrashed_set result;
	{
		const rsx::invalidation_cause cause = is_writing ? rsx::invalidation_cause::deferred_write : rsx::invalidation_cause::deferred_read;
		result = m_texture_cache.invalidate_address(*m_secondary_cb_list.get(), address, cause);
	}

	if (result.invalidate_samplers)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}

	if (!result.violation_handled)
	{
		return zcull_ctrl->on_access_violation(address);
	}

	if (result.num_flushable > 0)
	{
		if (g_fxo->get<rsx::dma_manager>().is_current_thread())
		{
			// The offloader thread cannot handle flush requests
			ensure(!(m_queue_status & flush_queue_state::deadlock));

			m_offloader_fault_range = g_fxo->get<rsx::dma_manager>().get_fault_range(is_writing);
			m_offloader_fault_cause = (is_writing) ? rsx::invalidation_cause::write : rsx::invalidation_cause::read;

			g_fxo->get<rsx::dma_manager>().set_mem_fault_flag();
			m_queue_status |= flush_queue_state::deadlock;
			m_eng_interrupt_mask |= rsx::backend_interrupt;

			// Wait for deadlock to clear
			while (m_queue_status & flush_queue_state::deadlock)
			{
				utils::pause();
			}

			g_fxo->get<rsx::dma_manager>().clear_mem_fault_flag();
			return true;
		}

		bool has_queue_ref = false;
		std::function<void()> data_transfer_completed_callback{};

		if (!is_current_thread()) [[likely]]
		{
			// Always submit primary cb to ensure state consistency (flush pending changes such as image transitions)
			vm::temporary_unlock();

			std::lock_guard lock(m_flush_queue_mutex);

			m_flush_requests.post(false);
			m_eng_interrupt_mask |= rsx::backend_interrupt;
			has_queue_ref = true;
		}
		else
		{
			if (mtl::is_uninterruptible())
			{
				rsx_log.error("Fault in uninterruptible code!");
			}

			// Flush primary cb queue to sync pending changes (e.g image transitions!)
			flush_command_queue();
		}

		if (has_queue_ref)
		{
			// Wait for the RSX thread to process request if it hasn't already
			m_flush_requests.producer_wait();

			data_transfer_completed_callback = [&]()
			{
				m_flush_requests.remove_one();
				has_queue_ref = false;
			};
		}

		auto secondary_cmd = m_secondary_cb_list.next();
		{
			// Metal objects created by the flush are autoreleased on this (foreign) thread
			mtl::autorelease_scope pool;
			m_texture_cache.flush_all(*secondary_cmd, result, data_transfer_completed_callback);
		}

		if (has_queue_ref)
		{
			// Release RSX thread if it's still locked
			m_flush_requests.remove_one();
		}

		// The flush is synchronous; recycle this thread's private scratch ring (no-op on the RSX thread)
		mtl::recycle_thread_scratch_heap(*secondary_cmd);
	}

	return true;
}

void MTLGSRender::on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause)
{
	mtl::autorelease_scope pool;
	std::lock_guard lock(m_secondary_cb_guard);

	auto secondary_cmd = m_secondary_cb_list.next();
	auto data = m_texture_cache.invalidate_range(*secondary_cmd, range, cause);
	AUDIT(data.empty());

	if (cause == rsx::invalidation_cause::unmap)
	{
		if (data.violation_handled)
		{
			m_texture_cache.purge_unreleased_sections();
			{
				std::lock_guard lock(m_sampler_mutex);
				m_samplers_dirty.store(true);
			}
		}

		mtl::unmap_dma(range.start, range.length());
	}

	// Recycle this thread's private scratch ring (no-op on the RSX thread)
	mtl::recycle_thread_scratch_heap(*secondary_cmd);
}

void MTLGSRender::on_semaphore_acquire_wait()
{
	if (m_flush_requests.pending() ||
		(async_flip_requested & flip_request::emu_requested) ||
		(m_queue_status & flush_queue_state::deadlock))
	{
		do_local_task(rsx::FIFO::state::lock_wait);
	}
}

bool MTLGSRender::on_vram_exhausted(rsx::problem_severity severity)
{
	ensure(!mtl::is_uninterruptible() && rsx::get_current_renderer()->is_current_thread());

	bool texture_cache_relieved = false;
	if (severity >= rsx::problem_severity::fatal)
	{
		// Hard sync before trying to evict anything. This guarantees no UAF on the GPU.
		// As a bonus, we also get a free gc pass
		flush_command_queue(true, true);

		if (m_texture_cache.is_overallocated())
		{
			// Evict some unused textures. Do not evict any active references
			std::set<u32> exclusion_list;
			auto scan_array = [&](const auto& texture_array)
			{
				for (auto i = 0ull; i < texture_array.size(); ++i)
				{
					const auto& tex = texture_array[i];
					const auto addr = rsx::get_address(tex.offset(), tex.location());
					exclusion_list.insert(addr);
				}
			};

			scan_array(rsx::method_registers.fragment_textures);
			scan_array(rsx::method_registers.vertex_textures);

			// Hold the secondary lock guard to prevent threads from trying to touch access violation handler stuff
			std::lock_guard lock(m_secondary_cb_guard);

			rsx_log.warning("Texture cache is overallocated. Will evict unnecessary textures.");
			texture_cache_relieved = m_texture_cache.evict_unused(exclusion_list);
		}
	}

	texture_cache_relieved |= m_texture_cache.handle_memory_pressure(severity);
	if (severity == rsx::problem_severity::low)
	{
		// Low severity only handles invalidating unused textures
		return texture_cache_relieved;
	}

	bool surface_cache_relieved = false;

	// NOTE: No VRAM spilling. Apple silicon has unified memory; there is nowhere to spill to.

	// Moderate severity and higher also starts removing stale render target objects
	if (m_rtts.handle_memory_pressure(*m_current_command_buffer, severity))
	{
		surface_cache_relieved = true;
		m_rtts.trim(*m_current_command_buffer, severity);
	}

	const bool any_cache_relieved = (texture_cache_relieved || surface_cache_relieved);
	if (severity < rsx::problem_severity::fatal)
	{
		return any_cache_relieved;
	}

	if (surface_cache_relieved && !m_samplers_dirty)
	{
		// If surface cache was modified destructively, then we must reload samplers touching the surface cache.
		bool invalidate_samplers = false;
		auto scan_array = [&](const auto& texture_array, const auto& sampler_states)
		{
			if (invalidate_samplers)
			{
				return;
			}

			for (auto i = 0ull; i < texture_array.size(); ++i)
			{
				if (texture_array[i].enabled() &&
					sampler_states[i] &&
					sampler_states[i]->upload_context == rsx::texture_upload_context::framebuffer_storage)
				{
					invalidate_samplers = true;
					break;
				}
			}
		};

		scan_array(rsx::method_registers.fragment_textures, fs_sampler_state);
		scan_array(rsx::method_registers.vertex_textures, vs_sampler_state);

		if (invalidate_samplers)
		{
			m_samplers_dirty.store(true);
		}
	}

	// Imminent crash, full GPU sync is the least of our problems
	flush_command_queue(true, true);

	return any_cache_relieved;
}

void MTLGSRender::notify_tile_unbound(u32 tile)
{
	//TODO: Handle texture writeback
	if (false)
	{
		u32 addr = rsx::get_address(tiles[tile].offset, tiles[tile].location);
		on_notify_pre_memory_unmapped(addr, tiles[tile].size, *std::make_unique<std::vector<std::pair<u64, u64>>>());
		m_rtts.invalidate_surface_address(addr, false);
	}

	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}
}

void MTLGSRender::check_present_status()
{
	while (!m_queued_frames.empty())
	{
		auto ctx = m_queued_frames.front();
		if (!ctx->swap_command_buffer->poke())
		{
			return;
		}

		frame_context_cleanup(ctx);
	}
}

void MTLGSRender::check_heap_status()
{
	// Heaps are growable. When one of the persistently bound rings swaps its backing buffer (heap_changed), views and
	// cached offsets into the old buffer become invalid. Whole-heap bindings are rebuilt on every draw.
	if (!mtl::test_status_interrupt(mtl::heap_changed))
	{
		return;
	}

	// Check for validity
	if (m_persistent_attribute_storage &&
		m_persistent_attribute_storage->parent != m_attrib_ring_info.value())
	{
		mtl::get_resource_manager()->dispose(m_persistent_attribute_storage);
	}

	if (m_volatile_attribute_storage &&
		m_volatile_attribute_storage->parent != m_attrib_ring_info.value())
	{
		mtl::get_resource_manager()->dispose(m_volatile_attribute_storage);
	}

	// Cached vertex ranges point into the previous attribute buffer
	m_vertex_cache->purge();

	mtl::clear_status_interrupt(mtl::heap_changed);
}

void MTLGSRender::set_viewport()
{
	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(
		resolution_scaling_config,
		rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height());

	//NOTE: The scale_offset matrix already has viewport matrix factored in
	m_viewport.originX = 0.;
	m_viewport.originY = 0.;
	m_viewport.width = clip_width;
	m_viewport.height = clip_height;

	// Metal has no unrestricted depth range; the vertex program applies the RSX clip range.
	m_viewport.znear = 0.;
	m_viewport.zfar = 1.;

	m_current_command_buffer->flags |= mtl::command_list::cb_reload_dynamic_state;
	m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
}

void MTLGSRender::set_scissor(bool clip_viewport)
{
	areau scissor;
	if (get_scissor(scissor, clip_viewport))
	{
		m_scissor.height = scissor.height();
		m_scissor.width = scissor.width();
		m_scissor.x = scissor.x1;
		m_scissor.y = scissor.y1;

		m_current_command_buffer->flags |= mtl::command_list::cb_reload_dynamic_state;
	}
}

void MTLGSRender::bind_viewport()
{
	if (m_graphics_state & rsx::pipeline_state::zclip_config_state_dirty)
	{
		// Depth range is fixed to [0, 1] (see set_viewport)
		m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
	}

	auto encoder = ensure(get_render_encoder());
	encoder->setViewport(m_viewport);

	MTL::ScissorRect scissor = get_clamped_scissor();
	if (!scissor.width || !scissor.height)
	{
		// Nothing can be drawn (emit_geometry skips the draws); keep a valid rectangle for the encoder
		scissor = { 0, 0, 1, 1 };
	}

	encoder->setScissorRect(scissor);
}

MTL::ScissorRect MTLGSRender::get_clamped_scissor() const
{
	// Metal requires the scissor rectangle to lie inside the render target
	MTL::ScissorRect scissor = m_scissor;
	const NS::UInteger fb_width = std::max(1u, m_draw_fbo.width);
	const NS::UInteger fb_height = std::max(1u, m_draw_fbo.height);
	scissor.x = std::min<NS::UInteger>(scissor.x, fb_width);
	scissor.y = std::min<NS::UInteger>(scissor.y, fb_height);
	scissor.width = std::min<NS::UInteger>(scissor.width, fb_width - scissor.x);
	scissor.height = std::min<NS::UInteger>(scissor.height, fb_height - scissor.y);
	return scissor;
}

void MTLGSRender::on_init_thread()
{
	if (!m_device)
	{
		fmt::throw_exception("No Metal device was created");
	}

	GSRender::on_init_thread();
	zcull_ctrl.reset(static_cast<::rsx::reports::ZCULL_control*>(this));

	// There is no shader interpreter on Metal, so the pipeline cache is always preloaded (every shader mode behaves
	// like the async recompiler).
	{
		mtl::autorelease_scope pool;

		if (!m_overlay_manager)
		{
			m_frame->hide();
			m_shaders_cache->load(nullptr);
			m_frame->show();
		}
		else
		{
			rsx::shader_loading_dialog_native dlg(this);

			// TODO: Handle window resize messages during loading
			m_shaders_cache->load(&dlg);
		}
	}

	// Saves every pipeline built so far to the pipeline archive (background thread)
	mtl::on_pipeline_cache_preloaded();
}

void MTLGSRender::on_exit()
{
	GSRender::on_exit();
	mtl::destroy_pipe_compiler(); // Ensure no pending shaders being compiled
	mtl::flush_pipeline_archive_async(); // Start saving pending pipelines; render_device::destroy() waits (bounded)
	zcull_ctrl.release();
}

void MTLGSRender::flush_command_queue(bool hard_sync, bool do_not_switch)
{
	close_and_submit_command_buffer();

	if (hard_sync)
	{
		// wait for the latest instruction to execute
		m_current_command_buffer->reset();

		// Clear all command buffer statuses
		m_primary_cb_list.poke_all();

		// Drain present queue. The main-queue work is complete here, but present lists wait for their drawables, which
		// the display releases at the paced rate: block (not spin) and count the wait as display back-pressure, so the
		// pacing does not mistake it for a slow guest frame.
		while (!m_queued_frames.empty())
		{
			const u64 wait_start = get_system_time();
			frame_context_cleanup(m_queued_frames.front()); // Bounded wait (FRAME_PRESENT_TIMEOUT); pops the frame
			m_present_pacing.blocked_time += get_system_time() - wait_start;
		}

		m_flush_requests.clear_pending_flag();
	}

	if (!do_not_switch)
	{
		// Grab next cb in line and make it usable
		// NOTE: Even in the case of a hard sync, this is required to free any waiters on the CB (ZCULL)
		m_current_command_buffer = m_primary_cb_list.next();
		m_current_command_buffer->reset();
	}
	else
	{
		// Special hard-sync where we must preserve the CB. This can happen when an emergency event handler is invoked and needs to flush to hw.
		ensure(hard_sync);
	}

	// Just in case a queued frame holds a ref to this cb, drain the present queue
	check_present_status();

	m_current_command_buffer->clear_flags();

	if (m_occlusion_query_active)
	{
		m_current_command_buffer->flags |= mtl::command_list::cb_load_occluson_task;
	}

	m_current_command_buffer->begin();
}

bool MTLGSRender::release_GCM_label(u32 /*type*/, u32 /*address*/, u32 /*args*/)
{
	// Host GPU labels require writing guest memory from the GPU timeline (VK: vkCmdUpdateBuffer into a DMA block).
	// Not supported by the Metal backend (backend_config.supports_host_gpu_labels = false).
	ensure(!backend_config.supports_host_gpu_labels);
	return false;
}

void MTLGSRender::on_guest_texture_read(mtl::command_list& /*cmd*/)
{
	if (!backend_config.supports_host_gpu_labels)
	{
		return;
	}
}

void MTLGSRender::write_barrier(u32 address, u32 range)
{
	ensure(is_current_thread());
	m_rtts.invalidate_range(utils::address_range32::start_length(address, range));
}

void MTLGSRender::sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload)
{
	rsx::thread::sync_hint(hint, payload);

	if (!(m_current_command_buffer->flags & mtl::command_list::cb_has_occlusion_task))
	{
		// Occlusion queries not enabled, do nothing
		return;
	}

	// Occlusion test result evaluation is coming up, avoid a hard sync
	switch (hint)
	{
	case rsx::FIFO::interrupt_hint::conditional_render_eval:
	{
		// If a flush request is already enqueued, do nothing
		if (m_flush_requests.pending())
		{
			return;
		}

		// If the result is not going to be read by CELL, do nothing
		const auto ref_addr = static_cast<u32>(payload.address);
		if (!zcull_ctrl->is_query_result_urgent(ref_addr))
		{
			// No effect on CELL behaviour, it will be faster to handle this in RSX code
			return;
		}

		// OK, cell will be accessing the results, probably.
		// Try to avoid flush spam, it is more costly to flush the CB than it is to just upload the vertex data
		// This is supposed to be an optimization afterall.
		const auto now = get_system_time();
		if ((now - m_last_cond_render_eval_hint) > 50)
		{
			// Schedule a sync on the next loop iteration
			m_flush_requests.post(false);
			m_flush_requests.remove_one();
		}

		m_last_cond_render_eval_hint = now;
		break;
	}
	case rsx::FIFO::interrupt_hint::zcull_sync:
	{
		// Check if the required report is synced to this CB
		auto& data = m_occlusion_map[payload.query->driver_handle];

		// NOTE: Currently, a special condition exists where the indices can be empty even with active draw count.
		// This is caused by async compiler and should be removed when ubershaders are added in
		if (!data.is_current(m_current_command_buffer) || data.indices.empty())
		{
			return;
		}

		// Unavoidable hard sync coming up, flush immediately
		// This heavyweight hint should be used with caution
		std::lock_guard lock(m_flush_queue_mutex);
		flush_command_queue();

		if (m_flush_requests.pending())
		{
			// Clear without wait
			m_flush_requests.clear_pending_flag();
		}
		break;
	}
	}
}

void MTLGSRender::do_local_task(rsx::FIFO::state state)
{
	mtl::autorelease_scope pool;

	if (m_queue_status & flush_queue_state::deadlock)
	{
		// Clear offloader deadlock
		// NOTE: It is not possible to handle regular flush requests before this is cleared
		// NOTE: This may cause graphics corruption due to unsynchronized modification
		on_invalidate_memory_range(m_offloader_fault_range, m_offloader_fault_cause);
		m_queue_status.clear(flush_queue_state::deadlock);
	}

	if (m_queue_status & flush_queue_state::flushing)
	{
		// Abort recursive CB submit requests.
		// When flushing flag is already set, only deadlock events may be processed.
		return;
	}
	else if (m_flush_requests.pending())
	{
		if (m_flush_queue_mutex.try_lock())
		{
			// TODO: Determine if a hard sync is necessary
			// Pipeline barriers later may do a better job synchronizing than wholly stalling the pipeline
			flush_command_queue();

			m_flush_requests.clear_pending_flag();
			m_flush_requests.consumer_wait();
			m_flush_queue_mutex.unlock();
		}
	}
	else if (!in_begin_end && state != rsx::FIFO::state::lock_wait)
	{
		if (m_graphics_state & rsx::pipeline_state::framebuffer_reads_dirty)
		{
			//This will re-engage locks and break the texture cache if another thread is waiting in access violation handler!
			//Only call when there are no waiters
			m_texture_cache.do_update();
			m_graphics_state.clear(rsx::pipeline_state::framebuffer_reads_dirty);
		}
	}

	rsx::thread::do_local_task(state);

	switch (state)
	{
	case rsx::FIFO::state::lock_wait:
		// Critical check finished
		return;
	default:
		break;
	}

	if (m_overlay_manager)
	{
		const auto should_ignore = in_begin_end && state != rsx::FIFO::state::empty;
		if ((async_flip_requested & flip_request::native_ui) && !should_ignore && !is_stopped())
		{
			flush_command_queue(true);
			rsx::display_flip_info_t info{};
			info.buffer = current_display_buffer;
			flip(info);
		}
	}
}

bool MTLGSRender::load_program()
{
	const auto shadermode = g_cfg.video.shadermode.get();

	const auto [primitive, emulated_primitive] = mtl::get_appropriate_topology(rsx::method_registers.current_draw_clause.primitive);
	const u8 topology_class = mtl::get_topology_class(primitive);

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		get_current_fragment_program(fs_sampler_state);
		ensure(current_fragment_program.valid);

		get_current_vertex_program(vs_sampler_state);

		m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
	}
	else if (!(m_graphics_state & rsx::pipeline_state::pipeline_config_dirty) &&
		m_program &&
		m_pipeline_properties.state.topology_class == topology_class &&
		m_pipeline_renderpass_key == m_current_renderpass_key)
	{
		return true;
	}

	auto &vertex_program = current_vertex_program;
	auto &fragment_program = current_fragment_program;

	if ((m_graphics_state & rsx::pipeline_state::pipeline_config_dirty) ||
		m_pipeline_renderpass_key != m_current_renderpass_key)
	{
		mtl::pipeline_props properties = mtl::decode_rsx_state(
			m_ctx,
			topology_class,
			m_fbo_images,
			m_rtts.m_bound_depth_stencil.second,
			backend_config,
			static_cast<u8>(m_draw_buffers.size()),
			m_draw_fbo.samples,
			!!(current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
		);

		m_graphics_state.clear(rsx::pipeline_state::pipeline_config_dirty);
		m_pipeline_renderpass_key = m_current_renderpass_key;

		if (m_program && m_pipeline_properties == properties)
		{
			// Nothing changed
			return true;
		}

		// Fallthrough
		m_pipeline_properties = properties;
	}
	else
	{
		// Update primitive type
		m_pipeline_properties.state.topology_class = topology_class;
	}

	m_vertex_prog = nullptr;
	m_fragment_prog = nullptr;

	mtl::enter_uninterruptible();

	if (g_cfg.video.debug_overlay)
	{
		m_frame_stats.program_cache_lookups_total += 2;
		if (m_program_cache_hint.has_fragment_program())
		{
			m_frame_stats.program_cache_lookups_ellided++;
		}
		if (m_program_cache_hint.has_vertex_program())
		{
			m_frame_stats.program_cache_lookups_ellided++;
		}
	}

	// Load current program from cache. The shader interpreter is not ported: every non-recompiler mode compiles
	// asynchronously and skips draws until the pipeline is ready.
	std::tie(m_program, m_vertex_prog, m_fragment_prog) = m_prog_buffer->get_graphics_pipeline(
		&m_program_cache_hint,
		vertex_program,
		fragment_program,
		m_pipeline_properties,
		shadermode != shader_mode::recompiler, true);

	// The pipeline is being compiled on a worker. Without an interpreter the draw would be skipped: wait a little for
	// it (compiles typically take a few ms on Apple silicon), within a per-frame budget so a burst of new shaders costs
	// at most a short hitch.
	if (!m_program && shadermode != shader_mode::recompiler && m_async_compile_wait_spent_us < async_compile_wait_budget_us)
	{
		const u64 wait_start = get_system_time();

		mtl::leave_uninterruptible();

		while (!m_program && get_system_time() - wait_start + m_async_compile_wait_spent_us < async_compile_wait_budget_us)
		{
			std::this_thread::sleep_for(std::chrono::microseconds(250));

			mtl::enter_uninterruptible();
			std::tie(m_program, m_vertex_prog, m_fragment_prog) = m_prog_buffer->get_graphics_pipeline(
				&m_program_cache_hint,
				vertex_program,
				fragment_program,
				m_pipeline_properties,
				true, true);
			mtl::leave_uninterruptible();
		}

		m_async_compile_wait_spent_us += get_system_time() - wait_start;
		mtl::enter_uninterruptible();
	}

	mtl::leave_uninterruptible();

	if (m_prog_buffer->check_cache_missed())
	{
		// Notify the user with HUD notification
		if (g_cfg.misc.show_shader_compilation_hint)
		{
			if (m_overlay_manager)
			{
				rsx::overlays::show_shader_compile_notification();
			}
		}
	}

	if (!m_program &&
		(shadermode == shader_mode::async_with_interpreter || shadermode == shader_mode::interpreter_only) &&
		!m_interpreter_warning_logged)
	{
		m_interpreter_warning_logged = true;
		rsx_log.warning("Metal: shader interpreter unavailable, skipping draws until pipelines finish compiling.");
	}

	if (m_program)
	{
		std::tie(m_vs_binding_table, m_fs_binding_table) = get_binding_table();
	}
	else
	{
		m_vs_binding_table = nullptr;
		m_fs_binding_table = nullptr;
	}

	return m_program != nullptr;
}

void MTLGSRender::load_program_env()
{
	if (!m_program)
	{
		fmt::throw_exception("Unreachable right now");
	}

	const auto& ctx = REGS(m_ctx);

	const u32 fragment_constants_size = current_fp_metadata.program_constants_buffer_length;

	const bool update_transform_constants = !!(m_graphics_state & rsx::pipeline_state::transform_constants_dirty);
	const bool update_fragment_constants = !!(m_graphics_state & rsx::pipeline_state::fragment_constants_dirty);
	const bool update_vertex_env = !!(m_graphics_state & rsx::pipeline_state::vertex_state_dirty);
	const bool update_fragment_env = !!(m_graphics_state & rsx::pipeline_state::fragment_state_dirty);
	const bool update_fragment_texture_env = !!(m_graphics_state & rsx::pipeline_state::fragment_texture_state_dirty);
	const bool update_raster_env = (ctx->polygon_stipple_enabled() && !!(m_graphics_state & rsx::pipeline_state::polygon_stipple_pattern_dirty));
	const bool update_instancing_data = ctx->current_draw_clause.is_trivial_instanced_draw;

	if (update_vertex_env)
	{
		// Vertex state. Note, we're now on std430 alignment here, not hardware alignment.
		// Use the bulk allocator here
		const auto mem = m_vertex_env_allocator->alloc();
		auto buf = m_vertex_env_ring_info.map<char>(mem, 96);

		m_draw_processor.fill_scale_offset_data(buf, false);
		m_draw_processor.fill_user_clip_data(buf + 64);
		*(reinterpret_cast<u32*>(buf + 68)) = ctx->transform_branch_bits();
		*(reinterpret_cast<f32*>(buf + 72)) = ctx->point_size() * resolution_scaling_config.scale_factor();
		*(reinterpret_cast<f32*>(buf + 76)) = ctx->clip_min();
		*(reinterpret_cast<f32*>(buf + 80)) = ctx->clip_max();

		m_vertex_env_ring_info.unmap();
		m_vertex_env_dynamic_offset = mem;
	}

	if (update_instancing_data)
	{
		// Combines transform load + instancing lookup table
		constexpr usz alignment = 256;
		usz indirection_table_offset = 0;
		usz constants_data_table_offset = 0;

		rsx::io_buffer indirection_table_buf([&](usz size) -> std::pair<void*, usz>
		{
			indirection_table_offset = m_instancing_buffer_ring_info.alloc<1>(utils::align(size, alignment));
			return std::make_pair(m_instancing_buffer_ring_info.map(indirection_table_offset, size), size);
		});

		rsx::io_buffer constants_array_buf([&](usz size) -> std::pair<void*, usz>
		{
			constants_data_table_offset = m_instancing_buffer_ring_info.alloc<1>(utils::align(size, alignment));
			return std::make_pair(m_instancing_buffer_ring_info.map(constants_data_table_offset, size), size);
		});

		m_draw_processor.fill_constants_instancing_buffer(indirection_table_buf, constants_array_buf, m_vertex_prog);
		m_instancing_buffer_ring_info.unmap();

		m_instancing_indirection_buffer_info = { m_instancing_buffer_ring_info.heap.get(), indirection_table_offset, indirection_table_buf.size() };
		m_instancing_constants_array_buffer_info = { m_instancing_buffer_ring_info.heap.get(), constants_data_table_offset, constants_array_buf.size() };
	}
	else if (update_transform_constants)
	{
		// Transform constants
		usz mem_offset = 0;
		auto alloc_storage = [&](usz size) -> std::pair<void*, usz>
		{
			mem_offset = m_transform_constants_allocator->alloc_bytes(size);
			return std::make_pair(m_transform_constants_ring_info.map(mem_offset, size), size);
		};

		auto io_buf = rsx::io_buffer(alloc_storage);
		upload_transform_constants(io_buf);

		if (!io_buf.empty())
		{
			m_transform_constants_ring_info.unmap();
			m_xform_constants_dynamic_offset = mem_offset;
		}
	}

	if (update_fragment_constants)
	{
		// Fragment constants
		if (fragment_constants_size)
		{
			m_fragment_constants_dynamic_offset = m_fragment_constants_allocator->alloc_bytes(fragment_constants_size);
			auto buf = m_fragment_constants_ring_info.map(m_fragment_constants_dynamic_offset, fragment_constants_size);

			m_prog_buffer->fill_fragment_constants_buffer({ reinterpret_cast<float*>(buf), fragment_constants_size },
				*ensure(m_fragment_prog), current_fragment_program, true);

			m_fragment_constants_ring_info.unmap();
		}
	}

	if (update_fragment_env)
	{
		m_fragment_env_dynamic_offset = m_fragment_env_ring_info.static_alloc<32>();
		auto buf = m_fragment_env_ring_info.map(m_fragment_env_dynamic_offset, 32);

		m_draw_processor.fill_fragment_state_buffer(buf, current_fragment_program);
		m_fragment_env_ring_info.unmap();
	}

	if (update_fragment_texture_env)
	{
		m_texture_parameters_dynamic_offset = m_fragment_texture_params_ring_info.static_alloc<256, 768>();
		auto buf = m_fragment_texture_params_ring_info.map(m_texture_parameters_dynamic_offset, 768);

		current_fragment_program.texture_params.write_to(buf, current_fp_metadata.referenced_textures_mask);
		m_fragment_texture_params_ring_info.unmap();
	}

	if (update_raster_env)
	{
		m_stipple_array_dynamic_offset = m_raster_env_ring_info.static_alloc<128>();
		auto buf = m_raster_env_ring_info.map(m_stipple_array_dynamic_offset, 128);

		std::memcpy(buf, ctx->polygon_stipple_pattern(), 128);
		m_raster_env_ring_info.unmap();

		m_graphics_state.clear(rsx::pipeline_state::polygon_stipple_pattern_dirty);
	}

	// Rings are bound whole; the shaders index them with the dynamic offsets packed into the vertex layout entry.
	// Bindings are rebuilt on every call since a ring may have swapped its backing buffer (growth).
	const auto whole_heap = [](const mtl::data_heap& heap)
	{
		return mtl::glsl::buffer_binding_info(heap.heap.get(), 0, heap.size());
	};

	m_program->bind_uniform(whole_heap(m_vertex_env_ring_info), mtl::glsl::binding_set_index_vertex, m_vs_binding_table->context_buffer_location);
	m_program->bind_uniform(whole_heap(m_vertex_layout_ring_info), mtl::glsl::binding_set_index_vertex, m_vs_binding_table->vertex_buffers_location + 2);
	m_program->bind_uniform(whole_heap(m_fragment_env_ring_info), mtl::glsl::binding_set_index_fragment, m_fs_binding_table->context_buffer_location);
	m_program->bind_uniform(whole_heap(m_fragment_texture_params_ring_info), mtl::glsl::binding_set_index_fragment, m_fs_binding_table->tex_param_location);
	m_program->bind_uniform(whole_heap(m_raster_env_ring_info), mtl::glsl::binding_set_index_fragment, m_fs_binding_table->polygon_stipple_params_location);

	if (m_vs_binding_table->cbuf_location != umax)
	{
		m_program->bind_uniform(whole_heap(m_transform_constants_ring_info), mtl::glsl::binding_set_index_vertex, m_vs_binding_table->cbuf_location);
	}

	if (m_fs_binding_table->cbuf_location != umax)
	{
		m_program->bind_uniform(whole_heap(m_fragment_constants_ring_info), mtl::glsl::binding_set_index_fragment, m_fs_binding_table->cbuf_location);
	}

	if (mtl::emulate_conditional_rendering() && m_vs_binding_table->cr_pred_buffer_location != umax)
	{
		// Word 1 of the predicate buffer is all ones (always draw)
		const u32 offset = cond_render_ctrl.hw_cond_active ? 0 : 4;
		m_program->bind_uniform({ m_cond_render_buffer.get(), offset, 4 }, mtl::glsl::binding_set_index_vertex, m_vs_binding_table->cr_pred_buffer_location);
	}

	if (current_vertex_program.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS)
	{
		m_program->bind_uniform(m_instancing_indirection_buffer_info, mtl::glsl::binding_set_index_vertex, m_vs_binding_table->instanced_lut_buffer_location);
		m_program->bind_uniform(m_instancing_constants_array_buffer_info, mtl::glsl::binding_set_index_vertex, m_vs_binding_table->instanced_cbuf_location);
	}

	// Clear flags
	rsx::flags32_t handled_flags =
		rsx::pipeline_state::fragment_state_dirty |
		rsx::pipeline_state::vertex_state_dirty |
		rsx::pipeline_state::fragment_texture_state_dirty;

	if (!update_instancing_data)
	{
		handled_flags |= rsx::pipeline_state::transform_constants_dirty;
	}

	if (update_fragment_constants)
	{
		handled_flags |= rsx::pipeline_state::fragment_constants_dirty;
	}

	m_graphics_state.clear(handled_flags);
}

std::pair<const MTLGSRender::vs_binding_table_t*, const MTLGSRender::fs_binding_table_t*> MTLGSRender::get_binding_table() const
{
	ensure(m_program && m_vertex_prog && m_fragment_prog);
	return { &m_vertex_prog->binding_table, &m_fragment_prog->binding_table };
}

bool MTLGSRender::is_current_program_interpreted() const
{
	// No shader interpreter on Metal
	return false;
}

std::pair<std::string, std::string> MTLGSRender::get_programs() const
{
	return
	{
		m_vertex_prog ? m_vertex_prog->shader.get_source() : std::string{},
		m_fragment_prog ? m_fragment_prog->shader.get_source() : std::string{}
	};
}

void MTLGSRender::upload_transform_constants(const rsx::io_buffer& buffer)
{
	const usz transform_constants_size = m_vertex_prog->has_indexed_constants ? 8192 : m_vertex_prog->constant_ids.size() * 16;

	if (transform_constants_size)
	{
		buffer.reserve(transform_constants_size);
		auto buf = buffer.data();

		const auto constant_ids = (transform_constants_size == 8192)
			? std::span<const u16>{}
			: std::span<const u16>(m_vertex_prog->constant_ids);
		m_draw_processor.fill_vertex_program_constants_data(buf, constant_ids);
	}
}

void MTLGSRender::update_vertex_env(u32 id, const mtl::vertex_upload_info& vertex_info)
{
#pragma pack(push, 1)
	struct rsx_prog_vertex_layout_entry_t
	{
		u32 vertex_base_index;
		u32 vertex_index_offset;
		u32 draw_id;
		u32 xform_constants_offset;
		u32 vs_context_offset;
		u32 fs_constants_offset;
		u32 fs_context_offset;
		u32 fs_texture_base_index;
		u32 fs_stipple_pattern_offset;
		u32 reserved;
		s32 attrib_data[1];
	};
#pragma pack(pop)

	// Actual allocation must have been done previously
	const u32 vs_constant_id_offset = static_cast<u32>(m_xform_constants_dynamic_offset) / 16u;
	const u32 vertex_context_offset = static_cast<u32>(m_vertex_env_dynamic_offset) / 96u;
	const u32 vertex_layout_offset = static_cast<u32>(m_vertex_layout_dynamic_offset) / 168u;
	const u32 fs_constant_id_offset = static_cast<u32>(m_fragment_constants_dynamic_offset) / 16u;
	const u32 fs_context_offset = static_cast<u32>(m_fragment_env_dynamic_offset) / 32u;
	const u32 fs_texture_base_index = static_cast<u32>(m_texture_parameters_dynamic_offset) / 48u;
	const u32 fs_stipple_pattern_offset = static_cast<u32>(m_stipple_array_dynamic_offset) / 16u;

	auto buf = m_vertex_layout_ring_info.map(m_vertex_layout_dynamic_offset + (168u * id), 168u);
	auto dst = reinterpret_cast<rsx_prog_vertex_layout_entry_t*>(buf);

	// Pack
	dst->vertex_base_index = vertex_info.vertex_index_base;
	dst->vertex_index_offset = vertex_info.vertex_index_offset;
	dst->draw_id = id;

	dst->xform_constants_offset = vs_constant_id_offset;
	dst->vs_context_offset = vertex_context_offset;

	dst->fs_constants_offset = fs_constant_id_offset;
	dst->fs_context_offset = fs_context_offset;
	dst->fs_texture_base_index = fs_texture_base_index;
	dst->fs_stipple_pattern_offset = fs_stipple_pattern_offset;

	// Push constants are staged by the program and uploaded to the scratch heap on bind()
	const u32 push_val = vertex_layout_offset + id;
	m_program->push_constants(mtl::glsl::binding_set_index_vertex, 0, 4, &push_val);

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		// TODO: This should be cached aggressively.
		u32 blend_config[7];
		blend_config[0] = REGS(m_ctx)->registers[NV4097_SET_BLEND_EQUATION];
		blend_config[1] = REGS(m_ctx)->registers[NV4097_SET_BLEND_FUNC_SFACTOR];
		blend_config[2] = REGS(m_ctx)->registers[NV4097_SET_BLEND_FUNC_DFACTOR];

		const auto blend_colors = rsx::get_constant_blend_colors();
		blend_config[3] = std::bit_cast<u32>(blend_colors[0]);
		blend_config[4] = std::bit_cast<u32>(blend_colors[1]);
		blend_config[5] = std::bit_cast<u32>(blend_colors[2]);
		blend_config[6] = std::bit_cast<u32>(blend_colors[3]);

		m_program->push_constants(mtl::glsl::binding_set_index_fragment, 4, 28, blend_config);
	}

	if (m_fragment_prog && m_fragment_prog->requires_lod_bias)
	{
		// Shader-side sampler LOD bias (samplers can't apply it before Apple10), see MTLFragmentProgram::requires_lod_bias
		m_program->push_constants(mtl::glsl::binding_set_index_fragment,
			MTLFragmentProgram::lod_bias_push_offset, MTLFragmentProgram::lod_bias_push_size, m_fs_lod_bias.data());
	}

	// Now actually fill in the data
	m_draw_processor.fill_vertex_layout_state(
		m_vertex_layout,
		current_vp_metadata,
		vertex_info.first_vertex,
		vertex_info.allocated_vertex_count,
		dst->attrib_data,
		vertex_info.persistent_window_offset,
		vertex_info.volatile_window_offset);

	m_vertex_layout_ring_info.unmap();
}

void MTLGSRender::patch_transform_constants(rsx::context* /*ctx*/, u32 index, u32 count)
{
	if (!m_program || !m_vertex_prog)
	{
		// Shouldn't be reachable, but handle it correctly anyway
		m_graphics_state |= rsx::pipeline_state::transform_constants_dirty;
		return;
	}

	if (!m_vertex_prog->overlaps_constants_range(index, count))
	{
		// Nothing meaningful to us
		return;
	}

	// Constants are addressed through the draw parameters, so a patch is just a new allocation
	auto allocate_mem = [&](usz size) -> std::pair<void*, usz>
	{
		m_xform_constants_dynamic_offset = m_transform_constants_ring_info.alloc<256>(utils::align(size, 256));
		return std::make_pair(m_transform_constants_ring_info.map(m_xform_constants_dynamic_offset, size), size);
	};

	rsx::io_buffer iobuf(allocate_mem);
	upload_transform_constants(iobuf);

	if (!iobuf.empty())
	{
		m_transform_constants_ring_info.unmap();
	}
}

void MTLGSRender::init_buffers(rsx::framebuffer_creation_context context, bool)
{
	prepare_rtts(context);
}

void MTLGSRender::close_and_submit_command_buffer(const mtl::submit_info_t& submit_info)
{
	ensure(!m_queue_status.test_and_set(flush_queue_state::flushing));

	// Host MM sync before executing anything on the GPU
	rsx::mm_flush();

	// Workaround for deadlock occuring during RSX offloader fault
	// TODO: Restructure command submission infrastructure to avoid this condition
	const bool sync_success = g_fxo->get<rsx::dma_manager>().sync();

	// DMA readbacks are fenced on the submission itself (mtl::dma_fence_t observes the list's timeline value), and
	// the waiter usually follows immediately: commit such lists inline instead of through the offloader.
	const bool force_flush = !sync_success || !!(m_current_command_buffer->flags & mtl::command_list::cb_has_dma_transfer);

	if (mtl::test_status_interrupt(mtl::heap_dirty))
	{
		// Unified memory: ring buffers never need a manual flush
		mtl::clear_status_interrupt(mtl::heap_dirty);
	}

	// End any active renderpasses; the caller should handle reopening
	close_render_pass();

	// End open queries. Flags will be automatically reset by the submit routine
	if (m_current_command_buffer->flags & mtl::command_list::cb_has_open_query)
	{
		auto open_query = m_occlusion_map[m_active_query_info->driver_handle].indices.back();
		m_occlusion_query_manager->end_query(*m_current_command_buffer, nullptr, open_query);
		m_current_command_buffer->flags &= ~mtl::command_list::cb_has_open_query;
	}

	m_current_command_buffer->end();
	m_current_command_buffer->tag();

	mtl::queue_submit(m_current_command_buffer, submit_info, force_flush);

	m_current_command_buffer->clear_flags();
	m_render_pass.reset();

	m_queue_status.clear(flush_queue_state::flushing);
}

void MTLGSRender::prepare_rtts(rsx::framebuffer_creation_context context)
{
	const bool clipped_scissor = (context == rsx::framebuffer_creation_context::context_draw);
	if (m_current_framebuffer_context == context && !m_graphics_state.test(rsx::rtt_config_dirty) && m_draw_pass_desc)
	{
		// Fast path
		// Framebuffer usage has not changed, framebuffer exists and config regs have not changed
		set_scissor(clipped_scissor);
		return;
	}

	m_graphics_state.clear(
		rsx::rtt_config_dirty |
		rsx::rtt_config_contested |
		rsx::rtt_config_valid |
		rsx::rtt_cache_state_dirty);

	get_framebuffer_layout(context, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	if (m_draw_pass_desc && m_framebuffer_layout.ignore_change)
	{
		// Nothing has changed, we're still using the same framebuffer
		// Update flags to match current
		set_scissor(clipped_scissor);
		return;
	}

	m_rtts.prepare_render_target(*m_current_command_buffer,
		m_framebuffer_layout.color_format, m_framebuffer_layout.depth_format,
		m_framebuffer_layout.width, m_framebuffer_layout.height,
		m_framebuffer_layout.target, m_framebuffer_layout.aa_mode, m_framebuffer_layout.raster_type,
		m_framebuffer_layout.color_addresses, m_framebuffer_layout.zeta_address,
		m_framebuffer_layout.actual_color_pitch, m_framebuffer_layout.actual_zeta_pitch,
		resolution_scaling_config);

	// Reset framebuffer information
	const auto color_bpp = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
	const auto samples = get_format_sample_count(m_framebuffer_layout.aa_mode);

	for (u8 i = 0; i < rsx::limits::color_buffers_count; ++i)
	{
		// Flush old address if we keep missing it
		if (m_surface_info[i].pitch && g_cfg.video.write_color_buffers)
		{
			const utils::address_range32 rsx_range = m_surface_info[i].get_memory_range();
			m_texture_cache.set_memory_read_flags(rsx_range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(*m_current_command_buffer, rsx_range);
		}

		m_surface_info[i].address = m_surface_info[i].pitch = 0;
		m_surface_info[i].width = m_framebuffer_layout.width;
		m_surface_info[i].height = m_framebuffer_layout.height;
		m_surface_info[i].color_format = m_framebuffer_layout.color_format;
		m_surface_info[i].bpp = color_bpp;
		m_surface_info[i].samples = samples;
	}

	// Process depth surface as well
	{
		if (m_depth_surface_info.pitch && g_cfg.video.write_depth_buffer)
		{
			const utils::address_range32 surface_range = m_depth_surface_info.get_memory_range();
			m_texture_cache.set_memory_read_flags(surface_range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(*m_current_command_buffer, surface_range);
		}

		m_depth_surface_info.address = m_depth_surface_info.pitch = 0;
		m_depth_surface_info.width = m_framebuffer_layout.width;
		m_depth_surface_info.height = m_framebuffer_layout.height;
		m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
		m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
		m_depth_surface_info.samples = samples;
	}

	// Bind created rtts as current fbo...
	const auto draw_buffers = rsx::utility::get_rtt_indexes(m_framebuffer_layout.target);
	m_draw_buffers.clear();
	m_fbo_images.clear();

	for (u8 index : draw_buffers)
	{
		if (auto surface = std::get<1>(m_rtts.m_bound_render_targets[index]))
		{
			m_fbo_images.push_back(surface);

			m_surface_info[index].address = m_framebuffer_layout.color_addresses[index];
			m_surface_info[index].pitch = m_framebuffer_layout.actual_color_pitch[index];
			ensure(surface->rsx_pitch == m_framebuffer_layout.actual_color_pitch[index]);

			m_texture_cache.notify_surface_changed(m_surface_info[index].get_memory_range(m_framebuffer_layout.aa_factors));
			m_draw_buffers.push_back(index);
		}
	}

	if (std::get<0>(m_rtts.m_bound_depth_stencil) != 0)
	{
		auto ds = std::get<1>(m_rtts.m_bound_depth_stencil);
		m_fbo_images.push_back(ds);

		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
		ensure(ds->rsx_pitch == m_framebuffer_layout.actual_zeta_pitch);

		m_texture_cache.notify_surface_changed(m_depth_surface_info.get_memory_range(m_framebuffer_layout.aa_factors));
	}

	// Before messing with memory properties, flush command queue if there are dma transfers queued up
	if (m_current_command_buffer->flags & mtl::command_list::cb_has_dma_transfer)
	{
		flush_command_queue();
	}

	if (!m_rtts.superseded_surfaces.empty())
	{
		for (auto& surface : m_rtts.superseded_surfaces)
		{
			m_texture_cache.discard_framebuffer_memory_region(*m_current_command_buffer, surface->get_memory_range());
		}

		m_rtts.superseded_surfaces.clear();
	}

	if (!m_rtts.orphaned_surfaces.empty())
	{
		for (auto& [base_addr, surface] : m_rtts.orphaned_surfaces)
		{
			bool lock = surface->is_depth_surface() ? !!g_cfg.video.write_depth_buffer :
				!!g_cfg.video.write_color_buffers;

			if (lock &&
#ifdef TEXTURE_CACHE_DEBUG
				!m_texture_cache.is_protected(
					base_addr,
					surface->get_memory_range(),
					rsx::texture_upload_context::framebuffer_storage)
#else
				!surface->is_locked()
#endif
				)
			{
				lock = false;
			}

			if (!lock) [[likely]]
			{
				m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer, surface->get_memory_range());
				continue;
			}

			m_texture_cache.lock_memory_region(
				*m_current_command_buffer, surface, surface->get_memory_range(), false,
				surface->template get_surface_width<rsx::surface_metrics::pixels>(), surface->template get_surface_height<rsx::surface_metrics::pixels>(), surface->get_rsx_pitch(),
				surface);
		}

		m_rtts.orphaned_surfaces.clear();
	}

	for (u8 index : m_draw_buffers)
	{
		if (!m_surface_info[index].address || !m_surface_info[index].pitch) continue;

		const utils::address_range32 surface_range = m_surface_info[index].get_memory_range();
		if (g_cfg.video.write_color_buffers)
		{
			m_texture_cache.lock_memory_region(
				*m_current_command_buffer, m_rtts.m_bound_render_targets[index].second, surface_range, true,
				m_surface_info[index].width, m_surface_info[index].height, m_framebuffer_layout.actual_color_pitch[index],
				m_rtts.m_bound_render_targets[index].second);
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer, surface_range);
		}
	}

	if (m_depth_surface_info.address && m_depth_surface_info.pitch)
	{
		const utils::address_range32 surface_range = m_depth_surface_info.get_memory_range();
		if (g_cfg.video.write_depth_buffer)
		{
			m_texture_cache.lock_memory_region(
				*m_current_command_buffer, m_rtts.m_bound_depth_stencil.second, surface_range, true,
				m_depth_surface_info.width, m_depth_surface_info.height, m_framebuffer_layout.actual_zeta_pitch,
				m_rtts.m_bound_depth_stencil.second);
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer, surface_range);
		}
	}

	m_current_renderpass_key = mtl::get_renderpass_key(m_fbo_images);

	// Build the framebuffer (attachment set) and its render pass descriptor
	const auto [fbo_width, fbo_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);

	mtl::framebuffer_info fbo{};
	for (auto& surface : m_fbo_images)
	{
		if (surface->aspect() & mtl::aspect_color)
		{
			fbo.color[fbo.color_count++] = surface;
		}
		else
		{
			fbo.depth_stencil = surface;
		}

		fbo.samples = std::max<u8>(fbo.samples, surface->samples());
	}

	fbo.width = fbo_width;
	fbo.height = fbo_height;

	const bool fbo_changed = !m_draw_pass_desc ||
		fbo.color_count != m_draw_fbo.color_count ||
		fbo.color != m_draw_fbo.color ||
		fbo.depth_stencil != m_draw_fbo.depth_stencil ||
		fbo.width != m_draw_fbo.width ||
		fbo.height != m_draw_fbo.height ||
		fbo.samples != m_draw_fbo.samples;

	if (fbo_changed)
	{
		// The active pass renders into the previous surface set
		close_render_pass();

		m_draw_fbo = fbo;
		update_render_pass_descriptor();
	}

	set_viewport();
	set_scissor(clipped_scissor);
	on_framebuffer_layout_updated();

	check_zcull_status(true);
}

void MTLGSRender::renderctl(u32 request_code, void* args)
{
	switch (request_code)
	{
	case mtl::rctrl_queue_submit:
	{
		mtl::autorelease_scope pool;
		const auto packet = reinterpret_cast<mtl::queue_submit_t*>(args);
		mtl::queue_submit(packet);
		delete packet;
		break;
	}
	default:
		rsx::thread::renderctl(request_code, args);
	}
}

bool MTLGSRender::scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate)
{
	if (swapchain_unavailable)
		return false;

	mtl::autorelease_scope pool;

	if (m_texture_cache.blit(src, dst, interpolate, m_rtts, *m_current_command_buffer))
	{
		m_samplers_dirty.store(true);
		m_current_command_buffer->set_flag(mtl::command_list::cb_has_blit_transfer);

		if (m_current_command_buffer->flags & mtl::command_list::cb_has_dma_transfer)
		{
			// A dma transfer has been queued onto this cb
			// This likely means that we're done with the tranfers to the target (writes_likely_completed=1)
			flush_command_queue();
		}
		return true;
	}

	return false;
}

void MTLGSRender::begin_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	ensure(!m_occlusion_query_active);

	query->result = 0;
	//query->sync_timestamp = get_system_time();
	m_active_query_info = query;
	m_occlusion_query_active = true;
	m_current_command_buffer->flags |= mtl::command_list::cb_load_occluson_task;
}

void MTLGSRender::end_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	ensure(query == m_active_query_info);

	// NOTE: flushing the queue is very expensive, do not flush just because query stopped
	if (m_current_command_buffer->flags & mtl::command_list::cb_has_open_query)
	{
		// End query. The visibility mode is encoder state; disarm it on the open pass.
		auto open_query = m_occlusion_map[m_active_query_info->driver_handle].indices.back();
		m_occlusion_query_manager->end_query(*m_current_command_buffer, get_render_encoder(), open_query);
		m_current_command_buffer->flags &= ~mtl::command_list::cb_has_open_query;
	}

	// Clear occlusion load flag
	m_current_command_buffer->flags &= ~mtl::command_list::cb_load_occluson_task;

	m_occlusion_query_active = false;
	m_active_query_info = nullptr;
}

bool MTLGSRender::check_occlusion_query_status(rsx::reports::occlusion_query_info* query)
{
	if (!query->num_draws)
		return true;

	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return true;

	if (data.is_current(m_current_command_buffer))
		return false;

	const u32 oldest = data.indices.front();
	return m_occlusion_query_manager->check_query_status(oldest);
}

void MTLGSRender::get_occlusion_query_result(rsx::reports::occlusion_query_info* query)
{
	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return;

	if (query->num_draws)
	{
		if (data.is_current(m_current_command_buffer))
		{
			std::lock_guard lock(m_flush_queue_mutex);
			flush_command_queue();

			if (m_flush_requests.pending())
			{
				m_flush_requests.clear_pending_flag();
			}

			rsx_log.warning("[Performance warning] Unexpected ZCULL read caused a hard sync");
			busy_wait();
		}

		data.sync();

		// Gather data
		for (const auto occlusion_id : data.indices)
		{
			query->result += m_occlusion_query_manager->get_query_result(occlusion_id);
			if (query->result && !g_cfg.video.precise_zpass_count)
			{
				// We only need one hit unless precise zcull is requested
				break;
			}
		}
	}

	m_occlusion_query_manager->free_queries(*m_current_command_buffer, data.indices);
	data.indices.clear();
}

void MTLGSRender::discard_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (m_active_query_info == query)
	{
		end_occlusion_query(query);
	}

	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return;

	m_occlusion_query_manager->free_queries(*m_current_command_buffer, data.indices);
	data.indices.clear();
}

void MTLGSRender::emergency_query_cleanup(mtl::command_list* commands)
{
	ensure(commands == static_cast<mtl::command_list*>(m_current_command_buffer));

	if (m_current_command_buffer->flags & mtl::command_list::cb_has_open_query)
	{
		auto open_query = m_occlusion_map[m_active_query_info->driver_handle].indices.back();
		m_occlusion_query_manager->end_query(*m_current_command_buffer, get_render_encoder(), open_query);
		m_current_command_buffer->flags &= ~mtl::command_list::cb_has_open_query;
	}
}

void MTLGSRender::begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources)
{
	ensure(!sources.empty());

	// Flag check whether to calculate all entries or only one
	bool partial_eval;

	// Try and avoid regenerating the data if its a repeat/spam
	// NOTE: The incoming list is reversed with the first entry being the newest
	if (m_cond_render_sync_tag == sources.front()->sync_tag)
	{
		// Already synched, check subdraw which is possible if last sync happened while query was active
		if (!m_active_query_info || m_active_query_info != sources.front())
		{
			rsx::thread::begin_conditional_rendering(sources);
			return;
		}

		// Partial evaluation only
		partial_eval = true;
	}
	else
	{
		m_cond_render_sync_tag = sources.front()->sync_tag;
		partial_eval = false;
	}

	mtl::autorelease_scope pool;

	u32 dst_offset = 0;
	u32 num_hw_queries = 0;
	usz first = 0;
	usz last = (!partial_eval) ? sources.size() : 1;

	// Count number of queries available. This is an "opening" evaluation, if there is only one source, read it as-is.
	// The idea is to avoid scheduling a compute task unless we have to.
	for (usz i = first; i < last; ++i)
	{
		auto& query_info = m_occlusion_map[sources[i]->driver_handle];
		num_hw_queries += ::size32(query_info.indices);
	}

	// NOTE: Every GPU read of visibility results goes through cmd.compute(), which ends the active render pass.
	// Metal only writes visibility results when a pass completes, so this split is unavoidable.
	if (num_hw_queries == 1 && !partial_eval) [[ likely ]]
	{
		// Accept the first available query handle as the source of truth. No aggregation is required.
		for (usz i = first; i < last; ++i)
		{
			auto& query_info = m_occlusion_map[sources[i]->driver_handle];
			if (!query_info.indices.empty())
			{
				const auto& index = query_info.indices.front();
				m_occlusion_query_manager->get_query_result_indirect(*m_current_command_buffer, index, 1, m_cond_render_buffer.get(), 0, 4);

				rsx::thread::begin_conditional_rendering(sources);
				return;
			}
		}

		// This is unreachable unless something went horribly wrong
		fmt::throw_exception("Unreachable");
	}
	else if (num_hw_queries > 0)
	{
		// We'll need to do some result aggregation using a compute shader.
		// Each visibility slot is 64-bit; the aggregator sums 32-bit words, so both halves are summed (high words are 0).
		mtl::buffer* scratch = nullptr;

		// Range latching. Because of how the query pool manages allocations using a stack, we get an inverse sequential set of handles/indices that we can easily group together.
		struct { u32 first, last; } query_range = { umax, 0 };

		auto copy_query_range_impl = [&]()
		{
			if (!scratch)
			{
				scratch = mtl::get_scratch_buffer(*m_current_command_buffer, num_hw_queries * mtl::query_pool::slot_size);
			}

			const auto count = (query_range.last - query_range.first + 1);
			m_occlusion_query_manager->get_query_result_indirect(*m_current_command_buffer, query_range.first, count, scratch, dst_offset);
			dst_offset += count * mtl::query_pool::slot_size;
		};

		for (usz i = first; i < last; ++i)
		{
			auto& query_info = m_occlusion_map[sources[i]->driver_handle];
			for (const auto& index : query_info.indices)
			{
				// First iteration?
				if (query_range.first == umax)
				{
					query_range = { index, index };
					continue;
				}

				// Head?
				if ((query_range.first - 1) == index)
				{
					query_range.first = index;
					continue;
				}

				// Tail?
				if ((query_range.last + 1) == index)
				{
					query_range.last = index;
					continue;
				}

				// Flush pending queue. In practice, this is never reached and we fall out to the spill block outside the loops
				copy_query_range_impl();

				// Start a new range for the current index
				query_range = { index, index };
			}
		}

		if (query_range.first != umax)
		{
			// Dangling queries, flush
			copy_query_range_impl();
		}

		// Sanity check
		ensure(scratch && dst_offset <= scratch->size());

		if (!partial_eval)
		{
			// Fast path should have been caught above
			ensure(dst_offset > mtl::query_pool::slot_size);

			// Clear result to zero
			m_current_command_buffer->compute()->fillBuffer(m_cond_render_buffer->value(), NS::Range::Make(0, 4), 0);
		}

		mtl::get_compute_task<mtl::cs_aggregator>()->run(*m_current_command_buffer, m_cond_render_buffer.get(), scratch, dst_offset / 4);
	}
	else if (m_program)
	{
		// This can sometimes happen when shaders are compiling, only log if there is a program hit
		rsx_log.warning("Dubious query data pushed to cond render! Please report to developers(q.pending=%d)", sources.front()->pending);
	}

	rsx::thread::begin_conditional_rendering(sources);
}

void MTLGSRender::end_conditional_rendering()
{
	thread::end_conditional_rendering();
}
