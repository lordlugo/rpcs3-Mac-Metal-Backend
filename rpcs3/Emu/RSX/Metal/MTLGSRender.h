#pragma once

// Native Metal 4 renderer (port of VK/VKGSRender.h). Plain C++ (metal-cpp); includable from the Qt UI.

#include "upscalers/upscaling.h"

#include "mtlutils/buffer_object.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"
#include "mtlutils/sync.h"

#include "MTLGSRenderTypes.hpp"
#include "MTLTextureCache.h"
#include "MTLRenderTargets.h"
#include "MTLOverlays.h"
#include "MTLProgramBuffer.h"
#include "MTLRenderPass.h"
#include "MTLQueryPool.h"

#include "Emu/RSX/GSRender.h"

#include <deque>
#include <functional>
#include <initializer_list>
#include <unordered_map>

using namespace mtl::upscaling_flags_; // clang workaround.

namespace CA
{
	class MetalLayer;
}

class MTLGSRender : public GSRender, public ::rsx::reports::ZCULL_control
{
private:
	enum frame_context_state : u32
	{
		dirty = 1
	};

	enum flush_queue_state : u32
	{
		ok = 0,
		flushing = 1,
		deadlock = 2
	};

	using vs_binding_table_t = decltype(MTLVertexProgram::binding_table);
	using fs_binding_table_t = decltype(MTLFragmentProgram::binding_table);

private:
	const MTLFragmentProgram* m_fragment_prog = nullptr;

	// Per fragment texture unit mip LOD bias, pushed to programs with requires_lod_bias (pre-Apple10 GPUs)
	std::array<f32, 16> m_fs_lod_bias{};
	const MTLVertexProgram* m_vertex_prog = nullptr;
	mtl::glsl::program* m_program = nullptr;
	mtl::glsl::program* m_prev_program = nullptr;
	mtl::pipeline_props m_pipeline_properties{};
	u64 m_pipeline_renderpass_key = umax;          // Attachment configuration m_pipeline_properties was built for
	mtl::rasterizer_state m_rasterizer_state{};
	mtl::encoder_state m_encoder_state{};

	const vs_binding_table_t* m_vs_binding_table = nullptr;
	const fs_binding_table_t* m_fs_binding_table = nullptr;

	mtl::texture_cache m_texture_cache;
	mtl::surface_cache m_rtts;

	std::unique_ptr<mtl::buffer> null_buffer;
	std::unique_ptr<mtl::buffer_view> null_buffer_view;

	// Placeholder depth textures for shadow samplers without a bound image (depth2d<>/depthcube<> cannot take colour views)
	// [0] = 2D (also 1D, declared as 2D), [1] = Cube
	std::array<std::unique_ptr<mtl::viewable_image>, 2> m_null_depth_textures;

	std::unique_ptr<mtl::upscaler> m_upscaler;
	output_scaling_mode m_output_scaling{output_scaling_mode::bilinear};

	std::unique_ptr<mtl::buffer> m_cond_render_buffer;
	u64 m_cond_render_sync_tag = 0;

	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = { true };
	std::unique_ptr<mtl::sampler> m_stencil_mirror_sampler;
	std::array<mtl::sampler*, rsx::limits::fragment_textures_count> fs_sampler_handles{};
	std::array<mtl::sampler*, rsx::limits::vertex_textures_count> vs_sampler_handles{};

	std::unique_ptr<mtl::buffer_view> m_persistent_attribute_storage;
	std::unique_ptr<mtl::buffer_view> m_volatile_attribute_storage;

	std::pair<const vs_binding_table_t*, const fs_binding_table_t*> get_binding_table() const;

public:
	std::unique_ptr<mtl::vertex_cache> m_vertex_cache;
	std::unique_ptr<mtl::shader_cache> m_shaders_cache;

private:
	std::unique_ptr<mtl::program_cache> m_prog_buffer;

	// Device and presentation surface
	std::unique_ptr<mtl::render_device> m_device;
	mtl::timeline m_timeline;
	void* m_view = nullptr;                      // NSView of the game window
	CA::MetalLayer* m_metal_layer = nullptr;     // Backing layer of m_view (retained, released on teardown)
	bool m_layer_framebuffer_only = true;

	// Occlusion queries (visibility result buffers)
	std::unique_ptr<mtl::query_pool_manager> m_occlusion_query_manager;
	bool m_occlusion_query_active = false;
	rsx::reports::occlusion_query_info* m_active_query_info = nullptr;
	std::vector<mtl::occlusion_data> m_occlusion_map;

	shared_mutex m_secondary_cb_guard;
	mtl::command_buffer_chain<MTL_MAX_ASYNC_CB_COUNT> m_secondary_cb_list;

	mtl::command_buffer_chain<MTL_MAX_ASYNC_CB_COUNT> m_primary_cb_list;
	mtl::command_buffer_chunk* m_current_command_buffer = nullptr;

	// Main render pass (VK: renderpass + framebuffer)
	mtl::framebuffer_info m_draw_fbo{};
	std::array<const MTL::Texture*, 5> m_draw_fbo_textures{};  // Textures m_draw_pass_desc was built with
	mtl::ref<MTL4::RenderPassDescriptor> m_draw_pass_desc;
	const MTL::Buffer* m_draw_pass_visibility_buffer = nullptr;
	mtl::render_pass_tracker m_render_pass;
	u64 m_current_renderpass_key = 0;
	std::vector<mtl::image*> m_fbo_images;
	u32 m_render_pass_splits = 0;               // Passes split this frame (feedback loops, depth reads)

	// Depth-stencil state objects (Metal has no dynamic stencil masks)
	std::unordered_map<u64, mtl::ref<MTL::DepthStencilState>> m_depth_stencil_states;

	sizeu m_swapchain_dims{};
	bool swapchain_unavailable = false;
	bool should_reinitialize_swapchain = false;

	u64 m_last_heap_sync_time = 0;
	u32 m_texbuffer_view_size = 0;

	mtl::data_heap m_attrib_ring_info;                         // Vertex data
	mtl::data_heap m_fragment_constants_ring_info;             // Fragment program constants
	mtl::data_heap m_transform_constants_ring_info;            // Transform program constants
	mtl::data_heap m_fragment_env_ring_info;                   // Fragment environment params
	mtl::data_heap m_vertex_env_ring_info;                     // Vertex environment params
	mtl::data_heap m_fragment_texture_params_ring_info;        // Fragment texture params
	mtl::data_heap m_vertex_layout_ring_info;                  // Vertex layout structure
	mtl::data_heap m_index_buffer_ring_info;                   // Index data
	mtl::data_heap m_texture_upload_buffer_ring_info;          // Texture upload heap
	mtl::data_heap m_raster_env_ring_info;                     // Raster control such as polygon and line stipple
	mtl::data_heap m_instancing_buffer_ring_info;              // Instanced rendering data (constants indirection table + instanced constants)

	// Ring buffers are bound whole (by GPU address); shaders index them with the dynamic offsets below.
	mtl::glsl::buffer_binding_info m_instancing_indirection_buffer_info{};
	mtl::glsl::buffer_binding_info m_instancing_constants_array_buffer_info{};

	u64 m_xform_constants_dynamic_offset = 0;          // We manage transform_constants dynamic offset manually to alleviate performance penalty of doing a hot-patch of constants.
	u64 m_vertex_env_dynamic_offset = 0;
	u64 m_vertex_layout_dynamic_offset = 0;
	u64 m_fragment_constants_dynamic_offset = 0;
	u64 m_fragment_env_dynamic_offset = 0;
	u64 m_texture_parameters_dynamic_offset = 0;
	u64 m_stipple_array_dynamic_offset = 0;

	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 96>> m_vertex_env_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_transform_constants_allocator;
	std::unique_ptr<rsx::data_heap::bulk_allocator<256, 16>> m_fragment_constants_allocator;

	std::vector<mtl::frame_context_t> m_frame_context_storage;
	u32 m_max_async_frames = 0u;
	// Temp frame context to use if the real frame queue is overburdened. Only used for storage
	mtl::frame_context_t m_aux_frame_context;

	u32 m_current_queue_index = 0;
	mtl::frame_context_t* m_current_frame = nullptr;
	std::deque<mtl::frame_context_t*> m_queued_frames;

	MTL::Viewport m_viewport{};
	MTL::ScissorRect m_scissor{};

	// Scissor clamped to the render target (Metal rejects rectangles outside the attachments). Empty -> draws skipped.
	MTL::ScissorRect get_clamped_scissor() const;

	std::vector<u8> m_draw_buffers;

	shared_mutex m_flush_queue_mutex;
	mtl::flush_request_task m_flush_requests;

	ullong m_last_cond_render_eval_hint = 0;

	// Offloader thread deadlock recovery
	rsx::atomic_bitmask_t<flush_queue_state> m_queue_status;
	utils::address_range32 m_offloader_fault_range;
	rsx::invalidation_cause m_offloader_fault_cause;

	mtl::draw_call_t m_current_draw {};

	std::unique_ptr<mtl::viewable_image> m_overlay_recording_img;

	//Vertex layout
	rsx::vertex_input_layout m_vertex_layout;

	// One-time diagnostics
	bool m_interpreter_warning_logged = false;
	bool m_wide_lines_warning_logged = false;
	bool m_depth_bounds_warning_logged = false;
	bool m_logic_op_warning_logged = false;

public:
	u64 get_cycles() final;
	~MTLGSRender() override;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}

private:
	void prepare_rtts(rsx::framebuffer_creation_context context);

	void close_and_submit_command_buffer(const mtl::submit_info_t& submit_info = {});

	void flush_command_queue(bool hard_sync = false, bool do_not_switch = false);
	void queue_swap_request();
	void frame_context_cleanup(mtl::frame_context_t *ctx);
	void advance_queued_frames();
	void present(mtl::frame_context_t *ctx);
	bool reinitialize_swapchain();
	void configure_metal_layer();

	mtl::viewable_image* get_present_source(mtl::present_surface_info* info, const rsx::avconf& avconfig);

	// Render pass management
	void update_render_pass_descriptor();
	void begin_render_pass(const mtl::attachment_clear_info* clear = nullptr);
	void close_render_pass();
	void invalidate_render_pass();
	void split_render_pass();
	bool is_render_pass_open() const;
	MTL4::RenderCommandEncoder* get_render_encoder() const;
	void on_render_pass_begin(MTL4::RenderCommandEncoder* encoder);

	MTL::DepthStencilState* get_depth_stencil_state(u64 key);

	void update_draw_state();
	void check_present_status();
	void check_heap_status();

	mtl::vertex_upload_info upload_vertex_data();
	rsx::simple_array<u8> m_scratch_mem;

	bool load_program();
	void load_program_env();
	void update_vertex_env(u32 id, const mtl::vertex_upload_info& vertex_info);
	void upload_transform_constants(const rsx::io_buffer& buffer);

	void load_texture_env();
	bool bind_texture_env();

	mtl::image_view* get_null_texture_view(rsx::texture_dimension_extended type, bool is_depth);

public:
	void init_buffers(rsx::framebuffer_creation_context context, bool skip_reading = false);
	void set_viewport();
	void set_scissor(bool clip_viewport);
	void bind_viewport();

	// Sync
	void write_barrier(u32 address, u32 range) override;
	void sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload) override;
	bool release_GCM_label(u32 type, u32 address, u32 data) override;

	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override;
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override;
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override;

	// External callback in case we need to suddenly submit a commandlist unexpectedly, e.g in a violation handler
	void emergency_query_cleanup(mtl::command_list* commands);

	// External callback to handle out of video memory problems
	bool on_vram_exhausted(rsx::problem_severity severity);

	// Conditional rendering
	void begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources) override;
	void end_conditional_rendering() override;

	// Host sync object (host GPU labels are not supported on Metal; kept for API parity)
	void on_guest_texture_read(mtl::command_list& cmd);

	// GRAPH backend
	void patch_transform_constants(rsx::context* ctx, u32 index, u32 count) override;

	// Misc
	bool is_current_program_interpreted() const override;
	std::pair<std::string, std::string> get_programs() const override;

	// The timeline signaled by every submission of this renderer
	mtl::timeline& get_timeline() { return m_timeline; }

protected:
	void clear_surface(u32 mask) override;
	void begin() override;
	void end() override;
	void emit_geometry(u32 sub_index) override;

	void on_init_thread() override;
	void on_exit() override;
	void flip(const rsx::display_flip_info_t& info) override;

	void renderctl(u32 request_code, void* args) override;

	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate) override;
	void notify_tile_unbound(u32 tile) override;

	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
};
