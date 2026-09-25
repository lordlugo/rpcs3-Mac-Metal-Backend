#pragma once

// MetalFX spatial upscaling (replaces the VK backend's FidelityFX FSR1 pass, output_scaling_mode::fsr), followed by
// optional FidelityFX RCAS sharpening (g_cfg.video.rcas_sharpening_intensity, 0 = off; same curve as FSR1 on VK).
//
// Scalers: an MTL4FX::SpatialScaler is fixed to one configuration (formats, input texture size, output size) and
// creating one compiles pipelines (tens to hundreds of ms). Scalers are therefore built on a worker thread and kept in
// a small MRU cache (windowed/fullscreen, stereo views, resolution changes). Until the scaler for the current
// configuration is ready, the pass behaves like bilinear_upscale_pass: present never waits for a scaler build.
//
// Only the spatial scaler applies: the temporal scaler and MTL4FXFrameInterpolator need motion vectors, depth and a
// jittered projection, none of which exist for the PS3's final (already composited) image.
//
// Falls back to bilinear when: MetalFX is unsupported, the scaler is not ready yet or could not be created, the
// content is not upscaled on both axes, the input is multisampled / not 2D, or it lacks scaler->colorTextureUsage().
// The caller (MTLPresent.cpp) does not use the pass for stereo 3D.
//
// Input: MetalFX reads the content from the texture origin and needs input texture size <= output size. If the
// content does not start at (0, 0) or the texture is larger than the output, the content region is first copied
// into a content-sized texture.
//
// Synchronization: the scaler encodes its own passes into the MTL4 command buffer (no encoder may be open). It waits
// on m_fence before reading its input and updates it after writing its output. m_fence is updated by a compute
// encoder that opens with the command list's queue barrier (i.e. after everything recorded before), and the encoder
// opened after the scaler waits on it; RCAS dispatches in that encoder, every later encoder opens with a queue barrier.
//
// Texture usage:
//  - input: scaler->colorTextureUsage() (MTL::TextureUsageShaderRead on current systems);
//  - MetalFX target: scaler->outputTextureUsage() | ShaderRead (intermediate when RCAS runs, else the view output);
//  - view output: additionally ShaderWrite (RCAS target). Sampled by video_out_calibration_pass / upscale_blit.

#include "upscaling.h"

#include "util/atomic.hpp"

#include <functional>
#include <memory>
#include <vector>

template <class Context>
class named_thread;

namespace mtl
{
	class metalfx_upscale_pass : public upscaler
	{
		struct scaler_config
		{
			MTL::PixelFormat input_format = MTL::PixelFormatInvalid;
			MTL::PixelFormat output_format = MTL::PixelFormatInvalid;
			u32 input_width = 0;    // Input texture size (the content may be smaller, see setInputContentWidth)
			u32 input_height = 0;
			u32 output_width = 0;
			u32 output_height = 0;

			bool operator==(const scaler_config&) const = default;
		};

		struct scaler_entry
		{
			scaler_config config{};
			MTL4FX::SpatialScaler* scaler = nullptr;   // Owned (+1)
			MTL::TextureUsage color_usage = MTL::TextureUsageUnknown;
			MTL::TextureUsage output_usage = MTL::TextureUsageUnknown;
			u64 last_used = 0;
		};

		// Shared with the builder thread. The worker writes `scaler` and then sets `done`; the RSX thread reads `scaler`
		// only after observing `done`.
		struct build_job
		{
			scaler_config config{};
			MTL4FX::SpatialScaler* scaler = nullptr;   // Owned (+1) result, nullptr on failure
			u64 duration_us = 0;
			atomic_t<bool> done = false;
		};

		static constexpr usz max_cached_scalers = 4;
		static constexpr usz max_failed_configs = 16;
		static constexpr u32 build_request_threshold = 3; // Consecutive requests before a build starts (skips transient sizes while resizing)

		std::vector<scaler_entry> m_scalers;
		std::vector<scaler_config> m_failed_configs;       // Never retried

		std::shared_ptr<build_job> m_pending_job;
		std::unique_ptr<named_thread<std::function<void()>>> m_builder;

		scaler_config m_requested_config{};
		u32 m_requested_count = 0;
		u64 m_use_counter = 0;

		std::unique_ptr<mtl::viewable_image> m_output_left;
		std::unique_ptr<mtl::viewable_image> m_output_right;
		std::unique_ptr<mtl::viewable_image> m_intermediate;  // MetalFX target when RCAS runs
		std::unique_ptr<mtl::viewable_image> m_input_copy;    // Content-sized copy of the input, see above

		MTL::Fence* m_fence = nullptr;   // Owned (+1)

		enum class support_state { unknown, supported, unsupported };
		support_state m_support = support_state::unknown;

		bool m_rcas_unavailable = false;

		// Log-once flags
		bool m_logged_usage = false;
		bool m_logged_input = false;
		bool m_logged_waiting = false;

		bool is_supported();

		// Scaler cache / builder (RSX thread)
		scaler_entry* find_scaler(const scaler_config& config);
		void request_scaler(const scaler_config& config);
		void poll_builder(bool wait);
		void add_scaler(const scaler_config& config, MTL4FX::SpatialScaler* scaler);
		static MTL4FX::SpatialScaler* create_scaler(const scaler_config& config); // Any thread

		static MTL::PixelFormat get_output_format(MTL::PixelFormat input_format);
		static void prepare_image(std::unique_ptr<mtl::viewable_image>& image, MTL::PixelFormat format, u32 width, u32 height,
			MTL::TextureUsage usage, const char* debug_name);
		void dispose_images();

		// Makes the scaler (encoded as its own passes) wait for all previously recorded work. Closes the encoder.
		void signal_fence(mtl::command_list& cmd);
		// Opens a compute encoder whose commands wait for the scaler's output. Leaves it open for RCAS.
		void wait_fence(mtl::command_list& cmd);

		// Returns the upscaled (and sharpened) content of src_area as an output_size image, or nullptr to fall back
		mtl::viewable_image* upscale(mtl::command_list& cmd, mtl::viewable_image* src, const areai& src_area,
			const size2u& input_size, const size2u& output_size, rsx::flags32_t mode);

	public:
		metalfx_upscale_pass() = default;
		~metalfx_upscale_pass() override;

		metalfx_upscale_pass(const metalfx_upscale_pass&) = delete;
		metalfx_upscale_pass& operator=(const metalfx_upscale_pass&) = delete;

		// Collects a finished scaler build and builds the RCAS kernel (once) while no drawable is held
		void prepare() override;

		mtl::viewable_image* scale_output(
			mtl::command_list& cmd,                 // CB
			mtl::viewable_image* src,               // Source input
			MTL::Texture* present_surface,          // Present target. May be nullptr for some passes
			const areai& src_area,                  // Scaling request: source region
			const areai& dst_area,                  // Scaling request: destination region
			rsx::flags32_t mode                     // Mode
		) override;
	};
}
