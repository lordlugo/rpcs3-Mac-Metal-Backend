#pragma once

// Bridges the Qt window (NSView*) to a CAMetalLayer. Implemented in Objective-C++ (metal_layer.mm).

namespace CA
{
	class MetalLayer;
}

namespace mtl
{
	// Returns the CAMetalLayer backing `ns_view` (a QWindow created with QSurface::MetalSurface), creating and
	// attaching one if the view is not layer-backed yet. Must be called on the main thread for creation; reading an
	// existing layer is thread-safe. The returned layer is not retained.
	CA::MetalLayer* get_metal_layer_from_view(void* ns_view);

	// Points of the view in pixels (backing scale applied).
	// NOTE: Blocks on the main thread (dispatch_sync) when called from another thread; never use it on the RSX thread.
	void get_view_pixel_size(void* ns_view, unsigned& width, unsigned& height);

	// Sets layer.contentsScale to the window's backing scale factor
	void sync_layer_scale(void* ns_view, CA::MetalLayer* layer);

	// Properties of the window/screen showing the game, sampled on the main thread
	struct surface_properties
	{
		double min_refresh_interval = 0.;  // NSScreen.minimumRefreshInterval: fastest refresh (1/120 s on ProMotion). 0 if unknown
		double max_refresh_interval = 0.;  // NSScreen.maximumRefreshInterval: slowest refresh (> min on variable refresh screens)
		double update_granularity = 0.;    // NSScreen.displayUpdateGranularity (informational)
		double backing_scale = 0.;         // NSWindow.backingScaleFactor (applied as layer.contentsScale)
		bool fullscreen = false;           // The window is in (native) fullscreen
		unsigned long long serial = 0;     // Incremented by every update; 0 until the first update has run
	};

	// Schedules a main-thread update (dispatch_async; runs inline on the main thread; never blocks) that
	//  - samples the window's screen into the cached surface_properties (read them with get_surface_properties())
	//  - applies the window-dependent layer state from the same values: contentsScale = backingScaleFactor, opaque.
	// Call it when the window may have changed (resize, fullscreen, other screen) and periodically.
	void request_surface_update(void* ns_view, CA::MetalLayer* layer);

	// Latest properties published by request_surface_update (thread-safe, non-blocking)
	surface_properties get_surface_properties();

	// CACurrentMediaTime(): the clock used by MTLDrawable presentedTime / presentAtTime
	double get_media_time();
}
