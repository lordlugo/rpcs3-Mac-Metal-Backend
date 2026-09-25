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

	// Points of the view in pixels (backing scale applied)
	void get_view_pixel_size(void* ns_view, unsigned& width, unsigned& height);

	// Sets layer.contentsScale to the window's backing scale factor
	void sync_layer_scale(void* ns_view, CA::MetalLayer* layer);
}
