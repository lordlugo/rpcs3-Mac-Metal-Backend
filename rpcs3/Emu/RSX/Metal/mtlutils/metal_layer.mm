#include "metal_layer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#pragma GCC diagnostic pop

#include <dispatch/dispatch.h>

namespace mtl
{
	static CAMetalLayer* find_or_create_layer(NSView* view)
	{
		if (!view)
		{
			return nil;
		}

		CALayer* layer = [view layer];
		if ([layer isKindOfClass:[CAMetalLayer class]])
		{
			return (CAMetalLayer*)layer;
		}

		// QSurface::MetalSurface with QT_MAC_NO_CONTAINER_LAYER=1 should already provide a CAMetalLayer.
		// Fall back to creating one (must happen on the main thread).
		__block CAMetalLayer* result = nil;
		auto create = ^{
			CAMetalLayer* metal_layer = [CAMetalLayer layer];
			[view setWantsLayer:YES];
			[view setLayer:metal_layer];
			result = metal_layer;
		};

		if ([NSThread isMainThread])
		{
			create();
		}
		else
		{
			dispatch_sync(dispatch_get_main_queue(), create);
		}

		return result;
	}

	CA::MetalLayer* get_metal_layer_from_view(void* ns_view)
	{
		NSView* view = (__bridge NSView*)ns_view;
		CAMetalLayer* layer = find_or_create_layer(view);
		return static_cast<CA::MetalLayer*>((__bridge void*)layer);
	}

	void get_view_pixel_size(void* ns_view, unsigned& width, unsigned& height)
	{
		NSView* view = (__bridge NSView*)ns_view;
		__block NSSize size = NSMakeSize(0, 0);
		__block CGFloat scale = 1.0;

		auto query = ^{
			size = [view bounds].size;
			scale = [view window] ? [[view window] backingScaleFactor] : [[NSScreen mainScreen] backingScaleFactor];
		};

		if ([NSThread isMainThread])
		{
			query();
		}
		else
		{
			dispatch_sync(dispatch_get_main_queue(), query);
		}

		width = static_cast<unsigned>(size.width * scale);
		height = static_cast<unsigned>(size.height * scale);
	}

	void sync_layer_scale(void* ns_view, CA::MetalLayer* ca_layer)
	{
		NSView* view = (__bridge NSView*)ns_view;
		CAMetalLayer* layer = (__bridge CAMetalLayer*)static_cast<void*>(ca_layer);
		if (!view || !layer)
		{
			return;
		}

		auto apply = ^{
			const CGFloat scale = [view window] ? [[view window] backingScaleFactor] : [[NSScreen mainScreen] backingScaleFactor];
			[layer setContentsScale:scale];
		};

		if ([NSThread isMainThread])
		{
			apply();
		}
		else
		{
			dispatch_async(dispatch_get_main_queue(), apply);
		}
	}
}
