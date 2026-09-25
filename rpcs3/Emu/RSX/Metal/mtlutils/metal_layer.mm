#include "metal_layer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#import <QuartzCore/CAMetalLayer.h>
#pragma GCC diagnostic pop

#include <dispatch/dispatch.h>

#include <mutex>

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

	// Written by the main thread (request_surface_update), read by the RSX thread. Static storage: a block that runs
	// after the renderer is gone still writes valid memory.
	static std::mutex s_surface_mutex;
	static surface_properties s_surface_properties;

	void request_surface_update(void* ns_view, CA::MetalLayer* ca_layer)
	{
		NSView* view = (__bridge NSView*)ns_view;
		CAMetalLayer* layer = (__bridge CAMetalLayer*)static_cast<void*>(ca_layer);
		if (!view)
		{
			return;
		}

		// MRC: the block retains view and layer while it is queued
		auto update = ^{
			NSWindow* window = [view window];
			NSScreen* screen = window ? [window screen] : nil;
			if (!screen)
			{
				screen = [NSScreen mainScreen];
			}

			surface_properties props{};
			props.backing_scale = window ? [window backingScaleFactor] : (screen ? [screen backingScaleFactor] : 1.0);
			props.fullscreen = window && (([window styleMask] & NSWindowStyleMaskFullScreen) != 0);

			if (screen)
			{
				props.min_refresh_interval = [screen minimumRefreshInterval];
				props.max_refresh_interval = [screen maximumRefreshInterval];
				props.update_granularity = [screen displayUpdateGranularity];
			}

			if (layer)
			{
				// Same source as the drawable size (QWindow size * devicePixelRatio == bounds * backingScaleFactor)
				if (props.backing_scale > 0. && [layer contentsScale] != props.backing_scale)
				{
					[layer setContentsScale:props.backing_scale];
				}

				// Every drawable pixel is written, but the alpha channel of guest images is meaningless (X8R8G8B8).
				// An opaque layer makes the compositor ignore it (no see-through/flicker), skips blending and is
				// required for direct-to-display in fullscreen.
				if (![layer isOpaque])
				{
					[layer setOpaque:YES];
				}

				// The emulated video output is sRGB (Rec.709 primaries). Tag the layer so macOS color-matches it to the
				// display like any other sRGB content: an untagged layer is shown in the display's own gamut, which on
				// P3/XDR screens oversaturates every color and shifts contrast compared with the rest of the desktop.
				static CGColorSpaceRef s_srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
				if (s_srgb && (![layer colorspace] || !CFEqual([layer colorspace], s_srgb)))
				{
					[layer setColorspace:s_srgb];
				}

				// SDR output: no extended range (EDR would let values above 1.0 through and change tone mapping)
				if ([layer wantsExtendedDynamicRangeContent])
				{
					[layer setWantsExtendedDynamicRangeContent:NO];
				}
			}

			std::lock_guard lock(s_surface_mutex);
			props.serial = s_surface_properties.serial + 1;
			s_surface_properties = props;
		};

		if ([NSThread isMainThread])
		{
			update();
		}
		else
		{
			dispatch_async(dispatch_get_main_queue(), update);
		}
	}

	surface_properties get_surface_properties()
	{
		std::lock_guard lock(s_surface_mutex);
		return s_surface_properties;
	}

	double get_media_time()
	{
		return CACurrentMediaTime();
	}
}
