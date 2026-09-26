#pragma once

// Persistent Metal 4 pipeline archive: compiled pipeline binaries survive between sessions of a title, so repeat boots
// (shader cache preload) and effects seen in earlier sessions no longer pay for a full pipeline compile.
//
// How it works (see MTLPipelineArchive.cpp for the details and the reasoning):
//  - Every pipeline state (render and compute) is built by new_render_pipeline_state / new_compute_pipeline_state.
//    Those go through a "capturing" MTL4Compiler that has an MTL4PipelineDataSetSerializer attached, and pass the
//    archives written by earlier sessions as MTL4CompilerTaskOptions::lookupArchives, so the compiler reuses their
//    GPU binaries instead of compiling.
//  - A background thread periodically retires the capturing compiler (a new one takes over for later builds) and
//    writes the retired serializer with serializeAsArchiveAndFlushToURL (temp file + rename). Each serializer is
//    written exactly once, when no build uses it any more.
//  - Files live next to the title's RSX shader cache (removed with it by "Remove shader cache"):
//      <cache>/shaders_cache/metal_pipeline_archive/<fast|precise>/
//    (not inside pipelines/metal/v1.0: rsx::shaders_cache::load() deletes unknown files there and only creates raw/
//    when that directory does not exist yet).
//    identity.txt pins the OS build and GPU; any mismatch discards the archives (they would only miss).
//
// Disabled when the on-disk shader cache is disabled, when the title has no cache directory, or with the environment
// variable RPCS3_METAL_PIPELINE_ARCHIVE=0. Then pipelines are compiled with the device compiler exactly as before.

#include "mtlutils/mtl_api.h"

namespace mtl
{
	// Renderer constructor, after the device exists and before any pipeline is built. Opens the archives of earlier
	// sessions and starts capturing. No-op when disabled.
	void initialize_pipeline_archive();

	// The boot-time shader cache preload has finished (MTLGSRender::on_init_thread). Writes everything built so far
	// in the background; if the preload was complete, that file supersedes the archives of earlier sessions.
	void on_pipeline_cache_preloaded();

	// The RSX thread is exiting (MTLGSRender::on_exit, after the pipe compiler workers are gone). Starts writing what
	// is still pending in the background and returns immediately.
	void flush_pipeline_archive_async();

	// Stops the archive thread, waiting at most `timeout_ms` for a write in progress, and releases every Metal object
	// the archive owns. Called by render_device::destroy() (UI thread). Safe to call when never initialized.
	void shutdown_pipeline_archive(u32 timeout_ms = 3000);

	// Build a pipeline state. Owned (+1) result, or nullptr with `error` set (autoreleased) on failure. Thread-safe.
	// Uses the archive when enabled, otherwise the device compiler without task options.
	// `key` identifies the pipeline across sessions: a hash of everything the descriptor is derived from (0 = unknown).
	// The archive uses it to recognize a preload that recorded exactly the pipelines its newest file already holds.
	MTL::RenderPipelineState* new_render_pipeline_state(const MTL4::RenderPipelineDescriptor* descriptor, NS::Error** error, u64 key = 0);
	MTL::ComputePipelineState* new_compute_pipeline_state(const MTL4::ComputePipelineDescriptor* descriptor, NS::Error** error, u64 key = 0);
}
