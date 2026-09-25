# RPCS3 native Metal 4 backend — design & team contract

Target: macOS 26+, Apple silicon (M1 = Apple7 = Metal 4 family) only. Metal is the ONLY renderer (Vulkan/OpenGL are
not built). Bindings: metal-cpp 381.0.0 (`3rdparty/metal-cpp`). Namespace: `mtl` (lowercase; `MTL`/`MTL4`/`NS`/`CA`
are metal-cpp). Files: `rpcs3/Emu/RSX/Metal/`, utilities in `Metal/mtlutils/`. The CMake globs every `*.cpp`/`*.mm`
here recursively — adding a file is enough.

## 1. Porting rule: mirror the VK backend

Every component is a port of its `rpcs3/Emu/RSX/VK/` counterpart. Keep the same structure, class names (in namespace
`mtl`), method names and algorithms, so upstream VK fixes can be carried over. Rename file prefixes `VK` -> `MTL`
(e.g. `VKTextureCache.cpp` -> `MTLTextureCache.cpp`, class `VKGSRender` -> `MTLGSRender`).

### Type substitution table (applies to ALL public signatures)

| Vulkan backend                                   | Metal backend                                                  |
|--------------------------------------------------|----------------------------------------------------------------|
| `const vk::command_buffer&` / `vk::command_buffer&` | `mtl::command_list&` (always non-const)                     |
| `vk::command_buffer_chunk`                        | `mtl::command_buffer_chunk` (MTLGSRenderTypes.hpp, derives command_list) |
| `vk::render_device`, `vk::g_render_device`         | `mtl::render_device`, `mtl::g_render_device`                   |
| `vk::buffer`, `vk::buffer_view`                    | `mtl::buffer`, `mtl::buffer_view` (mtlutils/buffer_object.h)   |
| `vk::image`, `vk::viewable_image`, `vk::image_view`| `mtl::image`, `mtl::viewable_image`, `mtl::image_view` (mtlutils/image.h) |
| `VkFormat`                                         | `MTL::PixelFormat`                                             |
| `VkComponentMapping`                               | `MTL::TextureSwizzleChannels`                                  |
| `VkImageAspectFlags` / `VK_IMAGE_ASPECT_*_BIT`     | `u32` / `mtl::aspect_color`, `aspect_depth`, `aspect_stencil`  |
| `VkBufferImageCopy`                                | `mtl::buffer_image_copy` (MTLHelpers.h)                        |
| `vk::data_heap`                                    | `mtl::data_heap` (mtlutils/data_heap.h)                        |
| `vk::glsl::program`, `program_input`, `shader`     | `mtl::glsl::program`, `program_input`, `shader` (MTLProgramPipeline.h) |
| `vk::pipeline_props`                               | `mtl::pipeline_props` (MTLProgramPipeline.h)                   |
| `vk::sampler`, `border_color_t`                    | `mtl::sampler`, `mtl::border_color_t` (mtlutils/sampler.h)     |
| `vk::fence` / `VkSemaphore` / timeline             | `mtl::fence` / `mtl::timeline` (MTLSharedEvent)                |
| `VkFilter`                                         | `bool linear` (or `MTL::SamplerMinMagFilter`)                  |
| `vk::framebuffer*` / `VkRenderPass` params          | `mtl::image* target` (+ optional depth target); passes build their own `MTL4::RenderPassDescriptor` |
| `vk::get_resource_manager()`, `vk::get_gc()`       | `mtl::get_resource_manager()`, `mtl::get_gc()` (MTLResourceManager.h) |
| `vk::get_event_id()` etc.                          | same names in `mtl::` (MTLResourceManager.h)                   |
| `vk::raise_status_interrupt` etc.                  | same names in `mtl::` (MTLHelpers.h)                           |
| `vk::get_compute_task<T>()`                        | `mtl::get_compute_task<T>()` (MTLCompute.h)                    |
| `vk::begin_renderpass/end_renderpass/is_renderpass_open(cmd)` | `cmd.begin_render_pass(desc)` / `cmd.end_render_pass()` / `cmd.is_render_pass_open()` |
| `vkCmdCopy*`, `vkCmdFillBuffer`                    | `cmd.compute()->copyFromBuffer/copyFromTexture/fillBuffer(...)` |
| image layouts, `change_layout`, `push_layout`       | **Delete.** Metal has no layouts.                              |
| `vk::insert_*_barrier(...)`                        | Usually delete (see §3); inside compute use `cmd.compute()` which serializes |

## 2. Ownership & lifetime

* metal-cpp: `new*`/`alloc`/`copy` return +1 references; release them. Use `mtl::ref<T>` for RAII. Everything else is
  autoreleased: wrap entry points on every thread in `mtl::autorelease_scope` (RSX thread: per draw/flip/task;
  compiler threads: per job).
* **Metal 4 command buffers do NOT retain resources.** Never destroy a buffer/texture/view/pipeline/sampler that
  in-flight work may use: hand it to `mtl::get_gc()->dispose(unique_ptr)` (deferred until the current event id
  completes), exactly as the VK backend does with `vk::get_gc()`.
* Residency: `mtl::buffer` and `mtl::image` register themselves in the device residency set on creation and evict on
  destruction. Raw `MTL::Buffer`/`MTL::Texture` you create yourself must call `g_render_device->make_resident()` /
  `evict()`. Views and texture-buffers created from a registered parent need nothing.

## 3. Synchronization model (Metal 4: all resources untracked)

`mtl::command_list` (mtlutils/commands.h) implements a conservative, always-correct model:
* Every new encoder (render pass or compute) begins with `barrierAfterQueueStages(all work -> this encoder's stages)`.
* Every command recorded via `cmd.compute()` gets an intra-encoder barrier before it (MTL4 compute commands run
  concurrently otherwise). Use `cmd.compute_unordered()` only for a command that is independent of the previous one
  (e.g. the dispatch right after `program::bind`).
* Inside a render pass nothing can wait: Apple GPUs do not support fragment->fragment barriers in a pass. To sample a
  surface that is bound as an attachment either (a) use framebuffer fetch (`[[color(n)]]`, same pixel only) or
  (b) end the pass, which the next encoder's barrier then orders.
* CPU/GPU sync: `mtl::timeline` (MTLSharedEvent). `command_list::submit()` commits and signals the next timeline
  value; `poke()`/`wait()` poll/wait. The renderer's `command_buffer_chunk` calls `mtl::on_event_completed(eid_tag)`.
* Shared (unified memory) buffers written by the CPU before `submit()` are visible to that submission. GPU writes are
  visible to the CPU after the submission's timeline value is signaled.

## 4. Binding model

One `MTL4::ArgumentTable` per stage slot lives in each `command_list` (`argument_table(table_vertex|table_fragment|
table_compute)`). `glsl::program::bind()` writes GPU addresses / resource IDs into them and calls `setArgumentTable`.
`glsl::build_binding_layout()` is the single source of truth for GLSL binding -> Metal index; the shader translator
uses the same layout for SPIRV-Cross `add_msl_resource_binding`. There is no `setBytes` in Metal 4: push constants and
small uniforms go to `mtl::get_scratch_heap()` and are bound by address.

Samplers must be created with `supportArgumentBuffers(true)` (mtl::sampler does). Stage limits: 31 buffers, 64
textures (we size tables to 64), 16 samplers.

## 5. Shader path

RSX ucode -> shared decompilers -> **Vulkan-flavoured GLSL 450** (same generator as VK, bindings set 0 = vertex,
set 1 = fragment) -> `spirv::compile_glsl_to_spv` (glslang, already in tree) -> SPIRV-Cross MSL (MSL 3.2, macOS, 
framebuffer fetch for subpass inputs, native texture buffers, argument-table friendly discrete bindings)
-> `MTL4Compiler::newLibrary` -> `MTL4::RenderPipelineDescriptor`/`ComputePipelineDescriptor` via `MTL4Compiler`.
Static GLSL used by compute kernels / overlays goes through the same path (`glsl::create_compute_program`,
`glsl::create_graphics_program`). Compile on RPCS3's pipe-compiler worker threads (thread QoS is inherited by Metal).
`g_cfg.video.disable_msl_fast_math` selects `MTL::MathModeSafe` vs `Fast`.

SPIRV-Cross throws `spirv_cross::CompilerError`; the translation TU (`MTLShaderCompiler.cpp`) is compiled with
`-fexceptions` so it can catch and log (the rest of RPCS3 is `-fno-exceptions`).

## 6. Metal gaps and the chosen workaround (owner in brackets)

| Gap | Workaround |
|---|---|
| No fragment/render-target barriers inside a pass | Programmable blending via `[[color(n)]]` (SPIRV-Cross maps `subpassInput`); set `backend_config.supports_programmable_blending = true` [S,R]. Feedback loops: same-pixel reads through framebuffer fetch when detectable, otherwise end the pass before sampling (count splits per frame, log in debug overlay) [R,T] |
| No depth framebuffer fetch | Depth read while bound: end the pass and sample the depth texture (VK "emulate_depth_compare" path) [R] |
| No D24S8 on Apple GPUs | Always `Depth32Float_Stencil8`; existing d24 gather/scatter compute kernels for memory transfers [T,C] |
| Depth bounds only on Apple10 | `setDepthTestBounds` when `caps().depth_bounds`, otherwise ignored (logged once) [R] |
| No hardware conditional rendering | Shader predicate path (`emulate_conditional_rendering()` = true) fed by `cs_aggregator` from visibility results [R,C] |
| Last provoking vertex | `supports_last_provoking_vertex = false` (smooth fallback) v1 [R] |
| Primitive restart always on | Common code rewrites restart index; widen u16->u32 if restart disabled and 0xFFFF appears [R] |
| Wide lines | 1px lines v1 (logged once) [R] |
| Only 3 border colors | `border_color_t` nearest enum [T,R] |
| No scaled blit | `mtl::copy_scaled_image` = sampled draw through the overlay blit pass [C,T] |
| MSAA writes from compute | Unresolve as per-sample fragment pass [C] |
| Stencil masks not dynamic | `MTLDepthStencilState` cache keyed on all depth/stencil state [R] |
| Logic ops | Framebuffer-fetch emulation in the fragment shader when a logic op is enabled (else ignored + logged) [S] |
| Y-up NDC | VK already flips via viewport; Metal: negate `gl_Position.y` in the vertex epilogue (SPIRV-Cross `flip_vert_y`) and keep top-left viewport origin [S,R] |
| 1D textures without mips | Decompiler emits 2D (height 1) for 1D samplers [S,T] |
| Texel buffer size | Vertex streams read via `texture_buffer` views; if > max width, bind a window (VK `window()` logic) [R] |
| Clears inside a pass | `attachment_clear_pass` quad, or begin the pass with `loadAction=Clear` for full clears [R,C] |

## 7. Component owners

* **Core** (done): mtlutils/*, MTLHelpers.{h,cpp} (runtime state), MTLResourceManager, MTLImpl.cpp, MTLDeviceQuery.h,
  MTLProgramPipeline.h (contract).
* **S — shaders & pipelines**: MTLProgramPipeline.cpp, MTLShaderCompiler.{h,cpp}, MTLPipelineCompiler.{h,cpp},
  MTLFragmentProgram.{h,cpp}, MTLVertexProgram.{h,cpp}, MTLCommonDecompiler.{h,cpp}, MTLProgramBuffer.h.
* **C — compute, overlays, resolve, upscaling**: MTLCompute.{h,cpp}, MTLOverlays.{h,cpp}, MTLResolveHelper.{h,cpp},
  upscalers/*.
* **T — textures & surfaces**: MTLFormats.{h,cpp}, MTLTexture.cpp, MTLTextureCache.{h,cpp},
  MTLRenderTargets.{h,cpp}, MTLDMA.{h,cpp}.
* **R — renderer core**: MTLGSRender.{h,cpp}, MTLGSRenderTypes.hpp, MTLDraw.cpp, MTLVertexBuffers.cpp,
  MTLPresent.cpp, MTLQueryPool.{h,cpp}, MTLRenderPass.{h,cpp}, MTLCommandStream.{h,cpp}, scratch heap.

## 8. Compile checking

No macOS toolchain in the dev container. `/home/claude/mtl-check.sh <files>` type-checks with clang against the real
metal-cpp headers and stub Apple SDK headers (RPCS3 warning-as-error set included). Every `.cpp` must pass it.
`.mm` files cannot be checked here — keep them tiny.

Include-order pitfalls found on the first real macOS builds:

- `<objc/runtime.h>` (pulled in by metal-cpp) declares a global `Method` typedef, and `gcm_enums.h` ends with a global
  `using namespace gcm;` (which contains `gcm::Method`). CMake force-includes `objc/runtime.h` into every file under
  `RSX/Metal/` so the order of includes doesn't matter. Code outside this folder must not include metal-cpp; UI code
  goes through `MTLDeviceQuery.h`, which declares `mtl::create_render_thread()`.
- Homebrew's `/opt/homebrew/include` is searched last (`-idirafter`, see `buildfiles/cmake/ForkMacOSHomebrew.cmake`)
  so Homebrew copies of bundled libraries (protobuf, libpng, ...) can't replace the bundled headers.
