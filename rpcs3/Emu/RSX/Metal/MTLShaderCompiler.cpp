// NOTE: This translation unit is built with -fexceptions (see rpcs3/Emu/CMakeLists.txt): SPIRV-Cross reports errors
// by throwing spirv_cross::CompilerError. Every SPIRV-Cross call below is wrapped so nothing escapes into the
// -fno-exceptions remainder of RPCS3.

#include "stdafx.h"
#include "MTLShaderCompiler.h"
#include "mtlutils/device.h"

#include "Emu/RSX/Program/SPIRVCommon.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Utilities/StrUtil.h"
#include "util/fnv_hash.hpp"

#include <algorithm>
#include <exception>
#include <unordered_set>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <SPIRV-Cross/spirv_msl.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace mtl::glsl
{
	atomic_t<bool> g_dump_msl_to_log{ false };

	namespace
	{
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

		spv::ExecutionModel to_execution_model(::glsl::program_domain domain)
		{
			switch (domain)
			{
			case ::glsl::glsl_vertex_program: return spv::ExecutionModelVertex;
			case ::glsl::glsl_fragment_program: return spv::ExecutionModelFragment;
			case ::glsl::glsl_compute_program: return spv::ExecutionModelGLCompute;
			default: fmt::throw_exception("Unexpected program domain %d", static_cast<int>(domain));
			}
		}

		const char* to_string(program_input_type type)
		{
			switch (type)
			{
			case input_type_uniform_buffer: return "uniform buffer";
			case input_type_texel_buffer: return "texel buffer";
			case input_type_texture: return "texture";
			case input_type_storage_buffer: return "storage buffer";
			case input_type_storage_texture: return "storage texture";
			case input_type_push_constant: return "push constant";
			case input_type_attachment: return "input attachment";
			default: return "undefined";
			}
		}

		// What a reflected SPIR-V resource needs from its resource_slot
		enum class metal_resource_class
		{
			buffer,          // UBO / SSBO -> [[buffer(n)]]
			texture,         // Separate image, storage image, texel buffer -> [[texture(n)]]
			sampled_texture, // Combined image sampler -> [[texture(n)]] + [[sampler(m)]]
			sampler          // Separate sampler -> [[sampler(m)]]
		};

		const char* to_string(metal_resource_class cls)
		{
			switch (cls)
			{
			case metal_resource_class::buffer: return "buffer";
			case metal_resource_class::texture: return "image";
			case metal_resource_class::sampled_texture: return "sampled image";
			case metal_resource_class::sampler: return "sampler";
			}
			return "resource";
		}

		// Constant sampler for sampled images left without a sampler slot. Identical to the renderer's stencil-mirror
		// sampler (nearest, clamp to opaque-black border, LOD 0), the only realistic users besides texelFetch-only
		// inputs, whose sampler is never used.
		const spirv_cross::MSLConstexprSampler& get_fallback_sampler()
		{
			static const spirv_cross::MSLConstexprSampler s_sampler = []()
			{
				spirv_cross::MSLConstexprSampler sampler{};
				sampler.coord = spirv_cross::MSL_SAMPLER_COORD_NORMALIZED;
				sampler.min_filter = spirv_cross::MSL_SAMPLER_FILTER_NEAREST;
				sampler.mag_filter = spirv_cross::MSL_SAMPLER_FILTER_NEAREST;
				sampler.mip_filter = spirv_cross::MSL_SAMPLER_MIP_FILTER_NEAREST;
				sampler.s_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER;
				sampler.t_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER;
				sampler.r_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_BORDER;
				sampler.border_color = spirv_cross::MSL_SAMPLER_BORDER_COLOR_OPAQUE_BLACK;
				sampler.lod_clamp_enable = true;
				sampler.lod_clamp_min = 0.f;
				sampler.lod_clamp_max = 0.f;
				return sampler;
			}();
			return s_sampler;
		}

		// Vertex fetch format of a vertex shader input (GLSL `layout(location = N) in <type>`)
		MTL::VertexFormat get_vertex_format(const spirv_cross::SPIRType& type)
		{
			if (type.columns != 1 || !type.array.empty() || type.vecsize < 1 || type.vecsize > 4)
			{
				return MTL::VertexFormatInvalid;
			}

			static constexpr MTL::VertexFormat float_formats[] = { MTL::VertexFormatFloat, MTL::VertexFormatFloat2, MTL::VertexFormatFloat3, MTL::VertexFormatFloat4 };
			static constexpr MTL::VertexFormat half_formats[] = { MTL::VertexFormatHalf, MTL::VertexFormatHalf2, MTL::VertexFormatHalf3, MTL::VertexFormatHalf4 };
			static constexpr MTL::VertexFormat int_formats[] = { MTL::VertexFormatInt, MTL::VertexFormatInt2, MTL::VertexFormatInt3, MTL::VertexFormatInt4 };
			static constexpr MTL::VertexFormat uint_formats[] = { MTL::VertexFormatUInt, MTL::VertexFormatUInt2, MTL::VertexFormatUInt3, MTL::VertexFormatUInt4 };

			switch (type.basetype)
			{
			case spirv_cross::SPIRType::Float: return float_formats[type.vecsize - 1];
			case spirv_cross::SPIRType::Half: return half_formats[type.vecsize - 1];
			case spirv_cross::SPIRType::Int: return int_formats[type.vecsize - 1];
			case spirv_cross::SPIRType::UInt: return uint_formats[type.vecsize - 1];
			default: return MTL::VertexFormatInvalid;
			}
		}

		void log_failure(::glsl::program_domain domain, std::string_view reason, const std::string& glsl_source, const std::string& msl)
		{
			rsx_log.error("[MSL] Failed to translate %s shader: %s", to_string(domain), reason);
			rsx_log.error("[MSL] GLSL source:\n%s", glsl_source);

			if (!msl.empty())
			{
				rsx_log.error("[MSL] Generated MSL:\n%s", msl);
			}
		}

		// Collect Metal slot assignments for every active resource the SPIR-V declares.
		// Returns an empty string on success, otherwise the error message.
		std::string apply_resource_bindings(spirv_cross::CompilerMSL& compiler, spv::ExecutionModel model, const binding_layout& layout)
		{
			const auto active = compiler.get_active_interface_variables();
			const spirv_cross::ShaderResources resources = compiler.get_shader_resources(active);

			// Sets used per binding location, to catch two sets aliasing one location (layout keys are per stage)
			std::unordered_map<u32, u32> binding_sets;

			// Sampled images that got a constexpr sampler because the stage ran out of sampler slots
			std::vector<std::string> constexpr_samplers;

			auto bind = [&](const spirv_cross::Resource& res, metal_resource_class cls) -> std::string
			{
				const u32 set = compiler.get_decoration(res.id, spv::DecorationDescriptorSet);
				const u32 binding = compiler.get_decoration(res.id, spv::DecorationBinding);

				if (const auto [it, inserted] = binding_sets.try_emplace(binding, set); !inserted && it->second != set)
				{
					return fmt::format("'%s' (set=%u, binding=%u) aliases binding %u of set %u; per-stage bindings must be unique",
						res.name, set, binding, binding, it->second);
				}

				const auto found = layout.slots.find(binding);
				if (found == layout.slots.end())
				{
					return fmt::format("'%s' (set=%u, binding=%u) is not declared in the program inputs", res.name, set, binding);
				}

				const resource_slot& slot = found->second;
				const spirv_cross::SPIRType& type = compiler.get_type(res.type_id);

				u32 count = 1;
				for (u32 i = 0; i < type.array.size(); ++i)
				{
					if (!type.array_size_literal[i] || type.array[i] == 0)
					{
						return fmt::format("'%s' (set=%u, binding=%u) is a runtime-sized or specialized descriptor array", res.name, set, binding);
					}
					count *= type.array[i];
				}

				if (count > slot.array_size)
				{
					return fmt::format("'%s' (set=%u, binding=%u) has %u elements but the input declares %u", res.name, set, binding, count, slot.array_size);
				}

				spirv_cross::MSLResourceBinding msl_binding{};
				msl_binding.stage = model;
				msl_binding.desc_set = set;
				msl_binding.binding = binding;
				msl_binding.count = count;

				auto type_matches = [&]()
				{
					switch (cls)
					{
					case metal_resource_class::buffer:
						return slot.type == input_type_uniform_buffer || slot.type == input_type_storage_buffer;
					case metal_resource_class::sampled_texture:
					case metal_resource_class::sampler:
						return slot.type == input_type_texture;
					case metal_resource_class::texture:
						return slot.type == input_type_texture || slot.type == input_type_texel_buffer || slot.type == input_type_storage_texture;
					}
					return false;
				};

				if (!type_matches())
				{
					return fmt::format("'%s' (set=%u, binding=%u) is a %s in the shader but a %s in the program inputs",
						res.name, set, binding, to_string(cls), to_string(slot.type));
				}

				switch (cls)
				{
				case metal_resource_class::buffer:
					if (slot.buffer_index == umax)
					{
						return fmt::format("'%s' (set=%u, binding=%u): no Metal buffer slot left in this stage", res.name, set, binding);
					}
					msl_binding.basetype = spirv_cross::SPIRType::Struct;
					msl_binding.msl_buffer = slot.buffer_index;
					break;
				case metal_resource_class::sampled_texture:
				case metal_resource_class::texture:
					if (slot.texture_index == umax)
					{
						return fmt::format("'%s' (set=%u, binding=%u): no Metal texture slot left in this stage", res.name, set, binding);
					}
					msl_binding.basetype = spirv_cross::SPIRType::SampledImage;
					msl_binding.msl_texture = slot.texture_index;

					if (cls == metal_resource_class::sampled_texture)
					{
						if (slot.sampler_index != umax)
						{
							msl_binding.msl_sampler = slot.sampler_index;
						}
						else if (count == 1)
						{
							// Sampler slots exhausted (layout order puts stencil mirrors and texelFetch-only inputs last):
							// sample with a constant sampler equal to the stencil-mirror sampler instead.
							compiler.remap_constexpr_sampler(res.id, get_fallback_sampler());
							constexpr_samplers.push_back(res.name);
						}
						else
						{
							return fmt::format("'%s' (set=%u, binding=%u): no sampler slots left for a sampler array", res.name, set, binding);
						}
					}
					break;
				case metal_resource_class::sampler:
					if (slot.sampler_index == umax)
					{
						if (count != 1)
						{
							return fmt::format("'%s' (set=%u, binding=%u): no sampler slots left for a sampler array", res.name, set, binding);
						}

						compiler.remap_constexpr_sampler(res.id, get_fallback_sampler());
						constexpr_samplers.push_back(res.name);
						return {};
					}
					msl_binding.basetype = spirv_cross::SPIRType::Sampler;
					msl_binding.msl_sampler = slot.sampler_index;
					break;
				}

				compiler.add_msl_resource_binding(msl_binding);
				return {};
			};

			std::string error;
			auto bind_all = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& list, metal_resource_class cls)
			{
				for (const auto& res : list)
				{
					if (error.empty())
					{
						error = bind(res, cls);
					}
				}
			};

			bind_all(resources.uniform_buffers, metal_resource_class::buffer);
			bind_all(resources.storage_buffers, metal_resource_class::buffer);
			bind_all(resources.storage_images, metal_resource_class::texture);
			bind_all(resources.separate_images, metal_resource_class::texture);
			bind_all(resources.separate_samplers, metal_resource_class::sampler);

			for (const auto& res : resources.sampled_images)
			{
				if (!error.empty())
				{
					break;
				}

				// Texel buffers (samplerBuffer) become native texture_buffer<T>: no sampler is involved
				const bool is_texel_buffer = compiler.get_type(res.type_id).image.dim == spv::DimBuffer;
				error = bind(res, is_texel_buffer ? metal_resource_class::texture : metal_resource_class::sampled_texture);
			}

			if (!error.empty())
			{
				return error;
			}

			if (!resources.acceleration_structures.empty() || !resources.atomic_counters.empty() || !resources.shader_record_buffers.empty())
			{
				return "unsupported resource class (acceleration structure / atomic counter / shader record)";
			}

			if (!constexpr_samplers.empty())
			{
				rsx_log.notice("[MSL] %u sampled image(s) use a constant nearest/clamp-to-border sampler (stage sampler slots exhausted): %s",
					::size32(constexpr_samplers), fmt::merge(constexpr_samplers, ", "));
			}

			// Subpass inputs need no binding: input_attachment_index N reads the pixel's [[color(N)]] (framebuffer fetch)

			if (!resources.push_constant_buffers.empty())
			{
				if (layout.push_constant_buffer_index == umax)
				{
					return layout.push_constant_size
						? fmt::format("push constant block '%s': no Metal buffer slot left in this stage", resources.push_constant_buffers.front().name)
						: fmt::format("push constant block '%s' is used but no push constant input was declared", resources.push_constant_buffers.front().name);
				}

				spirv_cross::MSLResourceBinding msl_binding{};
				msl_binding.stage = model;
				msl_binding.basetype = spirv_cross::SPIRType::Struct;
				msl_binding.desc_set = spirv_cross::kPushConstDescSet;
				msl_binding.binding = spirv_cross::kPushConstBinding;
				msl_binding.count = 1;
				msl_binding.msl_buffer = layout.push_constant_buffer_index;
				compiler.add_msl_resource_binding(msl_binding);
			}

			return {};
		}
	}

	std::string_view get_entry_point_name(::glsl::program_domain domain)
	{
		switch (domain)
		{
		case ::glsl::glsl_vertex_program: return "rpcs3_vs_main";
		case ::glsl::glsl_fragment_program: return "rpcs3_fs_main";
		case ::glsl::glsl_compute_program: return "rpcs3_cs_main";
		default: fmt::throw_exception("Unexpected program domain %d", static_cast<int>(domain));
		}
	}

	bool translate_glsl_to_msl(
		::glsl::program_domain domain,
		const std::string& glsl_source,
		const binding_layout& layout,
		msl_translation_result& result)
	{
		result = {};

		// 1. GLSL -> SPIR-V (glslang, Vulkan rules: same front-end as the VK backend)
		std::vector<u32> spv;
		{
			std::string source = glsl_source; // compile_glsl_to_spv takes a mutable reference
			if (!spirv::compile_glsl_to_spv(spv, source, domain, ::glsl::glsl_rules_vulkan) || spv.empty())
			{
				log_failure(domain, "glslang could not compile the GLSL source", glsl_source, {});
				return false;
			}
		}

		// 2. SPIR-V -> MSL (SPIRV-Cross)
		const spv::ExecutionModel model = to_execution_model(domain);
		const std::string entry_point(get_entry_point_name(domain));
		std::string error;

		try
		{
			spirv_cross::CompilerMSL compiler(std::move(spv));

			spirv_cross::CompilerMSL::Options msl_options = compiler.get_msl_options();
			msl_options.platform = spirv_cross::CompilerMSL::Options::macOS;
			msl_options.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(3, 2);
			msl_options.argument_buffers = false;                 // Discrete bindings: MTL4ArgumentTable slots
			msl_options.enable_decoration_binding = false;        // Every binding comes from the layout
			msl_options.use_framebuffer_fetch_subpasses = true;   // subpassInput(index N) -> [[color(N)]]
			msl_options.texture_buffer_native = true;             // samplerBuffer -> texture_buffer<T>
			msl_options.texture_1D_as_2D = true;                  // 1D images are 2D (height 1) textures on Metal
			msl_options.enable_point_size_builtin = true;
			msl_options.pad_fragment_output_components = true;    // Outputs always cover all attachment channels
			msl_options.enable_clip_distance_user_varying = false; // Nothing downstream reads clip distances
			msl_options.swizzle_texture_samples = false;          // Swizzles live in the texture views
			msl_options.force_native_arrays = false;
			msl_options.multiview = false;
			msl_options.dispatch_base = false;
			msl_options.capture_output_to_buffer = false;
			msl_options.emulate_cube_array = false;
			msl_options.ios_support_base_vertex_instance = false;

			// SSBO .length() support: sizes of the bound buffers live right after the stage's regular buffers
			// (push constants included). program::bind() fills this table from the bound ranges.
			msl_options.buffer_size_buffer_index = std::min<u32>(layout.buffer_count, gpu_capabilities::max_buffers_per_stage - 1);

			compiler.set_msl_options(msl_options);

			spirv_cross::CompilerGLSL::Options common_options = compiler.get_common_options();
			common_options.vertex.flip_vert_y = true;       // Vulkan clip space is Y-down, Metal Y-up (same as MoltenVK)
			common_options.vertex.fixup_clipspace = false;  // Both use a [0, 1] depth range
			compiler.set_common_options(common_options);

			compiler.rename_entry_point("main", entry_point, model);
			compiler.set_entry_point(entry_point, model);

			if (error = apply_resource_bindings(compiler, model, layout); error.empty())
			{
				result.msl = compiler.compile();
				result.entry_point = compiler.get_cleansed_entry_point_name(entry_point, model);
				result.needs_buffer_size_buffer = compiler.needs_buffer_size_buffer();

				if (domain == ::glsl::glsl_compute_program)
				{
					for (u32 i = 0; i < 3; ++i)
					{
						result.workgroup_size[i] = std::max(1u, compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, i));
					}
				}

				if (domain == ::glsl::glsl_vertex_program)
				{
					// Reflect [[stage_in]] attributes once here; pipelines build their vertex descriptor from this list
					const auto active = compiler.get_active_interface_variables();
					for (const auto& input : compiler.get_shader_resources(active).stage_inputs)
					{
						const u32 location = compiler.get_decoration(input.id, spv::DecorationLocation);
						const MTL::VertexFormat format = get_vertex_format(compiler.get_type(input.type_id));

						if (format == MTL::VertexFormatInvalid)
						{
							error = fmt::format("vertex input '%s' (location %u) cannot be fetched as a vertex attribute", input.name, location);
							break;
						}

						result.vertex_attributes.emplace_back(location, format);
					}

					std::sort(result.vertex_attributes.begin(), result.vertex_attributes.end(), FN(x.first < y.first));
				}

				if (!error.empty())
				{
					// Reported below
				}
				else if (compiler.needs_swizzle_buffer() || compiler.needs_view_mask_buffer() || compiler.needs_dispatch_base_buffer() ||
					compiler.needs_output_buffer() || compiler.needs_patch_output_buffer() || compiler.needs_input_threadgroup_mem())
				{
					error = "the generated MSL requires an auxiliary buffer the Metal backend does not provide";
				}
				else if (result.needs_buffer_size_buffer && layout.buffer_count >= gpu_capabilities::max_buffers_per_stage)
				{
					error = fmt::format("SSBO .length() needs a buffer-size table but all %u buffer slots are in use", layout.buffer_count);
				}
			}
		}
		catch (const std::exception& e)
		{
			error = fmt::format("SPIRV-Cross: %s", e.what());
		}

		if (!error.empty())
		{
			log_failure(domain, error, glsl_source, result.msl);
			result = {};
			return false;
		}

		if (g_dump_msl_to_log)
		{
			rsx_log.notice("[MSL] %s shader (%s):\n%s", to_string(domain), result.entry_point, result.msl);
		}

		return true;
	}

	MTL::Library* compile_msl_library(const std::string& msl, std::string_view label, bool fast_math)
	{
		ensure(g_render_device && g_render_device->compiler());
		mtl::autorelease_scope autorelease;

		auto options = mtl::ref(MTL::CompileOptions::alloc()->init());
		options->setLanguageVersion(MTL::LanguageVersion3_2);
		options->setMathMode(fast_math ? MTL::MathModeFast : MTL::MathModeSafe);
		options->setMathFloatingPointFunctions(fast_math ? MTL::MathFloatingPointFunctionsFast : MTL::MathFloatingPointFunctionsPrecise);
		options->setPreserveInvariance(true); // Honors [[invariant]] vertex positions (RSX vertex programs)
		options->setLibraryType(MTL::LibraryTypeExecutable);

		auto descriptor = mtl::ref(MTL4::LibraryDescriptor::alloc()->init());
		descriptor->setSource(mtl::ns_str(msl));
		descriptor->setName(mtl::ns_str(label));
		descriptor->setOptions(options.get());

		NS::Error* error = nullptr;
		MTL::Library* library = g_render_device->compiler()->newLibrary(descriptor.get(), &error);

		if (!library)
		{
			rsx_log.error("[MSL] MTL4Compiler failed to build library '%s': %s", label, mtl::to_string(error));
			rsx_log.error("[MSL] Source:\n%s", msl);
			return nullptr;
		}

		if (error)
		{
			// Warnings only
			rsx_log.warning("[MSL] Library '%s' built with diagnostics: %s", label, mtl::to_string(error));
		}

		return library;
	}

	// ---- shader --------------------------------------------------------------------------------------------------

	shader::~shader()
	{
		destroy();
	}

	void shader::create(::glsl::program_domain domain, const std::string& source)
	{
		std::lock_guard lock(m_compile_lock);

		if (m_library)
		{
			m_library->release();
			m_library = nullptr;
		}

		m_type = domain;
		m_source = source;
		m_msl.clear();
		m_entry_point = std::string(get_entry_point_name(domain));
		m_compile_failed = false;
		m_needs_buffer_sizes = false;
		m_vertex_attributes.clear();
	}

	bool shader::compile(const binding_layout& layout, bool fast_math)
	{
		std::lock_guard lock(m_compile_lock);

		if (m_library)
		{
			// Layouts are a deterministic function of the program inputs, so a shader only ever needs one translation
			return true;
		}

		if (m_compile_failed || m_source.empty())
		{
			return false;
		}

		mtl::autorelease_scope autorelease;

		msl_translation_result translated;
		if (!translate_glsl_to_msl(m_type, m_source, layout, translated))
		{
			m_compile_failed = true;
			return false;
		}

		// Stable (FNV-1a) source hash: names the library and the shaderlog dump
		usz source_hash = rpcs3::fnv_seed;
		for (const char c : m_source)
		{
			source_hash = rpcs3::hash64(source_hash, static_cast<u8>(c));
		}

		const std::string label = fmt::format("%s_%016llx", translated.entry_point, source_hash);

		if (g_cfg.video.log_programs)
		{
			fs::write_file(fs::get_cache_dir() + "shaderlog/" + label + ".msl", fs::rewrite, translated.msl);
		}

		m_library = compile_msl_library(translated.msl, label, fast_math);
		if (!m_library)
		{
			rsx_log.error("[MSL] GLSL source of '%s':\n%s", label, m_source);
			m_compile_failed = true;
			return false;
		}

		m_msl = std::move(translated.msl);
		m_entry_point = std::move(translated.entry_point);
		m_needs_buffer_sizes = translated.needs_buffer_size_buffer;
		m_vertex_attributes = std::move(translated.vertex_attributes);
		return true;
	}

	void shader::destroy()
	{
		std::lock_guard lock(m_compile_lock);

		if (m_library)
		{
			m_library->release();
			m_library = nullptr;
		}

		m_source.clear();
		m_msl.clear();
		m_compile_failed = false;
		m_needs_buffer_sizes = false;
		m_vertex_attributes.clear();
	}
}
