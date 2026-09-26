#include "stdafx.h"
#include "sampler.h"
#include "Emu/RSX/Utils/color_utils.hpp"

namespace mtl
{
	static MTL::SamplerBorderColor get_closest_border_color(const color4f& color4)
	{
		if ((color4.r + color4.g + color4.b) > 1.35f)
		{
			// If color elements are brighter than roughly 0.5 average, use white border
			return MTL::SamplerBorderColorOpaqueWhite;
		}

		if (color4.a > 0.5f)
		{
			return MTL::SamplerBorderColorOpaqueBlack;
		}

		return MTL::SamplerBorderColorTransparentBlack;
	}

	border_color_t::border_color_t(const color4f& color)
		: color_value(color)
	{
		value = get_closest_border_color(color);

		switch (value)
		{
		case MTL::SamplerBorderColorOpaqueWhite:
			exact = (color.r == 1.f && color.g == 1.f && color.b == 1.f && color.a == 1.f);
			break;
		case MTL::SamplerBorderColorOpaqueBlack:
			exact = (color.r == 0.f && color.g == 0.f && color.b == 0.f && color.a == 1.f);
			break;
		default:
			exact = (color.r == 0.f && color.g == 0.f && color.b == 0.f && color.a == 0.f);
			break;
		}
	}

	border_color_t::border_color_t(MTL::SamplerBorderColor color)
		: value(color)
	{
		switch (color)
		{
		case MTL::SamplerBorderColorOpaqueWhite: color_value = color4f(1.f, 1.f, 1.f, 1.f); break;
		case MTL::SamplerBorderColorOpaqueBlack: color_value = color4f(0.f, 0.f, 0.f, 1.f); break;
		default: color_value = color4f(0.f, 0.f, 0.f, 0.f); break;
		}
	}

	u64 sampler_create_info::hash() const
	{
		u64 h = 0;
		auto mix = [&h](u64 v)
		{
			h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		};

		mix(static_cast<u64>(clamp_u) | (static_cast<u64>(clamp_v) << 8) | (static_cast<u64>(clamp_w) << 16) |
			(static_cast<u64>(unnormalized_coordinates) << 24) | (static_cast<u64>(min_filter) << 25) |
			(static_cast<u64>(mag_filter) << 27) | (static_cast<u64>(mip_filter) << 29) |
			(static_cast<u64>(border_color.value) << 32) | (static_cast<u64>(depth_compare) << 36) |
			(static_cast<u64>(compare_function) << 40));
		mix(std::bit_cast<u32>(mip_lod_bias));
		mix(std::bit_cast<u32>(max_anisotropy));
		mix(std::bit_cast<u32>(min_lod));
		mix(std::bit_cast<u32>(max_lod));
		return h;
	}

	bool sampler_create_info::operator == (const sampler_create_info& that) const
	{
		return clamp_u == that.clamp_u && clamp_v == that.clamp_v && clamp_w == that.clamp_w &&
			unnormalized_coordinates == that.unnormalized_coordinates &&
			mip_lod_bias == that.mip_lod_bias && max_anisotropy == that.max_anisotropy &&
			min_lod == that.min_lod && max_lod == that.max_lod &&
			min_filter == that.min_filter && mag_filter == that.mag_filter && mip_filter == that.mip_filter &&
			border_color == that.border_color && depth_compare == that.depth_compare &&
			compare_function == that.compare_function;
	}

	sampler::sampler(const render_device& dev, const sampler_create_info& create_info)
		: info(create_info)
	{
		autorelease_scope pool;
		auto desc = ref(MTL::SamplerDescriptor::alloc()->init());

		if (info.unnormalized_coordinates)
		{
			// Metal restrictions for pixel coordinates: clamp-to-edge/zero addressing, no mipmapping, no anisotropy
			desc->setNormalizedCoordinates(false);
			auto fix_mode = [](MTL::SamplerAddressMode mode)
			{
				return (mode == MTL::SamplerAddressModeClampToZero) ? mode : MTL::SamplerAddressModeClampToEdge;
			};
			desc->setSAddressMode(fix_mode(info.clamp_u));
			desc->setTAddressMode(fix_mode(info.clamp_v));
			desc->setRAddressMode(fix_mode(info.clamp_w));
			desc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
			desc->setMaxAnisotropy(1);
			desc->setLodMinClamp(0.f);
			desc->setLodMaxClamp(0.f);
		}
		else
		{
			desc->setNormalizedCoordinates(true);
			desc->setSAddressMode(info.clamp_u);
			desc->setTAddressMode(info.clamp_v);
			desc->setRAddressMode(info.clamp_w);
			desc->setMipFilter(info.mip_filter);
			desc->setMaxAnisotropy(static_cast<NS::UInteger>(std::clamp(info.max_anisotropy, 1.f, 16.f)));
			desc->setLodMinClamp(info.min_lod);
			desc->setLodMaxClamp(std::max(info.min_lod, info.max_lod));

			if (info.mip_lod_bias != 0.f && dev.caps().apple10)
			{
				desc->setLodBias(info.mip_lod_bias);
			}
		}

		desc->setMinFilter(info.min_filter);
		desc->setMagFilter(info.mag_filter);
		desc->setBorderColor(info.border_color.value);

		if (info.depth_compare)
		{
			desc->setCompareFunction(info.compare_function);
		}

		// Required to bind through MTL4ArgumentTable::setSamplerState(gpuResourceID)
		desc->setSupportArgumentBuffers(true);

		value = dev.handle()->newSamplerState(desc.get());
		ensure(value, "Metal: failed to create sampler state");
		m_resource_id = value->gpuResourceID();
	}

	sampler::~sampler()
	{
		if (value)
		{
			value->release();
			value = nullptr;
		}
	}

	void sampler_pool_t::clear()
	{
		m_pool.clear();
	}

	cached_sampler_object_t* sampler_pool_t::find(const sampler_create_info& info) const
	{
		const auto found = m_pool.find(info.hash());
		if (found == m_pool.end())
		{
			return nullptr;
		}

		for (const auto& obj : found->second)
		{
			if (obj->info == info)
			{
				return obj.get();
			}
		}

		return nullptr;
	}

	cached_sampler_object_t* sampler_pool_t::emplace(std::unique_ptr<cached_sampler_object_t>& object)
	{
		object->key = object->info.hash();
		auto result = object.get();
		m_pool[object->key].emplace_back(std::move(object));
		return result;
	}

	std::vector<std::unique_ptr<cached_sampler_object_t>> sampler_pool_t::collect(std::function<bool(const cached_sampler_object_t&)> predicate)
	{
		std::vector<std::unique_ptr<cached_sampler_object_t>> result;

		for (auto it = m_pool.begin(); it != m_pool.end();)
		{
			auto& bucket = it->second;
			for (auto obj = bucket.begin(); obj != bucket.end();)
			{
				if (predicate(**obj))
				{
					result.emplace_back(std::move(*obj));
					obj = bucket.erase(obj);
				}
				else
				{
					++obj;
				}
			}

			it = bucket.empty() ? m_pool.erase(it) : std::next(it);
		}

		return result;
	}

	usz sampler_pool_t::size() const
	{
		usz count = 0;
		for (const auto& [key, bucket] : m_pool)
		{
			count += bucket.size();
		}
		return count;
	}

	MTL::SamplerAddressMode get_sampler_address_mode(rsx::texture_wrap_mode wrap, bool* uses_border)
	{
		if (uses_border)
		{
			*uses_border = false;
		}

		switch (wrap)
		{
		case rsx::texture_wrap_mode::wrap: return MTL::SamplerAddressModeRepeat;
		case rsx::texture_wrap_mode::mirror: return MTL::SamplerAddressModeMirrorRepeat;
		case rsx::texture_wrap_mode::clamp_to_edge: return MTL::SamplerAddressModeClampToEdge;
		case rsx::texture_wrap_mode::border:
			if (uses_border) *uses_border = true;
			return MTL::SamplerAddressModeClampToBorderColor;
		case rsx::texture_wrap_mode::clamp: return MTL::SamplerAddressModeClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_clamp_to_edge: return MTL::SamplerAddressModeMirrorClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_border: return MTL::SamplerAddressModeMirrorClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_clamp: return MTL::SamplerAddressModeMirrorClampToEdge;
		default:
			fmt::throw_exception("Unhandled texture clamp mode");
		}
	}

	MTL::CompareFunction get_compare_function(rsx::comparison_function op, bool reverse_direction)
	{
		switch (op)
		{
		case rsx::comparison_function::never: return MTL::CompareFunctionNever;
		case rsx::comparison_function::greater: return reverse_direction ? MTL::CompareFunctionLess : MTL::CompareFunctionGreater;
		case rsx::comparison_function::less: return reverse_direction ? MTL::CompareFunctionGreater : MTL::CompareFunctionLess;
		case rsx::comparison_function::less_or_equal: return reverse_direction ? MTL::CompareFunctionGreaterEqual : MTL::CompareFunctionLessEqual;
		case rsx::comparison_function::greater_or_equal: return reverse_direction ? MTL::CompareFunctionLessEqual : MTL::CompareFunctionGreaterEqual;
		case rsx::comparison_function::equal: return MTL::CompareFunctionEqual;
		case rsx::comparison_function::not_equal: return MTL::CompareFunctionNotEqual;
		case rsx::comparison_function::always: return MTL::CompareFunctionAlways;
		default:
			fmt::throw_exception("Unknown compare op: 0x%x", static_cast<u32>(op));
		}
	}
}
