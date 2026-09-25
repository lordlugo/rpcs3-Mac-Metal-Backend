R"(
vec4 texelFetch2DMS(in _MSAA_SAMPLER_TYPE_ tex, const in ivec2 clamp_bounds, const in vec2 sample_count, const in ivec2 icoords, const in ivec2 offset)
{
	const vec2 resolve_coords = vec2(clamp(icoords + offset, ivec2(0), clamp_bounds)); // Clamp to edge. Input offset is always zero or positive.
	const vec2 aa_coords = floor(resolve_coords / sample_count);                       // AA coords = real_coords / sample_count
	const vec2 sample_loc = fma(aa_coords, -sample_count, resolve_coords);             // Sample ID = real_coords % sample_count
	const float sample_index = fma(sample_loc.y, sample_count.y, sample_loc.x);

	// TODO: Hack. Filtering will break when sampling sub-pixel sample ids in wrap mode.
	return texelFetch(tex, ivec2(aa_coords), int(sample_index));
}

vec4 sampleTexture2DMS(in _MSAA_SAMPLER_TYPE_ tex, const in vec2 coords, const in sampler_info tex_params)
{
	const uint flags = tex_params.flags;
	const vec2 scaled_coords = _texcoord_xform(coords, tex_params);
	const vec2 normalized_coords = texture2DMSCoord(scaled_coords, flags);
	const vec2 sample_count = vec2(2., textureSamples(tex) * 0.5);
	const ivec2 image_size = ivec2(textureSize(tex) * sample_count);
	const ivec2 clamp_bounds = image_size - ivec2(1);

	// Position in the sample-expanded image, snapped to 1/128 texel like the fixed-point texel addressing of a texture
	// unit. A lookup at a pixel centre lands exactly on the boundary between the two samples of that pixel; without the
	// snap, float rounding (interpolation, fast math) picks either sample from one lookup to the next, so an edge pixel
	// could mix the depth of one surface with the normal or the light of another (outlines along geometry edges in
	// light pre-pass / deferred games). Multiplying by a power of two keeps the snapped value exact.
	const vec2 texel_coords = floor(fma(normalized_coords, vec2(image_size) * 128., vec2(0.5))) * (1. / 128.);
	const ivec2 icoords = ivec2(texel_coords);
	const vec4 sample0 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(0));

	if (_get_bits(flags, FILTERED_MAG_BIT, 2) == 0)
	{
		return sample0;
	}

	// Bilinear scaling, with upto 2x2 downscaling with simple weights
	const vec2 uv_step = 1.0 / vec2(image_size);
	const vec2 actual_step = vec2(dFdx(normalized_coords.x), dFdy(normalized_coords.y));

	const bvec2 no_filter = lessThan(abs(uv_step - actual_step), vec2(0.000001));
	if (no_filter.x && no_filter.y)
	{
		return sample0;
	}

	vec4 a, b;
	float factor;
	const vec4 sample2 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(0, 1));     // Top left

	if (no_filter.x)
	{
		// No scaling, 1:1
		a = sample0;
		b = sample2;
	}
	else
	{
		// Filter required, sample more data
		const vec4 sample1 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(1, 0));     // Bottom right
		const vec4 sample3 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(1, 1));     // Top right

		if (actual_step.x > uv_step.x)
		{
		    // Downscale in X, centered (in texel units, consistent with the snapped texel index)
		    const vec3 weights = compute2x2DownsampleWeights(texel_coords.x, 1.0, actual_step.x * image_size.x);

		    const vec4 sample4 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(2, 0));    // Further bottom right
		    a = fma(sample0, weights.xxxx, sample1 * weights.y) + (sample4 * weights.z);                   // Weighted sum

		    if (!no_filter.y)
		    {
		        const vec4 sample5 = texelFetch2DMS(tex, clamp_bounds, sample_count, icoords, ivec2(2, 1));    // Further top right
		        b = fma(sample2, weights.xxxx, sample3 * weights.y) + (sample5 * weights.z);                   // Weighted sum
		    }
		}
		else if (actual_step.x < uv_step.x)
		{
		    // Upscale in X
		    factor = fract(texel_coords.x);
		    a = mix(sample0, sample1, factor);
		    b = mix(sample2, sample3, factor);
		}
	}

	if (no_filter.y)
	{
		// 1:1 no scale
		return a;
	}
	else if (actual_step.y > uv_step.y)
	{
		// Downscale in Y
		const vec3 weights = compute2x2DownsampleWeights(texel_coords.y, 1.0, actual_step.y * image_size.y);
		// We only have 2 rows computed for performance reasons, so combine rows 1 and 2
		return a * weights.x + b * (weights.y + weights.z);
	}
	else if (actual_step.y < uv_step.y)
	{
		// Upscale in Y
		factor = fract(texel_coords.y);
		return mix(a, b, factor);
	}
}

)"
