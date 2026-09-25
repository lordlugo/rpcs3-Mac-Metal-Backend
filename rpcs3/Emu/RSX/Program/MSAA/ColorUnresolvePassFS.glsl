R"(
#version 420
#extension GL_ARB_separate_shader_objects: enable

// Fragment-shader variant of ColorUnresolvePass.glsl. Multisampled images cannot be written from compute on every
// API (Metal), so the multisampled target is rendered with per-sample shading: each sample fetches its texel from
// the sample-expanded (resolved) source image.

#ifdef VULKAN
layout(set=0, binding=0) uniform sampler2D fs0;
layout(push_constant) uniform static_data { ivec2 sample_count; };
#else
layout(binding=31) uniform sampler2D fs0;
uniform ivec2 sample_count;
#endif

layout(location=0) out vec4 out_color;

void main()
{
	ivec2 pixel_coord = ivec2(gl_FragCoord.xy);
	pixel_coord *= sample_count.xy;
	pixel_coord.x += (gl_SampleID % sample_count.x);
	pixel_coord.y += (gl_SampleID / sample_count.x);
	out_color = texelFetch(fs0, pixel_coord, 0);
}

)"
