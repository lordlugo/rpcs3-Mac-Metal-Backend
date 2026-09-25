#pragma once
#include "../Program/FragmentProgramDecompiler.h"
#include "../Program/GLSLTypes.h"
#include "MTLProgramPipeline.h"

// Port of VK/VKFragmentProgram. Emits the same Vulkan-flavoured GLSL 450 (set 1 = fragment resources, identical
// binding numbering and inputs list) which is translated to MSL by the pipe compiler. Metal-specific differences:
//  - 1D textures are declared/sampled as 2D (height 1) textures at t = 0.5 (Metal 1D textures cannot have mipmaps).
//  - No GL_EXT_fragment_shader_barycentric: the renderer must report supports_normalized_barycentrics = true
//    (Apple GPUs interpolate correctly), RSX_SHADER_CONTROL_ATTRIBUTE_INTERPOLATION is ignored if ever set.
//  - Programmable blending subpassInput frag_src_N reads [[color(N)]] via framebuffer fetch (no binding needed).
//  - Logic ops are not emulated: the shared ROP code exposes no logic-op control bits (see report / DESIGN.md).

class MTLFragmentDecompilerThread : public FragmentProgramDecompiler
{
	std::string& m_shader;
	ParamArray& m_parrDummy;
	std::vector<mtl::glsl::program_input> inputs;
	class MTLFragmentProgram* mtl_prog;
	glsl::shader_properties m_shader_props{};

	void prepareBindingTable();

public:
	MTLFragmentDecompilerThread(std::string& shader, ParamArray& parr, const RSXFragmentProgram& prog, u32& size, class MTLFragmentProgram& dst)
		: FragmentProgramDecompiler(prog, size)
		, m_shader(shader)
		, m_parrDummy(parr)
		, mtl_prog(&dst)
	{
	}

	void Task();
	const std::vector<mtl::glsl::program_input>& get_inputs() { return inputs; }

protected:
	std::string getFloatTypeName(usz elementCount) override;
	std::string getHalfTypeName(usz elementCount) override;
	std::string getFunction(FUNCTION f) override;
	std::string compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1) override;

	void insertHeader(std::stringstream& OS) override;
	void insertInputs(std::stringstream& OS) override;
	void insertOutputs(std::stringstream& OS) override;
	void insertConstants(std::stringstream& OS) override;
	void insertGlobalFunctions(std::stringstream& OS) override;
	void insertMainStart(std::stringstream& OS) override;
	void insertMainEnd(std::stringstream& OS) override;
};

/** Storage for a Fragment Program in the process of recompilation.
 *  Decompile() + Compile() run on the thread that first requests the program (RSX thread or shader-cache loader).
 *  The expensive GLSL -> MSL -> MTLLibrary step is deferred to the pipe compiler (shader.compile(), once).
 */
class MTLFragmentProgram
{
public:
	MTLFragmentProgram();
	~MTLFragmentProgram();

	ParamArray parr;
	mtl::glsl::shader* handle = nullptr;   // Set by Compile(); points at `shader` (VK: the VkShaderModule)
	u32 id = 0;
	mtl::glsl::shader shader;
	std::vector<u32> constant_offsets;

	std::array<u32, 4> output_color_masks{ {} };
	std::vector<mtl::glsl::program_input> uniforms;

	struct
	{
		u32 context_buffer_location = umax;           // Rasterizer context
		u32 cbuf_location = umax;                     // Constants register file
		u32 tex_param_location = umax;                // Texture configuration data
		u32 polygon_stipple_params_location = umax;   // Polygon stipple settings
		u32 ftex_location[16];                        // Texture locations array
		u32 ftex_stencil_location[16];                // Texture stencil mirror array
		u32 frag_depth_input_location = umax;         // Fragment depth compare
		u32 frag_src_location[4];                     // Fragment input attachment locations (framebuffer fetch on Metal)

	} binding_table;

	void SetInputs(std::vector<mtl::glsl::program_input>& inputs);
	/**
	 * Decompile a fragment shader located in the PS3's Memory.  This function operates synchronously.
	 * @param prog RSXShaderProgram specifying the location and size of the shader in memory
	 */
	void Decompile(const RSXFragmentProgram& prog);

	/** Prepare the decompiled shader for the pipe compiler (MSL translation happens on first pipeline build). */
	void Compile();

private:
	/** Deletes the shader and any stored information */
	void Delete();
};
