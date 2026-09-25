#pragma once
#include "../Program/VertexProgramDecompiler.h"
#include "../Program/program_util.h"
#include "MTLProgramPipeline.h"

// Port of VK/VKVertexProgram. Emits the same Vulkan-flavoured GLSL 450 (set 0 = vertex resources, identical binding
// numbering and inputs list). Metal-specific differences:
//  - Vertex texture 1D fetches are 2D fetches at t = 0.5 (1D textures are 2D textures of height 1 on Metal).
//  - No fp64 in MSL: the z-clip emulation always uses the fallback path.
//  - Clip-space Y is flipped by the MSL translator (SPIRV-Cross flip_vert_y), not by the GLSL.

struct MTLVertexDecompilerThread : public VertexProgramDecompiler
{
	std::string& m_shader;
	std::vector<mtl::glsl::program_input> inputs;
	class MTLVertexProgram* mtl_prog;

	struct
	{
		bool emulate_conditional_rendering{false};
	}
	m_device_props;

protected:
	std::string getFloatTypeName(usz elementCount) override;
	std::string getIntTypeName(usz elementCount) override;
	std::string getFunction(FUNCTION f) override;
	std::string compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1, bool scalar) override;

	void insertHeader(std::stringstream& OS) override;
	void insertInputs(std::stringstream& OS, const std::vector<ParamType>& inputs) override;
	void insertConstants(std::stringstream& OS, const std::vector<ParamType>& constants) override;
	void insertOutputs(std::stringstream& OS, const std::vector<ParamType>& outputs) override;
	void insertMainStart(std::stringstream& OS) override;
	void insertMainEnd(std::stringstream& OS) override;

	void prepareBindingTable();

	const RSXVertexProgram& rsx_vertex_program;
public:
	MTLVertexDecompilerThread(const RSXVertexProgram& prog, std::string& shader, ParamArray&, class MTLVertexProgram& dst)
		: VertexProgramDecompiler(prog)
		, m_shader(shader)
		, mtl_prog(&dst)
		, rsx_vertex_program(prog)
	{
	}

	void Task();
	const std::vector<mtl::glsl::program_input>& get_inputs() { return inputs; }

	void insertFSExport(std::stringstream& OS);
};

class MTLVertexProgram : public rsx::VertexProgramBase
{
public:
	MTLVertexProgram();
	~MTLVertexProgram();

	ParamArray parr;
	mtl::glsl::shader* handle = nullptr;   // Set by Compile(); points at `shader` (VK: the VkShaderModule)
	mtl::glsl::shader shader;
	std::vector<mtl::glsl::program_input> uniforms;

	// Quick attribute indices
	struct
	{
		u32 context_buffer_location = umax;        // Vertex program context
		u32 cr_pred_buffer_location = umax;        // Conditional rendering predicate
		u32 vertex_buffers_location = umax;        // Vertex input streams (3)
		u32 cbuf_location = umax;                  // Vertex program constants register file
		u32 instanced_lut_buffer_location = umax;  // Instancing redirection table
		u32 instanced_cbuf_location = umax;        // Instancing constants register file
		u32 vtex_location[4];                      // Vertex textures (inf)

	} binding_table;

	void Decompile(const RSXVertexProgram& prog);
	/** Prepare the decompiled shader for the pipe compiler (MSL translation happens on first pipeline build). */
	void Compile();
	void SetInputs(std::vector<mtl::glsl::program_input>& inputs);

	u32 Flags() const { return m_ctrl; }

private:
	void Delete();

	u32 m_ctrl = 0u;
};
