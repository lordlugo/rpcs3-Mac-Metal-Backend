#pragma once
#include "../Program/GLSLTypes.h"

#include <string_view>

namespace mtl
{
	using namespace ::glsl;

	// Same varying register locations as the VK backend (GLSL layout(location=N) <-> MSL [[user(locnN)]]).
	int get_varying_register_location(std::string_view varying_register_name);

	// "tex12" -> 12, "vtex3" -> 3
	int get_texture_index(std::string_view name);
}
