#pragma once

// Lightweight queries usable from the UI (no metal-cpp include needed).

#include <string>
#include <vector>

namespace mtl
{
	// Names of Metal devices usable by the renderer (Apple silicon Macs expose exactly one).
	std::vector<std::string> get_device_names();

	// True when the system GPU supports the Metal 4 feature family (Apple M1 or newer on macOS 26+).
	bool is_metal4_supported();
}
