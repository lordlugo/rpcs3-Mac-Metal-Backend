#pragma once

// Lightweight entry points usable from the UI. Deliberately free of metal-cpp: the Qt translation units include RSX
// headers (and with them gcm_enums.h's global `using namespace gcm;`), which clashes with <objc/runtime.h>.

#include <string>
#include <vector>

namespace utils
{
	struct serial;
}

namespace mtl
{
	// Creates the Metal renderer as the emulator's rsx::thread (g_fxo), like the VK/GL/Null cases in
	// gui_application / headless_application.
	void create_render_thread(utils::serial* ar);

	// Names of Metal devices usable by the renderer (Apple silicon Macs expose exactly one).
	std::vector<std::string> get_device_names();

	// True when the system GPU supports the Metal 4 feature family (Apple M1 or newer on macOS 26+).
	bool is_metal4_supported();
}
