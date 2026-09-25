#include "stdafx.h"
#include "Emu/Audio/CoreAudio/coreaudio_enumerator.h"
#include "Emu/Audio/CoreAudio/CoreAudioBackend.h"

#include <algorithm>

std::vector<audio_device_enumerator::audio_device> coreaudio_enumerator::get_output_devices()
{
	std::vector<audio_device> result;

	for (AudioDeviceID id : CoreAudioBackend::get_output_devices())
	{
		CoreAudioBackend::device_info info{};

		if (!CoreAudioBackend::query_device(id, info))
		{
			continue;
		}

		audio_device dev{};
		dev.id = info.uid; // Stable across reboots and reconnects (AudioDeviceIDs are not)
		dev.name = info.name.empty() ? info.uid : info.name;
		dev.max_ch = info.channels;
		result.emplace_back(std::move(dev));
	}

	std::sort(result.begin(), result.end(), [](const audio_device& a, const audio_device& b)
	{
		return a.name < b.name;
	});

	return result;
}
