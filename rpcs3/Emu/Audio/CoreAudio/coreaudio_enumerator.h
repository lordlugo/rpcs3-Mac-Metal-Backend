#pragma once

#include "Emu/Audio/audio_device_enumerator.h"

class coreaudio_enumerator final : public audio_device_enumerator
{
public:
	coreaudio_enumerator() = default;
	~coreaudio_enumerator() override = default;

	std::vector<audio_device> get_output_devices() override;
};
