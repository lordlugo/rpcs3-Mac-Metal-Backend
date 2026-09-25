#pragma once

#ifndef __APPLE__
#error "Core Audio is only available on macOS"
#endif

#include "util/atomic.hpp"
#include "Emu/Audio/AudioBackend.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>

#include <array>
#include <string>
#include <vector>

// Native macOS audio output (AUHAL), with AUSpatialMixer for multichannel content.
//
// Fidelity: the stream is 48 kHz float, exactly what cellAudio mixes. The output device is switched to 48 kHz when it
// supports it (and switched back when RPCS3 stops using it), so stereo reaches the device without resampling or any
// other processing ("Switch Device To 48 kHz"). The RSXAudio provider's port rate is used the same way.
//
// Multichannel (LPCM 5.1/7.1, Dolby Digital and DTS, which the emulated system mixes as 5.1 LPCM):
//  - Headphones (AirPods and other Bluetooth headphones, headphone jack) and the built-in speakers of a Mac:
//    AUSpatialMixer renders the channels as a speaker bed around the listener (binaural for headphones, Apple's
//    speaker virtualization for built-in speakers). Head tracking follows the listener's head with AirPods.
//  - Speakers / receivers: the channels are mapped to the device's speaker layout by channel label (Audio MIDI Setup >
//    Configure Speakers), using Core Audio's standard down/upmix only where the layouts differ.
// The "Spatial Audio" setting overrides the automatic choice.
class CoreAudioBackend final : public AudioBackend
{
public:
	CoreAudioBackend();
	~CoreAudioBackend() override;

	CoreAudioBackend(const CoreAudioBackend&) = delete;
	CoreAudioBackend& operator=(const CoreAudioBackend&) = delete;

	std::string_view GetName() const override { return "CoreAudio"sv; }

	bool Initialized() override { return true; }
	bool Operational() override;
	bool DefaultDeviceChanged() override;

	bool Open(std::string_view dev_id, AudioFreq freq, AudioSampleSize sample_size, AudioChannelCnt ch_cnt, audio_channel_layout layout) override;
	void Close() override;

	f64 GetCallbackFrameLen() override;

	void Play() override;
	void Pause() override;

	enum class output_kind : u32
	{
		headphones,
		builtin_speakers,
		external_speakers,
	};

	struct device_info
	{
		AudioDeviceID id = kAudioObjectUnknown;
		std::string uid;
		std::string name;
		u32 channels = 0;                        // Output channels of the device
		std::vector<AudioChannelLabel> labels;   // Speaker layout (preferred channel layout), may be empty
		output_kind kind = output_kind::external_speakers;
		u32 transport = 0;
	};

	// Shared with the enumerator
	static std::vector<AudioDeviceID> get_output_devices();
	static AudioDeviceID get_default_output_device();
	static bool query_device(AudioDeviceID id, device_info& info);

private:
	static constexpr f64 AUDIO_MIN_LATENCY = 512.0 / 48000;
	static constexpr u32 MAX_FRAMES_PER_SLICE = 8192; // Also covers large device buffers at low device rates

	enum class route_type : u32
	{
		direct,        // Interleaved stream straight into AUHAL (stereo, or channel mapped)
		spatial_bed,   // AUSpatialMixer, channels rendered as a speaker bed (headphones, built-in speakers)
		matrix,        // AUSpatialMixer in bypass mode: label based mapping / standard downmix to the device layout
	};

	AudioUnit m_output_unit = nullptr;
	AudioUnit m_mixer_unit = nullptr;
	route_type m_route = route_type::direct;

	device_info m_device{};
	bool m_use_default_device = true;
	f64 m_stream_rate = 48000.;                   // Rate of the emulated output
	f64 m_device_rate = 48000.;                   // Nominal rate of the device
	u32 m_buffer_frames = 512;
	std::string m_route_description;

	// Device whose nominal sample rate was changed by this backend, the rate set, and the rate to restore
	AudioDeviceID m_rate_device = kAudioObjectUnknown;
	f64 m_rate_set = 0.;
	f64 m_rate_to_restore = 0.;

	u32 m_frame_bytes = 0;                        // Bytes of one interleaved frame from cellAudio
	std::vector<f32> m_scratch;                   // Interleaved staging for the mixer input (render thread only)
	std::array<u8, sizeof(f32) * AUDIO_MAX_CHANNELS> m_last_sample{};

	atomic_t<bool> m_reset_req = false;           // The device is gone or failed: recreate the backend
	atomic_t<bool> m_reconfigure_req = false;     // Output route changed (headphones plugged in, speaker setup changed...)
	std::vector<std::pair<AudioObjectID, AudioObjectPropertyAddress>> m_listeners;

	void read_frames(u8* dst, u32 bytes);
	void set_rate_for_device(const device_info& device);
	void restore_device_rate();
	bool build_route(const device_info& device, audio_channel_layout layout);
	void destroy_units();
	void install_listeners();
	void remove_listeners();

	static OSStatus render_direct(void* ref, AudioUnitRenderActionFlags* flags, const AudioTimeStamp* time, UInt32 bus, UInt32 frames, AudioBufferList* data);
	static OSStatus render_mixer_input(void* ref, AudioUnitRenderActionFlags* flags, const AudioTimeStamp* time, UInt32 bus, UInt32 frames, AudioBufferList* data);
	static OSStatus property_listener(AudioObjectID object, UInt32 count, const AudioObjectPropertyAddress* addresses, void* ref);
};
