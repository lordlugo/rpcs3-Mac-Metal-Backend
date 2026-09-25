#include "stdafx.h"
#include "Emu/Audio/CoreAudio/CoreAudioBackend.h"
#include "Emu/Audio/audio_device_enumerator.h"
#include "Emu/system_config.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>

LOG_CHANNEL(coreaudio_log, "CoreAudio");

namespace
{
	constexpr UInt32 fourcc(const char (&code)[5])
	{
		return (static_cast<UInt32>(static_cast<u8>(code[0])) << 24) | (static_cast<UInt32>(static_cast<u8>(code[1])) << 16) |
			(static_cast<UInt32>(static_cast<u8>(code[2])) << 8) | static_cast<UInt32>(static_cast<u8>(code[3]));
	}

	constexpr AudioObjectPropertyAddress prop(AudioObjectPropertySelector selector, AudioObjectPropertyScope scope = kAudioObjectPropertyScopeGlobal)
	{
		return { selector, scope, kAudioObjectPropertyElementMain };
	}

	template <typename T>
	bool get_property(AudioObjectID object, const AudioObjectPropertyAddress& address, T& value)
	{
		UInt32 size = sizeof(T);
		return AudioObjectHasProperty(object, &address) && AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, &value) == noErr && size == sizeof(T);
	}

	template <typename T>
	std::vector<T> get_property_array(AudioObjectID object, const AudioObjectPropertyAddress& address)
	{
		UInt32 size = 0;
		if (!AudioObjectHasProperty(object, &address) || AudioObjectGetPropertyDataSize(object, &address, 0, nullptr, &size) != noErr || size < sizeof(T))
		{
			return {};
		}

		std::vector<T> result(size / sizeof(T));
		size = static_cast<UInt32>(result.size() * sizeof(T));

		if (AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, result.data()) != noErr)
		{
			return {};
		}

		result.resize(size / sizeof(T));
		return result;
	}

	std::string cfstring_to_string(CFStringRef str)
	{
		if (!str)
		{
			return {};
		}

		if (const char* ptr = CFStringGetCStringPtr(str, kCFStringEncodingUTF8))
		{
			return ptr;
		}

		const CFIndex max_size = CFStringGetMaximumSizeForEncoding(CFStringGetLength(str), kCFStringEncodingUTF8) + 1;
		std::string result(static_cast<usz>(max_size), '\0');

		if (!CFStringGetCString(str, result.data(), max_size, kCFStringEncodingUTF8))
		{
			return {};
		}

		result.resize(std::strlen(result.c_str()));
		return result;
	}

	// For properties that return a CFStringRef owned by the caller (device UID, object name)
	std::string get_string_property(AudioObjectID object, AudioObjectPropertySelector selector)
	{
		CFStringRef str = nullptr;
		if (!get_property(object, prop(selector), str) || !str)
		{
			return {};
		}

		std::string result = cfstring_to_string(str);
		CFRelease(str);
		return result;
	}

	u32 get_output_channel_count(AudioDeviceID device)
	{
		const AudioObjectPropertyAddress address = prop(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput);

		UInt32 size = 0;
		if (!AudioObjectHasProperty(device, &address) || AudioObjectGetPropertyDataSize(device, &address, 0, nullptr, &size) != noErr ||
			size < offsetof(AudioBufferList, mBuffers))
		{
			return 0;
		}

		std::vector<u64> storage((size + sizeof(u64) - 1) / sizeof(u64));
		auto* list = reinterpret_cast<AudioBufferList*>(storage.data());

		if (AudioObjectGetPropertyData(device, &address, 0, nullptr, &size, list) != noErr)
		{
			return 0;
		}

		u32 channels = 0;
		const AudioBuffer* buffers = list->mBuffers;

		for (UInt32 i = 0; i < list->mNumberBuffers; i++)
		{
			channels += buffers[i].mNumberChannels;
		}

		return channels;
	}

	std::vector<AudioChannelLabel> expand_layout(const AudioChannelLayout& layout)
	{
		std::vector<AudioChannelLabel> labels;

		if (layout.mChannelLayoutTag == kAudioChannelLayoutTag_UseChannelDescriptions)
		{
			const AudioChannelDescription* descriptions = layout.mChannelDescriptions;

			for (UInt32 i = 0; i < layout.mNumberChannelDescriptions; i++)
			{
				labels.push_back(descriptions[i].mChannelLabel);
			}

			return labels;
		}

		// Tag or bitmap: let Core Audio expand it into channel descriptions
		AudioFormatPropertyID property = kAudioFormatProperty_ChannelLayoutForTag;
		const void* specifier = &layout.mChannelLayoutTag;
		UInt32 specifier_size = sizeof(layout.mChannelLayoutTag);

		if (layout.mChannelLayoutTag == kAudioChannelLayoutTag_UseChannelBitmap)
		{
			property = kAudioFormatProperty_ChannelLayoutForBitmap;
			specifier = &layout.mChannelBitmap;
			specifier_size = sizeof(layout.mChannelBitmap);
		}

		UInt32 size = 0;
		if (AudioFormatGetPropertyInfo(property, specifier_size, specifier, &size) != noErr || size < offsetof(AudioChannelLayout, mChannelDescriptions))
		{
			return labels;
		}

		std::vector<u64> storage((size + sizeof(u64) - 1) / sizeof(u64));
		auto* full = reinterpret_cast<AudioChannelLayout*>(storage.data());

		if (AudioFormatGetProperty(property, specifier_size, specifier, &size, full) != noErr || full->mChannelLayoutTag != kAudioChannelLayoutTag_UseChannelDescriptions)
		{
			return labels;
		}

		const AudioChannelDescription* descriptions = full->mChannelDescriptions;

		for (UInt32 i = 0; i < full->mNumberChannelDescriptions; i++)
		{
			labels.push_back(descriptions[i].mChannelLabel);
		}

		return labels;
	}

	std::vector<AudioChannelLabel> get_device_labels(AudioDeviceID device)
	{
		const AudioObjectPropertyAddress address = prop(kAudioDevicePropertyPreferredChannelLayout, kAudioObjectPropertyScopeOutput);

		UInt32 size = 0;
		if (!AudioObjectHasProperty(device, &address) || AudioObjectGetPropertyDataSize(device, &address, 0, nullptr, &size) != noErr ||
			size < offsetof(AudioChannelLayout, mChannelDescriptions))
		{
			return {};
		}

		std::vector<u64> storage((size + sizeof(u64) - 1) / sizeof(u64));
		auto* layout = reinterpret_cast<AudioChannelLayout*>(storage.data());

		if (AudioObjectGetPropertyData(device, &address, 0, nullptr, &size, layout) != noErr)
		{
			return {};
		}

		return expand_layout(*layout);
	}

	// Channel order of the emulated audio output (same as the other backends and the downmixer in AudioBackend.h):
	// 5.1: L R C LFE Ls Rs, 7.1: L R C LFE (rear) Lb Rb (side) Ls Rs
	std::vector<AudioChannelLabel> ps3_labels(u32 channels)
	{
		switch (channels)
		{
		case 8:
			return { kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
				kAudioChannelLabel_RearSurroundLeft, kAudioChannelLabel_RearSurroundRight, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround };
		case 6:
			return { kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_Center, kAudioChannelLabel_LFEScreen,
				kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround };
		default:
			return { kAudioChannelLabel_Left, kAudioChannelLabel_Right };
		}
	}

	// "Audio Channel Layout" setting (output layout forced by the user)
	std::vector<AudioChannelLabel> user_layout_labels(audio_channel_layout layout)
	{
		switch (layout)
		{
		case audio_channel_layout::automatic: break;
		case audio_channel_layout::mono: return { kAudioChannelLabel_Mono };
		case audio_channel_layout::stereo: return { kAudioChannelLabel_Left, kAudioChannelLabel_Right };
		case audio_channel_layout::stereo_lfe: return { kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LFEScreen };
		case audio_channel_layout::quadraphonic: return { kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround };
		case audio_channel_layout::quadraphonic_lfe: return { kAudioChannelLabel_Left, kAudioChannelLabel_Right, kAudioChannelLabel_LeftSurround, kAudioChannelLabel_RightSurround, kAudioChannelLabel_LFEScreen };
		case audio_channel_layout::surround_5_1: return ps3_labels(6);
		case audio_channel_layout::surround_7_1: return ps3_labels(8);
		}

		return {};
	}

	bool is_speaker_label(AudioChannelLabel label)
	{
		return label != kAudioChannelLabel_Unknown && label != kAudioChannelLabel_Unused && label != kAudioChannelLabel_UseCoordinates &&
			label < kAudioChannelLabel_Discrete_0;
	}

	bool has_label(const std::vector<AudioChannelLabel>& labels, AudioChannelLabel label)
	{
		return std::find(labels.begin(), labels.end(), label) != labels.end();
	}

	// Channel layout storage (variable length struct)
	std::vector<u64> make_layout(const std::vector<AudioChannelLabel>& labels)
	{
		const usz bytes = offsetof(AudioChannelLayout, mChannelDescriptions) + std::max<usz>(labels.size(), 1) * sizeof(AudioChannelDescription);
		std::vector<u64> storage((bytes + sizeof(u64) - 1) / sizeof(u64));

		auto* layout = reinterpret_cast<AudioChannelLayout*>(storage.data());
		layout->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
		layout->mChannelBitmap = static_cast<AudioChannelBitmap>(0);
		layout->mNumberChannelDescriptions = static_cast<UInt32>(labels.size());

		AudioChannelDescription* descriptions = layout->mChannelDescriptions;

		for (usz i = 0; i < labels.size(); i++)
		{
			descriptions[i].mChannelLabel = labels[i];
			descriptions[i].mChannelFlags = kAudioChannelFlags_AllOff;
			descriptions[i].mCoordinates[0] = 0.f;
			descriptions[i].mCoordinates[1] = 0.f;
			descriptions[i].mCoordinates[2] = 0.f;
		}

		return storage;
	}

	AudioStreamBasicDescription make_format(f64 rate, u32 channels, bool interleaved, bool int16)
	{
		const u32 sample_bytes = int16 ? 2 : 4;

		AudioStreamBasicDescription format{};
		format.mSampleRate = rate;
		format.mFormatID = kAudioFormatLinearPCM;
		format.mFormatFlags = (int16 ? kAudioFormatFlagIsSignedInteger : kAudioFormatFlagIsFloat) | kAudioFormatFlagIsPacked;

		if (!interleaved)
		{
			format.mFormatFlags |= kAudioFormatFlagIsNonInterleaved;
		}

		format.mBytesPerFrame = interleaved ? sample_bytes * channels : sample_bytes;
		format.mFramesPerPacket = 1;
		format.mBytesPerPacket = format.mBytesPerFrame;
		format.mChannelsPerFrame = channels;
		format.mBitsPerChannel = sample_bytes * 8;
		return format;
	}

	// AUHAL channel map: one entry per device channel, the source channel routed there (-1 = silent)
	std::vector<SInt32> make_channel_map(const std::vector<AudioChannelLabel>& source, const CoreAudioBackend::device_info& device)
	{
		std::vector<SInt32> map(device.channels, -1);
		std::vector<bool> placed(source.size(), false);

		// 1. Same speaker label
		for (usz dst = 0; dst < map.size() && dst < device.labels.size(); dst++)
		{
			for (usz src = 0; src < source.size(); src++)
			{
				if (!placed[src] && device.labels[dst] == source[src])
				{
					map[dst] = static_cast<SInt32>(src);
					placed[src] = true;
					break;
				}
			}
		}

		// 2. Mono on a device with left/right speakers: both
		for (usz src = 0; src < source.size(); src++)
		{
			if (placed[src] || source[src] != kAudioChannelLabel_Mono)
			{
				continue;
			}

			for (usz dst = 0; dst < map.size() && dst < device.labels.size(); dst++)
			{
				if (map[dst] == -1 && (device.labels[dst] == kAudioChannelLabel_Left || device.labels[dst] == kAudioChannelLabel_Right))
				{
					map[dst] = static_cast<SInt32>(src);
					placed[src] = true;
				}
			}
		}

		// 3. Everything else, in order, on the free device channels without a speaker label (unlabeled devices,
		//    partially labeled layouts): a channel is never dropped while the device has room for it
		usz next = 0;
		for (usz src = 0; src < source.size(); src++)
		{
			if (placed[src])
			{
				continue;
			}

			while (next < map.size() && (map[next] != -1 || (next < device.labels.size() && is_speaker_label(device.labels[next]))))
			{
				next++;
			}

			if (next >= map.size())
			{
				break;
			}

			map[next] = static_cast<SInt32>(src);
			placed[src] = true;
		}

		return map;
	}

	bool is_identity_map(const std::vector<SInt32>& map, u32 source_channels)
	{
		if (map.size() != source_channels)
		{
			return false;
		}

		for (usz i = 0; i < map.size(); i++)
		{
			if (map[i] != static_cast<SInt32>(i))
			{
				return false;
			}
		}

		return true;
	}

	std::string_view kind_to_string(CoreAudioBackend::output_kind kind)
	{
		switch (kind)
		{
		case CoreAudioBackend::output_kind::headphones: return "headphones";
		case CoreAudioBackend::output_kind::builtin_speakers: return "built-in speakers";
		case CoreAudioBackend::output_kind::external_speakers: return "speakers";
		}

		return "unknown";
	}

	bool check(OSStatus status, std::string_view what)
	{
		if (status != noErr)
		{
			coreaudio_log.error("%s failed: %d", what, static_cast<s32>(status));
			return false;
		}

		return true;
	}

	// Property listeners run on a Core Audio notification thread. The registry makes sure a notification that is
	// still in flight while a backend is destroyed does not touch freed memory.
	struct listener_registry
	{
		std::mutex mutex;
		std::unordered_set<const CoreAudioBackend*> live;
	};

	listener_registry& get_listener_registry()
	{
		static listener_registry registry;
		return registry;
	}
}

std::vector<AudioDeviceID> CoreAudioBackend::get_output_devices()
{
	std::vector<AudioDeviceID> result;

	for (AudioDeviceID id : get_property_array<AudioDeviceID>(kAudioObjectSystemObject, prop(kAudioHardwarePropertyDevices)))
	{
		if (get_output_channel_count(id) > 0)
		{
			result.push_back(id);
		}
	}

	return result;
}

AudioDeviceID CoreAudioBackend::get_default_output_device()
{
	AudioDeviceID id = kAudioObjectUnknown;

	if (!get_property(kAudioObjectSystemObject, prop(kAudioHardwarePropertyDefaultOutputDevice), id))
	{
		return kAudioObjectUnknown;
	}

	return id;
}

bool CoreAudioBackend::query_device(AudioDeviceID id, device_info& info)
{
	info = {};
	info.id = id;
	info.uid = get_string_property(id, kAudioDevicePropertyDeviceUID);
	info.name = get_string_property(id, kAudioObjectPropertyName);
	info.channels = get_output_channel_count(id);
	info.labels = get_device_labels(id);

	get_property(id, prop(kAudioDevicePropertyTransportType), info.transport);

	if (info.labels.size() > info.channels)
	{
		info.labels.resize(info.channels);
	}

	// What is connected: the terminal type of the first output stream, the data source of built-in outputs
	UInt32 terminal = kAudioStreamTerminalTypeUnknown;
	if (const auto streams = get_property_array<AudioStreamID>(id, prop(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput)); !streams.empty())
	{
		get_property(streams[0], prop(kAudioStreamPropertyTerminalType), terminal);
	}

	UInt32 source = 0;
	get_property(id, prop(kAudioDevicePropertyDataSource, kAudioObjectPropertyScopeOutput), source);

	std::string lower_name = info.name;
	std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });

	if (terminal == kAudioStreamTerminalTypeHeadphones || source == fourcc("hdpn"))
	{
		info.kind = output_kind::headphones;
	}
	else if (info.transport == kAudioDeviceTransportTypeBluetooth || info.transport == kAudioDeviceTransportTypeBluetoothLE)
	{
		// AirPods and other Bluetooth headphones (a Bluetooth speaker reports a speaker terminal)
		info.kind = terminal == kAudioStreamTerminalTypeSpeaker ? output_kind::external_speakers : output_kind::headphones;
	}
	else if (info.transport == kAudioDeviceTransportTypeBuiltIn)
	{
		// Apple silicon Macs expose the headphone jack as its own built-in device
		info.kind = lower_name.find("headphone") != std::string::npos ? output_kind::headphones : output_kind::builtin_speakers;
	}
	else
	{
		info.kind = output_kind::external_speakers;
	}

	return !info.uid.empty();
}

CoreAudioBackend::CoreAudioBackend()
	: AudioBackend()
{
	listener_registry& registry = get_listener_registry();
	std::lock_guard lock(registry.mutex);
	registry.live.insert(this);
}

CoreAudioBackend::~CoreAudioBackend()
{
	Close();
	restore_device_rate();

	listener_registry& registry = get_listener_registry();
	std::lock_guard lock(registry.mutex);
	registry.live.erase(this);
}

bool CoreAudioBackend::Operational()
{
	return m_output_unit != nullptr && !m_reset_req.observe();
}

bool CoreAudioBackend::DefaultDeviceChanged()
{
	if (m_reset_req.observe())
	{
		return false;
	}

	if (m_reconfigure_req.exchange(false))
	{
		// Same device, different route (headphones plugged in, speaker configuration changed, ...): reopen
		return true;
	}

	return m_use_default_device && get_default_output_device() != m_device.id;
}

void CoreAudioBackend::set_rate_for_device(const device_info& device)
{
	if (m_rate_device != kAudioObjectUnknown && m_rate_device != device.id)
	{
		restore_device_rate();
	}

	const AudioObjectPropertyAddress address = prop(kAudioDevicePropertyNominalSampleRate);

	f64 rate = 0.;
	get_property(device.id, address, rate);
	m_device_rate = rate > 0. ? rate : m_stream_rate;

	if (rate == m_stream_rate)
	{
		return;
	}

	if (!g_cfg.audio.match_device_rate)
	{
		coreaudio_log.notice("Device runs at %.0f Hz; the %.0f Hz stream is resampled (Switch Device To 48 kHz is disabled)", rate, m_stream_rate);
		return;
	}

	const auto ranges = get_property_array<AudioValueRange>(device.id, prop(kAudioDevicePropertyAvailableNominalSampleRates));
	const f64 wanted = m_stream_rate;
	const bool supported = std::any_of(ranges.begin(), ranges.end(), [wanted](const AudioValueRange& range)
	{
		return range.mMinimum <= wanted && range.mMaximum >= wanted;
	});

	if (!supported)
	{
		coreaudio_log.warning("Device '%s' does not support %.0f Hz; the stream is resampled to %.0f Hz", device.name, m_stream_rate, rate);
		return;
	}

	const Float64 new_rate = m_stream_rate;

	if (OSStatus err = AudioObjectSetPropertyData(device.id, &address, 0, nullptr, sizeof(new_rate), &new_rate); err != noErr)
	{
		coreaudio_log.warning("Could not switch '%s' to %.0f Hz (error %d); the stream is resampled to %.0f Hz", device.name, m_stream_rate, static_cast<s32>(err), rate);
		return;
	}

	// The switch completes asynchronously (up to 250 ms)
	f64 current = rate;
	for (u32 i = 0; i < 50 && current != new_rate; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		get_property(device.id, address, current);
	}

	if (m_rate_device != device.id)
	{
		m_rate_device = device.id;
		m_rate_to_restore = rate;
	}

	m_rate_set = new_rate;
	m_device_rate = current > 0. ? current : new_rate;
	coreaudio_log.notice("Switched '%s' from %.0f Hz to %.0f Hz (no resampling; restored when RPCS3 stops using the device)", device.name, rate, m_device_rate);
}

void CoreAudioBackend::restore_device_rate()
{
	if (m_rate_device == kAudioObjectUnknown)
	{
		return;
	}

	const AudioDeviceID device = std::exchange(m_rate_device, kAudioObjectUnknown);
	const AudioObjectPropertyAddress address = prop(kAudioDevicePropertyNominalSampleRate);

	// Only if nobody else changed it in the meantime
	if (f64 current = 0.; m_rate_to_restore > 0. && get_property(device, address, current) && current == m_rate_set)
	{
		const Float64 rate = m_rate_to_restore;
		if (AudioObjectSetPropertyData(device, &address, 0, nullptr, sizeof(rate), &rate) == noErr)
		{
			coreaudio_log.notice("Restored the device sample rate to %.0f Hz", rate);
		}
	}

	m_rate_to_restore = 0.;
	m_rate_set = 0.;
}

bool CoreAudioBackend::build_route(const device_info& device, audio_channel_layout layout)
{
	const bool use_s16 = get_convert_to_s16();

	// 1. Target speaker layout: the user's "Audio Channel Layout", otherwise the device's speaker configuration
	std::vector<AudioChannelLabel> target = user_layout_labels(layout);

	if (!target.empty() && target.size() > device.channels)
	{
		coreaudio_log.warning("Can't use layout %s with %d device channels. Falling back to automatic layout.", layout, device.channels);
		target.clear();
		layout = audio_channel_layout::automatic;
	}

	// Speakers of the device (at most 8: the mixer's output limit; interfaces with many channels list unused ones too)
	std::vector<AudioChannelLabel> device_speakers;
	for (AudioChannelLabel label : device.labels)
	{
		if (is_speaker_label(label) && device_speakers.size() < 8)
		{
			device_speakers.push_back(label);
		}
	}

	const bool device_labels_usable = has_label(device_speakers, kAudioChannelLabel_Mono) ||
		(has_label(device_speakers, kAudioChannelLabel_Left) && has_label(device_speakers, kAudioChannelLabel_Right));

	bool target_is_device = false;

	if (target.empty())
	{
		if (device_labels_usable)
		{
			target = device_speakers;
			target_is_device = device_speakers == device.labels && device.labels.size() == device.channels;
		}
		else
		{
			target = device.channels >= 2 ? std::vector<AudioChannelLabel>{ kAudioChannelLabel_Left, kAudioChannelLabel_Right } : std::vector<AudioChannelLabel>{ kAudioChannelLabel_Mono };
		}
	}

	const u32 target_count = static_cast<u32>(target.size());

	// AUSpatialMixer: "A single output is presented with 2, 4, 5, 6, 7 or 8 channels"
	const bool mixer_can_output = target_count >= 2 && target_count <= 8 && target_count != 3;

	// 2. Rendering for the connected hardware
	output_kind kind = device.kind;

	switch (g_cfg.audio.spatial_mode.get())
	{
	case audio_spatial_mode::automatic: break;
	case audio_spatial_mode::headphones: kind = output_kind::headphones; break;
	case audio_spatial_mode::speakers: kind = device.kind == output_kind::builtin_speakers ? output_kind::builtin_speakers : output_kind::external_speakers; break;
	case audio_spatial_mode::off: kind = output_kind::external_speakers; break;
	}

	// 3. Route
	const u32 game_channels = m_channels;
	std::vector<AudioChannelLabel> output_labels;

	if (game_channels == 2 && has_label(target, kAudioChannelLabel_Left) && has_label(target, kAudioChannelLabel_Right))
	{
		// Stereo: untouched, only routed to the device's left/right channels
		m_route = route_type::direct;
		output_labels = ps3_labels(2);
	}
	else if (game_channels > 2 && device.channels >= 2 && kind != output_kind::external_speakers && (target_count <= 2 || kind == output_kind::headphones))
	{
		// Surround on headphones or built-in speakers: render the channels as a speaker bed around the listener
		m_route = route_type::spatial_bed;
		output_labels = { kAudioChannelLabel_Left, kAudioChannelLabel_Right };
	}
	else if (game_channels > 2 && mixer_can_output)
	{
		// Speakers: map the channels to the speaker layout by label; standard down/upmix only where the layouts differ
		m_route = route_type::matrix;
		output_labels = target;
	}
	else
	{
		// Layouts the mixer cannot produce (mono, 3 channels): cellAudio downmixes to them, as with the other backends
		setup_channel_layout(game_channels, target_count, layout, coreaudio_log);
		m_route = route_type::direct;
		output_labels = user_layout_labels(m_layout);
	}

	const u32 output_channels = static_cast<u32>(output_labels.size());

	// Output unit
	const AudioComponentDescription output_desc{ kAudioUnitType_Output, kAudioUnitSubType_HALOutput, kAudioUnitManufacturer_Apple, 0, 0 };
	AudioComponent output_component = AudioComponentFindNext(nullptr, &output_desc);

	if (!output_component || !check(AudioComponentInstanceNew(output_component, &m_output_unit), "AudioComponentInstanceNew(HALOutput)"))
	{
		m_output_unit = nullptr;
		return false;
	}

	if (!check(AudioUnitSetProperty(m_output_unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &device.id, sizeof(device.id)), "Set current device"))
	{
		return false;
	}

	// ~10 ms device buffer (the emulator adds its own buffering on top)
	{
		UInt32 frames = 512;
		AudioValueRange range{};
		if (get_property(device.id, prop(kAudioDevicePropertyBufferFrameSizeRange), range) && range.mMaximum >= range.mMinimum)
		{
			frames = static_cast<UInt32>(std::clamp<f64>(frames, range.mMinimum, range.mMaximum));
		}

		if (AudioUnitSetProperty(m_output_unit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &frames, sizeof(frames)) != noErr)
		{
			coreaudio_log.warning("Could not set the device buffer size to %u frames", frames);
		}

		UInt32 actual = 0;
		UInt32 size = sizeof(actual);
		if (AudioUnitGetProperty(m_output_unit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &actual, &size) == noErr && actual)
		{
			frames = actual;
		}

		m_buffer_frames = frames;
	}

	const UInt32 max_frames = MAX_FRAMES_PER_SLICE;
	AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &max_frames, sizeof(max_frames));

	// Only used when the device cannot run at the stream rate: best available converter
	const UInt32 complexity = kAudioUnitSampleRateConverterComplexity_Mastering;
	AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_SampleRateConverterComplexity, kAudioUnitScope_Global, 0, &complexity, sizeof(complexity));
	const UInt32 quality = kRenderQuality_Max;
	AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_RenderQuality, kAudioUnitScope_Global, 0, &quality, sizeof(quality));

	if (m_route == route_type::direct)
	{
		const AudioStreamBasicDescription format = make_format(m_stream_rate, m_channels, true, use_s16);
		if (!check(AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format, sizeof(format)), "Set output unit stream format"))
		{
			return false;
		}

		const AURenderCallbackStruct callback{ render_direct, this };
		if (!check(AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback, sizeof(callback)), "Set render callback"))
		{
			return false;
		}
	}
	else
	{
		// The mixer works on float; 16-bit input ("Convert to 16 bit") is converted in render_mixer_input.
		// m_sample_size must stay as requested: cellAudio sizes its ring buffer from the requested sample size.
		const AudioComponentDescription mixer_desc{ kAudioUnitType_Mixer, kAudioUnitSubType_SpatialMixer, kAudioUnitManufacturer_Apple, 0, 0 };
		AudioComponent mixer_component = AudioComponentFindNext(nullptr, &mixer_desc);

		if (!mixer_component || !check(AudioComponentInstanceNew(mixer_component, &m_mixer_unit), "AudioComponentInstanceNew(SpatialMixer)"))
		{
			m_mixer_unit = nullptr;
			return false;
		}

		const UInt32 buses = 1;
		AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_ElementCount, kAudioUnitScope_Input, 0, &buses, sizeof(buses));
		AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &max_frames, sizeof(max_frames));

		const std::vector<AudioChannelLabel> source = ps3_labels(game_channels);
		const AudioStreamBasicDescription input_format = make_format(m_stream_rate, game_channels, false, false);
		const AudioStreamBasicDescription output_format = make_format(m_stream_rate, output_channels, false, false);
		const std::vector<u64> input_layout = make_layout(source);
		const std::vector<u64> output_layout = make_layout(output_labels);
		const auto layout_size = [](const std::vector<AudioChannelLabel>& labels)
		{
			return static_cast<UInt32>(offsetof(AudioChannelLayout, mChannelDescriptions) + labels.size() * sizeof(AudioChannelDescription));
		};

		if (!check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &input_format, sizeof(input_format)), "Set mixer input format") ||
			!check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Input, 0, input_layout.data(), layout_size(source)), "Set mixer input layout") ||
			!check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &output_format, sizeof(output_format)), "Set mixer output format") ||
			!check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_AudioChannelLayout, kAudioUnitScope_Output, 0, output_layout.data(), layout_size(output_labels)), "Set mixer output layout"))
		{
			return false;
		}

		const UInt32 algorithm = kSpatializationAlgorithm_UseOutputType;
		const UInt32 source_mode = m_route == route_type::spatial_bed ? kSpatialMixerSourceMode_AmbienceBed : kSpatialMixerSourceMode_Bypass;
		const UInt32 output_type = kind == output_kind::headphones ? kSpatialMixerOutputType_Headphones :
			kind == output_kind::builtin_speakers ? kSpatialMixerOutputType_BuiltInSpeakers : kSpatialMixerOutputType_ExternalSpeakers;
		const UInt32 no_reverb = 0;

		check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SpatializationAlgorithm, kAudioUnitScope_Input, 0, &algorithm, sizeof(algorithm)), "Set spatialization algorithm");
		check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerSourceMode, kAudioUnitScope_Input, 0, &source_mode, sizeof(source_mode)), "Set spatial mixer source mode");
		check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerOutputType, kAudioUnitScope_Global, 0, &output_type, sizeof(output_type)), "Set spatial mixer output type");
		AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_UsesInternalReverb, kAudioUnitScope_Global, 0, &no_reverb, sizeof(no_reverb));

		if (m_route == route_type::spatial_bed && kind == output_kind::headphones)
		{
			// Personal spatial audio profile (Settings > your name > Personalized Spatial Audio) and AirPods head tracking.
			// Both need entitlements that only a signed distribution can carry (see README); without them the generic
			// HRTF is used and the sound field stays fixed to the head.
			const UInt32 hrtf = kSpatialMixerPersonalizedHRTFMode_Auto;
			AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerPersonalizedHRTFMode, kAudioUnitScope_Global, 0, &hrtf, sizeof(hrtf));

			const UInt32 head_tracking = g_cfg.audio.spatial_head_tracking ? 1 : 0;
			if (AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerEnableHeadTracking, kAudioUnitScope_Global, 0, &head_tracking, sizeof(head_tracking)) != noErr && head_tracking)
			{
				coreaudio_log.notice("Head tracking is not available");
			}
		}

		const AURenderCallbackStruct callback{ render_mixer_input, this };
		if (!check(AudioUnitSetProperty(m_mixer_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback, sizeof(callback)), "Set mixer render callback") ||
			!check(AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &output_format, sizeof(output_format)), "Set output unit stream format"))
		{
			return false;
		}

		const AudioUnitConnection connection{ m_mixer_unit, 0, 0 };
		if (!check(AudioUnitSetProperty(m_output_unit, kAudioUnitProperty_MakeConnection, kAudioUnitScope_Input, 0, &connection, sizeof(connection)), "Connect mixer to output unit") ||
			!check(AudioUnitInitialize(m_mixer_unit), "AudioUnitInitialize(SpatialMixer)"))
		{
			return false;
		}

		m_scratch.assign(usz{MAX_FRAMES_PER_SLICE} * game_channels, 0.f);
	}

	// Device channels fed by the output unit (after the formats: the map refers to the input format's channels)
	if (!(m_route == route_type::matrix && target_is_device))
	{
		const std::vector<SInt32> map = make_channel_map(output_labels, device);

		if (!map.empty() && !is_identity_map(map, output_channels))
		{
			if (AudioUnitSetProperty(m_output_unit, kAudioOutputUnitProperty_ChannelMap, kAudioUnitScope_Output, 0, map.data(), static_cast<UInt32>(map.size() * sizeof(SInt32))) != noErr)
			{
				coreaudio_log.warning("Could not set the output channel map");
			}
		}
	}

	if (!check(AudioUnitInitialize(m_output_unit), "AudioUnitInitialize(HALOutput)"))
	{
		return false;
	}

	if (m_mixer_unit && m_route == route_type::spatial_bed && kind == output_kind::headphones)
	{
		UInt32 personalized = 0, tracking = 0;
		UInt32 size = sizeof(UInt32);
		AudioUnitGetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerAnyInputIsUsingPersonalizedHRTF, kAudioUnitScope_Global, 0, &personalized, &size);
		size = sizeof(UInt32);
		AudioUnitGetProperty(m_mixer_unit, kAudioUnitProperty_SpatialMixerEnableHeadTracking, kAudioUnitScope_Global, 0, &tracking, &size);
		coreaudio_log.notice("Spatial audio for headphones: personalized HRTF %s, head tracking %s", personalized ? "on" : "off", tracking ? "on" : "off");
	}

	static constexpr std::string_view route_names[] = { "direct", "spatial speaker bed", "speaker layout mapping" };
	m_route_description = fmt::format("%d channel(s) -> %s -> %d channel(s) on '%s' (%s, %u device channels, %u frame buffer)",
		game_channels, ::at32(route_names, static_cast<u32>(m_route)), output_channels, device.name, kind_to_string(kind), device.channels, m_buffer_frames);

	return true;
}

void CoreAudioBackend::destroy_units()
{
	if (m_output_unit)
	{
		AudioUnitUninitialize(m_output_unit);
		AudioComponentInstanceDispose(m_output_unit);
		m_output_unit = nullptr;
	}

	if (m_mixer_unit)
	{
		AudioUnitUninitialize(m_mixer_unit);
		AudioComponentInstanceDispose(m_mixer_unit);
		m_mixer_unit = nullptr;
	}
}

void CoreAudioBackend::install_listeners()
{
	const auto add = [this](AudioObjectID object, const AudioObjectPropertyAddress& address)
	{
		if (AudioObjectHasProperty(object, &address) && AudioObjectAddPropertyListener(object, &address, property_listener, this) == noErr)
		{
			m_listeners.emplace_back(object, address);
		}
	};

	add(kAudioObjectSystemObject, prop(kAudioHardwarePropertyDefaultOutputDevice));
	add(m_device.id, prop(kAudioDevicePropertyDeviceIsAlive));
	add(m_device.id, prop(kAudioDevicePropertyNominalSampleRate));
	add(m_device.id, prop(kAudioDevicePropertyDataSource, kAudioObjectPropertyScopeOutput));
	add(m_device.id, prop(kAudioDevicePropertyPreferredChannelLayout, kAudioObjectPropertyScopeOutput));
	add(m_device.id, prop(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput));
}

void CoreAudioBackend::remove_listeners()
{
	for (const auto& [object, address] : m_listeners)
	{
		AudioObjectRemovePropertyListener(object, &address, property_listener, this);
	}

	m_listeners.clear();
}

bool CoreAudioBackend::Open(std::string_view dev_id, AudioFreq freq, AudioSampleSize sample_size, AudioChannelCnt ch_cnt, audio_channel_layout layout)
{
	Close();

	m_use_default_device = dev_id.empty() || dev_id == audio_device_enumerator::DEFAULT_DEV_ID;
	AudioDeviceID device_id = kAudioObjectUnknown;

	if (!m_use_default_device)
	{
		for (AudioDeviceID id : get_output_devices())
		{
			if (get_string_property(id, kAudioDevicePropertyDeviceUID) == dev_id)
			{
				device_id = id;
				break;
			}
		}

		if (device_id == kAudioObjectUnknown)
		{
			coreaudio_log.warning("Device '%s' not found, using the default output device", dev_id);
			m_use_default_device = true;
		}
	}

	if (m_use_default_device)
	{
		device_id = get_default_output_device();
	}

	device_info device{};

	if (device_id == kAudioObjectUnknown || !query_device(device_id, device) || device.channels == 0)
	{
		coreaudio_log.error("No usable output device");
		return false;
	}

	coreaudio_log.notice("Opening '%s' (uid=%s, %u channels, transport='%s', detected %s)", device.name, device.uid, device.channels,
		std::string{ static_cast<char>(device.transport >> 24), static_cast<char>(device.transport >> 16), static_cast<char>(device.transport >> 8), static_cast<char>(device.transport) },
		kind_to_string(device.kind));

	// cellAudio always uses 48 kHz; the RSXAudio provider passes the rate of its port
	m_sampling_rate = freq;
	m_stream_rate = static_cast<f64>(static_cast<u32>(freq));
	m_sample_size = sample_size;
	m_channels = static_cast<u32>(ch_cnt);
	m_layout = default_layout(m_channels);

	if (!build_route(device, layout))
	{
		destroy_units();
		return false;
	}

	// Only once the route works: a backend that keeps failing must not flip the device's rate back and forth
	set_rate_for_device(device);

	m_frame_bytes = m_channels * get_sample_size();
	m_last_sample.fill(0);
	m_device = std::move(device);
	m_reset_req.release(false);
	m_reconfigure_req.release(false);

	install_listeners();

	if (!check(AudioOutputUnitStart(m_output_unit), "AudioOutputUnitStart"))
	{
		remove_listeners();
		destroy_units();
		restore_device_rate();
		return false;
	}

	coreaudio_log.notice("Output route: %s at %.0f Hz (stream %.0f Hz)", m_route_description, m_device_rate, m_stream_rate);

	// The default device may have changed while this ran (before the listener existed)
	if (m_use_default_device && get_default_output_device() != m_device.id)
	{
		m_reconfigure_req.release(true);
	}

	return true;
}

void CoreAudioBackend::Close()
{
	if (m_output_unit)
	{
		// Synchronous: no render callback runs once this returns
		AudioOutputUnitStop(m_output_unit);
	}

	remove_listeners();
	destroy_units();

	std::lock_guard lock(m_cb_mutex);
	m_playing = false;
	m_last_sample.fill(0);
}

void CoreAudioBackend::Play()
{
	if (!m_output_unit)
	{
		coreaudio_log.error("Play() called uninitialized");
		return;
	}

	if (m_playing) return;

	std::lock_guard lock(m_cb_mutex);
	m_playing = true;
}

void CoreAudioBackend::Pause()
{
	if (!m_output_unit)
	{
		coreaudio_log.error("Pause() called uninitialized");
		return;
	}

	if (!m_playing) return;

	std::lock_guard lock(m_cb_mutex);
	m_playing = false;
	m_last_sample.fill(0);
}

f64 CoreAudioBackend::GetCallbackFrameLen()
{
	return std::max<f64>(AUDIO_MIN_LATENCY, m_buffer_frames / std::max<f64>(m_device_rate, 1.));
}

void CoreAudioBackend::read_frames(u8* dst, u32 bytes)
{
	const u32 frame_bytes = m_frame_bytes;

	// Never block the real-time thread: if Play()/Pause()/Close() holds the lock, this cycle is silent
	std::unique_lock lock(m_cb_mutex, std::try_to_lock);

	if (frame_bytes && !m_reset_req.observe() && lock.owns_lock() && m_write_callback && m_playing)
	{
		u32 written = std::min(m_write_callback(bytes, dst), bytes);
		written -= written % frame_bytes;

		if (written >= frame_bytes)
		{
			std::memcpy(m_last_sample.data(), dst + written - frame_bytes, frame_bytes);
		}

		// Underrun: hold the last sample (no click)
		for (u32 i = written; i + frame_bytes <= bytes; i += frame_bytes)
		{
			std::memcpy(dst + i, m_last_sample.data(), frame_bytes);
		}
	}
	else
	{
		std::memset(dst, 0, bytes);
	}
}

OSStatus CoreAudioBackend::render_direct(void* ref, AudioUnitRenderActionFlags* /*flags*/, const AudioTimeStamp* /*time*/, UInt32 /*bus*/, UInt32 frames, AudioBufferList* data)
{
	auto* const self = static_cast<CoreAudioBackend*>(ref);

	if (!data || data->mNumberBuffers == 0 || !data->mBuffers[0].mData)
	{
		return noErr;
	}

	AudioBuffer& buffer = data->mBuffers[0];
	const u32 bytes = std::min<u32>(frames * self->m_frame_bytes, buffer.mDataByteSize);
	self->read_frames(static_cast<u8*>(buffer.mData), bytes);
	return noErr;
}

OSStatus CoreAudioBackend::render_mixer_input(void* ref, AudioUnitRenderActionFlags* /*flags*/, const AudioTimeStamp* /*time*/, UInt32 /*bus*/, UInt32 frames, AudioBufferList* data)
{
	auto* const self = static_cast<CoreAudioBackend*>(ref);
	const u32 channels = self->m_channels;

	if (!data)
	{
		return noErr;
	}

	AudioBuffer* const buffers = data->mBuffers;

	if (data->mNumberBuffers < channels || self->m_scratch.size() < usz{MAX_FRAMES_PER_SLICE} * channels)
	{
		for (UInt32 i = 0; i < data->mNumberBuffers; i++)
		{
			if (buffers[i].mData) std::memset(buffers[i].mData, 0, buffers[i].mDataByteSize);
		}

		return noErr;
	}

	const bool input_s16 = self->get_convert_to_s16();

	// cellAudio produces interleaved frames; the mixer takes one float buffer per channel
	for (u32 done = 0; done < frames;)
	{
		const u32 count = std::min<u32>(frames - done, MAX_FRAMES_PER_SLICE);
		self->read_frames(reinterpret_cast<u8*>(self->m_scratch.data()), count * self->m_frame_bytes);

		const f32* src = self->m_scratch.data();
		const s16* src16 = reinterpret_cast<const s16*>(self->m_scratch.data());

		for (u32 ch = 0; ch < channels; ch++)
		{
			f32* dst = static_cast<f32*>(buffers[ch].mData);
			if (!dst || buffers[ch].mDataByteSize < (done + count) * sizeof(f32)) continue;

			dst += done;

			if (input_s16)
			{
				for (u32 i = 0; i < count; i++)
				{
					dst[i] = src16[i * channels + ch] * (1.0f / 32768.0f);
				}
			}
			else
			{
				for (u32 i = 0; i < count; i++)
				{
					dst[i] = src[i * channels + ch];
				}
			}
		}

		done += count;
	}

	return noErr;
}

OSStatus CoreAudioBackend::property_listener(AudioObjectID object, UInt32 count, const AudioObjectPropertyAddress* addresses, void* ref)
{
	auto* const self = static_cast<CoreAudioBackend*>(ref);

	listener_registry& registry = get_listener_registry();
	std::lock_guard lock(registry.mutex);

	if (!registry.live.contains(self))
	{
		return noErr;
	}

	bool reconfigure = false;
	bool failed = false;

	for (UInt32 i = 0; i < count; i++)
	{
		const AudioObjectPropertyAddress& address = addresses[i];

		// Notifications can be batched with addresses nobody asked for (input side, other devices)
		const bool this_device = object == self->m_device.id;
		const bool output_side = address.mScope == kAudioObjectPropertyScopeOutput || address.mScope == kAudioObjectPropertyScopeGlobal;

		switch (address.mSelector)
		{
		case kAudioDevicePropertyDeviceIsAlive:
		{
			// A device that is already gone may not answer at all: that is "dead" too
			UInt32 alive = 0;
			if (this_device && (!get_property(object, prop(kAudioDevicePropertyDeviceIsAlive), alive) || !alive))
			{
				coreaudio_log.warning("Output device disconnected");
				self->m_reset_req.release(true);
				failed = true;
			}

			break;
		}
		case kAudioHardwarePropertyDefaultOutputDevice:
		{
			reconfigure |= self->m_use_default_device;
			break;
		}
		case kAudioDevicePropertyNominalSampleRate:
		{
			// Ignore the switch made by this backend
			f64 rate = 0.;
			if (this_device && get_property(object, prop(kAudioDevicePropertyNominalSampleRate), rate) && rate != self->m_device_rate)
			{
				coreaudio_log.notice("Device sample rate changed to %.0f Hz", rate);
				reconfigure = true;
			}

			break;
		}
		case kAudioDevicePropertyDataSource:
		case kAudioDevicePropertyPreferredChannelLayout:
		case kAudioDevicePropertyStreamConfiguration:
		{
			// Headphones plugged in, speaker setup changed, Bluetooth profile switch (e.g. AirPods to hands-free)
			if (this_device && output_side)
			{
				coreaudio_log.notice("Output route changed");
				reconfigure = true;
			}

			break;
		}
		default:
			break;
		}
	}

	if (reconfigure || failed)
	{
		if (reconfigure)
		{
			self->m_reconfigure_req.release(true);
		}

		std::lock_guard cb_lock(self->m_state_cb_mutex);

		if (self->m_state_callback)
		{
			self->m_state_callback(failed ? AudioStateEvent::UNSPECIFIED_ERROR : AudioStateEvent::DEFAULT_DEVICE_MAYBE_CHANGED);
		}
	}

	return noErr;
}
