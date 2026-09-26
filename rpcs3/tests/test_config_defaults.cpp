#include <gtest/gtest.h>

#include "Emu/system_config.h"

// Fork defaults that fix observed issues: construction must reflect them so a
// fresh install gets them without any migration marker files.
TEST(ConfigDefaults, TimeStretchingIsOn)
{
	// Smooths audio over game-side production jitter (skips/silence/underrun chops
	// instead of cutting in and out). See cellAudio xrun stats in RPCS3.log.
	const cfg_root def;
	EXPECT_TRUE(def.audio.enable_time_stretching.get());
}

TEST(ConfigDefaults, VSyncMatchesPlatformDefault)
{
	const cfg_root def;

#ifdef __APPLE__
	// The Metal renderer paces presentation to the display only with VSync enabled
	EXPECT_EQ(def.video.vsync.get(), vsync_mode::adaptive);
#else
	EXPECT_EQ(def.video.vsync.get(), vsync_mode::off);
#endif
}
