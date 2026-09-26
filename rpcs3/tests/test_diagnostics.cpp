#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include "util/diagnostics.hpp"

LOG_CHANNEL(diag_test_channel, "DIAGTEST");

TEST(Diagnostics, NormalizeMasksNumbers)
{
	EXPECT_EQ(logs::diagnostics_listener::normalize_headline("Cached object for address 0xCE5E2000 was found"),
		"Cached object for address # was found");
	EXPECT_EQ(logs::diagnostics_listener::normalize_headline("2560x1440 -> 3456x1944, built in 1 ms"),
		"# -> #, built in # ms");
	EXPECT_EQ(logs::diagnostics_listener::normalize_headline("plain message without numbers"),
		"plain message without numbers");
	EXPECT_EQ(logs::diagnostics_listener::normalize_headline("first line\nsecond line 0x123"),
		"first line");
}

TEST(Diagnostics, GroupsRepeatsByNormalizedHeadline)
{
	logs::diagnostics_listener diag;

	diag.log(1'000'000, diag_test_channel.error, "PPU Thread", "Cached object for address 0xCE5E2000 was found");
	diag.log(2'000'000, diag_test_channel.error, "PPU Thread", "Cached object for address 0xDEADBEEF was found");
	diag.log(3'000'000, diag_test_channel.warning, "RSX Thread", "nextDrawable timed out. Frame skipped.");

	EXPECT_EQ(diag.total_count(), 3);

	const auto entries = diag.snapshot();
	ASSERT_EQ(entries.size(), 2);

	const logs::diagnostics_listener::entry* error_entry = nullptr;
	const logs::diagnostics_listener::entry* warning_entry = nullptr;

	for (const auto& e : entries)
	{
		if (e.level == "ERROR")
		{
			error_entry = &e;
		}
		else if (e.level == "WARNING")
		{
			warning_entry = &e;
		}
	}

	ASSERT_NE(error_entry, nullptr);
	ASSERT_NE(warning_entry, nullptr);
	EXPECT_EQ(error_entry->count, 2);
	EXPECT_EQ(error_entry->channel, "DIAGTEST");
	EXPECT_EQ(error_entry->headline, "Cached object for address # was found");
	EXPECT_EQ(warning_entry->count, 1);
}

TEST(Diagnostics, GroupsPerCallSite)
{
	logs::diagnostics_listener diag;

	diag.log(1000, diag_test_channel.warning, "PPU[0x1000000] Thread (main_thread) [HLE:0x01ff4b5c]", "same failure 0x1");
	diag.log(2000, diag_test_channel.warning, "PPU[0x1000001] Thread (main_thread) [HLE:0x01ff4b5c]", "same failure 0x2");

	// Same thread kind and call site, different instance: one entry
	ASSERT_EQ(diag.snapshot().size(), 1);

	diag.log(3000, diag_test_channel.warning, "PPU[0x1000000] Thread (Loading) [HLE:0x01ff9999]", "same failure 0x3");

	// Different call site: second entry
	const auto entries = diag.snapshot();
	ASSERT_EQ(entries.size(), 2);
	EXPECT_EQ(diag.total_count(), 3);
}

TEST(Diagnostics, TracksLastSeen)
{
	logs::diagnostics_listener diag;

	diag.log(1'000'000, diag_test_channel.error, "T", "boom");
	diag.log(9'000'000, diag_test_channel.error, "T", "boom");

	const auto entries = diag.snapshot();
	ASSERT_EQ(entries.size(), 1);
	EXPECT_EQ(entries[0].first_stamp, 1'000'000);
	EXPECT_EQ(entries[0].last_stamp, 9'000'000);
	EXPECT_EQ(entries[0].count, 2);
}

TEST(Diagnostics, IgnoresLowerSeverities)
{
	logs::diagnostics_listener diag;

	diag.log(1000, diag_test_channel.notice, "SYS", "background noise");
	diag.log(2000, diag_test_channel.trace, "SYS", "more noise");
	diag.log(3000, diag_test_channel.success, "SYS", "all good");

	EXPECT_EQ(diag.total_count(), 0);
	EXPECT_TRUE(diag.snapshot().empty());
}

TEST(Diagnostics, RecordsFatalErrorAndWarning)
{
	logs::diagnostics_listener diag;

	diag.log(1000, diag_test_channel.fatal, "SPU", "fatal boom");
	diag.log(2000, diag_test_channel.warning, "SPU", "warn once");

	EXPECT_EQ(diag.total_count(), 2);
	EXPECT_EQ(diag.snapshot().size(), 2);
}

TEST(Diagnostics, CapsUniqueEntries)
{
	logs::diagnostics_listener diag;

	// 600 distinct digit-free headlines (letters pass the normalizer through untouched)
	for (u32 i = 0; i < 600; i++)
	{
		const char a = static_cast<char>('a' + (i / 26) % 26);
		const char b = static_cast<char>('a' + i % 26);
		const char c = static_cast<char>('a' + (i / (26 * 26)) % 26);
		std::string text = "unique issue   ";
		text[13] = a;
		text[14] = b;
		text[15] = c;
		diag.log(1000 + i, diag_test_channel.warning, "T", text);
	}

	EXPECT_EQ(diag.total_count(), 600);
	EXPECT_EQ(diag.snapshot().size(), 512);
	EXPECT_EQ(diag.dropped_count(), 600 - 512);
}

TEST(Diagnostics, RefreshesReportWithoutShutdown)
{
	logs::diagnostics_listener diag;

	const std::string path = (std::filesystem::temp_directory_path() / "rpcs3_diag_periodic_test.log").string();
	std::remove(path.c_str());
	diag.set_output_path(path);

	// First qualifying message flushes immediately (no previous write)
	diag.log(1'000'000, diag_test_channel.error, "PPU", "early boom 0x1");

	std::FILE* f = std::fopen(path.c_str(), "rb");
	ASSERT_NE(f, nullptr);
	std::string content;
	char buf[1024];
	while (const usz n = std::fread(buf, 1, sizeof(buf), f))
	{
		content.append(buf, n);
	}
	std::fclose(f);

	EXPECT_NE(content.find("Totals: 0 fatal, 1 error, 0 warning"), std::string::npos);

	// A second message within the interval is recorded but does not rewrite the file
	diag.log(2'000'000, diag_test_channel.error, "PPU", "later boom 0x2");
	EXPECT_EQ(diag.total_count(), 2);

	f = std::fopen(path.c_str(), "rb");
	ASSERT_NE(f, nullptr);
	content.clear();
	while (const usz n = std::fread(buf, 1, sizeof(buf), f))
	{
		content.append(buf, n);
	}
	std::fclose(f);
	std::remove(path.c_str());

	EXPECT_EQ(content.find("later boom"), std::string::npos);
	EXPECT_NE(content.find("early boom"), std::string::npos);
}

TEST(Diagnostics, ReportRendersCountsAndHeadlines)
{
	logs::diagnostics_listener diag;

	diag.log(1'500'000, diag_test_channel.error, "PPU", "boom 0x1 happened");
	diag.log(2'500'000, diag_test_channel.error, "PPU", "boom 0x2 happened");

	const std::string path = (std::filesystem::temp_directory_path() / "rpcs3_diag_test.log").string();
	diag.write_report(path);

	std::FILE* f = std::fopen(path.c_str(), "rb");
	ASSERT_NE(f, nullptr);
	std::string content;
	char buf[1024];
	while (const usz n = std::fread(buf, 1, sizeof(buf), f))
	{
		content.append(buf, n);
	}
	std::fclose(f);
	std::remove(path.c_str());

	EXPECT_NE(content.find("Totals: 0 fatal, 2 error, 0 warning"), std::string::npos);
	EXPECT_NE(content.find("2 x [ERROR][DIAGTEST]"), std::string::npos);
	EXPECT_NE(content.find("boom # happened"), std::string::npos);
}
