#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "util/logs.hpp"

namespace logs
{
	// Aggregates error/warning/fatal log messages into a concise problem report.
	// Registered as a log listener in main(); the report is written to disk on shutdown,
	// so it can be inspected after RPCS3 closes (RPCS3_diagnostics.log next to RPCS3.log).
	// The full detail stays in RPCS3.log; each entry below points at the channel and thread
	// that reported the issue and how many times it repeated.
	class diagnostics_listener final : public listener
	{
	public:
		diagnostics_listener() = default;
		~diagnostics_listener() override = default;

		void log(u64 stamp, const message& msg, std::string_view prefix, std::string_view text) override;

		// Write the aggregated report to the given file path. Always writes something:
		// a short "no errors or warnings" note when the session was clean.
		void write_report(const std::string& path) const;

		// Remember where the report lives; every qualifying message then refreshes it at
		// most once per write_interval, so the file is usable even if RPCS3 hangs or is
		// force-quit and never reaches the shutdown write. Empty path disables this.
		void set_output_path(std::string path);

		struct entry
		{
			std::string level;   // FATAL, ERROR or WARNING
			std::string channel; // Log channel that reported the issue (e.g. RSX, PPU)
			std::string site;    // Normalized thread/call-site prefix (groups per code location)
			std::string headline; // First line of the message with numbers masked (groups repeats)
			std::string sample;  // First full message text seen (truncated)
			std::string thread;  // Full thread prefix of the first occurrence
			u64 first_stamp = 0; // Timestamp (microseconds) of the first occurrence
			u64 last_stamp = 0;  // Timestamp (microseconds) of the latest occurrence
			u64 count = 0;       // Occurrences including repeats
		};

		// Group repeat occurrences of the same underlying issue (masks addresses/sizes/counts)
		static std::string normalize_headline(std::string_view text);

		u64 total_count() const;
		u64 dropped_count() const;
		std::vector<entry> snapshot() const;

	private:
		static constexpr usz max_entries = 512;
		static constexpr usz max_sample_chars = 1024;
		static constexpr std::chrono::seconds write_interval{60};

		mutable std::mutex m_mutex;
		std::map<std::string, entry> m_entries;
		u64 m_dropped = 0; // Unique issues discarded once max_entries is reached
		u64 m_total = 0;   // Recorded messages including repeats
		std::string m_path; // Report file, refreshed periodically (empty: only explicit writes)
		std::chrono::steady_clock::time_point m_last_write{}; // Epoch: first write is immediate
	};
}
