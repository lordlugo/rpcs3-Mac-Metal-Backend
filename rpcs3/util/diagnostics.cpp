#include "util/diagnostics.hpp"

#include "Utilities/File.h"

#include <algorithm>

namespace logs
{
	static bool is_recorded_level(level lvl)
	{
		return lvl == level::fatal || lvl == level::error || lvl == level::warning;
	}

	static const char* level_name(level lvl)
	{
		switch (lvl)
		{
		case level::fatal: return "FATAL";
		case level::error: return "ERROR";
		case level::warning: return "WARNING";
		default: return "OTHER";
		}
	}

	std::string diagnostics_listener::normalize_headline(std::string_view text)
	{
		// Key on the first line only; multi-line texts share their headline
		if (const usz nl = text.find('\n'); nl != std::string_view::npos)
		{
			text = text.substr(0, nl);
		}

		// Mask numeric values (addresses, sizes, counts) so repeats group together
		std::string out;
		out.reserve(text.size());

		for (usz i = 0; i < text.size();)
		{
			const char c = text[i];

			if (c < '0' || c > '9')
			{
				out += c;
				i++;
				continue;
			}

			out += '#';
			while (i < text.size())
			{
				const char d = text[i];
				const bool cont = (d >= '0' && d <= '9') || (d >= 'a' && d <= 'f') || (d >= 'A' && d <= 'F') ||
					d == 'x' || d == 'X' || d == '.' || d == '_' || d == ':';
				if (!cont)
				{
					break;
				}
				i++;
			}
		}

		if (out.size() > 256)
		{
			out.erase(256);
		}

		return out;
	}

	void diagnostics_listener::log(u64 stamp, const message& msg, std::string_view prefix, std::string_view text)
	{
		const level lvl = msg;

		if (!is_recorded_level(lvl))
		{
			return;
		}

		const char* const ch_name = (msg->name && *msg->name) ? msg->name : "?";
		const std::string headline = normalize_headline(text);

		// The thread prefix carries the call site ([liblv2: 0x...], [HLE:...]); normalize it
		// the same way so repeats from one code location group together. Worker pool indexes
		// (RSX Worker 7) and PPU IDs normalize away: any worker/thread of the same kind matches.
		const std::string site = normalize_headline(prefix);

		// Unique key per severity, channel, call site and normalized headline
		std::string key;
		key += level_name(lvl);
		key += '\x1f';
		key += ch_name;
		key += '\x1f';
		key += site;
		key += '\x1f';
		key += headline;

		std::string flush_path;

		{
			std::lock_guard lock(m_mutex);
			m_total++;

			if (auto it = m_entries.find(key); it != m_entries.end())
			{
				it->second.count++;
				it->second.last_stamp = stamp;
			}
			else if (m_entries.size() >= max_entries)
			{
				m_dropped++;
			}
			else
			{
				entry e;
				e.level = level_name(lvl);
				e.channel = ch_name;
				e.site = site;
				e.headline = headline;
				e.sample = std::string(text.substr(0, max_sample_chars));
				e.thread = std::string(prefix.substr(0, 128));
				e.first_stamp = stamp;
				e.last_stamp = stamp;
				e.count = 1;
				m_entries.emplace(std::move(key), std::move(e));
			}

			// Refresh the on-disk report while the session runs, so a hang or force-quit
			// still leaves a usable file. Throttled; the file write runs without the lock.
			if (!m_path.empty())
			{
				const auto now = std::chrono::steady_clock::now();
				if (now - m_last_write >= write_interval)
				{
					m_last_write = now;
					flush_path = m_path;
				}
			}
		}

		if (!flush_path.empty())
		{
			write_report(flush_path);
		}
	}

	void diagnostics_listener::set_output_path(std::string path)
	{
		std::lock_guard lock(m_mutex);
		m_path = std::move(path);
	}

	u64 diagnostics_listener::total_count() const
	{
		std::lock_guard lock(m_mutex);
		return m_total;
	}

	u64 diagnostics_listener::dropped_count() const
	{
		std::lock_guard lock(m_mutex);
		return m_dropped;
	}

	std::vector<diagnostics_listener::entry> diagnostics_listener::snapshot() const
	{
		std::lock_guard lock(m_mutex);

		std::vector<entry> out;
		out.reserve(m_entries.size());

		for (const auto& [key, e] : m_entries)
		{
			out.push_back(e);
		}

		return out;
	}

	void diagnostics_listener::write_report(const std::string& path) const
	{
		std::vector<entry> sorted = snapshot();
		std::sort(sorted.begin(), sorted.end(), [](const entry& a, const entry& b)
		{
			return a.count > b.count;
		});

		u64 fatals = 0, errors = 0, warnings = 0;

		for (const entry& e : sorted)
		{
			if (e.level == "FATAL")
			{
				fatals += e.count;
			}
			else if (e.level == "ERROR")
			{
				errors += e.count;
			}
			else if (e.level == "WARNING")
			{
				warnings += e.count;
			}
		}

		std::string out = "RPCS3 diagnostics report\n"
			"Counts below include repeats. See RPCS3.log for the full log.\n";
		out += fmt::format("Totals: %llu fatal, %llu error, %llu warning message(s)\n", fatals, errors, warnings);

		if (sorted.empty())
		{
			out += "No errors or warnings were logged in this session.\n";
		}
		else
		{
			out += "Top issues (most frequent first):\n";

			for (const entry& e : sorted)
			{
				out += fmt::format("\n%llu x [%s][%s]%s%s%s\n", e.count, e.level.c_str(), e.channel.c_str(),
					e.thread.empty() ? "" : "[", e.thread.empty() ? "" : e.thread.c_str(), e.thread.empty() ? "" : "]");
				out += fmt::format("  first seen at %llu.%03llus, last seen at %llu.%03llus: %s\n",
					e.first_stamp / 1'000'000, (e.first_stamp / 1'000) % 1'000,
					e.last_stamp / 1'000'000, (e.last_stamp / 1'000) % 1'000, e.headline.c_str());

				if (!e.site.empty() && e.site != e.thread)
				{
					out += fmt::format("  call site: %s\n", e.site.c_str());
				}

				if (e.sample != e.headline)
				{
					out += "  e.g.:\n";
					out += e.sample;
					if (out.empty() || out.back() != '\n')
					{
						out += '\n';
					}
				}
			}

			if (m_dropped)
			{
				out += fmt::format("\n(%llu further unique issue(s) omitted: report capped)\n", m_dropped);
			}
		}

		fs::write_file(path, fs::rewrite, out);
	}
}
