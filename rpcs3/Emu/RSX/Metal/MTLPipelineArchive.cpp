#include "stdafx.h"
#include "MTLPipelineArchive.h"
#include "mtlutils/device.h"

#include "Emu/System.h"
#include "Emu/system_config.h"
#include "Emu/cache_utils.hpp"
#include "Utilities/File.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
#include <pthread.h>
#endif

// Design notes
// ============
// API facts this relies on (metal-cpp 381 / macOS 26 SDK headers):
//  - An MTL4PipelineDataSetSerializer attached to an MTL4CompilerDescriptor records every pipeline that compiler
//    creates. With ...CaptureBinaries it can be written with serializeAsArchiveAndFlushToURL. Whether "flush" also
//    empties the data set is not documented, and the serializer is not documented as thread-safe (MTL4Compiler and
//    MTL4Archive are Sendable, the serializer is not).
//  - MTLDevice::newArchiveWithURL opens such a file; MTL4CompilerTaskOptions::lookupArchives lets a compiler reuse the
//    binaries it finds there ("Only add MTL4Archive instances to the array that are compatible with the Metal device").
//
// Hence:
//  - Compatibility: archives are only opened if identity.txt (format version, OS build, GPU name and architecture)
//    matches the running system; otherwise the directory is emptied. A build that fails WITH the archives but works
//    WITHOUT them disables them for the session and deletes them.
//  - Lookup: through lookupArchives on every compile (not MTL4Archive::newRenderPipelineState). Pipelines found in an
//    archive are still created by the capturing compiler and therefore recorded again, so a complete preload re-records
//    everything and old files can be dropped (see "File policy"); it also lets the compiler reuse the binaries of
//    individual functions for new combinations of known shaders.
//  - Capture/write: a serializer is written exactly once, by the archive thread, after its compiler has been retired
//    (swapped out for a fresh compiler + serializer) and every build that was still using it has returned. This is
//    correct whatever "flush" does, never serializes concurrently with a compile, never blocks a build, and bounds the
//    memory the serializer holds to one retirement period.
//
// File policy
// -----------
// Directory: <cache>/shaders_cache/metal_pipeline_archive/<fast|precise>/ (math mode changes every binary).
// A session (renderer lifetime) gets number S = 1 + highest number on disk. Retired serializer k of session S is
// written to sSSSSSSSS-kkkk.mtl4archive (".tmp" first, then renamed). Files of one session never overlap.
// Serializer 0 lives from renderer creation until the shader cache preload has finished, so it builds (and records,
// even when the binary came from an archive) every pipeline the title ever stored in its RSX shader cache. If the
// preload ran to completion its file supersedes all older archives: it is named sSSSSSSSS-0000-base.mtl4archive, and the
// next boot deletes every file of older sessions before opening anything. Safety net: the file is only promoted when it
// is at least half the size of the archives that were loaded (a serializer that did not record archive hits would
// produce a much smaller file); otherwise older files are kept and the caps below eventually reset the directory.
// Files are only ever deleted at boot (before they are opened) or when the archives were found to be broken.
//
// Unchanged preloads: base.txt records, for the newest base file (by session and size), how many pipelines its preload
// built and an order-independent digest of their keys (a hash of each pipeline's translated shaders and fixed-function
// state, see new_render_pipeline_state). A later complete preload with the same count and digest recorded exactly the
// pipelines that file already holds, so it is not rewritten: the session only updates base.txt ("confirmed" = its
// number), and the next boot supersedes older sessions' files up to that session except the base file itself, as if a
// new base file had been written. A base confirmed 7 boots in a row is rewritten anyway. A missing, torn or stale
// base.txt never matches, so the worst case is a rewrite of the base file; it is deleted along with that file.

namespace mtl
{
	namespace
	{
		using namespace std::chrono_literals;
		using clock_type = std::chrono::steady_clock;

		// Bump when the naming or the policy changes, or when every shader binary changes: a different identity discards
		// every archive. 2: invariant vertex positions and snapped MSAA texture lookups.
		constexpr u32 archive_format_version = 2;

		constexpr std::string_view archive_extension = ".mtl4archive";
		constexpr std::string_view base_suffix = "-base";
		constexpr std::string_view temp_extension = ".tmp";
		constexpr std::string_view incoming_dir = "incoming/"; // Archives are written here first (with their final name), then moved
		constexpr std::string_view identity_file_name = "identity.txt";
		constexpr std::string_view base_record_file_name = "base.txt";

		// A base file confirmed by this many boots in a row is rewritten anyway, which bounds the cost of anything that
		// changes the binaries without changing the pipeline keys (normally archive_format_version covers that)
		constexpr u32 max_base_confirmations = 7;

		// Limits for what is opened at boot. Normally one base file plus the deltas of one session remain.
		constexpr usz max_archive_files = 48;
		constexpr u64 max_archive_bytes = 3ull << 30;

		// Retirement policy of the capturing compiler (runs on the archive thread; builds never wait for it)
		constexpr auto retire_min_interval = 30s;    // Do not write more often than this after the preload
		constexpr auto retire_quiet_time = 3s;       // Prefer writing once compilation has settled down...
		constexpr u32 retire_pipeline_count = 256;   // ...but not later than after this many new pipelines...
		constexpr auto retire_max_interval = 300s;   // ...or this long
		constexpr auto retire_wait_limit = 120s;     // Longest wait for builds still using a retired compiler
		constexpr auto final_retire_wait_limit = 2s; // Same during shutdown

		// sSSSSSSSS-kkkk[-base].mtl4archive
		struct archive_file
		{
			std::string name;
			u32 session = 0;
			u32 index = 0;
			bool base = false;
			u64 size = 0;
		};

		std::string make_archive_name(u32 session, u32 index, bool base)
		{
			return fmt::format("s%08u-%04u", session, index) + std::string(base ? base_suffix : std::string_view{}) + std::string(archive_extension);
		}

		bool parse_number(std::string_view str, u32& value)
		{
			const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
			return ec == std::errc{} && ptr == str.data() + str.size();
		}

		// base.txt: what the newest base file holds (see "File policy")
		struct base_record
		{
			u32 session = 0;       // Session that wrote the base file (index 0)
			u64 size = 0;          // Its size, which ties the record to that very file
			u32 count = 0;         // Pipelines its preload recorded...
			u64 digest = 0;        // ...and the digest of their keys (capture_bundle::key_digest)
			u32 confirmed = 0;     // Newest session whose complete preload recorded the same pipelines (>= session)
			u32 confirmations = 0; // Boots in a row that confirmed the file instead of rewriting it
		};

		std::string format_base_record(const base_record& record)
		{
			return fmt::format("base %u %llu %u %016llx %u %u\n", record.session, record.size, record.count, record.digest, record.confirmed, record.confirmations);
		}

		bool parse_base_record(std::string_view text, base_record& out)
		{
			if (!text.ends_with('\n'))
			{
				return false;
			}

			text.remove_suffix(1);

			std::vector<std::string_view> fields;
			while (!text.empty())
			{
				const usz end = std::min(text.find(' '), text.size());
				fields.push_back(text.substr(0, end));
				text.remove_prefix(std::min(end + 1, text.size()));
			}

			const auto parse_u64 = [](std::string_view str, u64& value, int base)
			{
				const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value, base);
				return ec == std::errc{} && ptr == str.data() + str.size();
			};

			base_record record{};
			if (fields.size() != 7 || fields[0] != "base" || !parse_number(fields[1], record.session) || !parse_u64(fields[2], record.size, 10) ||
				!parse_number(fields[3], record.count) || !parse_u64(fields[4], record.digest, 16) || !parse_number(fields[5], record.confirmed) ||
				!parse_number(fields[6], record.confirmations))
			{
				return false;
			}

			out = record;
			return true;
		}

		bool parse_archive_name(std::string_view name, archive_file& out)
		{
			if (!name.ends_with(archive_extension))
			{
				return false;
			}

			name.remove_suffix(archive_extension.size());

			out.base = name.ends_with(base_suffix);
			if (out.base)
			{
				name.remove_suffix(base_suffix.size());
			}

			if (name.size() != 14 || name[0] != 's' || name[9] != '-')
			{
				return false;
			}

			return parse_number(name.substr(1, 8), out.session) && parse_number(name.substr(10, 4), out.index);
		}

		// The serializer is expected to write one file, but directory bundles are handled as well
		u64 get_entry_size(const std::string& path)
		{
			fs::stat_t info{};
			if (!fs::get_stat(path, info))
			{
				return 0;
			}

			if (info.is_directory)
			{
				const u64 size = fs::get_dir_size(path);
				return size == umax ? 0 : size;
			}

			return info.size;
		}

		bool remove_entry(const std::string& path)
		{
			fs::stat_t info{};
			if (!fs::get_stat(path, info))
			{
				return true;
			}

			return info.is_directory ? fs::remove_all(path) : fs::remove_file(path);
		}

		double to_mib(u64 bytes)
		{
			return static_cast<double>(bytes) / 0x100000;
		}

		u64 elapsed_ms(clock_type::time_point since)
		{
			return static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now() - since).count());
		}

		std::string build_identity(MTL::Device* device)
		{
			std::string architecture;
			if (MTL::Architecture* arch = device->architecture())
			{
				architecture = mtl::to_string(arch->name());
			}

			return fmt::format("RPCS3 Metal pipeline archive %u\nos: %s\ngpu: %s\narchitecture: %s\n",
				archive_format_version,
				mtl::to_string(NS::ProcessInfo::processInfo()->operatingSystemVersionString()),
				mtl::to_string(device->name()),
				architecture);
		}

		// A capturing compiler: an MTL4Compiler with its own pipeline data set serializer
		struct capture_bundle
		{
			MTL4::PipelineDataSetSerializer* serializer = nullptr; // +1
			MTL4::Compiler* compiler = nullptr;                    // +1, records into `serializer`
			u32 index = 0;
			atomic_t<u32> pipeline_count{ 0 };                     // Pipelines successfully built with `compiler`
			atomic_t<u64> key_digest{ 0 };                         // Sum of their mixed keys (independent of build order)
			atomic_t<bool> keys_known{ true };                     // False once a pipeline without a key was built

			void add_key(u64 key)
			{
				if (!key)
				{
					keys_known = false;
					return;
				}

				// splitmix64 finalizer, so that summing related keys can't cancel out
				key ^= key >> 30;
				key *= 0xbf58476d1ce4e5b9ull;
				key ^= key >> 27;
				key *= 0x94d049bb133111ebull;
				key ^= key >> 31;
				key_digest += key;
			}

			capture_bundle() = default;
			capture_bundle(const capture_bundle&) = delete;
			capture_bundle& operator=(const capture_bundle&) = delete;

			~capture_bundle()
			{
				if (compiler)
				{
					compiler->release();
				}

				if (serializer)
				{
					serializer->release();
				}
			}
		};

		class pipeline_archive final : public std::enable_shared_from_this<pipeline_archive>
		{
			MTL::Device* m_device = nullptr; // +1
			std::string m_directory;         // With trailing '/'
			u32 m_session = 0;

			// Archives of earlier sessions (read-only after init)
			std::vector<MTL4::Archive*> m_archives;                // +1 each
			std::vector<std::string> m_archive_names;
			MTL4::CompilerTaskOptions* m_lookup_options = nullptr; // +1, lookupArchives = m_archives
			u64 m_archive_bytes = 0;                               // Archive thread only after init
			atomic_t<bool> m_lookup_enabled{ false };
			std::optional<base_record> m_base_record;              // Describes the loaded base file (archive thread only after init)

			// Current capturing compiler. Builds hold a reference while they use it.
			std::mutex m_bundle_lock;
			std::shared_ptr<capture_bundle> m_bundle;
			u32 m_next_bundle_index = 0; // Init, then archive thread only
			bool m_capture_enabled = false;
			u32 m_write_failures = 0;           // Consecutive failed writes (archive thread only)

			atomic_t<u64> m_last_build_time{ 0 }; // clock_type ticks
			clock_type::time_point m_init_time{};
			clock_type::time_point m_last_retire_time{}; // Archive thread only after init

			atomic_t<bool> m_preload_done{ false };
			atomic_t<bool> m_preload_complete{ false };

			// Archive thread
			std::thread m_thread;
			std::mutex m_mutex;
			std::condition_variable m_cv;
			bool m_retire_requested = false;
			bool m_final_requested = false;
			bool m_discard_requested = false;
			bool m_exited = false;

		public:
			pipeline_archive() = default;
			pipeline_archive(const pipeline_archive&) = delete;
			pipeline_archive& operator=(const pipeline_archive&) = delete;

			~pipeline_archive()
			{
				// Only reached once the archive thread is gone (joined, or it dropped the last reference itself)
				m_bundle.reset();

				if (m_lookup_options)
				{
					m_lookup_options->release();
				}

				for (MTL4::Archive* archive : m_archives)
				{
					archive->release();
				}

				if (m_device)
				{
					m_device->release();
				}
			}

			bool init(MTL::Device* device, std::string directory)
			{
				mtl::autorelease_scope pool;

				m_device = device->retain();
				m_directory = std::move(directory);
				m_init_time = m_last_retire_time = clock_type::now();

				if (!fs::create_path(m_directory.substr(0, m_directory.size() - 1)))
				{
					rsx_log.error("Metal: cannot create the pipeline archive directory %s (%s)", m_directory, fs::g_tls_error);
					return false;
				}

				std::vector<archive_file> files = scan_directory();

				// 1. Archives built by another OS (driver) version or GPU would only miss
				const std::string identity = build_identity(device);
				const std::string identity_path = m_directory + std::string(identity_file_name);

				std::string stored_identity;
				if (fs::file file(identity_path); file)
				{
					stored_identity = file.to_string();
				}

				if (stored_identity != identity)
				{
					if (!files.empty())
					{
						rsx_log.notice("Metal: discarding %u pipeline archive file(s) built for another macOS version or GPU", ::size32(files));
					}

					remove_files(files);
					remove_entry(record_path());

					if (!fs::write_file(identity_path, fs::rewrite, identity))
					{
						rsx_log.error("Metal: cannot write %s (%s); the pipeline archive is disabled", identity_path, fs::g_tls_error);
						return false;
					}
				}

				// 2. Files of sessions older than the newest complete one are superseded (see "File policy")
				u32 newest_session = 0;
				u32 newest_base = 0;

				for (const archive_file& file : files)
				{
					newest_session = std::max(newest_session, file.session);

					if (file.base)
					{
						newest_base = std::max(newest_base, file.session);
					}
				}

				// A later complete preload that recorded the same pipelines as the newest base file supersedes older
				// files just like a new base file would have (the base file itself is kept)
				std::optional<base_record> record;

				if (fs::file file(record_path()); file && newest_base)
				{
					base_record stored{};

					if (parse_base_record(file.to_string(), stored) && stored.session == newest_base && stored.confirmed >= stored.session &&
						std::any_of(files.begin(), files.end(), [&](const archive_file& f) { return f.base && f.session == newest_base && f.size == stored.size; }))
					{
						record = stored;
					}
				}

				const u32 superseding_session = record ? record->confirmed : newest_base;

				if (superseding_session)
				{
					std::vector<archive_file> superseded;
					std::erase_if(files, [&](const archive_file& file)
					{
						if (file.session >= superseding_session || (file.base && file.session == newest_base))
						{
							return false;
						}

						superseded.push_back(file);
						return true;
					});

					remove_files(superseded);
				}

				u64 total_size = 0;
				for (const archive_file& file : files)
				{
					total_size += file.size;
				}

				if (files.size() > max_archive_files || total_size > max_archive_bytes)
				{
					rsx_log.warning("Metal: resetting the pipeline archive (%u files, %.1f MiB): earlier archives were never superseded",
						::size32(files), to_mib(total_size));
					remove_files(files);
					record.reset();
				}

				// Session numbers keep increasing even when the confirming sessions wrote no file
				m_session = std::max(newest_session, superseding_session) + 1;

				// 3. Open them: base first (most hits), then newest first
				std::sort(files.begin(), files.end(), [](const archive_file& a, const archive_file& b)
				{
					if (a.base != b.base)
					{
						return a.base;
					}

					return a.session != b.session ? a.session > b.session : a.index > b.index;
				});

				for (const archive_file& file : files)
				{
					const std::string path = m_directory + file.name;

					NS::Error* error = nullptr;
					MTL4::Archive* archive = device->newArchive(NS::URL::fileURLWithPath(mtl::ns_str(path)), &error);

					if (!archive)
					{
						rsx_log.warning("Metal: pipeline archive %s cannot be opened (%s); deleting it", file.name, mtl::to_string(error));
						remove_entry(path);
						continue;
					}

					archive->setLabel(mtl::ns_str(file.name));
					m_archives.push_back(archive);
					m_archive_names.push_back(file.name);
					m_archive_bytes += file.size;
				}

				if (!m_archives.empty())
				{
					std::vector<const NS::Object*> objects(m_archives.begin(), m_archives.end());

					m_lookup_options = MTL4::CompilerTaskOptions::alloc()->init();
					m_lookup_options->setLookupArchives(NS::Array::array(objects.data(), objects.size())); // Copied (retained)
					m_lookup_enabled = true;
				}

				// The record only matters while the base file it describes is in use
				if (record && is_base_loaded(*record))
				{
					m_base_record = record;
				}
				else
				{
					remove_entry(record_path());
				}

				// 4. Start recording
				m_capture_enabled = true;
				m_bundle = create_bundle();

				if (!m_bundle)
				{
					m_capture_enabled = false;
					rsx_log.warning("Metal: pipelines built in this session will not be added to the pipeline archive");
				}

				rsx_log.notice("Metal: pipeline archive %s: %u file(s), %.1f MiB loaded (session %u)",
					m_directory, ::size32(m_archives), to_mib(m_archive_bytes), m_session);

				return !m_archives.empty() || m_capture_enabled;
			}

			void start()
			{
				// A plain std::thread (not named_thread): shutdown must be able to leave a slow write running (detach)
				m_thread = std::thread([self = shared_from_this()]()
				{
#ifdef __APPLE__
					pthread_setname_np("RSX Pipeline Archive");
#endif
					self->thread_main();
				});
			}

			// Stops the thread after it wrote what is pending. Bounded: returns false if it did not finish in time. The
			// thread then keeps running (it owns a reference to this object); call stop() again later, or detach().
			bool stop(u32 timeout_ms)
			{
				if (!m_thread.joinable())
				{
					return true;
				}

				std::unique_lock lock(m_mutex);
				m_final_requested = true;
				m_cv.notify_all();

				const bool exited = m_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() { return m_exited; });
				lock.unlock();

				if (!exited)
				{
					return false;
				}

				m_thread.join();
				return true;
			}

			// Last resort at process exit: never hang on a write that does not finish
			void detach()
			{
				if (m_thread.joinable())
				{
					m_thread.detach();
				}
			}

			void request_final()
			{
				std::lock_guard lock(m_mutex);
				m_final_requested = true;
				m_cv.notify_all();
			}

			void on_preload_finished(bool complete)
			{
				u32 built = 0;
				{
					std::lock_guard lock(m_bundle_lock);
					if (m_bundle && m_bundle->index == 0)
					{
						built = m_bundle->pipeline_count;
					}
				}

				rsx_log.notice("Metal: %u pipeline(s) built %s %.2f s after renderer start (%u pipeline archive file(s))",
					built, complete ? "for the shader cache preload" : "before the preload was interrupted",
					static_cast<double>(elapsed_ms(m_init_time)) / 1000., ::size32(m_archives));

				m_preload_complete = complete;
				m_preload_done = true;

				// Retire serializer 0 now: it holds every pipeline of the title (see "File policy")
				std::lock_guard lock(m_mutex);
				m_retire_requested = true;
				m_cv.notify_all();
			}

			template <typename T, typename D, typename F>
			T* build(const D* descriptor, NS::Error** error, u64 key, F&& compile)
			{
				std::shared_ptr<capture_bundle> bundle;
				{
					std::lock_guard lock(m_bundle_lock);
					bundle = m_bundle;
				}

				MTL4::Compiler* compiler = bundle ? bundle->compiler : g_render_device->compiler();
				const MTL4::CompilerTaskOptions* options = m_lookup_enabled ? m_lookup_options : nullptr;

				NS::Error* build_error = nullptr;
				T* result = compile(compiler, descriptor, options, &build_error);

				if (!result && options)
				{
					// The archives must never make a build fail: retry without them
					NS::Error* retry_error = nullptr;
					result = compile(compiler, descriptor, nullptr, &retry_error);

					if (result)
					{
						if (m_lookup_enabled.exchange(false))
						{
							rsx_log.error("Metal: a pipeline failed to build with the pipeline archive (%s) but built without it. "
								"The archive is ignored for the rest of the session and rebuilt.", mtl::to_string(build_error));

							std::lock_guard lock(m_mutex);
							m_discard_requested = true;
							m_cv.notify_all();
						}
					}
					else
					{
						build_error = retry_error;
					}
				}

				if (result)
				{
					if (bundle)
					{
						bundle->add_key(key);
						bundle->pipeline_count++;
						m_last_build_time = static_cast<u64>(clock_type::now().time_since_epoch().count());
					}
				}
				else if (error)
				{
					*error = build_error;
				}

				return result;
			}

		private:
			std::string record_path() const
			{
				return m_directory + std::string(base_record_file_name);
			}

			// The base file `record` describes is open and still used for lookups (not discarded)
			bool is_base_loaded(const base_record& record) const
			{
				const std::string name = make_archive_name(record.session, 0, true);
				return m_lookup_enabled && std::find(m_archive_names.begin(), m_archive_names.end(), name) != m_archive_names.end();
			}

			bool write_base_record(const base_record& record)
			{
				// A torn or stale record never matches (see init), which only costs a rewrite of the base file
				if (!fs::write_file(record_path(), fs::rewrite, format_base_record(record)))
				{
					rsx_log.warning("Metal: cannot write %s (%s)", record_path(), fs::g_tls_error);
					remove_entry(record_path());
					return false;
				}

				return true;
			}

			std::vector<archive_file> scan_directory()
			{
				std::vector<archive_file> files;

				std::vector<std::string> leftovers;

				if (fs::dir dir(m_directory); dir)
				{
					for (const fs::dir_entry& entry : dir)
					{
						if (entry.name == "." || entry.name == "..")
						{
							continue;
						}

						// Leftovers of an interrupted write
						if (entry.name.ends_with(temp_extension))
						{
							leftovers.push_back(entry.name);
							continue;
						}

						archive_file file{};
						if (!parse_archive_name(entry.name, file))
						{
							continue;
						}

						file.name = entry.name;
						file.size = entry.is_directory ? get_entry_size(m_directory + entry.name) : entry.size;
						files.push_back(std::move(file));
					}
				}

				for (const std::string& name : leftovers)
				{
					remove_entry(m_directory + name);
				}

				// Leftovers of an interrupted write (the staging directory)
				remove_entry(m_directory + std::string(incoming_dir.substr(0, incoming_dir.size() - 1)));

				return files;
			}

			void remove_files(std::vector<archive_file>& files)
			{
				for (const archive_file& file : files)
				{
					if (!remove_entry(m_directory + file.name))
					{
						rsx_log.warning("Metal: cannot delete pipeline archive %s (%s)", file.name, fs::g_tls_error);
					}
				}

				files.clear();
			}

			std::shared_ptr<capture_bundle> create_bundle()
			{
				mtl::autorelease_scope pool;

				auto bundle = std::make_shared<capture_bundle>();

				// Archives are made of the pipeline binaries (CaptureBinaries). CaptureDescriptors only serves pipeline scripts
				// for the offline binary generator (serializeAsPipelinesScript); with it, every archive write failed without
				// an error on macOS 26/27, so it is not requested.
				auto serializer_desc = mtl::ref(MTL4::PipelineDataSetSerializerDescriptor::alloc()->init());
				serializer_desc->setConfiguration(MTL4::PipelineDataSetSerializerConfigurationCaptureBinaries);

				bundle->serializer = m_device->newPipelineDataSetSerializer(serializer_desc.get());
				if (!bundle->serializer)
				{
					rsx_log.error("Metal: cannot create a pipeline data set serializer");
					return {};
				}

				auto compiler_desc = mtl::ref(MTL4::CompilerDescriptor::alloc()->init());
				compiler_desc->setLabel(mtl::ns_str("RSX pipeline compiler"));
				compiler_desc->setPipelineDataSetSerializer(bundle->serializer);

				NS::Error* error = nullptr;
				bundle->compiler = m_device->newCompiler(compiler_desc.get(), &error);
				if (!bundle->compiler)
				{
					rsx_log.error("Metal: cannot create a capturing MTL4Compiler: %s", mtl::to_string(error));
					return {};
				}

				bundle->index = m_next_bundle_index++;
				return bundle;
			}

			void thread_main()
			{
				std::unique_lock lock(m_mutex);

				while (true)
				{
					m_cv.wait_for(lock, 1s, [this]() { return m_retire_requested || m_final_requested || m_discard_requested; });

					const bool final = m_final_requested;
					const bool retire = std::exchange(m_retire_requested, false);
					const bool discard = std::exchange(m_discard_requested, false);

					lock.unlock();
					{
						mtl::autorelease_scope pool;

						if (discard)
						{
							discard_loaded_archives();
						}

						if (final || retire || should_retire())
						{
							retire_bundle(final);
						}
					}
					lock.lock();

					if (final)
					{
						break;
					}
				}

				m_exited = true;
				m_cv.notify_all();
			}

			bool is_final_requested()
			{
				std::lock_guard lock(m_mutex);
				return m_final_requested;
			}

			bool should_retire()
			{
				if (!m_preload_done)
				{
					return false;
				}

				u32 pending = 0;
				{
					std::lock_guard lock(m_bundle_lock);
					if (m_bundle)
					{
						pending = m_bundle->pipeline_count;
					}
				}

				if (!pending)
				{
					return false;
				}

				const auto now = clock_type::now();
				const auto since_last = now - m_last_retire_time;

				if (since_last < retire_min_interval)
				{
					return false;
				}

				const auto last_build = clock_type::time_point(clock_type::duration(static_cast<clock_type::rep>(m_last_build_time.load())));
				return (now - last_build) >= retire_quiet_time || pending >= retire_pipeline_count || since_last >= retire_max_interval;
			}

			// Swap in a fresh capturing compiler (none if `final`) and write the old one's serializer
			void retire_bundle(bool final)
			{
				std::shared_ptr<capture_bundle> next;
				if (!final && m_capture_enabled)
				{
					next = create_bundle();

					if (!next)
					{
						m_capture_enabled = false;
						rsx_log.warning("Metal: pipelines built from now on will not be added to the pipeline archive");
					}
				}

				std::shared_ptr<capture_bundle> retired;
				{
					std::lock_guard lock(m_bundle_lock);
					retired = std::exchange(m_bundle, std::move(next));
				}

				m_last_retire_time = clock_type::now();

				if (!retired)
				{
					return;
				}

				// Builds that picked the retired compiler before the swap may still be running
				const auto wait_start = clock_type::now();
				while (retired.use_count() > 1)
				{
					const auto limit = is_final_requested() ? final_retire_wait_limit : retire_wait_limit;
					if (clock_type::now() - wait_start > limit)
					{
						// The last build releases the bundle
						rsx_log.error("Metal: pipeline builds kept a retired compiler busy for too long; %u pipeline(s) are not saved to the pipeline archive",
							retired->pipeline_count.load());
						return;
					}

					std::this_thread::sleep_for(1ms);
				}

				std::atomic_thread_fence(std::memory_order_acquire);
				write(*retired);
			}

			void write(capture_bundle& bundle)
			{
				const u32 count = bundle.pipeline_count;
				if (!count)
				{
					return;
				}

				// Serializer 0 of a complete preload that recorded exactly the pipelines of the loaded base file (same keys):
				// rewriting the base (hundreds of MiB for a big title) on every boot would only reproduce it. This session
				// confirms it instead, which supersedes older files the same way (see "File policy").
				if (bundle.index == 0 && m_preload_complete && m_base_record && is_base_loaded(*m_base_record) && bundle.keys_known &&
					count == m_base_record->count && bundle.key_digest == m_base_record->digest && m_base_record->confirmations < max_base_confirmations)
				{
					base_record record = *m_base_record;
					record.confirmed = m_session;
					record.confirmations++;

					if (write_base_record(record))
					{
						m_base_record = record;
						rsx_log.notice("Metal: the pipeline archive %s already holds the %u pipeline(s) of the preload; not rewriting it",
							make_archive_name(record.session, 0, true), count);
						return;
					}

					// Otherwise the older files would not be superseded: write a new base file as usual
					m_base_record.reset();
				}

				mtl::autorelease_scope pool;
				const auto start = clock_type::now();

				// Written under its final name (Metal may key on the file extension) in a staging directory, then moved
				const std::string staging = m_directory + std::string(incoming_dir);
				if (!fs::create_path(staging.substr(0, staging.size() - 1)))
				{
					rsx_log.error("Metal: cannot create %s (%s)", staging, fs::g_tls_error);
					return;
				}

				const std::string temp_path = staging + make_archive_name(m_session, bundle.index, false);
				remove_entry(temp_path);

				NS::Error* error = nullptr;
				if (!bundle.serializer->serializeAsArchiveAndFlushToURL(NS::URL::fileURLWithPath(mtl::ns_str(temp_path)), &error))
				{
					rsx_log.error("Metal: cannot write the pipeline archive (%u pipeline(s)) to %s: %s", count, temp_path, mtl::to_string(error));
					remove_entry(temp_path);

					if (++m_write_failures >= 2 && m_capture_enabled)
					{
						// Don't keep capturing (memory) and failing (time) for the rest of the session
						m_capture_enabled = false;
						rsx_log.error("Metal: the pipeline archive is disabled for this session after repeated write failures");
					}

					return;
				}

				m_write_failures = 0;

				const u64 size = get_entry_size(temp_path);
				if (!size)
				{
					rsx_log.error("Metal: the pipeline archive written to %s is empty or missing", temp_path);
					remove_entry(temp_path);
					return;
				}

				// Serializer 0 of a complete preload supersedes the older archives (see "File policy")
				const bool supersedes = bundle.index == 0 && m_preload_complete;
				const bool base = supersedes && size >= m_archive_bytes / 2;

				const std::string name = make_archive_name(m_session, bundle.index, base);
				if (!fs::rename(temp_path, m_directory + name, true))
				{
					rsx_log.error("Metal: cannot move the pipeline archive into place (%s)", fs::g_tls_error);
					remove_entry(temp_path);
					return;
				}

				rsx_log.notice("Metal: saved %u pipeline(s) to the pipeline archive %s (%.1f MiB, %llu ms)", count, name, to_mib(size), elapsed_ms(start));

				if (base)
				{
					// Written after the file is in place: a missing or stale record only means the next boot rewrites it
					m_base_record.reset();

					if (const base_record record{ m_session, size, count, bundle.key_digest, m_session, 0 }; !bundle.keys_known || !write_base_record(record))
					{
						remove_entry(record_path());
					}
					else
					{
						m_base_record = record;
					}
				}

				if (supersedes && !base)
				{
					rsx_log.warning("Metal: the preload archive (%.1f MiB) is much smaller than the archives it would replace (%.1f MiB); keeping them",
						to_mib(size), to_mib(m_archive_bytes));
				}
			}

			// The archives of earlier sessions broke a build: nothing uses them any more (m_lookup_enabled is false), so
			// delete them. The MTL4Archive objects stay valid (open files) until the session ends.
			void discard_loaded_archives()
			{
				for (const std::string& name : m_archive_names)
				{
					remove_entry(m_directory + name);
				}

				rsx_log.notice("Metal: deleted %u pipeline archive file(s)", ::size32(m_archive_names));
				m_archive_names.clear();
				m_archive_bytes = 0;

				// The base file it describes is gone (the preload must be written again)
				m_base_record.reset();
				remove_entry(record_path());
			}
		};

		std::mutex g_archive_lock;
		std::shared_ptr<pipeline_archive> g_archive;

		std::shared_ptr<pipeline_archive> get_archive()
		{
			std::lock_guard lock(g_archive_lock);
			return g_archive;
		}

		// Archives whose final write outlasted shutdown_pipeline_archive's wait. Their threads keep running; they are
		// joined before the next archive opens the directory (both would work on the same files) and at process exit
		// (the thread logs and uses fs:: helpers, which must not run while static objects are destroyed).
		std::mutex g_lingering_lock;
		std::vector<std::shared_ptr<pipeline_archive>> g_lingering;

		// True when no archive thread is left running
		bool reap_lingering_archives(u32 timeout_ms)
		{
			std::lock_guard lock(g_lingering_lock);
			const auto deadline = clock_type::now() + std::chrono::milliseconds(timeout_ms);

			std::erase_if(g_lingering, [&](const std::shared_ptr<pipeline_archive>& archive)
			{
				const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - clock_type::now()).count();
				return archive->stop(static_cast<u32>(std::max<s64>(left, 0)));
			});

			return g_lingering.empty();
		}

		void reap_lingering_archives_at_exit()
		{
			if (reap_lingering_archives(10'000))
			{
				return;
			}

			// Never hang the exit. Each thread owns its archive and releases it when it returns.
			std::lock_guard lock(g_lingering_lock);
			for (const auto& archive : g_lingering)
			{
				archive->detach();
			}

			g_lingering.clear();
		}

		void add_lingering_archive(std::shared_ptr<pipeline_archive> archive)
		{
			std::lock_guard lock(g_lingering_lock);

			// Registered after g_lingering_lock/g_lingering were constructed, so it runs before they are destroyed
			static const bool s_exit_hook = (std::atexit(reap_lingering_archives_at_exit), true);
			static_cast<void>(s_exit_hook);

			g_lingering.push_back(std::move(archive));
		}
	}

	void initialize_pipeline_archive()
	{
		shutdown_pipeline_archive();

		if (!g_render_device || !g_render_device->handle())
		{
			return;
		}

		if (!reap_lingering_archives(5000))
		{
			rsx_log.warning("Metal: the previous pipeline archive is still being written; the archive is disabled for this session");
			return;
		}

		if (const char* setting = ::getenv("RPCS3_METAL_PIPELINE_ARCHIVE"); setting && setting[0] == '0')
		{
			rsx_log.notice("Metal: the pipeline archive is disabled (RPCS3_METAL_PIPELINE_ARCHIVE=0)");
			return;
		}

		// Same conditions and root as the RSX shader cache (rsx::shaders_cache: <ppu cache>/shaders_cache/)
		if (g_cfg.video.disable_on_disk_shader_cache)
		{
			return;
		}

		const std::string cache_path = rpcs3::cache::get_ppu_cache();
		if (cache_path.empty())
		{
			return;
		}

		// Not inside pipelines/metal/v1.0: rsx::shaders_cache::load() deletes every unknown file there, and it only creates
		// raw/ when that directory does not exist yet (this runs first). "Remove shader cache" deletes shaders_cache/.
		// Fast math changes every binary, so each math mode keeps its own archives.
		const std::string directory = cache_path + "shaders_cache/metal_pipeline_archive/" +
			(g_cfg.video.disable_msl_fast_math ? "precise/" : "fast/");

		auto archive = std::make_shared<pipeline_archive>();
		if (!archive->init(g_render_device->handle(), directory))
		{
			return;
		}

		archive->start();

		std::lock_guard lock(g_archive_lock);
		g_archive = std::move(archive);
	}

	void on_pipeline_cache_preloaded()
	{
		if (const auto archive = get_archive())
		{
			archive->on_preload_finished(!Emu.IsStopped());
		}
	}

	void flush_pipeline_archive_async()
	{
		if (const auto archive = get_archive())
		{
			archive->request_final();
		}
	}

	void shutdown_pipeline_archive(u32 timeout_ms)
	{
		std::shared_ptr<pipeline_archive> archive;
		{
			std::lock_guard lock(g_archive_lock);
			archive = std::move(g_archive);
			g_archive.reset();
		}

		if (archive && !archive->stop(timeout_ms))
		{
			rsx_log.error("Metal: the pipeline archive is still being written after %u ms; letting it finish in the background", timeout_ms);
			add_lingering_archive(std::move(archive));
		}

		// Otherwise the Metal objects are released here (last reference)
	}

	MTL::RenderPipelineState* new_render_pipeline_state(const MTL4::RenderPipelineDescriptor* descriptor, NS::Error** error, u64 key)
	{
		if (const auto archive = get_archive())
		{
			return archive->build<MTL::RenderPipelineState>(descriptor, error, key,
				[](MTL4::Compiler* compiler, const MTL4::RenderPipelineDescriptor* desc, const MTL4::CompilerTaskOptions* options, NS::Error** err)
				{
					return compiler->newRenderPipelineState(desc, options, err);
				});
		}

		return g_render_device->compiler()->newRenderPipelineState(descriptor, nullptr, error);
	}

	MTL::ComputePipelineState* new_compute_pipeline_state(const MTL4::ComputePipelineDescriptor* descriptor, NS::Error** error, u64 key)
	{
		if (const auto archive = get_archive())
		{
			return archive->build<MTL::ComputePipelineState>(descriptor, error, key,
				[](MTL4::Compiler* compiler, const MTL4::ComputePipelineDescriptor* desc, const MTL4::CompilerTaskOptions* options, NS::Error** err)
				{
					return compiler->newComputePipelineState(desc, options, err);
				});
		}

		return g_render_device->compiler()->newComputePipelineState(descriptor, nullptr, error);
	}
}
