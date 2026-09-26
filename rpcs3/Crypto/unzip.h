#pragma once

std::vector<u8> unzip(const void* src, usz size);

template <typename T>
inline std::vector<u8> unzip(const T& src)
{
	return unzip(src.data(), src.size());
}

// Unzip a whole gzip stream into dst, which must be exactly as large as the uncompressed data (e.g. the size
// recorded in the gzip trailer). Returns false unless the stream is complete and its CRC and length check out.
bool unzip_exact(const void* src, usz size, void* dst, usz dst_size);

bool unzip(const void* src, usz size, fs::file& out);

template <typename T>
inline bool unzip(const std::vector<u8>& src, fs::file& out)
{
	return unzip(src.data(), src.size(), out);
}

bool zip(const void* src, usz size, fs::file& out, bool multi_thread_it = false);

template <typename T>
inline bool zip(const T& src, fs::file& out)
{
	return zip(src.data(), src.size(), out);
}
