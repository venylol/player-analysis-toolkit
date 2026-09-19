// C++17 16-thread materializer for independent current-board CNN pretraining.
// TXT input is UTF-8; all tokens in the dataset contract are ASCII subsets.

#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace fs = std::filesystem;

constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kRecordSize = 25;
constexpr std::size_t kBoardBytes = 16;
constexpr std::uint64_t kFull = ~std::uint64_t{0};
constexpr std::uint64_t kNotA = 0xfefefefefefefefeULL;
constexpr std::uint64_t kNotH = 0x7f7f7f7f7f7f7f7fULL;

#pragma pack(push, 1)
struct ShardHeader {
    char magic[8];
    std::uint16_t version;
    std::uint16_t record_size;
    std::uint64_t records;
    char reserved[12];
};
#pragma pack(pop)

static_assert(sizeof(ShardHeader) == 32);

struct Options {
    fs::path source_dir;
    fs::path output_dir;
    unsigned threads = 16;
    int first_index = 0;
    int last_index = 25;
    std::uint64_t max_samples = 0;
};

struct ShardResult {
    std::string file_name;
    fs::path source;
    fs::path shard;
    std::uint64_t source_bytes = 0;
    std::uint64_t records = 0;
    std::uint64_t shard_bytes = 0;
    std::string sha256;
    bool reused = false;
};

std::mutex g_output_mutex;

std::string utf8(const fs::path& path) {
    const std::wstring value = path.wstring();
    if (value.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                         static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("failed to encode path as UTF-8");
    std::string output(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                        output.data(), size, nullptr, nullptr);
    return output;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &value);
    std::ostringstream output;
    output << std::put_time(&local, "%Y%m%dT%H%M%S");
    return output.str();
}

std::string sha256_file(const fs::path& path) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0;
    DWORD hash_size = 0;
    DWORD returned = 0;
    std::vector<unsigned char> object;
    std::vector<unsigned char> digest;
    auto check = [](NTSTATUS status, const char* operation) {
        if (status < 0) throw std::runtime_error(std::string("BCrypt failure: ") + operation);
    };
    try {
        check(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0), "open SHA-256");
        check(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                                reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &returned, 0),
              "get object length");
        check(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                                reinterpret_cast<PUCHAR>(&hash_size), sizeof(hash_size), &returned, 0),
              "get hash length");
        object.resize(object_size);
        digest.resize(hash_size);
        check(BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0), "create hash");
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open shard for SHA-256: " + utf8(path));
        std::vector<char> buffer(1024 * 1024);
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = input.gcount();
            if (count > 0) {
                check(BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                     static_cast<ULONG>(count), 0), "hash data");
            }
        }
        check(BCryptFinishHash(hash, digest.data(), hash_size, 0), "finish hash");
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
    std::ostringstream output;
    for (const unsigned char byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return output.str();
}

std::uint64_t shift_bits(std::uint64_t bits, int direction) {
    switch (direction) {
        case 0: return (bits & kNotH) << 1;  // east
        case 1: return (bits & kNotA) >> 1;  // west
        case 2: return bits << 8;            // south
        case 3: return bits >> 8;            // north
        case 4: return (bits & kNotH) << 9;  // south-east
        case 5: return (bits & kNotA) << 7;  // south-west
        case 6: return (bits & kNotH) >> 7;  // north-east
        case 7: return (bits & kNotA) >> 9;  // north-west
        default: throw std::logic_error("invalid direction");
    }
}

std::uint64_t legal_moves(std::uint64_t player, std::uint64_t opponent) {
    const std::uint64_t empty = ~(player | opponent) & kFull;
    std::uint64_t moves = 0;
    for (int direction = 0; direction < 8; ++direction) {
        std::uint64_t frontier = shift_bits(player, direction) & opponent;
        std::uint64_t captured = frontier;
        for (int extension = 0; extension < 5; ++extension) {
            frontier = shift_bits(frontier, direction) & opponent;
            captured |= frontier;
        }
        moves |= shift_bits(captured, direction) & empty;
    }
    return moves;
}

void append_record(std::vector<char>& buffer, const std::string& line, std::uint64_t line_number) {
    std::string_view view(line);
    if (!view.empty() && view.back() == '\r') view.remove_suffix(1);
    if (view.size() < 66 || view[64] != ' ') {
        throw std::runtime_error("invalid record at line " + std::to_string(line_number));
    }
    std::array<unsigned char, kBoardBytes> packed{};
    std::uint64_t player = 0;
    std::uint64_t opponent = 0;
    for (std::size_t square = 0; square < 64; ++square) {
        unsigned char code = 0;
        if (view[square] == 'X') {
            code = 1;
            player |= std::uint64_t{1} << square;
        } else if (view[square] == 'O') {
            code = 2;
            opponent |= std::uint64_t{1} << square;
        } else if (view[square] != '-') {
            throw std::runtime_error("invalid board character at line " + std::to_string(line_number));
        }
        packed[square / 4] |= static_cast<unsigned char>(code << ((square % 4) * 2));
    }
    int score = 0;
    const char* begin = view.data() + 65;
    const char* end = view.data() + view.size();
    const auto parsed = std::from_chars(begin, end, score);
    if (parsed.ec != std::errc{} || parsed.ptr != end || score < -64 || score > 64) {
        throw std::runtime_error("invalid score at line " + std::to_string(line_number));
    }
    const std::uint64_t legal = legal_moves(player, opponent);
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(packed.data()),
                  reinterpret_cast<const char*>(packed.data() + packed.size()));
    for (int byte = 0; byte < 8; ++byte) {
        buffer.push_back(static_cast<char>((legal >> (byte * 8)) & 0xff));
    }
    buffer.push_back(static_cast<char>(static_cast<std::int8_t>(score)));
}

ShardHeader read_header(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    ShardHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        throw std::runtime_error("cannot read shard header: " + utf8(path));
    }
    if (std::string(header.magic, 8) != "BCNNDS01" || header.version != kVersion ||
        header.record_size != kRecordSize) {
        throw std::runtime_error("invalid shard header: " + utf8(path));
    }
    return header;
}

void write_shard_manifest(const ShardResult& result, const fs::path& manifest_path,
                          const std::string& status, const std::string& attempt_id = "") {
    if (fs::exists(manifest_path)) throw std::runtime_error("refusing to overwrite manifest: " + utf8(manifest_path));
    std::ofstream output(manifest_path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write UTF-8 manifest: " + utf8(manifest_path));
    output << "{\n"
           << "  \"status\": \"" << status << "\",\n"
           << "  \"generator\": \"board-cnn-cpp17-v1\",\n"
           << "  \"attemptId\": \"" << json_escape(attempt_id) << "\",\n"
           << "  \"source\": \"" << json_escape(utf8(result.source)) << "\",\n"
           << "  \"sourceBytes\": " << result.source_bytes << ",\n"
           << "  \"records\": " << result.records << ",\n"
           << "  \"bytes\": " << result.shard_bytes << ",\n"
           << "  \"sha256\": \"" << result.sha256 << "\",\n"
           << "  \"shard\": \"" << json_escape(utf8(result.shard)) << "\",\n"
           << "  \"recordLayout\": {\"board2BitBytes\": 16, \"legalMoveUint64Bytes\": 8, \"scoreInt8Bytes\": 1},\n"
           << "  \"encoding\": \"UTF-8\"\n"
           << "}\n";
}

ShardResult materialize_one(const Options& options, int index) {
    std::ostringstream file_name_builder;
    file_name_builder << std::setw(7) << std::setfill('0') << index << ".txt";
    ShardResult result;
    result.file_name = file_name_builder.str();
    result.source = fs::absolute(options.source_dir / fs::path(result.file_name));
    const std::string stem = result.file_name.substr(0, 7);
    result.shard = fs::absolute(options.output_dir / fs::path(stem + ".bcnn"));
    if (!fs::is_regular_file(result.source)) throw std::runtime_error("source missing: " + utf8(result.source));
    result.source_bytes = fs::file_size(result.source);

    if (fs::exists(result.shard)) {
        const ShardHeader header = read_header(result.shard);
        const std::uint64_t expected = sizeof(ShardHeader) + header.records * kRecordSize;
        if (fs::file_size(result.shard) != expected) {
            throw std::runtime_error("existing completed shard size mismatch: " + utf8(result.shard));
        }
        result.records = header.records;
        result.shard_bytes = expected;
        result.sha256 = sha256_file(result.shard);
        result.reused = true;
        return result;
    }

    const std::ostringstream thread_id_builder = [&]() {
        std::ostringstream value;
        value << std::this_thread::get_id();
        return value;
    }();
    const std::string attempt_id = timestamp() + "-t" + thread_id_builder.str();
    const fs::path attempt = options.output_dir / fs::path(stem + ".attempt-" + attempt_id + ".part");
    const fs::path attempt_manifest = fs::path(attempt.wstring() + L".manifest.json");
    try {
        std::ifstream input(result.source, std::ios::binary);
        std::ofstream output(attempt, std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open source or attempt output");
        ShardHeader header{{'B','C','N','N','D','S','0','1'}, kVersion, kRecordSize, 0, {}};
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        std::vector<char> buffer;
        buffer.reserve(kRecordSize * 65536ULL);
        std::string line;
        std::uint64_t line_number = 0;
        while (std::getline(input, line)) {
            if (options.max_samples && result.records >= options.max_samples) break;
            ++line_number;
            append_record(buffer, line, line_number);
            ++result.records;
            if (buffer.size() >= kRecordSize * 65536ULL) {
                output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                buffer.clear();
            }
        }
        if (!buffer.empty()) output.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (!input.eof() && !options.max_samples) throw std::runtime_error("source read failure");
        header.records = result.records;
        output.seekp(0);
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.flush();
        if (!output) throw std::runtime_error("attempt write failure");
        output.close();
        result.shard_bytes = fs::file_size(attempt);
        const std::uint64_t expected = sizeof(ShardHeader) + result.records * kRecordSize;
        if (result.shard_bytes != expected) throw std::runtime_error("attempt shard size mismatch");
        result.sha256 = sha256_file(attempt);
        fs::rename(attempt, result.shard);
        const fs::path final_manifest = fs::path(result.shard.wstring() + L".manifest.json");
        write_shard_manifest(result, final_manifest, "complete", attempt_id);
        return result;
    } catch (const std::exception& error) {
        std::ofstream evidence(attempt_manifest, std::ios::binary);
        if (evidence) {
            evidence << "{\n  \"status\": \"failed\",\n  \"generator\": \"board-cnn-cpp17-v1\",\n"
                     << "  \"attemptId\": \"" << json_escape(attempt_id) << "\",\n"
                     << "  \"source\": \"" << json_escape(utf8(result.source)) << "\",\n"
                     << "  \"attemptPath\": \"" << json_escape(utf8(attempt)) << "\",\n"
                     << "  \"recordsWritten\": " << result.records << ",\n"
                     << "  \"error\": \"" << json_escape(error.what()) << "\",\n"
                     << "  \"encoding\": \"UTF-8\"\n}\n";
        }
        throw;
    }
}

void write_overall_manifest(const Options& options, const std::vector<ShardResult>& results) {
    const fs::path path = options.output_dir / L"manifest.json";
    if (fs::exists(path)) throw std::runtime_error("refusing to overwrite existing overall manifest: " + utf8(path));
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write overall UTF-8 manifest");
    output << "{\n  \"format\": \"board-cnn-shards-v1\",\n"
           << "  \"generator\": \"board-cnn-cpp17-v1\",\n"
           << "  \"sourceDirectory\": \"" << json_escape(utf8(fs::absolute(options.source_dir))) << "\",\n"
           << "  \"outputDirectory\": \"" << json_escape(utf8(fs::absolute(options.output_dir))) << "\",\n"
           << "  \"threads\": " << options.threads << ",\n"
           << "  \"maxSamplesPerFile\": " << options.max_samples << ",\n"
           << "  \"dataContract\": {\"planeOrder\": [\"current_empty\", \"current_X\", \"current_O\"], "
              "\"boardOrder\": \"a1,b1,...,h8\", \"valueTarget\": \"score/64.0\"},\n"
           << "  \"shards\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& item = results[i];
        output << "    {\"status\": \"complete\", \"source\": \"" << json_escape(utf8(item.source))
               << "\", \"sourceBytes\": " << item.source_bytes
               << ", \"records\": " << item.records << ", \"bytes\": " << item.shard_bytes
               << ", \"sha256\": \"" << item.sha256 << "\", \"shard\": \""
               << json_escape(utf8(item.shard)) << "\", \"reused\": " << (item.reused ? "true" : "false") << "}";
        output << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"encoding\": \"UTF-8\"\n}\n";
}

Options parse_options(int argc, wchar_t** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::wstring argument = argv[i];
        auto value = [&]() -> std::wstring {
            if (++i >= argc) throw std::runtime_error("missing command-line option value");
            return argv[i];
        };
        if (argument == L"--source-dir") options.source_dir = value();
        else if (argument == L"--output-dir") options.output_dir = value();
        else if (argument == L"--threads") options.threads = static_cast<unsigned>(std::stoul(value()));
        else if (argument == L"--first-index") options.first_index = std::stoi(value());
        else if (argument == L"--last-index") options.last_index = std::stoi(value());
        else if (argument == L"--max-samples") options.max_samples = std::stoull(value());
        else throw std::runtime_error("unknown argument");
    }
    if (options.source_dir.empty() || options.output_dir.empty()) throw std::runtime_error("--source-dir and --output-dir are required");
    if (options.threads == 0 || options.first_index < 0 || options.last_index > 25 || options.first_index > options.last_index) {
        throw std::runtime_error("invalid threads or file-index range");
    }
    return options;
}

int wmain(int argc, wchar_t** argv) {
    SetConsoleOutputCP(CP_UTF8);
    try {
        const Options options = parse_options(argc, argv);
        fs::create_directories(options.output_dir);
        const int file_count = options.last_index - options.first_index + 1;
        std::vector<ShardResult> results(static_cast<std::size_t>(file_count));
        std::atomic<int> next{0};
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::string first_error;
        const unsigned worker_count = (std::min)(options.threads, static_cast<unsigned>(file_count));
        std::vector<std::thread> workers;
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&]() {
                while (!failed.load()) {
                    const int local = next.fetch_add(1);
                    if (local >= file_count) return;
                    try {
                        results[static_cast<std::size_t>(local)] = materialize_one(options, options.first_index + local);
                        const auto& result = results[static_cast<std::size_t>(local)];
                        std::lock_guard<std::mutex> lock(g_output_mutex);
                        std::cout << result.file_name << " records=" << result.records
                                  << " bytes=" << result.shard_bytes << " sha256=" << result.sha256
                                  << (result.reused ? " reused" : " generated") << std::endl;
                    } catch (const std::exception& error) {
                        failed.store(true);
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (first_error.empty()) first_error = error.what();
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();
        if (failed.load()) throw std::runtime_error(first_error);
        write_overall_manifest(options, results);
        std::uint64_t total_records = 0;
        std::uint64_t total_bytes = 0;
        for (const auto& result : results) {
            total_records += result.records;
            total_bytes += result.shard_bytes;
        }
        std::cout << "complete files=" << results.size() << " records=" << total_records
                  << " shard_bytes=" << total_bytes << " threads=" << worker_count << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
}
