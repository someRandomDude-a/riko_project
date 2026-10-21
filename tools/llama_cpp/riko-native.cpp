// Private in-process bridge. No socket or HTTP listener is created.
#include "arg.h"
#include "common.h"
#include "server-context.h"
#include "server-http.h"
#include "json.h"
#include "log.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define RIKO_EXPORT __declspec(dllexport)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#define RIKO_EXPORT __attribute__((visibility("default")))
#include <dlfcn.h>
#endif

struct riko_runtime {
    common_params params;
    server_context context;
    std::unique_ptr<server_routes> routes;
    std::thread loop;
    std::atomic<bool> closing{false};

    ~riko_runtime() {
        closing.store(true);
        context.terminate();
        if (loop.joinable()) {
            loop.join();
        }
    }
};

using riko_output = int (*)(int, const char *, size_t, void *);
using riko_cancel = int (*)(void *);

namespace {

// llama.cpp logs why a load failed (OOM, no split buffers, KV type, bad GGUF) instead of returning it.
// While riko_create runs, keep its WARN/ERROR lines so the returned error carries the cause, and the
// layer-offload and flash-attention outcome so a successful create can report them. Every message still
// reaches the normal llama.cpp log (stderr).
struct riko_log_capture {
    std::mutex mutex;
    bool active = false;
    ggml_log_level level = GGML_LOG_LEVEL_NONE; // the level a GGML_LOG_LEVEL_CONT fragment continues
    std::string partial;
    std::deque<std::pair<ggml_log_level, std::string>> lines;
    std::vector<std::string> notes;
};

riko_log_capture & riko_log() {
    static riko_log_capture capture;
    return capture;
}

void riko_log_line(riko_log_capture & log, std::string line) {
    if (line.size() > 300) line.resize(300);
    // The --fit dry run logs below ERROR as DEBUG, so these come from the real load ("enabled" is INFO, "not supported" WARN).
    if ((line.rfind("load_tensors: offloaded ", 0) == 0 || line.rfind("resolve_fused_ops: Flash Attention", 0) == 0
            || line.rfind("riko-native: ", 0) == 0) && log.notes.size() < 4) {
        log.notes.push_back(line);
    }
    if (log.level == GGML_LOG_LEVEL_INFO) return;
    for (const auto & seen : log.lines) if (seen.second == line) return; // --fit loads the model twice
    log.lines.emplace_back(log.level, std::move(line));
    if (log.lines.size() > 12) log.lines.pop_front();
}

void riko_log_callback(ggml_log_level level, const char * text, void * user_data) {
    common_log_default_callback(level, text, user_data);
    if (!text) return;
    auto & log = riko_log();
    std::lock_guard<std::mutex> lock(log.mutex);
    if (!log.active) return;
    if (level != GGML_LOG_LEVEL_CONT) {
        if (!log.partial.empty()) riko_log_line(log, log.partial);
        log.partial.clear();
        log.level = level;
    }
    if (log.level != GGML_LOG_LEVEL_WARN && log.level != GGML_LOG_LEVEL_ERROR && log.level != GGML_LOG_LEVEL_INFO) return;
    log.partial += text;
    for (size_t end; (end = log.partial.find('\n')) != std::string::npos; log.partial.erase(0, end + 1)) {
        riko_log_line(log, log.partial.substr(0, end));
    }
    if (log.partial.size() > 4096) log.partial.clear();
}

// Captures for one riko_create call and stops on every exit path.
struct riko_log_scope {
    riko_log_scope() {
        auto & log = riko_log();
        std::lock_guard<std::mutex> lock(log.mutex);
        log.active = true;
        log.level = GGML_LOG_LEVEL_NONE;
        log.partial.clear();
        log.lines.clear();
        log.notes.clear();
    }
    ~riko_log_scope() {
        auto & log = riko_log();
        std::lock_guard<std::mutex> lock(log.mutex);
        log.active = false;
    }
    // The failure, then llama.cpp's errors before its warnings.
    std::string explain(const char * message) const {
        auto & log = riko_log();
        std::lock_guard<std::mutex> lock(log.mutex);
        if (!log.partial.empty()) riko_log_line(log, log.partial);
        std::string result(message);
        if (!log.lines.empty()) result += "; llama.cpp reported:";
        for (const auto level : {GGML_LOG_LEVEL_ERROR, GGML_LOG_LEVEL_WARN}) {
            for (const auto & line : log.lines) if (line.first == level) result += "\n" + line.second;
        }
        return result;
    }
    std::string notes() const {
        auto & log = riko_log();
        std::lock_guard<std::mutex> lock(log.mutex);
        std::string result;
        for (const auto & note : log.notes) result += (result.empty() ? "" : "\n") + note;
        return result;
    }
};

// The directory this library was loaded from, or empty when the loader cannot say.
std::filesystem::path riko_directory() {
    std::error_code failed;
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&riko_directory), &module)) return {};
    std::wstring path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, path.data(), (DWORD) path.size());
    if (length == 0 || length >= path.size()) return {};
    path.resize(length);
    const auto file = std::filesystem::absolute(std::filesystem::path(path), failed);
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&riko_directory), &info) || !info.dli_fname) return {};
    const auto file = std::filesystem::absolute(std::filesystem::path(info.dli_fname), failed);
#endif
    return failed ? std::filesystem::path() : file.parent_path();
}

// A GGML_BACKEND_DL build keeps each ggml backend in a module ([lib]ggml-<name>[-<variant>].so|dll; .so on macOS too) that
// ggml looks for beside the executable (python, riko-backend) and in the working directory, never beside this library,
// where a bundle keeps them. Load them from here first. A build that links its backends registers them on first use, so
// the registry is not empty and this loads nothing; llama_backend_init loads ggml's defaults only into an empty registry.
void riko_load_backends() {
    if (ggml_backend_reg_count() > 0) return;
    const auto directory = riko_directory();
    if (directory.empty()) return;
#if defined(_WIN32)
    const std::wstring prefix = L"ggml-", extension = L".dll";
#else
    const std::string prefix = "libggml-", extension = ".so";
#endif
    const char * cpu_side[] = {"ggml-base", "ggml-cpu", "ggml-blas"};
    std::vector<std::filesystem::path> modules;
    std::error_code failed;
    for (std::filesystem::directory_iterator it(directory, failed), end; !failed && it != end; it.increment(failed)) {
        const auto name = it->path().filename().native();
        if (name.rfind(prefix, 0) == 0 && it->path().extension().native() == extension) modules.push_back(it->path());
    }
#if defined(_WIN32)
    // ggml loads a module with LoadLibraryW, which looks for its own dependencies (cudart, cuBLAS) beside the executable
    // and on PATH. Loaded here first with the module's directory searched, ggml's load of the same path reuses it.
    std::vector<HMODULE> preloaded;
    for (const auto & module : modules) {
        if (HMODULE handle = LoadLibraryExW(module.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) preloaded.push_back(handle);
    }
#endif
    const auto utf8 = directory.u8string();
    ggml_backend_load_all_from_path(std::string(utf8.begin(), utf8.end()).c_str());
#if defined(_WIN32)
    for (HMODULE handle : preloaded) FreeLibrary(handle); // ggml holds its own reference to the backends it kept
#endif
    std::string others; // the GPU modules: all but ggml-base and the CPU-side ones
    for (const auto & module : modules) {
        const auto name = module.filename().u8string();
        const std::string stem(name.begin(), name.end() - (std::ptrdiff_t) extension.size());
        if (std::none_of(std::begin(cpu_side), std::end(cpu_side), [&](const char * part) { return stem.find(part) != std::string::npos; })) {
            others += (others.empty() ? "" : ", ") + stem;
        }
    }
    if (others.empty()) return;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) return;
    }
    // The module did not load (a missing loader such as libvulkan, or ROCm's libamdhip64 for ggml-hip) or found no device
    // (no driver): say so rather than run on the CPU unannounced. riko_create returns it with its notes.
    const std::string warning = "riko-native: no GPU device from " + others + "; llama.cpp runs on the CPU (is the GPU driver installed? ggml-hip also needs ROCm)\n";
    riko_log_callback(GGML_LOG_LEVEL_WARN, warning.c_str(), nullptr);
}

void riko_error(char * error, size_t capacity, const std::string & message) {
    if (!error || capacity == 0) return;
    const size_t length = std::min(capacity - 1, message.size());
    std::memcpy(error, message.data(), length);
    error[length] = 0;
}

// ggml's backends initialise once per process, inside the first riko_create (successful or not), so its log scope keeps
// their warnings (riko_load_backends' "no GPU device"). riko_devices only reads what this set up.
std::once_flag riko_initialized;
std::atomic<bool> riko_ready{false};

void riko_init() {
    std::call_once(riko_initialized, []() {
        common_init(); llama_log_set(riko_log_callback, nullptr); riko_load_backends(); llama_backend_init();
        riko_ready.store(true);
    });
}

const char * riko_device_type(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default:                             return "other";
    }
}

} // namespace

extern "C" {

RIKO_EXPORT void * riko_create(const char * arguments, int interval, char * error, size_t capacity) {
    riko_log_scope scope;
    try {
        if (!arguments) {
            throw std::runtime_error("missing native arguments");
        }
        riko_init();
        const auto input = json::parse(arguments);
        if (!input.is_array() || input.empty() || input.size() > 256) {
            throw std::runtime_error("invalid native arguments");
        }
        std::vector<std::string> strings = input.get<std::vector<std::string>>();
        std::vector<char *> argv;
        for (auto & value : strings) {
            argv.push_back(value.data());
        }
        // Match the normal argc/argv contract without including the terminator in argc.
        argv.push_back(nullptr);
        auto runtime = std::make_unique<riko_runtime>();
        if (!common_params_parse((int) strings.size(), argv.data(), runtime->params, LLAMA_EXAMPLE_SERVER)) {
            throw std::runtime_error("llama.cpp rejected the native arguments (its reason is on the backend's stderr)");
        }
        runtime->context.set_emotion_probe_interval(interval);
        runtime->routes = std::make_unique<server_routes>(runtime->params, runtime->context);
        if (!runtime->context.load_model(runtime->params)) {
            throw std::runtime_error("native model initialization failed");
        }
        runtime->routes->update_meta(runtime->context);
        riko_error(error, capacity, scope.notes()); // on success the buffer reports the offload and flash attention
        runtime->loop = std::thread([ptr = runtime.get()]() {
            try {
                ptr->context.start_loop();
            } catch (const std::exception & exception) {
                std::fprintf(stderr, "riko-native context loop failed: %s\n", exception.what());
            } catch (...) {
                std::fprintf(stderr, "riko-native context loop failed: unknown exception\n");
            }
            ptr->closing.store(true);
        });
        return runtime.release();
    } catch (const std::exception & exception) {
        riko_error(error, capacity, scope.explain(exception.what()));
        return nullptr;
    } catch (...) {
        riko_error(error, capacity, scope.explain("unknown native initialization failure"));
        return nullptr;
    }
}

RIKO_EXPORT int riko_request(void * handle, const char * path, const char * body, riko_output output, riko_cancel cancel, void * data) {
    if (!handle || !path || !body || !output || !cancel) {
        return -1;
    }
    auto * runtime = static_cast<riko_runtime *>(handle);
    std::function<bool()> stop;
    std::unique_ptr<server_http_req> request;
    server_http_res_ptr response;
    try {
        stop = [&]() { return runtime->closing.load() || cancel(data) != 0; };
        request.reset(new server_http_req{{}, {}, path, "", body, {}, stop});
        if (stop()) {
            return 1;
        }
        const std::string route(path);
        server_http_context::handler_t handler;
        if (route == "/v1/responses") handler = runtime->routes->post_responses_oai;
        else if (route == "/apply-template") handler = runtime->routes->post_apply_template;
        else if (route == "/tokenize") handler = runtime->routes->post_tokenize;
        else if (route == "/props") handler = runtime->routes->get_props;
        else if (route == "/slots") handler = runtime->routes->get_slots;
        else throw std::runtime_error("unsupported native operation");
        response = handler(*request);
        if (!response) {
            throw std::runtime_error("empty native response");
        }
        // The native stream owns the first chunk and flushes it through next().
        const bool streaming = response->is_stream();
        const std::string initial = streaming ? "" : response->data;
        if (output(response->status, initial.data(), initial.size(), data) != 0 && streaming) {
            bool more = true;
            while (more && !stop()) {
                std::string chunk;
                more = response->next(chunk);
                if (!chunk.empty() && output(response->status, chunk.data(), chunk.size(), data) == 0) {
                    break;
                }
            }
        }
        // Move ownership first: a throwing cleanup must never be invoked twice.
        auto completed = std::move(response);
        completed->on_complete();
        return 0;
    } catch (const std::exception & exception) {
        if (response) {
            try { response->on_complete(); } catch (...) {}
            response.reset();
        }
        const std::string error = json{{"error", {{"message", exception.what()}}}}.dump();
        output(500, error.data(), error.size(), data);
        return -1;
    } catch (...) {
        if (response) {
            try { response->on_complete(); } catch (...) {}
            response.reset();
        }
        const char * error = "{\"error\":{\"message\":\"unknown native request failure\"}}";
        output(500, error, std::strlen(error), data);
        return -1;
    }
}

RIKO_EXPORT int riko_set_interval(void * handle, int interval) {
    if (!handle || interval < 1 || interval > 512) return -1;
    static_cast<riko_runtime *>(handle)->context.set_emotion_probe_interval(interval);
    return 0;
}

RIKO_EXPORT void riko_stop(void * handle) {
    if (!handle) return;
    static_cast<riko_runtime *>(handle)->closing.store(true);
}

RIKO_EXPORT void riko_destroy(void * handle) {
    delete static_cast<riko_runtime *>(handle);
}

// ggml's devices, NUL-terminated in the caller's buffer, as {"version":1,"devices":[{"index":i,"name":"CUDA0",
// "description":"NVIDIA GeForce RTX 4090","backend":"CUDA","type":"cpu|gpu|igpu|accel|other","device_id":"0000:01:00.0"|null,
// "memory_free":bytes,"memory_total":bytes}]}. Returns the JSON's length; a result >= capacity means the buffer was too small
// and nothing was written (call again with a larger one). -2: no riko_create has run in this process yet (it initialises the
// backends); -1: a failure, its message in the buffer. Any thread, any time after that, also while contexts load, run or are
// destroyed: the registry is fixed once initialised, and each device reports its memory as llama.cpp's own load asks for it
// (cudaMemGetInfo/hipMemGetInfo, Vulkan's memory budget, Metal's working set, which counts this process's allocations only).
RIKO_EXPORT int riko_devices(char * buffer, size_t capacity) {
    try {
        if (!riko_ready.load()) {
            riko_error(buffer, capacity, "riko_devices before riko_create: the backends are not initialised");
            return -2;
        }
        json devices = json::array();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
            devices.push_back({{"index", i}, {"name", props.name ? props.name : ""}, {"description", props.description ? props.description : ""},
                {"backend", reg ? ggml_backend_reg_name(reg) : ""}, {"type", riko_device_type(props.type)},
                {"device_id", props.device_id ? json(props.device_id) : json(nullptr)},
                {"memory_free", (unsigned long long) props.memory_free}, {"memory_total", (unsigned long long) props.memory_total}});
        }
        const std::string text = json{{"version", 1}, {"devices", devices}}.dump_safe(); // a driver's description in a legacy code page must not fail the call
        if (text.size() > (size_t) std::numeric_limits<int>::max()) throw std::runtime_error("device list too large");
        if (buffer && text.size() < capacity) {
            std::memcpy(buffer, text.data(), text.size());
            buffer[text.size()] = 0;
        }
        return (int) text.size();
    } catch (const std::exception & exception) {
        riko_error(buffer, capacity, exception.what());
        return -1;
    } catch (...) {
        riko_error(buffer, capacity, "unknown native device failure");
        return -1;
    }
}

}
