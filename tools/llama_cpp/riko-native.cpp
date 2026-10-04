// Private in-process bridge. No socket or HTTP listener is created.
#include "arg.h"
#include "common.h"
#include "server-context.h"
#include "server-http.h"
#include "json.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define RIKO_EXPORT __declspec(dllexport)
#else
#define RIKO_EXPORT __attribute__((visibility("default")))
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

extern "C" {

RIKO_EXPORT void * riko_create(const char * arguments, int interval, char * error, size_t capacity) {
    try {
        if (!arguments) {
            throw std::runtime_error("missing native arguments");
        }
        static std::once_flag initialized;
        std::call_once(initialized, []() { common_init(); llama_backend_init(); });
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
            throw std::runtime_error("invalid llama.cpp parameters");
        }
        runtime->context.set_emotion_probe_interval(interval);
        runtime->routes = std::make_unique<server_routes>(runtime->params, runtime->context);
        if (!runtime->context.load_model(runtime->params)) {
            throw std::runtime_error("native model initialization failed");
        }
        runtime->routes->update_meta(runtime->context);
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
        if (error && capacity > 0) {
            const size_t length = std::min(capacity - 1, std::strlen(exception.what()));
            std::memcpy(error, exception.what(), length);
            error[length] = 0;
        }
        return nullptr;
    } catch (...) {
        if (error && capacity > 0) {
            const char * message = "unknown native initialization failure";
            const size_t length = std::min(capacity - 1, std::strlen(message));
            std::memcpy(error, message, length);
            error[length] = 0;
        }
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

}
