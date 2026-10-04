# Native emotion probe (private extension)

Generation remains in llama.cpp, loaded in the Python process through a small
C ABI DLL. No HTTP listener or llama-server child process is created. Native
server-context still owns its model, slots, templates and Responses parser.
Python's small probe uses CPU torch only for training/readout; no Transformers
generation backend or second copy of the main LLM is loaded. Julia-1 is the
teacher, CPU by default with optional CUDA via `emotion.device`.

The patch targets llama.cpp commit
`b92761a515ea31e852e7fbc1fad5f874b46f3718`. It is a private server extension,
not an upstream contribution. Upstream explicitly excludes activation APIs
from its supported server scope. Do not assume the patch applies to another
revision or that the stock WinGet binary has this feature.

### Minimal upstream changes

Only five upstream files are patched: the server CMake target, context `.cpp/.h`
for opt-in hidden-state capture, and task `.cpp/.h` for feature events aligned to
the preceding visible UTF-8 prefix. CUDA/Vulkan kernels, sampling, quantization,
tokenization and KV-cache algorithms stay upstream. The C ABI transport and its
exception/cancellation cleanup live in our separate `riko-native.cpp`.

Windows NVCC can repeatedly warn #221 when MSVC expands the upstream `INFINITY`
max-reduction sentinel. Release CUDA builds suppress that diagnostic through
`--diag-suppress=221`, without changing the sentinel or disabling other diagnostic
numbers. This suppresses all #221 warnings in that CUDA build, not just one line.
Remove that build flag when auditing new CUDA overflow warnings.

## Build

Use a separate checkout at the exact revision above. Apply `emotion-probe.patch`
with `git apply --check` first, then `git apply`. Build normally with CUDA or
Vulkan, for example:

```powershell
cmake -S .native/llama.cpp -B .native/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DRIKO_NATIVE_BRIDGE_SOURCE="$PWD/tools/llama_cpp/riko-native.cpp"
cmake --build .native/llama.cpp/build --config Release --target riko-native -j 2
```

A CPU-only build can check compilation and the bridge without a CUDA toolset:

```powershell
cmake -S .native/llama.cpp -B .native/llama.cpp/build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DRIKO_NATIVE_BRIDGE_SOURCE="$PWD/tools/llama_cpp/riko-native.cpp"
cmake --build .native/llama.cpp/build-cpu --config Release --target riko-native -j 2
```

This CPU DLL does not provide GPU main-model inference. The probe's CPU-only
setting is independent of the backend used to build llama.cpp.

Configure `runtime.native_library` to that build's `riko-native.dll`, keeping
its dependent llama/ggml DLLs beside it. The external llama-server backend is
removed; leaving `native_library` unset reports an actionable initialization error.
Existing mmap, quantization, KV cache, Flash Attention, slot scheduling and
Responses settings remain unchanged; routes are invoked as native functions,
not through a network. In-process mode cannot isolate a native crash from the
Python application. Live validation is still required for a given build and device.

## Enable (opt-in)

Add to your existing configuration; do not overwrite other settings:

```yaml
emotion:
  enabled: true
  device: cpu  # Optional cuda/cuda:0 for Julia only
  probe:
    enabled: true
    interval_tokens: 32
    hidden_units: [16384, 8192]
    rank: 32
    min_samples: 256
    retrain_every: 128
    max_samples: 4096
    epochs: 12
    min_agreement: 0.8
    min_macro_f1: 0.65
    max_rmse: 0.2
    min_confidence: 0.6
    use_for_expression: true
```

The probe is always CPU-only. Julia's optional CUDA execution requires a
CUDA-enabled torch wheel and compatible native Julia runtime; nothing installs
those packages automatically or changes the current configuration. Benchmark
Julia CUDA memory and contention before selecting it on an 8 GB GPU.

The Models settings tab exposes an Emotion probe interval slider (1-512 tokens,
default 32). Saving applies it to the loaded native context immediately without
changing other settings. Main LLM generation still evaluates every token.

## Capture and lifecycle

- The app enables the native hook only when the probe is enabled.
- At the configured generated-token interval, a single-token, non-speculative decode batch for
  slot 0 can capture its `result_norm` vector. Prefill and background slots are
  excluded. Mixed multi-slot batches are deliberately skipped rather than
  guessing tensor-row ownership. Pause-background-on-live is recommended. Very
  short visible replies may not reach an eligible sampling boundary.
- Captures use the existing evaluation callback, not embedding mode. No extra
  prefill outputs, weight changes or additional LLM pass are requested.
- A native capture copies one bounded F32 hidden vector, then averages it into
  256 features. The SSE event contains features and the byte length of the
  **previous visible prefix**, because the activation belongs to the consumed
  token, not the token subsequently sampled from it.
- Only visible-text parser transitions publish samples. Reasoning-only, tools,
  mismatched prefixes, malformed/nonfinite vectors and cancelled work are not
  admitted. UTF-8 offsets count bytes, not Python characters.
- Julia labels the same bounded visible prefix in a stateless call. Heuristic
  fallback is never accepted as a training label.
- Qualified predictions suppress assistant teacher events for that turn. A
  low-confidence sample clears that suppression and publishes its genuine Julia
  label as fallback. Late results from cancelled or superseded turns cannot
  replace the current expression.
- Sample queues and retained datasets are bounded. Training waits for idle
  foreground/background inference and yields between minibatches.
- Validation splits entire user turns by stable hash. Promotion requires at
  least 16 held-out samples and three held-out classes as well as agreement,
  macro-F1 and regression thresholds. These measure imitation of Julia, not
  independently established emotional correctness. Julia supplies arousal via
  intensity, so arousal is not an independently supervised signal here.
- A new GGUF checksum, server build, template, KV configuration or feature/
  teacher identity selects a separate dataset/checkpoint and starts unqualified.
  Native libraries and local teacher files are fingerprinted, so overwriting
  weights or rebuilding the bridge at the same path also isolates artifacts.
  Returning to an unchanged model can resume its qualified artifact and keep
  retraining as fresh samples accumulate. There is no cross-model weight reuse.
- Private activation/label/text datasets live in ignored
  `models/training/expression/<model-name>/<fingerprint>/`. Trained weights live
  in `models/<model-name>/expression probe/<fingerprint>/`. Legacy artifacts remain
  readable without deletion. Retained text and activations are sensitive local data.

The network has 24,576 hidden units with low-rank connections, not a dense
16,384-by-8,192 matrix. Width is configurable; quality must be evaluated.

## Performance and verification

Evaluation callbacks can split CUDA graphs and force synchronization on capture
steps. Preserving inference settings does **not** prove zero overhead. Before
enabling this permanently, compare identical seeded requests with capture off/
on and measure prefill latency, tokens/sec, peak VRAM, multi-slot correctness,
  and CPU probe latency. Also check actual Julia agreement on diverse held-out
turns, interruption handling and expression timing. No automatic model download,
training corpus fabrication, app launch or configuration change is performed.

The CPU DLL has compiled and loaded successfully on Windows/MSVC. Automated
tests use mocked native requests and teacher outputs to check UTF-8 streaming,
errors, cancellation, timeouts, cleanup ordering, CPU placement, stale-turn
rejection, fallback and artifact isolation. These checks do not establish live
hidden-state alignment, Julia training quality, CUDA compatibility or speed.
