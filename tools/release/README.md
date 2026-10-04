# Desktop distribution

The manual/tag-triggered `.github/workflows/release.yml` builds Windows x64 NSIS
and Linux x64 AppImage/deb packages. Each includes frozen Python, Electron and
separate pinned llama.cpp CUDA/Vulkan libraries. It uploads CI artifacts; it does
not publish a GitHub release automatically. Builds must pass loader/ABI smoke
checks. These do not replace GPU/model/audio tests on clean machines.

Windows lets the user select the application installation directory. The private
data directory is selected separately by the first-launch wizard on both systems
(AppImage has no installer). Its location is recorded under Electron userData.
Upgrades/uninstall must not remove it. Hugging Face cache, probe data, memories,
config and logs use this directory. Local GGUF selections remain at their original
paths. No model weights, personal config, .env, memories or unlicensed avatar/voice
assets are copied from a developer's machine.

CUDA is preferred when nvidia-smi detects NVIDIA; Vulkan is also included. Users
need compatible GPU drivers, not a build toolchain. No silent backend fallback.
Python torch is CPU-only in these packages (Julia, embeddings and probe training);
CUDA main-model inference is provided by the native llama.cpp backend, not torch.
Linux packages currently require the system Vulkan loader, PortAudio, libsndfile
and an audio/desktop stack. CUDA runtime/cuBLAS are copied from the toolkit; do
not redistribute NVIDIA drivers. Review NVIDIA's redistributable list/EULA,
Python package licenses, Vulkan SDK/loader notices and avatar/model/voice licenses
before public distribution. Signing certificates and publishing credentials are
not configured; unsigned Windows installers will trigger trust warnings.

GPT-SoVITS is an optional externally installed API executable. Its executable,
argument list, URL and reference audio are editable in Settings. Auto-start is
explicitly opt-in in packaged mode; the app terminates its owned direct process,
not unrelated services or process trees. Service readiness is not guaranteed at
launch. External MCP tools, FFmpeg media decoding and Discord must also be tested
and provisioned separately; packaging Python does not install arbitrary tools.

Release dependency versions (except pinned llama.cpp/build tools) are not yet fully
locked. CI builds need clean-machine acceptance tests and a license/SBOM audit
before being described as production-ready. CUDA/Vulkan hardware execution is not
verified by the source tests. No installer or GPU build was run on the developer
machine while adding this pipeline.
