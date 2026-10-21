# Riko: what this package contains

- `backend/`: Riko's Python backend, frozen with PyInstaller, with the Python packages it imports. Each package keeps its
  own license; many carry it in a `*.dist-info` folder under `backend/_internal/`.
- `native/<backend>/`: llama.cpp (MIT, `llama.cpp-LICENSE` beside this file) built as Riko's in-process inference library,
  one folder per GPU backend (CUDA and Vulkan on Windows and Linux, Metal on macOS, and ROCm (HIP) in Linux packages built
  with it). The CUDA folder also holds NVIDIA's CUDA runtime and cuBLAS libraries, redistributed under the NVIDIA CUDA
  Toolkit license. The HIP folder uses the ROCm 7 installed on your machine (HIP runtime, hipBLAS, rocBLAS).
- `SHA256SUMS.json`: the hash of every file above when the package was built (macOS signing changes the binaries).

Not included: GPU drivers, the Vulkan loader, ROCm, model weights, voices or avatars. Install your GPU vendor's driver for the
backend you choose in setup; without one, llama.cpp runs on the CPU. Models and voices are downloaded or chosen by you.

Your data (configuration, memories, chat history, downloaded models and logs) lives in the data folder you choose at
first launch, not inside the application, and updating or uninstalling Riko leaves it in place.
