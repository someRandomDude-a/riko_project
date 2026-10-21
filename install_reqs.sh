#!/usr/bin/env bash
# Installs the Python runtime into the active environment: activate the virtual environment first, or set PYTHON.
# The PyTorch wheels follow the machine: PyPI on macOS (Metal/MPS), CUDA when nvidia-smi sees a GPU, ROCm when
# rocminfo exists on Linux, CPU otherwise. Override with RIKO_TORCH_INDEX=default|cpu|cu126|cu130|rocm7.2|...
set -euo pipefail
cd "$(dirname "$0")"

# Without an active environment, prefer this repository's .venv over whatever python is first on PATH.
if [ -z "${PYTHON:-}" ]; then
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python
  elif [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/Scripts/python.exe ]; then PYTHON=.venv/Scripts/python.exe
  else PYTHON="$(command -v python || command -v python3 || true)"; fi
fi
if [ -z "$PYTHON" ]; then echo "No Python found: activate the virtual environment or set PYTHON." >&2; exit 1; fi
echo "Installing into $PYTHON"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "$PYTHON is not a working Python 3.11 or newer." >&2; exit 1; }
if command -v uv >/dev/null 2>&1; then pip_install() { uv pip install --python "$PYTHON" "$@"; }
else pip_install() { "$PYTHON" -m pip install "$@"; }; fi

index="${RIKO_TORCH_INDEX:-}"
if [ -z "$index" ]; then
  system="$(uname -s)"
  if [ "$system" = Darwin ]; then index=default
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    major="${cap%%.*}"; minor="${cap#*.}"; old_gpu=no; blackwell=no
    case "$major" in ''|*[!0-9]*) ;; *)
      if [ "$major" -lt 7 ] || { [ "$major" -eq 7 ] && [ "$minor" -lt 5 ] 2>/dev/null; }; then old_gpu=yes; fi
      if [ "$major" -ge 10 ]; then blackwell=yes; fi ;;
    esac
    # CUDA 13 wheels need driver 580+ and compute capability 7.5+ (Turing or newer). CUDA 12.6 still runs
    # Maxwell to Volta but has no Blackwell kernels, so a Blackwell GPU needs the newer driver.
    if [ "$old_gpu" = yes ]; then index=cu126
    elif [ "${driver%%.*}" -ge 580 ] 2>/dev/null; then index=cu130
    elif [ "$blackwell" = yes ]; then
      echo "This GPU (compute capability $cap) needs NVIDIA driver 580 or newer for PyTorch: update the driver, or set RIKO_TORCH_INDEX." >&2; exit 1
    else index=cu126; fi
  elif [ "$system" = Linux ] && { command -v rocminfo >/dev/null 2>&1 || [ -x /opt/rocm/bin/rocminfo ]; }; then index=rocm7.2
  else index=cpu; fi
fi
case "$index" in
  default) pip_install torch ;;
  cpu|xpu|cu[0-9]*|rocm[0-9]*) pip_install torch --index-url "https://download.pytorch.org/whl/$index" ;;
  *) echo "Unknown RIKO_TORCH_INDEX: $index" >&2; exit 1 ;;
esac
pip_install -r requirements-runtime.txt
# Wake enrollment uses only EfficientWord-Net's ONNX module; PyAudio and its other declared dependencies are unused.
pip_install --no-deps EfficientWord-Net
"$PYTHON" - <<'PY'
import torch
if torch.cuda.is_available():
    try: torch.ones(1, device='cuda').add_(1); print('torch', torch.__version__, 'GPU')
    except Exception as exc: print('torch', torch.__version__, 'sees a GPU these wheels cannot run on:', exc)
else: print('torch', torch.__version__, 'MPS' if torch.backends.mps.is_available() else 'CPU only')
PY
echo "Place your VRM avatar model in electron/public/models/ after installing the Electron dependencies."
