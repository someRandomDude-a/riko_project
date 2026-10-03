pip install uv
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements-runtime.txt
uv pip install --no-deps EfficientWord-Net
echo "Place your VRM avatar model in electron/public/models/ after installing the Electron dependencies."

python - <<PYCODE
import nltk
for pkg in ["averaged_perceptron_tagger", "cmudict"]:
    nltk.download(pkg)
PYCODE
