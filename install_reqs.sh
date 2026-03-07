pip install uv
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

python - <<PYCODE
import nltk
for pkg in ["averaged_perceptron_tagger", "cmudict"]:
    nltk.download(pkg)
PYCODE
