from transformers import AutoProcessor
import os
try:
    emotion_model_name = "superb/wav2vec2-base-superb-er"

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=emotion_model_name, local_files_only=False,)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
