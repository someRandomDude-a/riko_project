import torch
from transformers import pipeline
from PIL import Image
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
try:
    captioner = pipeline(
        "image-to-text", 
        model="microsoft/git-large-textcaps",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


def get_embedding(image_path):
    global captioner
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = Image.open(image_path)
    result = captioner(image)
    caption = result[0]['generated_text']
    print(f"Generated caption: '{caption}'")
    return caption

if __name__ == "__main__":
    from get_screenshot import get_screenshot
    print("Testing image captioning...")
    test_image_path = get_screenshot()  # Replace with your test image path if u want
    caption = get_embedding(test_image_path)
    if caption:
        print(f"Caption: {caption}")