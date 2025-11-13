import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
import torch
import yaml

# === Load Character Configuration ===
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)



# Constants (now converted to variables)
MODEL_NAME = char_config['RAG_params']['embedding_model_id']  # Sentence-BERT model name
BART_MODEL_NAME = char_config['RAG_params']['summarization_model_id']   # BART model name for summarization
FAISS_INDEX_PATH = 'faiss_index.index'  # File path for FAISS index
MEMORY_STORE_PATH = 'memory_store.json'  # File path for memory store
EMBEDDING_DIM = char_config['RAG_params']['text_embedding_dim']  # Embedding dimension from config  
FAISS_DECAY_FACTOR_HIGH = char_config['RAG_params']['high_importance_decay_factor']  # Decay factor for high-importance memories
FAISS_DECAY_FACTOR_LOW = char_config['RAG_params']['low_importance_decay_factor']   # Decay factor for low-importance memories
SUMMARY_MIN_LENGTH = char_config['RAG_params']['summary_min_length']  # Minimum length for summary
SUMMARY_MAX_LENGTH = char_config['RAG_params']['summary_max_length'] # Maximum length for summary
SUMMARY_NUM_BEAMS = char_config['RAG_params']['summary_beam_size']  # Beam search size for summarization
SUMMARY_MAX_TOKENS = char_config['RAG_params']['summary_max_tokens'] # Maximum number of tokens for BART input

# Load SentenceTransformer model
model = SentenceTransformer(MODEL_NAME)

summarizerPipeline = pipeline("summarization",model=BART_MODEL_NAME)


Example_text = """
Considerations for This Pipeline

GGUF-Specific Optimizations: The main advantage of using GGUF would be faster execution and reduced resource consumption. This assumes that GGUF is optimized for your hardware, such as quantization and sparse computations.

Hardware: If you’re running this on NVIDIA GPUs, TensorRT optimizations could also apply, assuming GGUF is compatible with it.

Model Size: GGUF might reduce the model’s memory footprint, which is especially useful when deploying very large models like BART in resource-constrained environments.

Where We Are Today

As of now, Hugging Face doesn’t support GGUF natively, and there’s no direct way to run a Hugging Face model (like BART) in a GGUF-optimized pipeline. However, GGUF-based optimizations (such as low-precision formats, memory optimizations, etc.) could eventually be implemented with tools that support this format.

Alternatives for Performance Optimization Today:

If you’re looking for fast inference with BART, here are some immediate alternatives you can explore:

ONNX: Convert the model to ONNX and run it using ONNX Runtime, which can be optimized for faster execution.

TensorRT: Convert the model to TensorRT for accelerated inference on NVIDIA GPUs.

Triton Inference Server: Deploy models with NVIDIA Triton for multi-model inference, batch processing, and GPU optimizations.

Quantization: Use Hugging Face's transformers or TensorFlow to reduce model precision to FP16 or INT8 for better performance.

Conclusion

While running BART as a GGUF model isn't directly possible within the Hugging Face pipeline right now, the concept of integrating performance-optimized formats into a pipeline is sound. The future may bring more standardized ways to convert Hugging Face models into GGUF or similar formats for deployment, especially if GGUF becomes more widely adopted.

In the meantime, optimizing BART using other frameworks like ONNX, TensorRT, or Triton should yield performance improvements without needing a GGUF-specific pipeline. Let me know if you'd like further guidance on any of these optimizations!
"""
# Function to summarize memory using BART


def summarize_text(text):
    global summarizerPipeline
    summary = summarizerPipeline(text,max_length= SUMMARY_MAX_LENGTH, min_length=SUMMARY_MIN_LENGTH)
    summary_text = ""
    for text in summary:
        summary_text += text['summary_text']
    return summary_text

if __name__ == "__main__":
    

    #input_text = input("Enter the text:")

    print(summarize_text(Example_text))