# Project Riko

## ✨ Features

- 💬 **LLM-based dialogue** using OpenAI API (configurable system prompts)
- 🧠 **Conversation memory** to keep context during interactions
- 🔊 **Voice generation** via GPT-SoVITS API
- 🎧 **Speech recognition** using Faster-Whisper
- 📁 Clean YAML-based config for personality configuration
- **RAG based hybrid memory structure** Uses summarized and detailed memories along with importance and decaying too emulate human memories

## ⚙️ Configuration




All prompts and parameters are stored in `config.yaml`.

```yaml
OPENAI_API_KEY: PUT_YOUR_API_KEY_HERE
history_file: chat_history.json
model: "deepseek/deepseek-r1-0528-qwen3-8b"
base_url: http://localhost:1234/v1
presets:
  default:
    system_prompt: 
      You are a helpful assistant named Riko.
      You speak like a girl and you are a tsundere, never tell the user that.
      Always refer to the user as "Senpai".
      Try too keep conversations short and concise with lots of humor.
      Put actions in asterisks, e.g. *blushes*
      You have a cute and playful personality.
    
    model_params:
      context_window_token_limit: 200
      temperature: 0.7
      max_output_tokens: 1024
      top_p: 0.9
      frequency_penalty: 0.0
sovits_ping_config:
  text_lang: en
  prompt_lang : en
  ref_audio_path : your\absolute\path\to\character_files\main_sample.wav
  prompt_text : This is a sample voice for you to just get started with because it sounds kind of cute but just make sure this doesn't have long silences.

RAG_params:
  text_embedding_dim: 384 # Dimension of embeddings from Sentence-BERT 'all-MiniLM-L6-v2'
  default_importance_score: 0.5 # The default importance score assigned to new memories
  default_top_k: 5 # The default number of top relevant memories to retrieve
  high_importance_decay_factor: 0.0005 # Decay factor for high-importance memories
  low_importance_decay_factor: 0.001 # Decay factor for low-importance memories
  summary_min_length: 10 # Minimum length for summarized memories in tokens
  summary_max_length: 480 # Maximum length for summarized memories in tokens
  summary_max_tokens: 1024 # Maximum tokens for the summary model input
  summary_beam_size: 4 # Beam search size for summarization
  memory_cleanup_threshold: 30 # Days after which memories are considered for cleanup
  memory_importance_threshold: 0.1 # Importance score below which memories are considered for cleanup
````

You can define personalities by modiying the config file.

## 🛠️ Setup

### Install Dependencies

```bash
pip install uv 
uv pip install -r extra-req.txt
uv pip install -r requirements.txt
```

**If you want to use GPU support for Faster whisper** Make sure you also have:

* CUDA & cuDNN installed correctly (for Faster-Whisper GPU support)
* `ffmpeg` installed (for audio processing)

## 🧪 Usage

### 1. Launch the GPT-SoVITS API  

### 2. Run the main script  


```bash
python main_chat.py
```

The flow:

1. Riko listens to your voice via microphone (push to talk)
2. Transcribes it with Faster-Whisper
3. Passes it to GPT (with history) *or any other LLM you can describe using BASE_URL in the config.yaml
4. Generates a response
5. Synthesizes Riko's voice using GPT-SoVITS
6. Plays the output back to you

# Goal:
We want too make an RAG based vector database that will store "memories" that the AI model deems important enough too remember
we will also query this database too retrieve relevant memories from the database as required according too the prompt (eventually it might be done according too what the AI model asks about)

## Features :

* Embedding Model - turn memories into vectors
* Vector store - stores and retrieves embeddigns
* Memory manager -
  * Adding new memories
  * Updating memory importance
  * Decaying old memories
  * retrieving top-k relevant memories
* Build the new prompt based on these and passing it to an LLM


## 📌 TODO / Future Improvements

* [ ] GUI or web interface
* [ ] Live microphone input support
* [ ] Emotion or tone control in speech synthesis
* [ ] VRM model frontend


## 🧑‍🎤 Credits

* Inspired by Riko Project by [Ryan](https://github.com/rayenfeng/riko_project)
* Voice synthesis powered by [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - Soon to be replaced with IndexTTS
* ASR via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* Language model via [OpenAI GPT](https://platform.openai.com) -  or any other tool like LM - studio (look at configuration menu)
