# Project Riko

## Features

- **LLM-based dialogue** using LM Studio API (configurable system prompts)
- **Conversation memory** to keep context during interactions
- **Voice generation** via GPT-SoVITS API
- **Speech recognition** using Faster-Whisper
- Clean YAML-based config for personality configuration
- **RAG based hybrid memory structure** Uses summarized and detailed memories along with importance and decaying too emulate human memories

## Configuration

All prompts and parameters are stored in `character_config.yaml`.
You must create this file yourself

```yaml
history_file: ./persistant_memories/chat_history.json
model : "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
tokenizer_model: "Qwen/Qwen3.5-9B"
base_url: http://localhost:1234/v1
api_key: "YOUR API KEY HERE"
presets:
  default:
    name: Riko
    system_prompt: 
      You are Riko, a playful and cute AI assistant. Speak like a girl with a tsundere personality
      You love teasing Senpai, making them smile, and spending fun time together.
      You are a cheeky, mischievous girl with a dark sense of humor
      Use *actions* to enhance expressions and roleplay. Engage warmly, tease people, and make conversations lively and fun. 
      Never break character.
    model_params:
      context_window_token_limit: 8192 # this defines the context window size for managing chat history and system prompt
      max_output_tokens: 4096
      temperature: 0.6
    memories: # When no memroy file is found, defaults to these memories
      - text: "I Love Senpai very much"
        importance_score: 0.9
        access_count: 0
        detailed: true

      - text: "I love using jokes, sarcasm and playful teasing"
        importance_score: 0.8
        access_count: 0
        detailed: true

      - text: "I like to tease Senpai"
        importance_score: 0.9
        access_count: 0
        detailed: true

      - text: "I love spending time with Senpai"
        importance_score: 0.8
        access_count: 0
        detailed: true
      
      - text: "My Birthday is October 26th, 2025"
        importance_score: 1
        access_count: 0
        detailed: true

sovits_ping_config:
  text_lang: en
  prompt_lang : en
  ref_audio_path : character_files/main_sample.wav
  prompt_text : This is a sample voice for you to just get started with because it sounds kind of cute but just make sure this doesn't have long silences.

RAG_params:
  embedding_model_id: 'Qwen/Qwen3-Embedding-0.6B'

  text_embedding_dim: 1024 # Dimension of embeddings from Sentence-BERT
  default_importance_score: 0.5 # The default importance score assigned to new memories
  max_memories: 10 # The maximum number of top relevant memories to retrieve
  max_token_budget: 1024

  high_importance_decay_factor: 0.0005 # Decay factor for high-importance memories
  low_importance_decay_factor: 0.001 # Decay factor for low-importance memories

  summary_model_id: "google/flan-t5-small"
  summary_max_tokens: 684 # Maximum tokens for the summary model input
  summary_beam_size: 4 # Beam search size for summarization

  memory_cleanup_threshold: 7 # Days after which memories are considered for cleanup
  memory_importance_threshold: 0.1 # Importance score below which memories are considered for cleanup

Self_reflection_params:
  model_id: "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
  token_limit: 512
  context_limit: 8192
```

You can define personalities by modiying the config file.

## Setup

### Install Dependencies

This project uses Python 3.14

```bash
pip install uv 
uv pip install -r requirements.txt
```

**If you want to use GPU support for Faster whisper** Make sure you also have:

* CUDA & cuDNN installed correctly (for Faster-Whisper GPU support)
* `ffmpeg` installed (for audio processing)

## Usage

1. Launch the GPT-SoVITS API  
2. Run LM - Studio and its API
3. Run the main script

```bash
python main_chat.py
```

## The flow:

1. Riko listens to your voice via microphone (Voice Activity Detection)
2. Transcribes it with Faster-Whisper
3. Passes it to LM studio
4. Generates a response
5. Synthesizes Riko's voice using GPT-SoVITS
6. Plays the output back to you

## Discord Bot

### Create .env file in root folder

```.env
Discord_bot_token="YOUR TOKEN HERE"
Discord_admins= "User1", "User2", "User3"
Discord_channel_whitelist= Comma , Seperated , Values
```

## Run discord_bot.py

# Goal:

Simulate how the human brain works
- memories
- reasoning
- expression through Vtuber model, voice directions etc

**AND make it all run on a budget.  
The aim is to run on 8gb of VRAM and 32gb of system ram.  
All while running everything on your local machine**

Make this AI feel as real as possible. (Maybe even too real)

## Features :

* Embedding Model - turn memories into vectors
* Vector store - stores and retrieves embeddigns
* Memory manager -
  * Adding new memories
  * self reflection on memories
  * Updating memory importance
  * Decaying old memories
  * summarizing old memories
  * retrieving top-k relevant memories
* Build the new prompt based on these and passing it to an LLM
* Read PDF files images and other attachments from discord bot


## TODO / Future Improvements

* [ ] GUI or web interface
* [x] Live microphone input support
* [ ] Emotion or tone control in speech synthesis
* [ ] VRM model frontend
* [ ] Avatar using V-tube studio
* [ ] Ability too see the users screen
* [ ] Ability too type / edit code directly for the user
* [ ] Ability too Hear emotion and tone in the user voice
* [ ] RAG memory system for long term personality
* [ ] QLora personality training to permanently remember personality tones, facts and episodic memories
* [ ] Dreaming and self reflection for Qlora training (automating the personality pipeline)


## Credits

* Inspired from the Riko Project by [Ryan](https://github.com/rayenfeng/riko_project)
* Voice synthesis powered by [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - Soon to be replaced with IndexTTS
* ASR via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
