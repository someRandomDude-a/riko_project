import requests
import numpy as np
import sounddevice as sd
import yaml
import time
import threading

# Load YAML config
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

def play_raw_stream(response, sample_rate: int, channels: int, dtype: np.dtype, timeout_secs: float = 0.1, start_secs: float = 10):
    """
    Streams raw PCM from response, waits for first chunk till start_secs, and stops when no data for timeout_secs.
    """
    bytes_per_sample = np.dtype(dtype).itemsize
    bytes_per_frame = bytes_per_sample * channels

    first_chunk_event = threading.Event()
    last_chunk_time = time.time()

    def callback(outdata, frames, time_info, status):
        nonlocal last_chunk_time
        if status:
            print("[WARN] stream status:", status)

        to_read = frames * bytes_per_frame
        chunk = response.raw.read(to_read)

        if not chunk:
            # No data → stop streaming
            raise sd.CallbackStop()

        if not first_chunk_event.is_set():
            # First actual chunk arrived
            print("[INFO] First chunk received")
            first_chunk_event.set()

        last_chunk_time = time.time()
        audio_data = np.frombuffer(chunk, dtype=dtype)
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
        else:
            audio_data = audio_data.reshape(-1, 1)

        outdata[:len(audio_data)] = audio_data

    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=dtype, callback=callback):
        print("[INFO] Waiting for first audio chunk …")
        # Wait for first data or timeout
        if not first_chunk_event.wait(start_secs):
            print(f"[ERROR] No audio chunk received within {timeout_secs} seconds.")
            return
        print("[INFO] Audio started; streaming now …")

        # Now keep stream alive until callback raises stop
        while True:
            if time.time() - last_chunk_time > timeout_secs:
                print(f"[INFO] No data received for {timeout_secs} secs → terminating.")
                break
            time.sleep(0.1)

def sovits_gen_stream(in_text, sample_rate=32000, channels=1, dtype=np.int16):
    url = "http://127.0.0.1:9880/tts"
    payload = {
        "text": in_text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": char_config['sovits_ping_config']['ref_audio_path'],
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang'],
        "media_type": "raw",
        "streaming_mode": True
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        play_raw_stream(response, sample_rate, channels, dtype)
    except Exception as e:
        print("[ERROR] sovits_gen_raw:", e)

if __name__ == "__main__":
    text_input = input("Enter text to speak here:\n")
    start_time = time.time()
    sovits_gen_stream(text_input)
    elapsed = time.time() - start_time
    print(f"[INFO] Finished in {elapsed:.2f} seconds")
