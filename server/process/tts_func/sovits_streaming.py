import requests
import numpy as np
import sounddevice as sd
import threading
import time
import yaml

with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)

audio_buffer = bytearray()
buffer_lock = threading.Lock()

end_of_stream = threading.Event()

def stream_audio(response, warmup_event, warmup_bytes, end_of_stream):
    global audio_buffer

    buffered = 0

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            with buffer_lock:
                audio_buffer.extend(chunk)

            buffered += len(chunk)

            if buffered >= warmup_bytes:
                warmup_event.set()

    except Exception as e:
        print("[ERROR] stream thread:", e)
    
    finally:
        print("[INFO] server finished streaming")
        end_of_stream.set()

def play_stream(end_of_stream,sample_rate=32000, channels=1, dtype=np.int16):

    bytes_per_frame = np.dtype(dtype).itemsize * channels

    last_data_time = time.time()

    def callback(outdata, frames, time_info, status):
        nonlocal last_data_time

        required_bytes = frames * bytes_per_frame

        with buffer_lock:
            available = len(audio_buffer)

            if available >= required_bytes:
                data = audio_buffer[:required_bytes]
                del audio_buffer[:required_bytes]
                last_data_time = time.time()
            elif end_of_stream.is_set() and available > 0:
                data = audio_buffer[:available]
                del audio_buffer[:available]
                data += bytes(required_bytes - len(data))
        
            else:
                data = bytes(required_bytes)

        audio = np.frombuffer(data, dtype=dtype).reshape(-1, channels)
        outdata[:] = audio

    with sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        callback=callback,
        blocksize=256
    ):
        while True:
            with buffer_lock:
                remaining = len(audio_buffer)
            if end_of_stream.is_set() and remaining == 0:
                break
            time.sleep(0.1)


def sovits_stream(text: str, sample_rate: int = 32000, channels: int = 1, dtype: np.dtype = np.int16):

    url = "http://127.0.0.1:9880/tts"

    payload = {
        "text": text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": char_config['sovits_ping_config']['ref_audio_path'],
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang'],
        "media_type": "raw",
        "streaming_mode": True,
    }

    response = requests.post(url, json=payload, stream=True)

    response.raise_for_status()

    warmup_event = threading.Event()

    bytes_per_frame = np.dtype(dtype).itemsize * channels

    warmup_ms = 60
    warmup_bytes = int(sample_rate * warmup_ms / 1000) * bytes_per_frame

    threading.Thread(
        target=stream_audio,
        args=(response, warmup_event, warmup_bytes, end_of_stream),
        daemon=True
    ).start()

    print("[INFO] buffering audio...")

    if not warmup_event.wait(10):
        print("[ERROR] no audio received")
        return

    print("[INFO] starting playback")

    play_stream(end_of_stream,sample_rate, channels, dtype)


if __name__ == "__main__":
    text = input("Enter text:\n")
    sovits_stream(text)