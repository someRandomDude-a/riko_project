"""GPT-SoVITS request construction and bounded audio export; no local playback."""
import io
import time
import wave


def request_payload(config, text):
    sovits = config.sovits  # sovits_ping_config, checked by load_config (SovitsConfig.from_raw)
    payload = {**sovits.request(), 'text': text, 'media_type': 'raw', 'streaming_mode': sovits.streaming_mode}
    payload['ref_audio_path'] = str((config.root / sovits.ref_audio_path).resolve())
    return sovits.url, payload


def synthesize_wav(config, text, *, max_seconds=60):
    import requests
    url, payload = request_payload(config, text)
    rate = config.sovits.sample_rate  # 8000-192000 Hz
    limit = min(rate * 2 * max_seconds, 8 * 1024 * 1024)
    pcm = bytearray()
    deadline = time.monotonic() + 90
    with requests.post(url, json=payload, stream=True, timeout=(5, 30)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(8192):
            if time.monotonic() > deadline: raise ValueError('Speech export timed out')
            if len(pcm) + len(chunk) > limit: raise ValueError('Speech exceeds audio export limit; use shorter text')
            pcm.extend(chunk)
    if not pcm or len(pcm) % 2: raise ValueError('GPT-SoVITS returned empty or invalid PCM audio')
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm)
    return output.getvalue()
