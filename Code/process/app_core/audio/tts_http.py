"""GPT-SoVITS request construction and bounded audio export; no local playback."""
import io
import time
import wave


def request_payload(config, text):
    settings = config.raw.get('sovits_ping_config', {})
    keys = ('text_lang', 'prompt_lang', 'prompt_text', 'batch_size', 'text_split_method',
        'split_bucket', 'parallel_infer', 'fragment_interval', 'speed_factor', 'min_chunk_length', 'overlap_length')
    payload = {key: settings[key] for key in keys if key in settings}
    payload.update(text=text, media_type='raw', streaming_mode=settings.get('streaming_mode', True))
    payload['ref_audio_path'] = str((config.root / settings.get('ref_audio_path', 'character_files/main_sample.wav')).resolve())
    return settings.get('url', 'http://127.0.0.1:9880/tts'), payload


def synthesize_wav(config, text, *, max_seconds=60):
    import requests
    url, payload = request_payload(config, text)
    rate = int(config.raw.get('sovits_ping_config', {}).get('sample_rate', 32000))
    if not 8000 <= rate <= 192000: raise ValueError('Invalid speech sample rate')
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
