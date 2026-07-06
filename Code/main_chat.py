from __future__ import annotations

import threading
import queue

from process.common.config import char_config
from process.voice_scripts.speech_recognition import monitor_and_transcribe
from process.voice_scripts.voice_generator import stream as tts_stream
from process.llm_scripts.llama_server import SentenceStreamer
from process.llm_scripts.module import llm_response

user_name = char_config["your_name"]
print(' \n ========= Starting Chat... ================ \n')

def _tts_worker(jobs: "queue.Queue[str]"):
    """Process TTS jobs serially so audio doesn't overlap and clip."""
    while True:
        sentence = jobs.get()
        if sentence is None:
            break
        try:
            tts_stream(sentence)
        except Exception as e:  # noqa: BLE001
            print(f"[TTS] error: {e!r}")


_tts_queue: "queue.Queue[str]" = queue.Queue(maxsize=8)
threading.Thread(target=_tts_worker, args=(_tts_queue,), daemon=True).start()


def _on_token(delta: str):
    """Per-token callback — print for visibility."""
    print(delta, end="", flush=True)


def _on_sentence(sentence: str):
    """Per-sentence callback — push to TTS worker."""
    print()  # newline after the streamed sentence
    try:
        _tts_queue.put_nowait(sentence)
    except queue.Full:
        # Don't block the LLM stream on a stuck TTS pipeline; drop the fragment.
        print(f"[TTS] queue full, dropping: {sentence[:40]}…")


def main_loop():
    while True:
        try:
            user_spoken_text = f"{user_name}: " + monitor_and_transcribe()
            print(f"\n[USER] {user_spoken_text}\n[Riko] ", end="", flush=True)

            tts_read_text, reasoning = llm_response(
                user_spoken_text,
                user_name=user_name,
                on_token=_on_token,
                on_sentence=_on_sentence,
            )
            print(f"\n\n[reasoning]\n{reasoning}\n")
        except KeyboardInterrupt:
            print("\n[exit] KeyboardInterrupt")
            break
        except Exception as e:
            print(f"[ERROR] {e!r}")
            # Don't `break` on transient errors — keep the chat loop alive.


if __name__ == "__main__":
    main_loop()
