from faster_whisper import WhisperModel
from process.asr_func.auto_transcriber import record_and_transcribe
from process.llm_funcs.llm_scr import Riko_Response
from process.tts_func.sovits_streaming import sovits_stream

from pathlib import Path


### transcribe audio 
import uuid
import soundfile as sf
import sounddevice as sd

def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')
whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="int8_float16")

while True:
    try:
        conversation_recording = output_wav_path = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)

        user_spoken_text = "Senpai: " + record_and_transcribe(whisper_model, conversation_recording)

        ### pass to LLM and get a LLM output.
        tts_read_text = Riko_Response(user_spoken_text)
    
        print(tts_read_text)

        # Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        # generate audio and save it to client/audio 
        # Remove timestamp from tts_read_text before passing it spoken text part that we care about ->  timestamp
        
        tts_read_text = tts_read_text.rsplit("timestamp:", 1)[0].split("Riko: ", 1)[1].strip()
        try:
            sovits_stream(tts_read_text)
        except KeyboardInterrupt as e:
            print(f"[Keyboard Interrupt Called]: {e}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        break
