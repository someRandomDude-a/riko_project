from faster_whisper import WhisperModel
#from process.asr_func.asr_push_to_talk import record_and_transcribe # we replace this with vad version
from process.asr_func.auto_transcriber import record_and_transcribe # our new VAD version
from process.llm_funcs.llm_scr import llm_response

##Variable to set streaming mode or normal mode for sovits gen
streamSovitsGen = True
from process.tts_func.sovits_streaming import sovits_gen_stream #Import the streaming mode function
from process.tts_func.sovits_ping import sovits_gen, play_audio #Import the ping mode functions

from pathlib import Path
import os
import time
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

        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)

        ### pass to LLM and get a LLM output.

        llm_output = llm_response(user_spoken_text)

        tts_read_text = llm_output
        print(llm_output)
        ### file organization 

        # 1. Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        # generate audio and save it to client/audio 
        # Remove timestamp from tts_read_text before passing it spoken text part that we care about->timestamp
        tts_read_text = tts_read_text.split("timestamp:")[0].strip()
        #for non streaming, use this:
        if streamSovitsGen:
            sovits_gen_stream(tts_read_text)
        else:
            gen_aud_path = sovits_gen(tts_read_text,output_wav_path)
            play_audio(output_wav_path)
            [fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]        # clean up audio files

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        break
