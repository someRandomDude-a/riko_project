from process.voice_scripts.speech_recognition import monitor_and_transcribe
from process.llm_scripts.module import llm_response
from process.voice_scripts.voice_generator import stream
from process.common.config import char_config

user_name = char_config["your_name"]
print(' \n ========= Starting Chat... ================ \n')

while True:
    try:
        user_spoken_text = "Senpai: " + monitor_and_transcribe()

        # pass to LLM and get a LLM output.
        tts_read_text, reasoning = llm_response(user_spoken_text, user_name=user_name)
        print(f"\n{tts_read_text}\n\nReasoning:\n{reasoning}")

        # generate audio and save it to client/audio 
        try:
            stream(tts_read_text)
        except KeyboardInterrupt as e:
            print(f"[Keyboard Interrupt Called]: {e}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        break
