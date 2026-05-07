from process.voice_scripts.speech_recognition import monitor_and_transcribe
from process.llm_scripts.module import Riko_Response
from process.voice_scripts.voice_generator import stream

print(' \n ========= Starting Chat... ================ \n')

while True:
    try:
        user_spoken_text = "(Senpai) " + monitor_and_transcribe()

        ### pass to LLM and get a LLM output.
        tts_read_text, reasoning = Riko_Response(user_spoken_text)
    
        print(tts_read_text + "\n\nReasoning:\n" + reasoning)

        # generate audio and save it to client/audio 
        # Remove timestamp from tts_read_text before passing it spoken text part that we care about ->  timestamp
        
        tts_read_text = tts_read_text.rsplit("timestamp:", 1)[0].split("(Riko) ", 1)[-1].strip()
        try:
            stream(tts_read_text)
        except KeyboardInterrupt as e:
            print(f"[Keyboard Interrupt Called]: {e}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        break
