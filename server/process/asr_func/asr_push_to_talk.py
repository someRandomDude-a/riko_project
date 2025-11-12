import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

def record_and_transcribe(model, output_file="recording.wav", samplerate=44100):
    """
    Simple push-to-talk recorder: record -> save -> transcribe -> return text
    """
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("Press ENTER to start recording...")
    input()
    
    print("🔴 Recording... Press ENTER to stop")
    
    # Record audio using streaming approach
    audio_data = []
    is_recording = False
    
    def callback(indata, frames, time, status):
        nonlocal audio_data, is_recording
        if is_recording:
            audio_data.append(indata.copy().flatten())
    
    # Start streaming recording
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback, blocksize=1024):
        is_recording = True
        input()  # Wait for stop
        is_recording = False
    
    print("⏹️  Stopping recording...")
    
    # Combine all audio chunks
    if audio_data:
        recording = np.concatenate(audio_data, axis=0)
    else:
        print("⚠️  No audio recorded")
        return ""
    
    print("💾 Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)
    
    print("🎯 Transcribing...")
    
    # Transcribe
    segments, info = model.transcribe(output_file, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    
    print(f"Transcription: {transcription}")
    return transcription.strip()


# Example usage
if __name__ == "__main__":
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="int8_float16")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")