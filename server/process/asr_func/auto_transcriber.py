import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import torch
import time

# Global Silero VAD model cache for performance
_CACHED_VAD_MODEL = None

def get_silero_vad_model():
    """
    Load and cache Silero VAD model globally for performance
    """
    global _CACHED_VAD_MODEL
    
    # Return cached model if loaded
    if _CACHED_VAD_MODEL is not None:
        return _CACHED_VAD_MODEL
    
    try:
        # Pip method
        from silero_vad import load_silero_vad, get_speech_timestamps
        
        # Load model
        model = load_silero_vad()
        
        # Soundfile backend read function
        def read_audio_with_soundfile(file_path):
            audio, sample_rate = sf.read(file_path)
            # Convert to torch tensor
            return torch.from_numpy(audio).float()
        
        # Cache for fast reuse
        _CACHED_VAD_MODEL = {
            'model': model,
            'read_audio': read_audio_with_soundfile,
            'get_speech_timestamps': get_speech_timestamps
        }
        
        return _CACHED_VAD_MODEL
        
    except ImportError:
        print("Install: pip install silero-vad soundfile torch")
        return None

def silero_vad_detection(audio_chunk, vad_model, sampling_rate=16000, threshold=0.5):
    """
    Use Silero VAD for speech detection
    """
    if vad_model is None:
        # Fallback to basic energy detection
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms > threshold
    
    try:
        # Convert audio chunk to tensor and ensure correct format
        if sampling_rate != 16000:
            # Resample to 16kHz for Silero VAD
            import scipy.signal
            audio_chunk = scipy.signal.resample(audio_chunk, int(len(audio_chunk) * 16000 / sampling_rate))
        
        # Convert to tensor with proper shape [batch, time]
        waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)  # Add batch dimension
        
        # Get VAD prediction using Silero API
        model = vad_model['model']
        get_speech_timestamps = vad_model['get_speech_timestamps']
        
        # Get speech timestamps for this chunk
        speech_timestamps = get_speech_timestamps(
            waveform, 
            model, 
            return_seconds=True,
            threshold=threshold
        )
        
        # If we found speech in this chunk, return True
        return len(speech_timestamps) > 0
        
    except ImportError:
        print("⚠️ scipy not available, falling back to energy detection")
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms > threshold
    except Exception as e:
        print(f"VAD error: {e}, using energy detection")
        # Fallback to energy detection
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms > threshold

def record_and_transcribe(model, output_file="recording.wav", samplerate=44100):
    """
    Smart recorder with Silero VAD: automatically detect speech -> record -> transcribe -> return text
    
    Same function signature as your original implementation:
    - model: WhisperModel instance
    - output_file: Output audio file path  
    - samplerate: Audio sampling rate
    - Returns: transcription string (same as original)
    """
    # Remove existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print("🎤 Smart recording mode with Silero VAD - speak naturally!")
    print("⏳ Listening for speech...")
    
    # Load VAD model (cached globally for performance)
    vad_model = get_silero_vad_model()
    
    # Audio recording parameters
    chunk_duration = 0.5  # Process audio in 0.5 second chunks
    chunk_size = int(samplerate * chunk_duration)
    min_speech_duration = 1.5  # Minimum seconds of speech to record
    silence_timeout = 2.0  # Stop recording after X seconds of silence
    max_silence_samples = int(silence_timeout / chunk_duration)
    
    # Recording state
    is_recording = False
    audio_data = []
    speech_duration = 0
    silence_count = 0
    should_stop = False
    
    def callback(indata, frames, time, status):
        nonlocal is_recording, audio_data, speech_duration, silence_count, should_stop
        
        if status:
            print(f"Stream status: {status}")
        
        # Convert to mono and normalize
        audio_chunk = indata.flatten() if len(indata.shape) > 1 else indata[:, 0]
        
        if not is_recording:
            # Check for speech start with VAD
            speech_detected = silero_vad_detection(audio_chunk, vad_model, samplerate)
            
            if speech_detected:
                print("🟢 Speech detected! Starting recording...")
                is_recording = True
                speech_duration = 0
                silence_count = 0
                audio_data = [audio_chunk]
            else:
                # Still listening for speech
                print("👂 Listening...                                   \r", end="")
        else:
            # Currently recording
            audio_data.append(audio_chunk)
            speech_duration += chunk_duration
            
            # Check speech activity
            speech_detected = silero_vad_detection(audio_chunk, vad_model, samplerate)
            
            if speech_detected:
                # Speech detected - reset silence counter
                silence_count = 0
                print(f"🔊 Recording... ({speech_duration:.1f}s)              \r", end="")
            else:
                # Silence detected
                silence_count += 1
                print(f"🔇 Silence... ({silence_count*chunk_duration:.1f}s)   \r", end="")
                
                # Check if we've had enough silence to stop
                if silence_count >= max_silence_samples and speech_duration >= min_speech_duration:
                    print(f"\n⏹️ Stopping recording after {speech_duration:.1f} seconds of speech")
                    should_stop = True
                    return  # Exit callback to trigger stream closure
    
    # Start recording stream
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', 
                      callback=callback, blocksize=chunk_size):
        # Keep stream alive until should_stop is set
        while not should_stop:
            time.sleep(0.1)
    
    print("✅ Recording completed")
    
    # Process recorded audio
    if not audio_data:
        print("⚠️ No speech detected")
        return ""
    
    # Combine all audio chunks
    recording = np.concatenate(audio_data, axis=0)
    
    print("💾 Saving audio...")
    
    # Write the file
    sf.write(output_file, recording, samplerate)
    
    print("🎯 Transcribing...")
    
    # Transcribe using faster-whisper (same as your original)
    segments, info = model.transcribe(output_file, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    
    print(f"Transcription: {transcription}")
    return transcription.strip()


# Example usage - EXACT SAME as your original
if __name__ == "__main__":
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="int8_float16")
    result = record_and_transcribe(model)
    print(f"Got: '{result}'")