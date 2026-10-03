"""Desktop master gain; preserves PCM timing and leaves generated audio untouched."""
def scale_pcm16(pcm, volume):
    if volume == 1: return pcm
    if volume == 0: return bytes(len(pcm))
    import numpy as np
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32)
    return (samples * volume).astype('<i2').tobytes()
