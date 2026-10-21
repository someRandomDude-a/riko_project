"""/api/voice, /api/mic, /api/audio and /api/sleep: the microphone, wake word and calibration, barge-in reports from the
renderer, playback mute and volume, and sleep."""
import logging
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..kernel.cancellation import TurnCancelled
from .backend import Services

router = APIRouter()
logger = logging.getLogger(__name__)


class VoiceActivityRequest(BaseModel):
    speech_seconds: float

class InterjectionRequest(BaseModel):
    text: str
    started_at: float
    ended_at: float

@router.post("/api/voice/activity")
def voice_activity(request: VoiceActivityRequest, backend: Services):
    backend.session.voice_activity(request.speech_seconds)
    return {"received": True}

@router.post("/api/voice/interjection")
def voice_interjection(request: InterjectionRequest, backend: Services):
    # 'fresh' words belong to no reply in flight: the client sends them as a chat turn instead. 'reply' words cut the
    # reply in flight and are already in history, so the reply is redone here, as VoiceInput does.
    session = backend.session
    anchor = session.voice_anchor()
    disposition = session.voice_transcript(request.text, request.started_at, request.ended_at, anchor) if anchor else 'fresh'
    if disposition == 'reply':
        def redo():
            try: session.respond(request.text, record_user=False, reply_to=anchor[0], wait=lambda: False, origin={'source': 'microphone'})
            except TurnCancelled: pass
            except Exception: logger.exception('Could not redo the reply an interjection cut')
        threading.Thread(target=redo, daemon=True, name='interjection-redo').start()
    return {"accepted": disposition in {'preserved', 'reply'}}

@router.post("/api/mic/toggle")
def toggle_mic(backend: Services):
    enabled = backend.state.toggle_mic()
    if enabled: backend.session.start_listening()
    else: backend.session.stop_listening()
    return {"enabled": enabled}

@router.post("/api/audio/toggle")
def toggle_audio(backend: Services): return {"enabled": backend.state.toggle_audio()}

class AudioVolumeRequest(BaseModel):
    volume: float = Field(ge=0, le=1, strict=True, allow_inf_nan=False)

@router.patch('/api/audio/volume')
def set_audio_volume(request: AudioVolumeRequest, backend: Services):
    try: return {'volume': backend.state.set_audio_volume(request.volume), 'enabled': backend.state.audio_enabled}
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc

@router.post("/api/sleep/toggle")
def toggle_sleep(backend: Services):
    enabled = not backend.state.sleep_mode
    backend.state.set_sleep(enabled)
    if enabled: backend.session.cancel()
    return {"sleep": enabled}

@router.post("/api/voice/start")
def start_voice(backend: Services):
    backend.state.set_mic(True)
    backend.session.start_listening()
    return {"starting": True}

@router.post("/api/voice/stop")
def stop_voice(backend: Services):
    backend.state.set_mic(False)
    backend.session.stop_listening()
    return {"stopped": True}

@router.get("/api/voice/status")
def voice_status(backend: Services): return backend.voice()

@router.post("/api/voice/activate")
def activate_voice(backend: Services):
    backend.state.set_mic(True)
    backend.session.start_listening()
    backend.session.wake.activate()
    return {"active": True}

class CalibrationRequest(BaseModel):
    action: str
    threshold: float | None = None

@router.post("/api/voice/calibration")
def voice_calibration(request: CalibrationRequest, backend: Services):
    session = backend.session
    operations = {"begin": session.wake.begin_calibration, "record": session.wake.record,
                  "save": session.wake.finish_calibration, "cancel": session.wake.cancel_calibration,
                  "test": lambda: session.wake.set_testing(True),
                  "stop_test": lambda: session.wake.set_testing(False),
                  "threshold": lambda: session.wake.set_threshold(request.threshold)}
    try:
        if request.action not in operations: raise ValueError("Unknown calibration operation")
        # Also between sentences of a reply: Riko's own voice must never become a wake-word sample.
        if request.action in {"begin", "test"} and session.status().replying():
            raise ValueError("Stop the current reply before calibration or testing")
        operations[request.action]()
        return session.wake.status()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/api/voice/devices")
def voice_devices():
    import sounddevice as sd
    return {"devices": [{"index": index, "name": device["name"]}
                        for index, device in enumerate(sd.query_devices()) if device["max_input_channels"] > 0]}
