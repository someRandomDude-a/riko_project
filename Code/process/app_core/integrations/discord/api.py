"""Discord-specific transport endpoints over the one existing session/model."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...audio.tts_http import synthesize_wav
from ...runtime.cancellation import TurnCancelled

MAX_AUDIO_SECONDS = 60
MAX_PCM_BYTES = 16000 * 2 * MAX_AUDIO_SECONDS


def transcribe_pcm(session, pcm):
    import numpy as np
    from faster_whisper import WhisperModel
    with session.asr_lock:
        model = getattr(session, 'warmed_asr', None) or getattr(getattr(session, 'voice', None), 'model', None)
        if model is None:
            config = session.config.raw.get('voice', {})
            model = WhisperModel(config.get('asr_model', 'distil-small.en'),
                device=config.get('asr_device', 'cuda'), compute_type=config.get('asr_compute_type', 'int8_float16'))
        session.warmed_asr = model
        samples = np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768
        segments, _ = model.transcribe(samples, beam_size=1, vad_filter=True, condition_on_previous_text=False)
        return ' '.join(segment.text.strip() for segment in segments).strip()


class DiscordChat(BaseModel):
    text: str = Field(min_length=1, max_length=16000)
    user_name: str = Field(default='Discord user', max_length=200)
    turn_id: UUID


class SpeechExport(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class StopTurn(BaseModel):
    turn_id: UUID


def create_router(get_session):
    router = APIRouter(prefix='/api/discord', tags=['discord'])

    def current():
        session = get_session()
        if session is None or session._closed: raise HTTPException(503, 'Companion runtime unavailable')
        return session

    @router.post('/chat')
    async def chat(request: DiscordChat):
        session = current()
        try:
            response = await run_in_threadpool(session.respond, request.text, request.user_name,
                speak=False, turn_id=str(request.turn_id))
            return {'text': response.message.content, 'turn_id': str(request.turn_id)}
        except TurnCancelled: return {'cancelled': True, 'turn_id': str(request.turn_id)}
        except RuntimeError as exc:
            if str(exc) == 'Riko is already handling another turn': raise HTTPException(409, 'Companion is busy; retry when its active turn finishes') from exc
            raise

    @router.post('/stop')
    def stop(request: StopTurn):
        session = current()
        with session._voice_lock:
            matched = session._active_turn == str(request.turn_id) and session._generation_active
            if matched: session.cancel()
        return {'stopped': matched}

    @router.post('/transcribe')
    async def transcribe(request: Request):
        session = current()
        if request.headers.get('content-type', '').split(';')[0] != 'application/octet-stream':
            raise HTTPException(415, 'Send mono 16 kHz signed little-endian PCM16')
        pcm = bytearray()
        async for chunk in request.stream():
            if len(pcm) + len(chunk) > MAX_PCM_BYTES: raise HTTPException(413, 'Audio exceeds 60 seconds')
            pcm.extend(chunk)
        if not pcm or len(pcm) % 2: raise HTTPException(400, 'Empty or invalid PCM16')
        try: text = await run_in_threadpool(transcribe_pcm, session, bytes(pcm))
        except Exception as exc: raise HTTPException(503, 'Speech recognition unavailable; text chat remains usable') from exc
        return {'text': text}

    @router.post('/speech')
    async def speech(request: SpeechExport):
        try: audio = await run_in_threadpool(synthesize_wav, current().config, request.text)
        except Exception as exc: raise HTTPException(503, 'GPT-SoVITS audio export unavailable; try again on the next reply') from exc
        return Response(audio, media_type='audio/wav')

    return router
