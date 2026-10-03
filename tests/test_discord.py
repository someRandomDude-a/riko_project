import asyncio
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from uuid import uuid4
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from process.app_core.integrations.discord import api
from process.app_core.integrations.discord.config import BotSettings, BASIC_SETTINGS
from process.app_core.integrations.discord.media import decode_audio, split_text
from process.app_core.integrations.discord.preferences import Preferences
from process.app_core.integrations.discord.replies import StreamReply
from process.app_core.integrations.discord.voice import VoiceCapture, MAX_CALL_BYTES, downsample_call
from process.app_core.audio.tts_http import request_payload, synthesize_wav
from process.app_core.tools.approval import ToolApprovals, approval_turn


def test_discord_policy_fails_closed_and_separates_admins(tmp_path):
    policy = BotSettings(tmp_path, admins=frozenset({1}), users=frozenset({2}), channels=frozenset({3}))
    assert policy.allows(1, 3, guild=True, admin=True)
    assert policy.allows(2, 3, guild=True)
    assert not policy.allows(2, 3, guild=True, admin=True)
    assert not policy.allows(99, 3, guild=True)
    assert not policy.allows(1, 4, guild=True)
    assert policy.allows(2, 4, guild=False)
    assert not BotSettings(tmp_path).allows(1, 3)
    assert not any('key' in key or 'path' in key or 'command' in key for key in BASIC_SETTINGS)


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost/secret', 'http://user:pass@localhost', 'http://localhost?key=secret'])
def test_discord_backend_url_cannot_send_runtime_secrets_remotely(tmp_path, url):
    with pytest.raises(ValueError): BotSettings.from_env(tmp_path, {'Discord_backend_url': url})


def test_discord_preferences_roundtrip_and_bad_data_preserved(tmp_path):
    path = tmp_path / 'discord.json'
    prefs = Preferences(path)
    assert not prefs.get(42, 'audio')
    prefs.set(42, 'audio', True)
    assert Preferences(path).get(42, 'audio')
    path.write_text('{invalid')
    with pytest.raises(ValueError): Preferences(path)
    assert path.read_text() == '{invalid'


def test_audio_decode_is_bounded_and_cannot_open_network_or_local_playlists(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'\0\0' * 1600)
    monkeypatch.setattr('process.app_core.integrations.discord.media.subprocess.run', run)
    assert len(decode_audio(b'fake')) == 3200
    command, options = calls[0]
    assert command[command.index('-protocol_whitelist') + 1] == 'pipe'
    assert command[command.index('-t') + 1] == '61'
    assert options['timeout'] == 30
    with pytest.raises(ValueError): decode_audio(b'x' * (8 * 1024 * 1024 + 1))
    assert len(calls) == 1


def test_streamed_discord_edits_split_long_replies_and_keep_reasoning_separate():
    async def run():
        sent = []
        class Message:
            def __init__(self, content): self.content, self.deleted = content, False
            async def edit(self, *, content): self.content = content
            async def delete(self): self.deleted = True
        async def send(content):
            message = Message(content); sent.append(message); return message
        reply = StreamReply(send, prefix='Reasoning: ')
        reply.feed('a' * 4000)
        await reply.flush()
        assert len(sent) == 3
        assert all(len(message.content) <= 2000 for message in sent)
        await reply.finish('short')
        assert sent[0].content == 'Reasoning: short'
        assert sent[1].deleted and sent[2].deleted
        assert reply.timer is None
    asyncio.run(run())


def test_voice_receive_requires_consent_and_discards_buffers_on_revoke():
    async def run():
        delivered = []
        user = SimpleNamespace(id=1, bot=False)
        capture = VoiceCapture(asyncio.get_running_loop(), lambda user: True,
            lambda user, pcm: delivered.append((user.id, pcm)))
        pcm = b'\1\0' * 1920
        capture.write(user, pcm); await asyncio.sleep(0)
        assert not capture.buffers
        capture.allow(1)
        for _ in range(10): capture.write(user, pcm)
        await asyncio.sleep(0)
        assert len(capture.buffers[1][1]) < MAX_CALL_BYTES
        capture.revoke(1)
        capture.flush(1)
        assert not delivered and not capture.buffers and not capture.timers
        capture.allow(1)
        for _ in range(10): capture.write(user, pcm)
        await asyncio.sleep(0)
        capture.flush(1)
        assert delivered[0][0] == 1
        capture.close()
        assert not capture.consent
    asyncio.run(run())


def test_call_pcm_is_resampled_to_mono_16khz():
    pcm = b'\1\0' * (48000 * 2)
    assert len(downsample_call(pcm)) == 16000 * 2


def test_discord_api_uses_existing_session_without_local_speech_and_scopes_stop(monkeypatch):
    identity = str(uuid4())
    calls = []
    session = SimpleNamespace(_closed=False, _voice_lock=threading.RLock(),
        _active_turn=identity, _generation_active=True, cancel=lambda: calls.append('cancel'),
        respond=lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(message=SimpleNamespace(content='hello')))
    app = FastAPI(); app.include_router(api.create_router(lambda: session))
    client = TestClient(app)
    assert client.post('/api/discord/chat', json={'text':'hi', 'turn_id':identity}).json()['text'] == 'hello'
    assert calls[0] == {'speak':False, 'turn_id':identity}
    assert not client.post('/api/discord/stop', json={'turn_id':str(uuid4())}).json()['stopped']
    assert 'cancel' not in calls
    assert client.post('/api/discord/stop', json={'turn_id':identity}).json()['stopped']
    assert calls[-1] == 'cancel'
    assert client.post('/api/discord/chat', json={'text':'hi', 'turn_id':'bad'}).status_code == 422


def test_discord_asr_api_validates_limits_and_recovers_after_failure(monkeypatch):
    calls = []
    session = SimpleNamespace(_closed=False)
    monkeypatch.setattr(api, 'transcribe_pcm', lambda session, pcm: calls.append(len(pcm)) or 'heard you')
    app = FastAPI(); app.include_router(api.create_router(lambda: session))
    client = TestClient(app)
    headers = {'Content-Type':'application/octet-stream'}
    assert client.post('/api/discord/transcribe', content=b'\0\0', headers=headers).json()['text'] == 'heard you'
    assert client.post('/api/discord/transcribe', content=b'\0', headers=headers).status_code == 400
    assert client.post('/api/discord/transcribe', content=b'\0\0').status_code == 415
    assert client.post('/api/discord/transcribe', content=b'\0' * (api.MAX_PCM_BYTES + 2), headers=headers).status_code == 413
    assert calls == [2]


def test_audio_export_failure_does_not_disable_subsequent_requests(monkeypatch):
    state = [False]
    def synthesize(config, text):
        if not state[0]: raise RuntimeError('offline')
        return b'RIFFfake'
    monkeypatch.setattr(api, 'synthesize_wav', synthesize)
    app = FastAPI(); app.include_router(api.create_router(lambda: SimpleNamespace(_closed=False, config=None)))
    client = TestClient(app)
    assert client.post('/api/discord/speech', json={'text':'hello'}).status_code == 503
    state[0] = True
    result = client.post('/api/discord/speech', json={'text':'hello'})
    assert result.status_code == 200 and result.headers['content-type'] == 'audio/wav'


def test_exported_tts_reuses_http_payload_and_never_opens_sound_device(tmp_path, monkeypatch):
    config = SimpleNamespace(root=tmp_path, raw={'sovits_ping_config':{'sample_rate':16000, 'text_lang':'en'}})
    url, payload = request_payload(config, 'hello')
    assert payload['media_type'] == 'raw' and payload['text_lang'] == 'en'
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield b'\1\0' * 16000
    monkeypatch.setattr('requests.post', lambda *args, **kwargs: Response())
    audio = synthesize_wav(config, 'hello')
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 16000 and wav.getnframes() == 16000


def test_approval_requests_include_transport_turn_identity(tmp_path):
    gate = ToolApprovals(tmp_path / 'approvals.json', True)
    from process.app_core.events.bus import event_bus
    captured = []
    def listener(event):
        if event.type == 'tool.approval_requested':
            captured.append(event)
            gate.resolve(event.payload['id'], False)
    unsubscribe = event_bus.subscribe(listener)
    token = approval_turn.set('discord-turn')
    try:
        assert not gate.authorize('tool', {}, 'call')
        assert captured[0].turn_id == 'discord-turn'
    finally: approval_turn.reset(token); unsubscribe(); gate.close()


def test_bot_import_and_command_registration_do_not_construct_models(tmp_path):
    discord = pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot
    from process.app_core.integrations.discord.commands import CompanionCommands
    async def run():
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})))
        await bot.add_cog(CompanionCommands(bot))
        names = {command.name for command in bot.tree.get_commands()}
        assert {'chat','speak','transcribe','join','leave','listen','stop','settings','messages','tool_policy','camera','tasks','initiative','whiteboard','resources','animation','memory','memories','history','edit','task_create','task_update','status','reasoning'} <= names
        assert 'audio' not in names
        assert bot.processor is None and bot.backend.http is None
        await bot.close()
    asyncio.run(run())


def test_stop_is_scoped_to_user_and_turn_and_retains_other_queued_jobs(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        calls = []
        class Backend:
            def __init__(self, *args): pass
            async def request(self, method, path, **kwargs): calls.append((path, kwargs['body']))
            async def close(self): pass
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), backend_factory=Backend)
        channel = SimpleNamespace(id=42)
        active = Job(channel, SimpleNamespace(id=1))
        active.task = asyncio.create_task(asyncio.Event().wait())
        bot.active = active
        mine, other = Job(channel, SimpleNamespace(id=2)), Job(channel, SimpleNamespace(id=1))
        bot.queue.put_nowait(mine); bot.queue.put_nowait(other)
        await bot.stop_channel(42, 2)
        assert not calls and not active.task.cancelled()
        assert bot.queue.get_nowait() is other; bot.queue.task_done()
        await bot.stop_channel(42, 1)
        assert calls == [('/api/discord/stop', {'turn_id':active.id})]
        with pytest.raises(asyncio.CancelledError): await active.task
        bot.active = None
        await bot.close()
    asyncio.run(run())


def test_approval_snapshots_do_not_relay_unrelated_desktop_turns(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        sent = []
        class Backend:
            def __init__(self, *args): pass
            async def close(self): pass
        async def send(content=None, **kwargs):
            sent.append(content)
            return SimpleNamespace(edit=edit)
        async def edit(**kwargs): pass
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), backend_factory=Backend)
        job = Job(SimpleNamespace(id=42, send=send), SimpleNamespace(id=1))
        bot.active = job
        unrelated = {'id':'desktop', 'name':'tool', 'arguments':{}, 'turn_id':'desktop-turn', 'expires_at':time.time()+120}
        matching = {**unrelated, 'id':'discord', 'turn_id':job.id}
        await bot.backend_event({'type':'resource.approvals','payload':{'pending':[unrelated, matching]}})
        assert set(bot.approvals) == {'discord'} and len(sent) == 1
        await bot.backend_event({'type':'resource.approvals','payload':{'pending':[matching]}})
        assert len(sent) == 1
        await bot.backend_event({'type':'tool.approval_resolved','payload':{'id':'discord'}})
        assert not bot.approvals
        bot.active = None
        await bot.close()
    asyncio.run(run())


def test_audio_attachments_are_rejected_before_download_and_pcm_cannot_bypass_pause(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        class Backend:
            def __init__(self, *args): self.ready = asyncio.Event(); self.ready.set()
            async def close(self): pass
        async def read(): raise AssertionError('Paused audio must never be downloaded')
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), backend_factory=Backend)
        channel, user = SimpleNamespace(id=42), SimpleNamespace(id=1)
        with pytest.raises(ValueError, match='paused'):
            await bot.process_job(Job(channel, user, attachment=SimpleNamespace(size=4, filename='recording.wav', read=read)))
        with pytest.raises(ValueError, match='paused'):
            await bot.process_job(Job(channel, user, pcm=b'\0\0'))
        await bot.close()
    asyncio.run(run())


def test_backend_close_cancels_subscription_and_clears_readiness():
    from process.app_core.integrations.discord.backend import BackendClient
    async def run():
        closed = []
        async def close_http(): closed.append(True)
        client = BackendClient('http://127.0.0.1:8765', None, None)
        client.http = SimpleNamespace(close=close_http)
        client.worker = asyncio.create_task(asyncio.Event().wait())
        client.ready.set()
        await client.close()
        assert client.closed and not client.ready.is_set() and client.worker.cancelled() and closed == [True]
    asyncio.run(run())


def test_paused_calling_cannot_load_voice_dependencies_and_text_chat_still_works(tmp_path, monkeypatch):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        calls, sent = [], []
        class Backend:
            def __init__(self, *args): self.ready = asyncio.Event(); self.ready.set()
            async def request(self, method, path, **kwargs):
                calls.append(path)
                if path == '/api/discord/chat': return {'text':'Text reply'}
                raise AssertionError('Calling/media must not run for text chat')
            async def close(self): pass
        async def send(content=None, **kwargs):
            assert 'file' not in kwargs
            sent.append(content)
            return SimpleNamespace(edit=None)
        user = SimpleNamespace(id=1, display_name='Owner')
        channel = SimpleNamespace(id=42, guild=None, send=send)
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), backend_factory=Backend)
        monkeypatch.setattr('process.app_core.integrations.discord.bot.receive_extension', lambda: (_ for _ in ()).throw(AssertionError('Voice dependency loaded')))
        bot.preferences.set(42, 'audio', True) # Old preference cannot re-enable attachments.
        assert bot.voice_for(channel) is None
        with pytest.raises(ValueError, match='paused'): await bot.join_call(None, receive=True)
        await bot.process_job(Job(channel, user, 'hi'))
        assert sent == ['Text reply'] and calls == ['/api/discord/chat']
        await bot.close()
    asyncio.run(run())


def test_whiteboard_changes_send_images_without_polling_or_audio(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot
    async def run():
        calls, sent = [], []
        class Backend:
            def __init__(self, *args): self.ready = asyncio.Event()
            async def request(self, method, path, **kwargs):
                calls.append(path)
                assert path == '/api/whiteboard/image'
                return b'PNG'
            async def close(self): pass
        async def send(content=None, **kwargs): sent.append(kwargs['file'].filename)
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), backend_factory=Backend)
        bot.board_target = SimpleNamespace(id=42, guild=None, send=send)
        await bot.backend_event({'type':'whiteboard.changed','payload':{'revision':'a'}})
        worker = bot.board_update
        await bot.backend_event({'type':'whiteboard.image','payload':{'revision':'a'}})
        assert bot.board_update is worker
        await worker
        assert calls == ['/api/whiteboard/image'] and sent == ['whiteboard.png']
        await bot.close()
    asyncio.run(run())
