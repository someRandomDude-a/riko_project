import asyncio
import io
import json
from pathlib import Path
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
from process.app_core.kernel.audio_config import audio_sections


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


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost/secret', 'http://user:pass@localhost', 'http://localhost?key=secret', 'http://[::1]:8765'])
def test_discord_backend_url_cannot_send_runtime_secrets_remotely(tmp_path, url):
    with pytest.raises(ValueError): BotSettings.from_env(tmp_path, {'Discord_backend_url': url})


def test_existing_localhost_backend_urls_keep_working_over_ipv4_loopback(tmp_path):
    assert BotSettings.from_env(tmp_path, {'Discord_backend_url': 'http://localhost:8765/'}).backend_url == 'http://127.0.0.1:8765'


def test_the_backend_url_defaults_to_the_backends_port_and_a_started_worker_reaches_its_own_backend(tmp_path, monkeypatch):
    from process.app_core.configuration.paths import DataPaths
    from process.app_core.integrations.discord.access import DiscordAccess
    assert BotSettings.from_env(tmp_path, {}).backend_url == 'http://127.0.0.1:8765'
    assert BotSettings.from_env(tmp_path, {'RIKO_PORT': '9123'}).backend_url == 'http://127.0.0.1:9123'  # as run_server binds
    assert BotSettings.from_env(tmp_path, {'RIKO_PORT': '9123', 'Discord_backend_url': 'http://127.0.0.1:9000'}).backend_url == 'http://127.0.0.1:9000'
    with pytest.raises(ValueError): BotSettings.from_env(tmp_path, {'RIKO_PORT': '80'})
    paths = DataPaths.at(tmp_path)
    paths.env_file.write_text('Discord_backend_url=http://127.0.0.1:8765\n', encoding='utf-8')
    monkeypatch.setenv('Discord_backend_url', 'http://127.0.0.1:9123')  # what DiscordLauncher hands the worker
    assert DiscordAccess(paths).settings().backend_url == 'http://127.0.0.1:9123'
    monkeypatch.delenv('Discord_backend_url')
    monkeypatch.setenv('DISCORD_BACKEND_URL', 'http://127.0.0.1:9124')  # how Windows reports the launcher's variable
    assert DiscordAccess(paths).settings().backend_url == 'http://127.0.0.1:9124'


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
        async def send(content, **kwargs):
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


def test_a_failed_streamed_update_costs_only_that_update_and_the_final_reply_is_still_sent():
    aiohttp = pytest.importorskip('aiohttp')
    async def run():
        sent, drops = [], [aiohttp.ServerDisconnectedError()] # Not an HTTPException or OSError: discord.py passes it through.
        class Message:
            def __init__(self, content): self.content = content
            async def edit(self, *, content): self.content = content
            async def delete(self): raise AssertionError('Nothing is superseded')
        async def send(content, **kwargs):
            if drops: raise drops.pop()
            message = Message(content); sent.append(message); return message
        reply = StreamReply(send)
        reply.feed('Hel'); await reply.timer # The coalesced send hits a brief network drop.
        assert not sent and reply.timer is None
        reply.feed('lo'); await reply.timer # Streaming carries on with the next update.
        assert [message.content for message in sent] == ['Hello']
        await reply.finish('Hello there')
        assert [message.content for message in sent] == ['Hello there']
    asyncio.run(run())


def test_one_failed_discord_chunk_does_not_stop_the_others_and_finish_is_idempotent():
    async def run():
        calls, fail, messages = [], set(), []
        class Message:
            def __init__(self, content): self.index, self.content = len(messages), content
            async def edit(self, *, content):
                calls.append(('edit', self.index))
                if ('edit', self.index) in fail: raise OSError('connection reset')
                self.content = content
            async def delete(self): calls.append(('delete', self.index))
        async def send(content, **kwargs):
            calls.append(('send', len(messages)))
            if ('send', len(messages)) in fail: raise ConnectionResetError('dropped')
            message = Message(content); messages.append(message); return message
        reply = StreamReply(send)
        reply.text = 'a' * 1900 + 'b' * 1900; await reply.flush()
        reply.text = 'A' * 1900 + 'B' * 1900 + 'c' * 1900 + 'd' * 1900 + 'e'
        fail.update({('edit', 0), ('send', 3)})
        with pytest.raises(OSError, match='connection reset'): await reply.flush() # The first failure, after the rest ran.
        assert [message.content[0] for message in messages] == ['a', 'B', 'c'] # Chunk 4 never posts ahead of chunk 3.
        fail.clear(); await reply.flush()
        assert [message.content[0] for message in messages] == ['A', 'B', 'c', 'd', 'e']
        await reply.finish('short')
        assert messages[0].content == 'short' and calls[-4:] == [('delete', 1), ('delete', 2), ('delete', 3), ('delete', 4)]
        count = len(calls); await reply.finish(); await reply.finish('other')
        assert len(calls) == count and messages[0].content == 'short'
        attempts = []
        async def offline(content, **kwargs): attempts.append(content); raise ConnectionResetError('offline')
        lost = StreamReply(offline)
        with pytest.raises(ConnectionResetError): await lost.finish('final') # An undelivered final reply is reported once.
        await lost.finish() # The turn queue's cleanup call neither retries nor raises again.
        assert attempts == ['final']
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
    session = SimpleNamespace(is_open=True, cancel_turn=lambda turn_id: turn_id == identity and not calls.append('cancel'),
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
    session = SimpleNamespace(is_open=True)
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
    app = FastAPI(); app.include_router(api.create_router(lambda: SimpleNamespace(is_open=True, config=None)))
    client = TestClient(app)
    assert client.post('/api/discord/speech', json={'text':'hello'}).status_code == 503
    state[0] = True
    result = client.post('/api/discord/speech', json={'text':'hello'})
    assert result.status_code == 200 and result.headers['content-type'] == 'audio/wav'


def test_exported_tts_reuses_http_payload_and_never_opens_sound_device(tmp_path, monkeypatch):
    config = SimpleNamespace(root=tmp_path, **audio_sections({'sovits_ping_config':{'sample_rate':16000, 'text_lang':'en'}}))
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
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json')
        await bot.add_cog(CompanionCommands(bot))
        names = {command.name for command in bot.tree.get_commands()}
        assert {'chat','speak','transcribe','join','leave','listen','stop','settings','messages','tool_policy','camera','tasks','initiative','whiteboard','resources','animation','memory','memories','history','edit','task_create','task_update','status','reasoning'} <= names
        assert 'audio' not in names
        assert bot.processor is None and bot.backend.http is None
        await bot.close()
    asyncio.run(run())


def test_resources_summary_keeps_memory_kind_scope_and_the_model_gpu_but_no_process_ids():
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.commands import resource_summary
    metal = {'key': 'ggml:MTL0', 'index': 0, 'name': 'Apple M5 Pro', 'backend': 'MTL', 'ggml_device': 'MTL0', 'llm': True, 'main': True,
        'memory_kind': 'unified', 'memory_scope': 'process', 'total_mib': 38339, 'used_mib': 548, 'free_mib': 37791, 'owned_mib': 548,
        'other_mib': None, 'attribution_complete': False, 'owned_processes': [{'pid': 4242, 'used_mib': 548, 'category': 'Python'}],
        'pci_bus_id': None, 'uuid': None}
    summary = resource_summary({'available': True, 'gpus': [metal], 'main_key': 'ggml:MTL0', 'source': 'llama.cpp devices (riko_devices)',
        'ggml_devices': [{'index': 2, 'name': 'CPU'}], 'errors': {}})
    assert summary == {'available': True, 'source': 'llama.cpp devices (riko_devices)', 'main_key': 'ggml:MTL0', 'gpus': [{'key': 'ggml:MTL0',
        'name': 'Apple M5 Pro', 'backend': 'MTL', 'memory_kind': 'unified', 'memory_scope': 'process', 'llm': True, 'total_mib': 38339,
        'used_mib': 548, 'free_mib': 37791, 'owned_mib': 548, 'other_mib': None, 'attribution_complete': False}]}
    assert resource_summary({'available': False, 'gpus': [], 'error': 'none'}) == {'available': False, 'source': None, 'main_key': None, 'gpus': []}


def test_stop_is_scoped_to_user_and_turn_and_retains_other_queued_jobs(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        calls = []
        class Backend:
            def __init__(self, *args): pass
            async def request(self, method, path, **kwargs): calls.append((path, kwargs['body']))
            async def close(self): pass
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
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
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
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
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
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
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
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
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
        bot.board_target = SimpleNamespace(id=42, guild=None, send=send)
        await bot.backend_event({'type':'whiteboard.changed','payload':{'revision':'a'}})
        worker = bot.board_update
        await bot.backend_event({'type':'whiteboard.image','payload':{'revision':'a'}})
        assert bot.board_update is worker
        await worker
        assert calls == ['/api/whiteboard/image'] and sent == ['whiteboard.png']
        await bot.close()
    asyncio.run(run())


def test_discord_transport_errors_never_end_the_turn_queue(tmp_path):
    pytest.importorskip('discord'); aiohttp = pytest.importorskip('aiohttp')
    from process.app_core.integrations.discord.backend import BackendError
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        sent, cleaned = [], []
        class Backend:
            def __init__(self, *args): self.ready = asyncio.Event(); self.ready.set()
            async def request(self, method, path, **kwargs):
                job, text = bot.active, kwargs['body']['text']
                if text == 'busy': raise BackendError(409, 'Riko is already handling another turn')
                if job.channel is offline:
                    await bot.backend_event({'type':'chat.delta', 'turn_id':job.id, 'payload':{'text':'partial'}})
                    job.reasoning = StreamReply(job.send); job.reasoning.feed('thinking')
                    async def finish_view(): cleaned.append('view'); raise aiohttp.ServerDisconnectedError()
                    bot.approvals['call'] = SimpleNamespace(job=job, finish=finish_view)
                return {'text': 'Reply to ' + text}
            async def close(self): pass
        async def drop(content=None, **kwargs): raise aiohttp.ServerDisconnectedError() # Every Discord call drops (Wi-Fi roam).
        async def deliver(content=None, **kwargs): sent.append(content); return SimpleNamespace(content=content)
        offline, online = SimpleNamespace(id=42, guild=None, send=drop), SimpleNamespace(id=42, guild=None, send=deliver)
        user = SimpleNamespace(id=1, display_name='Owner')
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
        bot.processor = asyncio.create_task(bot.process_queue())
        for channel, text in ((offline, 'lost'), (online, 'busy'), (online, 'hello')): bot.queue.put_nowait(Job(channel, user, text))
        await asyncio.wait_for(bot.queue.join(), 10) # The bound only catches a dead queue; a passing run returns at once.
        assert not bot.processor.done() and bot.active is None and not bot.approvals and cleaned == ['view']
        assert sent == ['Riko is already handling another turn', 'Reply to hello'] # The backend's busy detail still reaches the user.
        await bot.close()
        assert bot.processor.cancelled()
    asyncio.run(run())


def test_turn_queue_restarts_after_an_unexpected_failure_and_stops_only_when_cancelled(tmp_path):
    pytest.importorskip('discord')
    from process.app_core.integrations.discord.bot import CompanionBot, Job
    async def run():
        sent, entered = [], asyncio.Event()
        class Backend:
            def __init__(self, *args): self.ready = asyncio.Event(); self.ready.set()
            async def request(self, method, path, **kwargs):
                if kwargs['body']['text'] == 'hang': entered.set(); await asyncio.Event().wait()
                return {'text': 'Reply to ' + kwargs['body']['text']}
            async def close(self): pass
        async def deliver(content=None, **kwargs): sent.append(content); return SimpleNamespace(content=content)
        channel, user = SimpleNamespace(id=42, guild=None, send=deliver), SimpleNamespace(id=1, display_name='Owner')
        bot = CompanionBot(BotSettings(tmp_path, admins=frozenset({1})), preferences=tmp_path / 'discord_preferences.json', backend_factory=Backend)
        finish_job = bot.finish_job
        async def bug(job): bot.finish_job = finish_job; raise RuntimeError('unexpected bug')
        bot.finish_job = bug
        bot.start_processor(); first = bot.processor
        bot.queue.put_nowait(Job(channel, user, 'one')); bot.queue.put_nowait(Job(channel, user, 'two'))
        await asyncio.wait_for(bot.queue.join(), 10)
        assert first.done() and isinstance(first.exception(), RuntimeError)
        assert bot.processor is not first and not bot.processor.done() and bot.active is None
        assert sent == ['Reply to one', 'Reply to two']
        second = bot.processor
        bot.queue.put_nowait(Job(channel, user, 'hang')); await asyncio.wait_for(entered.wait(), 10)
        second.cancel() # Cancelled mid-turn outside close() (loop shutdown): the queue stops instead of swallowing it.
        await asyncio.wait_for(asyncio.gather(second, return_exceptions=True), 10)
        assert second.cancelled() and bot.processor is second and bot.active is None
        await bot.close()
    asyncio.run(run())


def test_a_chunk_retried_after_an_ambiguous_failure_is_posted_once():
    import aiohttp
    async def scenario():
        posted = {}  # Discord's view: one message per nonce, as with enforce_nonce
        class Message:
            def __init__(self, content): self.content = content
            async def edit(self, content): self.content = content
        calls = []
        async def send(content, nonce=None, **kwargs):
            calls.append(nonce)
            message = posted.setdefault(nonce, Message(content))
            if len(calls) == 1: raise aiohttp.ServerDisconnectedError()  # created, but the response was lost
            return message
        reply = StreamReply(send)
        reply.feed('Hel')
        await reply.timer  # the coalesced update fails after Discord made the message
        await reply.finish('Hello there')
        return posted, calls
    posted, calls = asyncio.run(scenario())
    assert len(posted) == 1 and [message.content for message in posted.values()] == ['Hello there'] and len(set(calls)) == 1
