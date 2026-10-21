"""User files are replaced durably: a brief Windows sharing violation is retried, a failed write leaves the original
and no temporary file, macOS flushes the drive cache, and a history save that fails after a reply never fails the turn.
Sharing violations are simulated (os.replace raising PermissionError, as MoveFileEx does), so this runs on every OS."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from process.app_core.configuration.config import MemoryConfig
from process.app_core.conversation.chat import ChatService
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import event_bus
from process.app_core.integrations.discord.preferences import Preferences
from process.app_core.persistence import atomic
from process.app_core.persistence.atomic import atomic_write
from process.app_core.persistence.memory import MemoryStore
from process.app_core.tools.approval import ToolApprovals
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch): monkeypatch.setattr(atomic, 'REPLACE_BACKOFF_SECONDS', 0)


def sharing_violation(monkeypatch, directory, times):
    """os.replace as Windows behaves while another program holds a file in directory: denied `times` times, then it
    works. Other files (another thread's) are never affected."""
    real, calls = os.replace, []
    def replace(source, target):
        if Path(target).parent != directory: return real(source, target)
        calls.append(Path(target).name)
        if len(calls) <= times: raise PermissionError(13, 'The process cannot access the file because it is being used by another process', str(target))
        return real(source, target)
    monkeypatch.setattr(os, 'replace', replace)
    return calls


def provider(text='hi'):
    def generate(messages, on_delta=None, **options):
        if on_delta: on_delta(text)  # streamed (and spoken) before the history is saved
        return ModelResponse(ChatMessage('assistant', text))
    stub = BaseProvider()
    stub.generate = generate
    return stub


def test_a_brief_sharing_violation_is_retried_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    path.write_text('old', encoding='utf-8')
    calls = sharing_violation(monkeypatch, tmp_path, 2)
    atomic_write(path, 'new')
    assert path.read_text(encoding='utf-8') == 'new' and calls == ['data.json'] * 3
    assert sorted(os.listdir(tmp_path)) == ['data.json']


def test_a_lasting_violation_fails_after_bounded_retries_with_the_original_untouched(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    path.write_text('old', encoding='utf-8')
    calls = sharing_violation(monkeypatch, tmp_path, 10 ** 6)
    with pytest.raises(PermissionError): atomic_write(path, 'new')
    assert len(calls) == atomic.REPLACE_ATTEMPTS
    assert path.read_text(encoding='utf-8') == 'old' and sorted(os.listdir(tmp_path)) == ['data.json']


def test_a_write_that_cannot_be_synced_removes_its_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    path.write_text('old', encoding='utf-8')
    def full(descriptor): raise OSError(28, 'No space left on device')
    monkeypatch.setattr(atomic, 'sync_file', full)
    with pytest.raises(OSError): atomic_write(path, 'new')
    assert path.read_text(encoding='utf-8') == 'old' and sorted(os.listdir(tmp_path)) == ['data.json']


def test_macos_flushes_the_drive_cache_and_falls_back_where_a_volume_cannot(monkeypatch):
    assert (atomic.F_FULLFSYNC is not None) == (sys.platform == 'darwin')  # detected on every OS this runs on
    calls, fsync, ours = [], os.fsync, 10 ** 6  # a descriptor no other thread can be using
    monkeypatch.setattr(os, 'fsync', lambda descriptor: calls.append(('fsync', descriptor)) if descriptor == ours else fsync(descriptor))
    monkeypatch.setattr(atomic, 'F_FULLFSYNC', 51)
    monkeypatch.setattr(atomic, 'fcntl', lambda descriptor, command: calls.append(('F_FULLFSYNC', descriptor, command)))
    atomic.sync_file(ours)
    assert calls == [('F_FULLFSYNC', ours, 51)]  # plain fsync stops at the drive's cache on macOS
    def unsupported(descriptor, command): raise OSError(45, 'Operation not supported')
    monkeypatch.setattr(atomic, 'fcntl', unsupported)
    calls.clear()
    atomic.sync_file(ours)
    assert calls == [('fsync', ours)]
    monkeypatch.setattr(atomic, 'F_FULLFSYNC', None)  # Linux and Windows
    calls.clear()
    atomic.sync_file(ours)
    assert calls == [('fsync', ours)]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits')
def test_an_existing_file_keeps_its_permissions_and_a_new_one_is_owner_only(tmp_path):
    shared, private = tmp_path / 'shared.json', tmp_path / 'private.json'
    shared.write_text('old', encoding='utf-8')
    shared.chmod(0o644)
    atomic_write(shared, 'new')
    atomic_write(private, 'new')
    assert (shared.stat().st_mode & 0o777, private.stat().st_mode & 0o777) == (0o644, 0o600)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits')
def test_an_explicit_mode_replaces_the_existing_files_permissions(tmp_path):
    backup = tmp_path / 'character_config.yaml.previous'
    backup.write_text('old', encoding='utf-8'); backup.chmod(0o644)
    atomic_write(backup, 'api_key: x', mode=0o600)  # settings_store's backup: owner-only whatever an older build left
    assert backup.stat().st_mode & 0o777 == 0o600 and backup.read_text() == 'api_key: x'


def test_a_leftover_fixed_temporary_name_no_longer_blocks_history_saves(tmp_path):
    path = tmp_path / 'chat_history.json'
    (tmp_path / 'chat_history.json.tmp').mkdir()  # e.g. a stale or still-scanned temporary file under the old fixed name
    service = ChatService(provider(), system_prompt='test', history_file=path)
    service.respond('hello')
    assert [record['content'] for record in json.loads(path.read_text(encoding='utf-8'))] == ['User: hello', 'hi']


def test_memory_capture_before_inference_survives_a_brief_sharing_violation(tmp_path, monkeypatch):
    config = MemoryConfig(store_file=tmp_path / 'memories.json', system1_enabled=False, embeddings_enabled=False)
    memory = MemoryStore(config, start_worker=False)
    try:
        calls = sharing_violation(monkeypatch, tmp_path, 1)
        record = memory.remember('User: my favourite colour is violet')
        assert calls == ['memories.json'] * 2
    finally: memory.close()
    restarted = MemoryStore(config, start_worker=False)
    try: assert [item['id'] for item in restarted.list_records()] == [record.id]
    finally: restarted.close()


def test_other_json_stores_retry_a_brief_sharing_violation(tmp_path, monkeypatch):
    gate = ToolApprovals(tmp_path / 'tool_approvals.json')
    try:
        sharing_violation(monkeypatch, tmp_path, 1)
        gate.configure({'calculator': True}, names=['calculator'])
    finally: gate.close()
    sharing_violation(monkeypatch, tmp_path, 1)
    Preferences(tmp_path / 'discord_preferences.json').set(42, 'audio', True)
    assert json.loads((tmp_path / 'tool_approvals.json').read_text(encoding='utf-8')) == {'calculator': True, 'version': 2, 'sources': {'riko': {'calculator': True}}}
    assert Preferences(tmp_path / 'discord_preferences.json').get(42, 'audio') is True


class FakeSpeech:
    def __init__(self, *args): pass
    def submit(self, *args): return False
    def submit_clip(self, *args, **kwargs): return False
    def cancel(self): pass
    def close(self): pass


def test_a_history_save_that_fails_after_the_reply_keeps_the_finished_turn_and_the_next_save_retries(tmp_path, monkeypatch):
    from process.app_core.runtime import session as session_module
    monkeypatch.setattr(session_module, 'SpeechQueue', FakeSpeech)
    path = tmp_path / 'chat_history.json'
    chat = ChatService(provider('A complete answer.'), system_prompt='test', history_file=path)
    config = SimpleNamespace(raw={}, root=tmp_path, paths=DataPaths.at(tmp_path), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = session_module.SessionManager(config, chat, DesktopState())
    events, real = [], os.replace
    unsubscribe = event_bus.subscribe(lambda event: events.append(event.type))
    try:
        sharing_violation(monkeypatch, tmp_path, 10 ** 6)  # held for the whole turn: every save of it fails
        assert session.respond('hello').message.content == 'A complete answer.'
        assert 'chat.completed' in events and 'model.error' not in events and 'chat.cancelled' not in events
        assert [(m.role, m.content) for m in chat.history] == [('user', 'User: hello'), ('assistant', 'A complete answer.')]
        assert not path.exists()
        monkeypatch.setattr(os, 'replace', real)  # released: the next save writes the whole history
        session.respond('again')
    finally:
        unsubscribe()
        session.close()
    assert [record['content'] for record in json.loads(path.read_text(encoding='utf-8'))] == [
        'User: hello', 'A complete answer.', 'User: again', 'A complete answer.']


def test_a_temporary_file_left_by_a_killed_write_is_swept_by_the_next_one(tmp_path):
    import os, time
    from process.app_core.persistence.atomic import atomic_write
    target = tmp_path / 'chat_history.json'
    stale, live = tmp_path / '.chat_history.json.abc123.tmp', tmp_path / '.chat_history.json.def456.tmp'
    stale.write_text('[]', encoding='utf-8'); live.write_text('[]', encoding='utf-8')
    old = time.time() - 3600
    os.utime(stale, (old, old))
    atomic_write(target, '[1]')
    assert target.read_text(encoding='utf-8') == '[1]' and not stale.exists() and live.exists()  # a recent one may be in use
