"""User data is never silently erased or overwritten: finished replies survive Stop/Sleep/shutdown,
failed or interrupted turns keep the user's message, and unreadable stores fail closed."""
import asyncio
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from process.app_core.conversation.chat import ChatDeps, ChatService
from process.app_core.conversation.history import ConversationHistory
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.desktop.state import DesktopState
from process.app_core.emotion.julia import JuliaEmotionEngine
from process.app_core.emotion.probe import EmotionProbe, ProbeConfig, build_network
from process.app_core.events.bus import EventBus, RuntimeEvent, event_bus
from process.app_core.kernel.cancellation import TurnCancelled
from process.app_core.runtime.session import SessionManager
from process.app_core.tools.approval import ToolApprovals
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


class FakeSpeech:
    """Accepts nothing, like muted audio or an offline GPT-SoVITS."""
    def __init__(self, *args): pass
    def submit(self, *args): return False
    def submit_clip(self, *args, **kwargs): return False
    def cancel(self): pass
    def close(self): pass


@pytest.fixture
def session_parts(monkeypatch):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', FakeSpeech)
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    config = SimpleNamespace(raw={}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = SessionManager(config, chat, DesktopState())
    events = []
    unsubscribe = event_bus.subscribe(lambda event: events.append(event.type))
    yield session, chat, events
    unsubscribe()
    session.close()


def answer(chat, text):
    def respond(user_text, user_name, **kwargs):
        kwargs['on_delta'](text)
        chat.conversation.append([ChatMessage('user', f'{user_name}: {user_text}'), *kwargs['response_history'](text)])
        return ModelResponse(ChatMessage('assistant', text))
    return respond


def contents(chat): return [(message.role, message.content) for message in chat.conversation.snapshot()]


def test_stop_sleep_and_shutdown_after_a_finished_reply_keep_it(session_parts):
    session, chat, events = session_parts
    chat.respond = answer(chat, 'A complete answer.')
    session.respond('hello')
    finished = contents(chat)
    session.cancel()  # Stop or Sleep after the reply ended
    session.close()  # backend shutdown
    assert contents(chat) == finished == [('user', 'User: hello'), ('assistant', 'A complete answer.')]
    assert 'chat.interrupted' not in events


def test_interrupting_a_reply_whose_speech_failed_keeps_the_text_the_user_read(session_parts):
    session, chat, _ = session_parts
    def respond(text, user_name, **kwargs):
        kwargs['on_delta']('Visible but never spoken. ')
        session._playback_event(RuntimeEvent('speech.error', {'error': 'TTS offline'}, turn_id=session._active_turn))
        session.cancel()
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('hello')
    assert contents(chat) == [('user', 'User: hello'), ('assistant', 'Visible but never spoken. ')]


def test_a_failed_reply_keeps_the_users_message_and_is_not_an_interruption(session_parts):
    session, chat, events = session_parts
    def respond(text, user_name, **kwargs):
        kwargs['on_delta']('Partial ')
        raise RuntimeError('Maximum tool-call iterations exceeded')
    chat.respond = respond
    with pytest.raises(RuntimeError): session.respond('hello')
    assert contents(chat) == [('user', 'User: hello'), ('assistant', 'Partial ')]
    assert 'model.error' in events and 'chat.interrupted' not in events


def test_shutdown_during_a_reply_keeps_the_users_message(session_parts):
    session, chat, _ = session_parts
    def respond(text, user_name, **kwargs):
        kwargs['on_delta']('Half an ans')
        session.close()  # shutdown arrives mid-reply
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('hello')
    assert chat.conversation.snapshot()[0].content == 'User: hello'


def test_a_bad_voice_setting_fails_the_turn_with_the_real_error_and_keeps_the_message(session_parts):
    session, chat, events = session_parts
    # load_config rejects this value (tests/test_config.py); a voice section built another way must still fail the turn cleanly.
    session.config.voice = replace(session.config.voice, interjection_debounce_seconds=0)
    with pytest.raises(ValueError, match='debounce'): session.respond('hello')
    assert contents(chat) == [('user', 'User: hello')]
    assert 'model.error' in events


def test_only_real_speech_failures_count_as_not_heard(session_parts):
    session, _, _ = session_parts
    session._active_turn = 'turn'
    session._playback_event(RuntimeEvent('speech.error', {'error': 'Speech queue full', 'queued': False}, turn_id='turn'))
    assert session._speech_heard()  # later sentences still play
    session._playback_event(RuntimeEvent('speech.error', {'error': 'TTS offline'}, turn_id='turn'))
    assert not session._speech_heard()


def test_muted_voice_continuation_drops_the_cut_reply_fragment(session_parts):
    session, chat, _ = session_parts
    session.state.toggle_audio()  # muted
    def respond(text, user_name, **kwargs):
        anchor = session.voice_anchor()  # the user is still talking; nothing is visible yet
        kwargs['on_delta']('Sure, the capital of')
        assert session.voice_transcript('of France', 1, 2, anchor)
        raise TurnCancelled()
    chat.respond = respond
    with pytest.raises(TurnCancelled): session.respond('What is the capital')
    assert contents(chat) == [('user', 'User: What is the capital\nof France')]


def test_failed_replies_are_archived_as_errors_not_interruptions(tmp_path):
    from process.app_core.persistence.conversation_store import ConversationStore
    store = ConversationStore(tmp_path / 'conversations.sqlite3')
    try:
        store.observe(RuntimeEvent('chat.delta', {'text': 'partial'}, turn_id='turn'))
        store.observe(RuntimeEvent('model.error', {'error': 'provider failed'}, turn_id='turn'))
        message = store.page()['messages'][-1]
        assert message['status'] == 'error' and not message.get('interrupted')
    finally: store.close()


def test_turn_lock_is_released_when_superseding_previous_speech_fails(session_parts):
    session, chat, _ = session_parts
    session._speech_pending = 1  # the previous reply is still queued for playback
    def broken(): raise OSError('audio device busy')
    session.speech.cancel = broken
    with pytest.raises(OSError): session.respond('first')
    session.speech.cancel = lambda: None
    chat.respond = answer(chat, 'Second reply.')
    session.respond('second')  # must not fail with "already handling another turn"
    assert chat.conversation.snapshot()[-1].content == 'Second reply.'


def provider(text='hi'):
    return SimpleNamespace(generate=lambda *args, **kwargs: ModelResponse(ChatMessage('assistant', text)))


def test_unreadable_chat_history_is_kept_aside_before_the_next_save(tmp_path):
    path = tmp_path / 'chat_history.json'
    path.write_text('[{"role": "user", "content": "keep me"},]', encoding='utf-8')  # hand edit left a trailing comma
    original = path.read_bytes()
    service = ChatService(provider(), system_prompt='test', history_file=path)
    assert service.history == []
    service.respond('hello')
    backups = list(tmp_path.glob('chat_history.json.unreadable-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert [record['content'] for record in json.loads(path.read_text(encoding='utf-8'))] == ['User: hello', 'hi']


def test_chat_history_is_never_saved_over_a_file_that_could_not_be_kept(tmp_path, monkeypatch):
    path = tmp_path / 'chat_history.json'
    path.write_text('{"not": "a list"}', encoding='utf-8')
    original = path.read_bytes()
    def locked(path): raise PermissionError('locked by another program')
    monkeypatch.setattr('process.app_core.conversation.history.preserve_unreadable', locked)
    service = ChatService(provider(), system_prompt='test', history_file=path)
    service.respond('hello')
    assert path.read_bytes() == original


def test_permission_denied_files_are_moved_aside_instead_of_blocking_forever(tmp_path, monkeypatch):
    history, approvals = tmp_path / 'chat_history.json', tmp_path / 'tool_approvals.json'
    history.write_text('[]', encoding='utf-8')
    approvals.write_text('{"calculator": true}', encoding='utf-8')
    # e.g. files left behind by a run under another account. Simulated rather than chmod(0), which on Windows only sets
    # the read-only attribute: the first read of each file is denied, as the OS would deny it.
    denied, read_text = {history, approvals}, Path.read_text
    def guarded(self, *args, **kwargs):
        if self in denied: denied.discard(self); raise PermissionError(13, 'Permission denied', str(self))
        return read_text(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    service = ChatService(provider(), system_prompt='test', history_file=history)
    service.respond('hello')
    gate = ToolApprovals(approvals)
    try:
        assert gate.snapshot()['default_required'] is True  # the UI shows what is enforced
        gate.configure({'calculator': False}, names=['calculator', 'task_create'])
    finally: gate.close()
    assert not denied  # both reads were denied once
    assert [record['content'] for record in json.loads(history.read_text(encoding='utf-8'))] == ['User: hello', 'hi']
    assert json.loads(approvals.read_text(encoding='utf-8')) == {'calculator': False, 'task_create': True, 'version': 2, 'sources': {'riko': {'calculator': False, 'task_create': True}}}
    assert {kept.name.split('.unreadable-')[0] for kept in tmp_path.glob('*.unreadable-*')} == {history.name, approvals.name}


def test_unreadable_tool_approval_policy_requires_approval_for_every_tool(tmp_path):
    path = tmp_path / 'tool_approvals.json'
    path.write_text('{"task_create": true,', encoding='utf-8')
    gate = ToolApprovals(path, default=False)
    try:
        assert gate.snapshot()['error']
        assert gate.authorize('calculator', {}, 'call-1', timeout=0.05) is False  # not silently allowed
        gate.configure({'calculator': False}, names=['calculator', 'task_create'])
        assert json.loads(path.read_text(encoding='utf-8')) == {'calculator': False, 'task_create': True, 'version': 2, 'sources': {'riko': {'calculator': False, 'task_create': True}}}
        assert gate.snapshot()['error'] == ''
        assert len(list(tmp_path.glob('tool_approvals.json.unreadable-*'))) == 1
    finally: gate.close()


@pytest.fixture
def todo(monkeypatch, tmp_path):
    from process.app_core.tools.builtin import todo_list
    monkeypatch.setattr(todo_list.Tool, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(todo_list.Tool, 'DATA_FILE', tmp_path / 'tasks.json')
    return todo_list.Tool({})


def saved_tasks(tmp_path): return json.loads((tmp_path / 'tasks.json').read_text(encoding='utf-8'))['tasks']


def test_todo_ids_stay_stable_across_removals_in_one_reply(todo, tmp_path):
    for text in ('one', 'two', 'three', 'four'): todo.execute(action='add', task=text)
    todo.execute(action='remove', task_id='2')
    todo.execute(action='remove', task_id='3')  # the model's second call, using IDs from the same list
    assert [task['text'] for task in saved_tasks(tmp_path)] == ['one', 'four']


def test_todo_ids_are_never_reused_after_removal_or_clear(todo, tmp_path):
    for text in ('one', 'two', 'three'): todo.execute(action='add', task=text)
    todo.execute(action='remove', task_id='3')
    assert '(ID: 4)' in todo.execute(action='add', task='four')
    with pytest.raises(ValueError): todo.execute(action='complete', task_id='3')  # stale ID: no silent match
    todo.execute(action='clear')
    assert '(ID: 5)' in todo.execute(action='add', task='five')


def test_todo_lists_saved_before_ids_keep_their_numbers(todo, tmp_path):
    (tmp_path / 'tasks.json').write_text(json.dumps([{'text': 'a', 'done': False}, {'text': 'b', 'done': False}]), encoding='utf-8')
    todo.execute(action='complete', task_id='2')
    assert [(task['id'], task['done']) for task in saved_tasks(tmp_path)] == [(1, False), (2, True)]


def test_unreadable_todo_list_is_kept_and_reported_not_replaced(todo, tmp_path):
    (tmp_path / 'tasks.json').write_text('[{"text": "buy milk"', encoding='utf-8')
    original = (tmp_path / 'tasks.json').read_bytes()
    with pytest.raises(RuntimeError, match='kept as'): todo.execute(action='add', task='new')
    backups = list(tmp_path.glob('tasks.json.unreadable-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    with pytest.raises(ValueError): todo.execute(action='remove', task_id='9')  # bad input is a tool error


def teacher():
    engine = JuliaEmotionEngine(None)
    engine._load_attempted = True
    engine._model = SimpleNamespace(predict=lambda **kw: {'answers': {
        'emotion': {'choice': 'love', 'max_probability': .9}, 'intensity': {'score': 2}, 'valence': {'score': 3}}})
    return engine


def probe_config():
    return ProbeConfig.from_raw({'hidden_units': [32, 16], 'rank': 8, 'min_samples': 32, 'retrain_every': 32, 'epochs': 2})


def trained_probe(directory, training):
    probe = EmotionProbe(directory, {'model': 'one'}, teacher(), probe_config(), idle=lambda: False, training_directory=training)
    probe.samples.append((torch.ones(256), [0, .5, 0., .5], 'message-one'))
    probe.network = build_network(probe.config).eval().requires_grad_(False)
    probe.validation = dict(agreement=.9, macro_f1=.8, rmse=.1, classes=3, validation_samples=32)  # passes the defaults
    probe.close()  # writes training.pt, examples.json and probe.pt (weights)
    return probe


def reopen(directory, training, **config):
    config = ProbeConfig.from_raw({'hidden_units': [32, 16], 'rank': 8, 'min_samples': 32, 'retrain_every': 32, 'epochs': 2, **config})
    return EmotionProbe(directory, {'model': 'one'}, teacher(), config, idle=lambda: False, training_directory=training)


def test_trained_weights_survive_a_restart_with_stricter_thresholds(tmp_path):
    directory, training = tmp_path / 'probe', tmp_path / 'training'
    trained_probe(directory, training)
    shutil.rmtree(training)  # the weights in probe.pt are now the only copy
    strict = reopen(directory, training, min_macro_f1=0.95)
    strict.validation = {}  # something saved later (e.g. an unqualified training run)
    strict.close()
    assert not strict.ready
    restored = reopen(directory, training)  # thresholds relaxed again
    try: assert restored.ready
    finally: restored.close()


def test_unreadable_legacy_probe_file_is_ignored_not_moved_and_saving_still_works(tmp_path):
    legacy = tmp_path / 'legacy'
    probe = EmotionProbe(tmp_path / 'probe', {'model': 'one'}, teacher(), probe_config(), idle=lambda: False,
        training_directory=tmp_path / 'training', legacy_directory=legacy)
    probe.close()
    probe.legacy_path.parent.mkdir(parents=True, exist_ok=True)
    probe.legacy_path.write_bytes(b'written by an old build')
    probe.path.unlink(); probe.data_path.unlink()  # only the unreadable legacy file is left
    restored = EmotionProbe(tmp_path / 'probe', {'model': 'one'}, teacher(), probe_config(), idle=lambda: False,
        training_directory=tmp_path / 'training', legacy_directory=legacy)
    restored.close()
    assert not restored.save_blocked and restored.path.exists()
    assert probe.legacy_path.read_bytes() == b'written by an old build'


def test_unreadable_probe_training_data_is_kept_and_the_weights_still_restore(tmp_path):
    directory, training = tmp_path / 'probe', tmp_path / 'training'
    first = trained_probe(directory, training)
    first.data_path.write_bytes(b'truncated by a disk error')
    restored = reopen(directory, training)
    try:
        assert restored.ready and restored.error
    finally: restored.close()
    kept = list(first.data_path.parent.glob('training.pt.unreadable-*'))
    assert len(kept) == 1 and kept[0].read_bytes() == b'truncated by a disk error'


def test_trained_probe_restores_after_its_training_data_was_deleted(tmp_path):
    directory, training = tmp_path / 'probe', tmp_path / 'training'
    trained_probe(directory, training)
    shutil.rmtree(training)
    restored = reopen(directory, training)
    try: assert restored.ready and len(restored.samples) == 0
    finally: restored.close()
    assert torch.load(restored.path, weights_only=True)['weights'] is not None  # shutdown kept the weights


def test_event_socket_bootstrap_builds_its_snapshot_off_the_event_loop():
    from process.app_core.events.stream import stream_events
    seen = {}
    def snapshot():
        try: asyncio.get_running_loop(); seen['on_loop'] = True
        except RuntimeError: seen['on_loop'] = False
        return {}
    class Socket:
        def __init__(self): self.sent = []
        async def accept(self): pass
        async def send_json(self, data): self.sent.append(data)
        async def receive(self):
            await asyncio.sleep(0.2)
            return {'type': 'websocket.disconnect'}
        async def close(self, code=1000): pass
    socket = Socket()
    asyncio.run(stream_events(socket, EventBus(), snapshot))
    assert seen == {'on_loop': False} and socket.sent[0]['type'] == 'state.snapshot'
