"""One owner of the conversation history (ConversationHistory): each change is one step under its lock and is saved, a
failed save is retried by the next change, readers get copies, and nothing else in the code mutates a history list."""
import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from process.app_core.conversation.chat import ChatDeps, ChatService
from process.app_core.conversation.history import ConversationHistory
from process.app_core.desktop.state import DesktopState
from process.app_core.kernel.cancellation import TurnCancelled
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.runtime.session import SessionManager
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths
from test_private_access import CODE


def saved(path): return [(record['role'], record['content']) for record in json.loads(path.read_text(encoding='utf-8'))]


def test_every_change_is_saved_and_a_snapshot_is_a_copy(tmp_path, monkeypatch):
    path = tmp_path / 'chat_history.json'
    history = ConversationHistory(path)
    assert history.append([ChatMessage('user', 'User: hi'), ChatMessage('assistant', 'Hello')]) == 0
    assert saved(path) == [('user', 'User: hi'), ('assistant', 'Hello')]
    copy = history.snapshot()
    copy.append(ChatMessage('user', 'not recorded'))
    assert len(history) == 2 and history.snapshot(1) == [copy[1]]
    assert history.replace(1, 2, [ChatMessage('assistant', 'Hel')]) == 2
    assert saved(path)[-1] == ('assistant', 'Hel')
    writes = []
    monkeypatch.setattr('process.app_core.conversation.history.atomic_write', lambda *args: writes.append(args))
    assert history.rewrite(lambda messages: 'nothing changed') == 'nothing changed' and writes == []  # nothing to save
    assert ConversationHistory(path).snapshot()[0].content == 'User: hi'  # what the next start loads


def test_a_failed_save_is_retried_by_the_next_change_even_one_that_changes_nothing(tmp_path, monkeypatch):
    from process.app_core.persistence import atomic
    path, failures = tmp_path / 'chat_history.json', [OSError('disk full')]
    def flaky(*args):
        if failures: raise failures.pop()
        atomic.atomic_write(*args)
    monkeypatch.setattr('process.app_core.conversation.history.atomic_write', flaky)
    history = ConversationHistory(path)
    history.append([ChatMessage('assistant', 'Shown before the save failed')])  # logged, never raised: the reply was shown
    assert not path.exists() and len(history) == 1
    history.rewrite(lambda messages: None)  # a cut that changed nothing still catches the file up
    assert saved(path) == [('assistant', 'Shown before the save failed')]


def test_a_rewrite_is_one_step_for_readers_and_the_other_writer(tmp_path):
    history = ConversationHistory(tmp_path / 'chat_history.json')
    history.append([ChatMessage('user', 'User: hi'), ChatMessage('assistant', 'A long reply')])
    inside, release, results = threading.Event(), threading.Event(), {}
    def cut(messages):  # SessionManager's rewrite of a cut reply, held open half way
        messages[1:2] = [ChatMessage('assistant', 'A long')]
        inside.set()
        assert release.wait(5)
        messages.append(ChatMessage('user', '[speaking over you] wait'))
    def read(): results['read'] = [m.content for m in history.snapshot()]
    def commit(): history.append([ChatMessage('assistant', 'Committed meanwhile')])  # ChatService's commit, on the turn thread
    rewriting = threading.Thread(target=history.rewrite, args=(cut,), daemon=True)
    rewriting.start()
    assert inside.wait(5)
    reader, writer = threading.Thread(target=read, daemon=True), threading.Thread(target=commit, daemon=True)
    reader.start(); writer.start()
    reader.join(.2); writer.join(.2)
    assert reader.is_alive() and writer.is_alive()  # both wait for the rewrite instead of seeing or racing half of it
    release.set()
    for thread in (rewriting, reader, writer): thread.join(5)
    after = ['User: hi', 'A long', '[speaking over you] wait']
    assert results['read'] in (after, [*after, 'Committed meanwhile'])  # whichever waiter went first, never the half-cut reply
    assert [m.content for m in history.snapshot()] == ['User: hi', 'A long', '[speaking over you] wait', 'Committed meanwhile']  # nothing lost


def test_chat_history_is_a_copy_and_the_reply_is_committed_through_the_owner(tmp_path):
    provider = SimpleNamespace(generate=lambda *args, **kwargs: ModelResponse(ChatMessage('assistant', 'Hello')))
    chat = ChatService(provider, system_prompt='test', history_file=tmp_path / 'chat_history.json')
    chat.respond('hi')
    chat.history.append(ChatMessage('user', 'User: not recorded'))
    assert [m.content for m in chat.history] == ['User: hi', 'Hello'] and len(chat.conversation) == 2
    assert saved(tmp_path / 'chat_history.json') == [('user', 'User: hi'), ('assistant', 'Hello')]
    with pytest.raises(AttributeError): chat.history = []  # read-only: change it through chat.conversation


def test_continued_input_replaces_the_message_a_reader_holds_instead_of_editing_it(monkeypatch):
    class Speech:
        def __init__(self, *args): pass
        def submit(self, *args): return False
        def cancel(self): pass
        def close(self): pass
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    chat = SimpleNamespace(conversation=ConversationHistory(), deps=ChatDeps(), provider=BaseProvider())
    config = SimpleNamespace(raw={'animation': {'enabled': False}}, root=Path('.'), paths=DataPaths.at(Path('.')), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    session = SessionManager(config, chat, DesktopState())
    held = []
    def respond(text, user_name, **kwargs):
        chat.conversation.append([ChatMessage('user', f'{user_name}: {text}')])  # committed, then speech continues it
        held.extend(session.history_snapshot())
        kwargs['on_reasoning']('Thinking')
        assert session.voice_transcript('and tomorrow', 1, 2, session.voice_anchor()) == 'reply'
        raise TurnCancelled()
    chat.respond = respond
    try:
        with pytest.raises(TurnCancelled): session.respond('What about today?')
        assert [m.content for m in held] == ['User: What about today?']  # the reader's copy is untouched
        assert [m.content for m in chat.conversation.snapshot()] == ['User: What about today?\nand tomorrow']
    finally: session.close()


MUTATORS = {'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'sort', 'reverse'}


def history_mutations(tree):
    """Writes to an `<x>.history` list: assignment, item or slice assignment and deletion, or a mutating method call."""
    found = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else node.targets if isinstance(node, ast.Delete) else []
        for target in targets:
            base = target.value if isinstance(target, ast.Subscript) else target
            if isinstance(base, ast.Attribute) and base.attr == 'history' and not (isinstance(base.value, ast.Name) and base.value.id == 'self'): found.append(node.lineno)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'history'): found.append(node.lineno)
    return found


def test_nothing_but_the_owner_mutates_the_history():
    paths = [*sorted((CODE / 'process' / 'app_core').rglob('*.py')), *sorted(CODE.glob('*.py'))]
    found = [f'{path.relative_to(CODE).as_posix()}:{line}' for path in paths for line in history_mutations(ast.parse(path.read_text(encoding='utf-8')))]
    assert not found, 'Change the conversation history through ConversationHistory (chat.conversation):\n' + '\n'.join(found)
    sample = 'chat.history.append(m)\nsession.chat.history[1:2] = []\ndel chat.history[0]\nchat.history = []\nx = chat.history[0]\nchannel.history(limit=3)\n'
    assert sorted(history_mutations(ast.parse(sample))) == [1, 2, 3, 4]
