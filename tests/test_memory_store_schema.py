"""The memory store reads its past and its future: a versioned layout with migrations, fields from another schema kept
and written back, wrongly typed values repaired (the file as it was kept aside) instead of failing every turn, and only
structural damage failing closed with the file untouched."""
from dataclasses import asdict, fields
import json

import pytest

from process.app_core.configuration.config import MemoryConfig
from process.app_core.persistence import memory as memory_module
from process.app_core.persistence.memory import FIELD_CHECKS, STORE_VERSION, MemoryRecord, MemoryStore


def store(path): return MemoryStore(MemoryConfig(store_file=path, system1_enabled=False, embeddings_enabled=False), start_worker=False)


def backups(tmp_path): return sorted(tmp_path.glob('memories.json.unreadable-*'))


def test_every_record_field_is_checked_and_every_older_version_has_a_migration():
    assert set(FIELD_CHECKS) | {'text', 'id'} == {item.name for item in fields(MemoryRecord)}
    assert set(memory_module.STORE_MIGRATIONS) == set(range(1, STORE_VERSION))


def test_wrongly_typed_fields_are_repaired_so_recall_still_works_and_the_file_as_it_was_is_kept(tmp_path):
    path = tmp_path / 'memories.json'
    path.write_text(json.dumps([
        {'id': 'tea', 'text': 'User: I like green tea', 'importance': '0.8', 'created_at': None, 'tags': 'drinks'},  # hand edits
        {'id': 'cats', 'text': 'User: I have two cats', 'importance': 7, 'confidence': True, 'access_count': -1, 'active': 1,
         'classification_status': 'queued', 'formation_context': None},
        {'id': 'old', 'text': 'User: an archived cat memory', 'active': 'false', 'revision': '2'}]), encoding='utf-8')
    original = path.read_bytes()
    memory = store(path)
    try:
        tea, cats, old = memory.list_records()
        assert (tea['importance'], tea['tags'], isinstance(tea['created_at'], str)) == (.8, ['drinks'], True)
        assert (cats['importance'], cats['confidence'], cats['access_count'], cats['active'], cats['classification_status'],
                cats['formation_context']) == (1.0, .5, 0, True, 'complete', {})
        assert (old['active'], old['revision']) == (False, 1)
        recalled = memory.retrieve('green tea cats')  # a TypeError here used to fail every turn
        assert 'green tea' in recalled and 'two cats' in recalled and 'archived' not in recalled
        assert memory.status()['error']  # the user is told, not only the log
        assert path.read_bytes() == original and not backups(tmp_path)  # loading never writes
        memory.remember('User: something new')
    finally: memory.close()
    kept = backups(tmp_path)
    assert len(kept) == 1 and kept[0].read_bytes() == original
    reloaded = store(path)  # repaired values were saved: nothing left to repair, no second copy
    try:
        assert not reloaded.status()['error'] and reloaded.list_records()[0]['importance'] == .8
        assert len(reloaded.list_records()) == 4
    finally: reloaded.close()
    assert backups(tmp_path) == kept


def test_a_repaired_store_is_never_saved_over_when_the_file_as_it_was_cannot_be_kept(tmp_path, monkeypatch):
    path = tmp_path / 'memories.json'
    path.write_text(json.dumps([{'id': 'tea', 'text': 'User: I like green tea', 'importance': '0.8'}]), encoding='utf-8')
    original = path.read_bytes()
    memory = store(path)
    try:
        def locked(path, **kwargs): raise PermissionError('locked by another program')
        monkeypatch.setattr(memory_module, 'preserve_unreadable', locked)
        with pytest.raises(OSError): memory.remember('User: something new')
        assert path.read_bytes() == original
        monkeypatch.undo()
    finally: memory.close()
    assert [kept.read_bytes() for kept in backups(tmp_path)] == [original]


def test_fields_from_another_schema_are_kept_and_written_back_unchanged(tmp_path):
    path = tmp_path / 'memories.json'
    newer = {'id': 'tea', 'text': 'User: I like green tea', 'embedding_version': 3, 'mood': {'valence': .4}}
    path.write_text(json.dumps({'version': STORE_VERSION, 'records': [newer], 'index': {'model': 'x'}}), encoding='utf-8')
    memory = store(path)
    try:
        assert 'green tea' in memory.retrieve('tea')
        assert 'mood' not in memory.list_records()[0]  # not part of this build's records, events or prompts
        memory.update('tea', importance=.9)
        memory.remember('User: something new')
    finally: memory.close()
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['version'] == STORE_VERSION and saved['index'] == {'model': 'x'}
    tea, new = saved['records']
    assert (tea['embedding_version'], tea['mood'], tea['importance']) == (3, {'valence': .4}, .9)
    assert set(new) == {item.name for item in fields(MemoryRecord)}
    assert not backups(tmp_path)  # nothing was repaired


def test_a_legacy_list_with_unknown_keys_loads_untouched_and_stays_a_list_older_builds_can_read(tmp_path):
    path = tmp_path / 'memories.json'
    legacy = [{**asdict(MemoryRecord(text='User: an old memory', id='old')), 'faiss_row': 4}]  # e.g. a downgrade
    path.write_text(json.dumps(legacy), encoding='utf-8')
    original = path.read_bytes()
    memory = store(path)
    try: assert path.read_bytes() == original and memory.list_records()[0]['text'] == 'User: an old memory'
    finally: memory.close()
    assert json.loads(path.read_text(encoding='utf-8')) == legacy  # main and older branches read only the list


def test_a_new_store_is_a_list_and_a_version_bump_wraps_it(tmp_path, monkeypatch):
    from process.app_core.persistence import memory as module
    path = tmp_path / 'memories.json'
    memory = store(path)
    try: memory.remember('User: I like green tea')
    finally: memory.close()
    assert isinstance(json.loads(path.read_text(encoding='utf-8')), list)
    monkeypatch.setattr(module, 'STORE_VERSION', 2)
    monkeypatch.setitem(module.STORE_MIGRATIONS, 1, lambda records: records)
    memory = store(path)
    memory.close()
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['version'] == 2 and saved['records'][0]['text'] == 'User: I like green tea'
    [kept] = backups(tmp_path)  # the list as it was before the migration, for a downgrade
    assert isinstance(json.loads(kept.read_text(encoding='utf-8')), list)


def test_migrations_upgrade_an_older_layout_step_by_step(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_module, 'STORE_VERSION', 3)
    monkeypatch.setattr(memory_module, 'STORE_MIGRATIONS', {
        1: lambda records: [{'text': record.pop('body'), **record} for record in records],
        2: lambda records: [{**record, 'tags': ['migrated']} for record in records]})
    path = tmp_path / 'memories.json'
    path.write_text(json.dumps([{'id': 'a', 'body': 'User: hello'}]), encoding='utf-8')
    memory = store(path)
    try: assert [(r['text'], r['tags']) for r in memory.list_records()] == [('User: hello', ['migrated'])]
    finally: memory.close()
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['version'] == 3 and saved['records'][0]['text'] == 'User: hello' and 'body' not in saved['records'][0]


@pytest.mark.parametrize('content', [
    json.dumps({'version': STORE_VERSION + 1, 'records': []}),  # a newer, incompatible layout
    json.dumps({'records': []}), json.dumps({'version': True, 'records': []}), json.dumps({'version': 0, 'records': []}),
    json.dumps({'version': STORE_VERSION, 'records': {}}), json.dumps('memories'),
    json.dumps([1]), json.dumps([{'importance': .5}]), json.dumps([{'text': 5}]),
    json.dumps([{'text': 'a', 'id': 7}]), json.dumps([{'text': 'a', 'id': ''}]),
    json.dumps([{'text': 'a', 'id': 'x'}, {'text': 'b', 'id': 'x'}]),
    '[{"text": "a"},]'])
def test_structural_damage_or_a_newer_layout_fails_closed_with_the_file_untouched(tmp_path, content):
    path = tmp_path / 'memories.json'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(RuntimeError, match='Unable to read memory store'): store(path)
    assert path.read_text(encoding='utf-8') == content and not backups(tmp_path)


def test_recall_survives_an_unusable_date(tmp_path):
    memory = store(tmp_path / 'memories.json')
    try:
        memory.remember('User: I like green tea')
        memory.records[0].created_at = None
        assert 'green tea' in memory.retrieve('tea')
    finally: memory.close()


def test_a_repaired_store_is_never_moved_aside_when_its_backup_cannot_be_copied(tmp_path, monkeypatch):
    import errno, shutil
    path = tmp_path / 'memories.json'
    path.write_text(json.dumps([{**asdict(MemoryRecord(text='User: I like green tea', id='tea')), 'importance': '0.8'}]), encoding='utf-8')
    original = path.read_bytes()
    memory = store(path)
    try:
        def full(*args, **kwargs): raise OSError(errno.ENOSPC, 'No space left on device')
        monkeypatch.setattr(shutil, 'copy2', full)
        with pytest.raises(OSError): memory._save()  # the disk is full: neither the backup nor the store is written
        assert path.read_bytes() == original and not backups(tmp_path)
        monkeypatch.undo()
        memory._save()  # space again: the original is kept once, then the repaired store is written
        assert [json.loads(kept.read_bytes()) for kept in backups(tmp_path)] == [json.loads(original)]
        assert json.loads(path.read_text(encoding='utf-8'))[0]['importance'] == .8
    finally: memory.close()
