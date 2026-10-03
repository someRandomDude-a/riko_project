import importlib.util
import json
from pathlib import Path

import pytest

from process.app_core.persistence.memory import MemoryRecord

spec = importlib.util.spec_from_file_location('legacy_migration', Path(__file__).resolve().parents[1] / 'tools' / 'migrate_legacy_memories.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def legacy():
    return {'id': 123, 'text': 'A memory\nwith unicode: café', 'importance_score': .8,
            'created_on': '2025-01-02T03:04', 'last_access': '2025-03-04T05:06:07',
            'access_count': 5, 'tokens': 20, 'detailed': False}


def test_converts_to_runtime_schema_without_changing_text_or_legacy_metadata():
    old = legacy()
    record = migration.convert_records([old], 'hash')[0]
    loaded = MemoryRecord(**record)
    assert loaded.text == old['text']
    assert loaded.importance == .8 and loaded.access_count == 5
    assert loaded.created_at == old['created_on']
    assert loaded.formation_context['migration']['legacy_record'] == old
    assert loaded.reflection_status == 'skipped'
    assert record == migration.convert_records([old], 'hash')[0]
    assert migration.convert_records([old], 'hash', classify=True)[0]['classification_status'] == 'pending'


def test_dry_run_writes_nothing_and_write_never_overwrites(tmp_path):
    source, output = tmp_path / 'old.json', tmp_path / 'new.json'
    original = json.dumps([legacy()]).encode()
    source.write_bytes(original)
    assert migration.migrate(source, output)['mode'] == 'dry-run'
    assert not output.exists()
    migration.migrate(source, output, write=True)
    saved = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        migration.migrate(source, output, write=True)
    assert source.read_bytes() == original and output.read_bytes() == saved
    with pytest.raises(ValueError, match='different'):
        migration.migrate(source, source, write=True)


@pytest.mark.parametrize('change', [{'importance_score': float('nan')}, {'importance_score': 8}, {'access_count': -1}, {'created_on': 'bad'}, {'text': ''}])
def test_invalid_records_fail_without_silent_loss(change):
    with pytest.raises(ValueError):
        migration.convert_records([{**legacy(), **change}], 'hash')


def test_missing_ids_remain_distinct_and_duplicate_ids_fail():
    old = legacy()
    with pytest.raises(ValueError, match='duplicate'):
        migration.convert_records([old, old], 'hash')
    old.pop('id')
    records = migration.convert_records([old, old], 'hash')
    assert records[0]['id'] != records[1]['id']


def test_enrichment_queues_the_current_pipeline_without_changing_originals(tmp_path):
    old = legacy()
    record = migration.convert_records([old], 'hash', enrich=True)[0]
    loaded = MemoryRecord(**record)
    assert loaded.classification_status == loaded.reflection_status == 'pending'
    assert loaded.text == old['text'] and loaded.importance == old['importance_score']
    assert loaded.user_fields == ['active', 'importance']
    assert not loaded.derived and loaded.source_ids == []
    assert loaded.formation_context['migration']['legacy_record'] == old
    source, output = tmp_path / 'old.json', tmp_path / 'new.json'
    source.write_text(json.dumps([old]), encoding='utf-8')
    report = migration.migrate(source, output, enrich=True)
    assert report['enrichment_executed'] is False
    assert report['classification'].startswith('queued') and report['reflection'].startswith('queued')
    assert not output.exists()


def test_no_partial_output_on_validation_failure(tmp_path):
    source, output = tmp_path / 'old.json', tmp_path / 'new.json'
    source.write_text(json.dumps([legacy(), None]), encoding='utf-8')
    with pytest.raises(ValueError):
        migration.migrate(source, output, write=True)
    assert not output.exists()
