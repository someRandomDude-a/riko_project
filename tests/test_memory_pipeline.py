from types import SimpleNamespace
import threading
import json
import pytest

from process.app_core.configuration.config import MemoryConfig
from process.app_core.persistence.memory import MemoryStore


def store(tmp_path, **options):
    config = MemoryConfig(store_file=tmp_path / 'memories.json', index_file=tmp_path / 'index',
                          system1_enabled=False, embeddings_enabled=False)
    return MemoryStore(config, start_worker=False, **options)


def test_capture_is_durable_and_searchable_before_classification(tmp_path):
    memory = store(tmp_path)
    record = memory.remember('My favorite color is violet')
    assert record.classification_status == 'pending'
    assert 'violet' in memory.retrieve('favorite color')
    restarted = store(tmp_path)
    assert restarted.list_records()[0]['id'] == record.id
    assert restarted._next_job()[0] == 'classify'
    assert memory.remember('My favorite color is violet').id == record.id


def test_recall_content_budget_uses_token_counter_not_words(tmp_path):
    memory = store(tmp_path)
    memory.config.token_budget = 20
    memory.token_counter = lambda text: len(text.encode('utf-8'))
    original = '界' * 100
    try:
        memory.remember(original)
        records = memory.retrieve('界', return_records=True)
        assert len(records[0]['text'].encode('utf-8')) <= 20
        assert records[0]['text_truncated']
        assert memory.list_records()[0]['text'] == original
    finally: memory.close()


def test_reflection_keeps_original_available_and_links_provenance(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        evidence = json.loads(args[0][1].content)
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({'memories': [
            {'text': 'User likes violet.', 'kind': 'distillation', 'source_ids': [evidence['focal_id']], 'confidence': .2}]})))
    memory = store(tmp_path, reflection_provider=SimpleNamespace(generate=generate))
    record = memory.remember('My favorite color is violet')
    memory._classify(record)
    snapshot = memory._find(record.id)
    worker = threading.Thread(target=memory._reflect, args=(snapshot,))
    worker.start()
    assert entered.wait(2)
    assert 'My favorite color is violet' in memory.retrieve('color')
    release.set()
    worker.join(2)
    records = memory.list_records()
    assert len(records) == 2
    assert records[0]['active']
    assert records[1]['source_ids'] == [record.id]
    assert records[1]['source_revisions'] == {record.id: 1}


def test_stale_classification_cannot_overwrite_correction_or_delete(tmp_path):
    memory = store(tmp_path)
    record = memory.remember('I like violet')
    memory.update(record.id, text='I like green', importance=.9)
    memory._classify(record)
    current = memory.list_records()[0]
    assert current['text'] == 'I like green'
    assert current['classification_status'] == 'pending'
    memory._classify(memory._find(record.id))
    assert memory.list_records()[0]['importance'] == .9
    memory.delete(record.id)
    memory._classify(record)
    assert memory.list_records() == []


def test_foreground_defers_reflection(tmp_path):
    memory = store(tmp_path, reflection_provider=SimpleNamespace(generate=lambda *a, **k: None))
    record = memory.remember('A useful experience')
    memory._classify(record)
    memory.set_foreground(True)
    memory._reflect(memory._find(record.id))
    assert memory.list_records()[0]['reflection_status'] == 'pending'


def test_context_and_related_evidence_support_contradiction(tmp_path):
    prompts = []
    def generate(messages, **kwargs):
        payload = json.loads(messages[1].content)
        prompts.append(payload)
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({'memories': [{
            'text': 'Color preference changed from violet to green following an explicit correction.',
            'kind': 'contradiction', 'source_ids': [r['id'] for r in payload['evidence']], 'confidence': .2}]})))
    memory = store(tmp_path, reflection_provider=SimpleNamespace(generate=generate))
    old = memory.remember('My favorite color is violet')
    context = {'history': [{'role': 'user', 'content': 'I used to like violet'}],
               'runtime': {'generating': True}, 'desktop': {'actions': [{'status': 'complete'}]}}
    new = memory.remember('Actually my favorite color is green now', context=context)
    context['runtime']['generating'] = False
    memory._classify(old)
    memory._classify(new)
    from copy import deepcopy
    memory._reflect(deepcopy(memory._find(new.id)))
    assert prompts[0]['evidence'][0]['formation_context']['runtime']['generating'] is True
    derived = memory.list_records()[-1]
    assert derived['reflection_kind'] == 'contradiction'
    assert set(derived['source_ids']) == {old.id, new.id}
    assert len(memory.list_records()) == 3
    memory.update(old.id, text='My favorite color was blue')
    assert memory.list_records()[-1]['active'] is False


def test_related_correction_during_reflection_rejects_result(tmp_path):
    memory = store(tmp_path)
    old = memory.remember('Violet preference')
    new = memory.remember('Green preference')
    memory._classify(old)
    memory._classify(new)
    def generate(messages, **kwargs):
        payload = json.loads(messages[1].content)
        memory.delete(old.id)
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({'memories': [{
            'text': 'Preferences changed.', 'kind': 'insight',
            'source_ids': [r['id'] for r in payload['evidence']], 'confidence': .2}]})))
    memory.reflection_provider = SimpleNamespace(generate=generate)
    from copy import deepcopy
    memory._reflect(deepcopy(memory._find(new.id)))
    assert len(memory.list_records()) == 1
    assert memory.list_records()[0]['reflection_status'] == 'pending'


def test_reflection_uses_independent_context_and_output_budgets(tmp_path):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content='{"memories":[]}'))
    memory = store(tmp_path, reflection_provider=SimpleNamespace(generate=generate))
    memory.config.reflection_context_window_tokens = 8192
    memory.config.reflection_max_output_tokens = 1536
    try:
        record = memory.remember('A useful preference')
        memory._classify(record)
        memory._reflect(memory._find(record.id))
        assert calls[0]['context_limit'] == 8192
        assert calls[0]['max_output_tokens'] == 1536
    finally: memory.close()


@pytest.mark.parametrize('content,finish', [('', 'stop'), ('prose instead of JSON', 'stop'), ('{"memories":[]}', 'length')])
def test_invalid_reflection_does_not_restart_inference(tmp_path, content, finish):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    memory = store(tmp_path, reflection_provider=SimpleNamespace(generate=generate))
    try:
        record = memory.remember('A useful preference')
        memory._classify(record)
        with pytest.raises(ValueError, match='without a repair retry'): memory._reflect(memory._find(record.id))
        assert len(calls) == 1
        assert len(memory.list_records()) == 1
    finally: memory.close()
