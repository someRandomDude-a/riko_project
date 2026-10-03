"""Convert the old llm_scripts JSON store without importing either runtime.

Dry-run (default):
  python tools/migrate_legacy_memories.py OLD_STORE --output NEW_STORE
Write a NEW file only:
  python tools/migrate_legacy_memories.py OLD_STORE --output NEW_STORE --write
Migrate and queue the current tagging + reflection pipeline:
  python tools/migrate_legacy_memories.py OLD_STORE --output NEW_STORE --write --enrich

Stop the companion before installing the output as memory.store_file. This tool
does not install it, change configuration, merge stores, or touch FAISS indexes.
The new MemoryStore builds its own optional retrieval index on startup.

Enrichment steps (--enrich):
  1. Preserve original text, dates, importance, access counts and legacy metadata.
  2. Set classification and reflection to durable pending jobs.
  3. After installing the output, start the app with memory.system1_enabled and
     memory.reflection_enabled configured, plus a working reflection provider.
  4. The existing CPU Julia classifier assigns memory type, tags and confidence.
     Migration protects original importance and active state from reclassification.
  5. The existing memory pipeline retrieves related evidence and runs reflection
     subject to its importance threshold, context budget and foreground priority.
     It creates separate consolidation/contradiction/insight/distillation records
     with source IDs/revisions, confidence limits and reflection-kind metadata.
     Originals are not overwritten. Malformed output fails without repair retries.
  6. Retrieval indexes are rebuilt by the current system, not copied from FAISS.

This script only queues enrichment. It never loads models, starts servers or
changes app settings. Disabled/unavailable stages cannot perform enrichment;
inspect memory processing statuses/errors in the app after startup. Formation
context contains the actual legacy record, not invented conversation evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid
from datetime import datetime

NAMESPACE = uuid.UUID('acf9778d-fca9-56d6-84b4-cda8cac16956')
MEMORY_TYPES = {'episodic', 'factual', 'semantic', 'preference', 'relationship', 'procedural'}


def unit_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{label} must be a finite number between 0 and 1; got {value!r}')
    return float(value)


def timestamp(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty ISO date/time string')
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{label} is not an ISO date/time: {value!r}') from exc
    # Preserve naive dates rather than guessing the old machine's timezone.
    return value


def convert_records(raw, source_hash, *, classify=False, enrich=False):
    if not isinstance(raw, list):
        raise ValueError('Legacy memory_store.json must contain a JSON list')
    result, ids = [], set()
    for index, old in enumerate(raw):
        label = f'record {index + 1}'
        if not isinstance(old, dict):
            raise ValueError(f'{label} must be an object (no records are silently dropped)')
        text = old.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{label}.text must be nonempty text')
        created = timestamp(old.get('created_on'), label + '.created_on')
        accessed = timestamp(old.get('last_access', created), label + '.last_access')
        importance = unit_number(old.get('importance_score', .5), label + '.importance_score')
        confidence = unit_number(old.get('confidence', .2), label + '.confidence')
        count = old.get('access_count', 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f'{label}.access_count must be a nonnegative integer')
        kind = old.get('memory_type', 'episodic')
        if kind not in MEMORY_TYPES:
            raise ValueError(f'{label}.memory_type is not supported: {kind!r}')
        tags = old.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f'{label}.tags must be a list of strings')
        legacy_id = old.get('id')
        if legacy_id is not None:
            if isinstance(legacy_id, bool) or not isinstance(legacy_id, (str, int)):
                raise ValueError(f'{label}.id must be a string or integer')
            identity = json.dumps(['legacy-llm-scripts', legacy_id], ensure_ascii=False)
        else:
            # Stable on reruns; keep distinct rows even when their text is identical.
            identity = json.dumps(['legacy-llm-scripts', index, old], sort_keys=True, ensure_ascii=False, allow_nan=False)
        record_id = str(uuid.uuid5(NAMESPACE, identity))
        if record_id in ids:
            raise ValueError(f'{label}: duplicate legacy ID {legacy_id!r}; resolve it in a separate copy first')
        ids.add(record_id)
        result.append({
            'text': text, 'memory_type': kind, 'importance': importance,
            'confidence': confidence, 'tags': tags, 'source': 'legacy_memory_migration',
            'created_at': created, 'last_accessed': accessed, 'access_count': count,
            'id': record_id, 'revision': 1, 'active': True,
            'classification_status': 'pending' if classify or enrich else 'complete',
            'reflection_status': 'pending' if enrich else 'skipped', 'processing_error': '', 'derived': False,
            'source_ids': [], 'source_revisions': {},
            # Optional classification must not discard or reprioritize old memories.
            'user_fields': ['active', 'importance'],
            'formation_context': {'migration': {
                'format': 'old_llm_scripts.Memory_system', 'version': 1,
                'source_sha256': source_hash, 'source_row': index + 1,
                'legacy_record': old,
                'note': 'Legacy text may already be reflected or summarized; original conversation provenance is unavailable.',
            }}, 'reflection_kind': '',
        })
    return result


def read_json(path):
    content = path.read_bytes()
    def invalid_constant(value):
        raise ValueError(f'Invalid JSON number {value}')
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key {key!r}')
            obj[key] = value
        return obj
    return content, json.loads(content.decode('utf-8-sig'), parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def migrate(source, output, *, write=False, classify=False, enrich=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Source and output must be different files')
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing output: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Output directory must already exist: {output.parent}')
    content, raw = read_json(source)
    digest = hashlib.sha256(content).hexdigest()
    records = convert_records(raw, digest, classify=classify, enrich=enrich)
    payload = (json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if write:
        # Exclusive creation also protects against a destination created after preflight.
        with output.open('xb') as stream:
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                output.unlink()  # Only our newly created, incomplete output.
                raise
    return {'mode': 'written' if write else 'dry-run', 'records': len(records),
            'source': str(source), 'source_sha256': digest, 'output': str(output),
            'classification': 'queued for next app startup' if classify or enrich else 'not queued',
            'reflection': 'queued for next app startup, subject to runtime settings and thresholds' if enrich else 'not queued',
            'enrichment_executed': False, 'source_modified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='Old persistant_memories/memory_store.json (not the code directory or FAISS index)')
    parser.add_argument('--output', required=True, type=Path, help='New JSON file; existing files are never overwritten')
    parser.add_argument('--write', action='store_true', help='Create output; without this flag only validate and report')
    parser.add_argument('--classify', action='store_true', help='Queue new-system classification on startup, preserving importance and active state; never queue reflection')
    parser.add_argument('--enrich', action='store_true', help='Queue current-system type/tag/confidence classification AND evidence-grounded self-reflection after installation; no models run in this script')
    args = parser.parse_args(argv)
    try:
        report = migrate(args.source, args.output, write=args.write, classify=args.classify, enrich=args.enrich)
    except (OSError, ValueError, TypeError) as exc:
        print(f'Migration failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
