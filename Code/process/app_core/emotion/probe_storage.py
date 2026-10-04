"""Per-model expression datasets, with read-only discovery of legacy storage."""
from pathlib import Path
import re


def probe_directory(root, runtime):
    if runtime.model_path:
        name = Path(runtime.model_path).stem
    else:
        name = (runtime.hf_repo_id or 'unknown-model').rstrip('/').rsplit('/', 1)[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .') or 'unknown-model'
    if name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        name = '_' + name
    return Path(root) / 'models' / name / 'expression probe'


def corpus_files(root):
    root = Path(root)
    seen = set()
    for pattern in ('models/training/expression/*/*/examples.json',
            'models/*/expression probe/*/examples.json',
            'persistent_memories/emotion_probes/*/examples.json'):
        for path in sorted(root.glob(pattern)):
            key = path.parent.name
            if not re.fullmatch(r'[0-9a-f]{64}', key) or key in seen: continue
            if not path.resolve().is_relative_to(root.resolve()): continue
            seen.add(key)
            yield path


def training_directory(root, runtime):
    name = probe_directory(root, runtime).parent.name
    return Path(root) / 'models' / 'training' / 'expression' / name
