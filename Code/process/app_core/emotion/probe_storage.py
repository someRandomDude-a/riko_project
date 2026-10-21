"""Per-model expression datasets, with read-only discovery of legacy storage. paths: the DataPaths (configuration/paths.py)
the datasets live under (models/, and persistent_memories/emotion_probes/ from earlier releases)."""
from pathlib import Path
import re


def probe_directory(paths, runtime):
    if runtime.model_path:
        name = Path(runtime.model_path).stem
    else:
        name = (runtime.hf_repo_id or 'unknown-model').rstrip('/').rsplit('/', 1)[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .') or 'unknown-model'
    if name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        name = '_' + name
    return Path(paths.models) / name / 'expression probe'


def corpus_files(paths):
    root, seen = Path(paths.root), set()
    for folder, pattern in ((paths.models, 'training/expression/*/*/examples.json'), (paths.models, '*/expression probe/*/examples.json'),
            (paths.emotion_probes, '*/examples.json')):
        for path in sorted(Path(folder).glob(pattern)):
            key = path.parent.name
            if not re.fullmatch(r'[0-9a-f]{64}', key) or key in seen: continue
            if not path.resolve().is_relative_to(root.resolve()): continue
            seen.add(key)
            yield path


def training_directory(paths, runtime):
    name = probe_directory(paths, runtime).parent.name
    return Path(paths.models) / 'training' / 'expression' / name
