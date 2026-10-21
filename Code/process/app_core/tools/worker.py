"""Disposable built-in tool process. No parent desktop/model objects are imported."""
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

if __name__ == '__main__':
    # The parent reads and writes UTF-8 (isolation.py). Without UTF-8 mode, Windows pipes use the ANSI code page,
    # which cannot encode results such as todo_list's emoji.
    sys.stdin.reconfigure(encoding='utf-8'); sys.stdout.reconfigure(encoding='utf-8')
    try:
        request = json.load(sys.stdin)
        cls = getattr(importlib.import_module(request['module']), request['class'])
        tool = cls(request.get('config', {}), request.get('context', {}))
        result = tool.execute(**request['arguments'])
        print(json.dumps({'result': result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)
