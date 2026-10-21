from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def requirement(text):
    name, spec = re.match(r'\s*([A-Za-z0-9._-]+)\s*(.*)', text).groups()
    return re.sub(r'[-_.]+', '-', name).lower(), spec.replace(' ', '')


def test_runtime_extra_mirrors_requirements_runtime_and_covers_every_extra():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    extras = project['optional-dependencies']
    lines = [line.split('#')[0].strip() for line in (ROOT / 'requirements-runtime.txt').read_text(encoding='utf-8').splitlines()]
    runtime = {requirement(line) for line in lines if line}
    assert {requirement(item) for item in extras['runtime']} == runtime
    for name in ('llama', 'desktop', 'discord'): assert {requirement(item) for item in extras[name]} <= runtime, name
    assert {requirement(item) for item in project['dependencies']} <= runtime
    assert 'riko-companion[runtime]' in extras['test']


def locked(path):
    """{name: [(version, hash count)]} from a uv-generated, hash-locked requirements file (one entry per marker)."""
    entries = {}
    for line in path.read_text(encoding='utf-8').replace('\\\n', ' ').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, version = re.match(r'([A-Za-z0-9._-]+)==(\S+)', line).groups()
            entries.setdefault(requirement(name)[0], []).append((version, line.count('--hash=sha256:')))
    return entries


def test_release_locks_pin_every_runtime_and_build_requirement_with_hashes():
    # Run `python tools/release/lock.py` after editing requirements-runtime.txt or tools/release/requirements-build.in.
    from packaging.specifiers import SpecifierSet
    release = ROOT / 'tools/release'
    lock, no_deps = locked(release / 'requirements-lock.txt'), locked(release / 'requirements-lock-no-deps.txt')
    inputs = [line.split('#')[0].strip() for path in (ROOT / 'requirements-runtime.txt', release / 'requirements-build.in')
        for line in path.read_text(encoding='utf-8').splitlines()]
    for name, spec in (requirement(line) for line in inputs if line):
        assert name in lock, f'{name} is not locked'
        assert all(SpecifierSet(spec).contains(version, prereleases=True) for version, _ in lock[name]), (name, spec, lock[name])
    assert {'pyinstaller', 'pyinstaller-hooks-contrib', 'pytest', 'pytest-timeout'} <= set(lock)  # pytest-timeout: ci.yml's --timeout
    assert set(no_deps) == {'efficientword-net'} and 'efficientword-net' not in lock
    assert all(hashes for entries in (*lock.values(), *no_deps.values()) for _, hashes in entries)
