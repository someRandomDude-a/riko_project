"""What CI runs (.github/workflows): the whole suite on every push, checks before the long native builds, time limits on
every job, and a native smoke build that loads a real model."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    data = yaml.safe_load((ROOT / '.github/workflows' / name).read_text(encoding='utf-8'))
    data['on'] = data.pop(True, data.get('on'))  # PyYAML reads the bare key `on` as YAML 1.1's boolean true
    return data


def steps(job):
    return {step.get('name') or step.get('run') or step.get('uses'): step for step in job['steps']}


def test_ci_runs_the_whole_suite_and_the_electron_checks_on_every_push_with_time_limits():
    ci = workflow('ci.yml')
    assert {'push', 'pull_request', 'workflow_dispatch'} <= set(ci['on']) and 'tags' not in (ci['on']['push'] or {})
    assert all(job.get('timeout-minutes') for job in ci['jobs'].values())
    python = ci['jobs']['python']
    assert {'ubuntu-22.04', 'macos-15', 'windows-2022'} <= set(python['strategy']['matrix']['os']) and python['env']['HF_HUB_OFFLINE'] == '1'
    commands = '\n'.join(step.get('run', '') for step in python['steps'])
    assert 'pip install --require-hashes -r tools/release/requirements-lock.txt' in commands  # the versions a release freezes
    assert 'pip install --require-hashes --no-deps -r tools/release/requirements-lock-no-deps.txt' in commands
    assert 'npm ci' in commands  # else tests/test_settings_store.py skips the Electron first-run setup check
    arguments = next(line for line in commands.splitlines() if 'pytest' in line).split('pytest', 1)[1].split()
    assert any(re.fullmatch(r'--timeout=\d+', argument) for argument in arguments)
    assert not any(argument.startswith(('tests', '-m', '-k', '--deselect', '--ignore')) for argument in arguments)  # every test
    electron = ci['jobs']['electron']
    assert electron['defaults']['run']['working-directory'] == 'electron'
    runs = [step.get('run', '') for step in electron['steps']]
    assert runs.index('npm ci') < runs.index('npm test') < runs.index('npm run build')
    assert any('node --check main.cjs' in run and 'node --check preload.cjs' in run for run in runs)


def test_release_runs_electron_checks_before_the_native_build_and_has_time_limits():
    job = workflow('release.yml')['jobs']['package']
    names = [step.get('name') for step in job['steps']]
    assert job['timeout-minutes'] <= 360
    assert names.index('Frontend dependencies') < names.index('Frontend tests and build') < names.index('Build the native backends and freeze Python')
    by_name = steps(job)
    assert by_name['Frontend tests and build']['run'].split() == ['npm', 'test', 'npm', 'run', 'build']
    assert by_name['Frontend tests and build'].get('timeout-minutes') and by_name['Backend release regression checks'].get('timeout-minutes')
    assert '--timeout=' in by_name['Backend release regression checks']['run']


def test_native_smoke_builds_the_release_bridge_for_metal_and_loads_a_pinned_model():
    smoke = workflow('native-smoke.yml')
    triggers = smoke['on']
    assert {'schedule', 'workflow_dispatch'} <= set(triggers)
    for event in ('push', 'pull_request'):
        assert {'tools/llama_cpp/**', 'Code/process/app_core/inference/**', 'Code/process/app_core/kernel/**', 'Code/process/app_core/emotion/**',
            'tools/release/**'} <= set(triggers[event]['paths'])
    job = smoke['jobs']['metal']
    assert job['runs-on'] == 'macos-15' and job['timeout-minutes']  # Apple silicon
    by_name = steps(job)
    assert by_name['Restore the llama.cpp checkout']['with']['key'] == 'llama.cpp-${{ steps.pin.outputs.commit }}'
    assert "runpy.run_path('tools/release/build.py')['PIN']" in by_name['Pinned llama.cpp commit']['run']
    assert by_name['Build the Metal bundle']['run'].startswith('python tools/release/build.py --smoke-bundle metal ')
    # A revision, not a branch, and a hash the download must match.
    assert re.search(r'/resolve/[0-9a-f]{40}/[^/]+\.gguf$', job['env']['MODEL_URL']) and re.fullmatch(r'[0-9a-f]{64}', job['env']['MODEL_SHA256'])
    assert 'shasum -a 256 -c' in by_name['Verify the model']['run']
    native = by_name['Native tests']
    assert {'RIKO_TEST_NATIVE_LIBRARY', 'RIKO_TEST_GGUF'} <= set(native['env']) and native['env']['RIKO_TEST_EXPECT_GPU'] == '1' and native['env']['RIKO_TEST_EXPECT_DEVICES'] == '1'
    assert native['run'].startswith('python -m pytest -m native ') and 'tests/test_native_smoke.py' in native['run']
    markers = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['pytest']['ini_options']['markers']
    assert any(marker.startswith('native:') for marker in markers)


def test_smoke_bundle_builds_only_the_pinned_commit_and_a_shipped_backend(tmp_path, monkeypatch):
    if not shutil.which('git'): pytest.skip('Git unavailable')
    spec = importlib.util.spec_from_file_location('release_build', ROOT / 'tools/release/build.py')
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    built = []
    monkeypatch.setattr(release, 'apply_patch', lambda checkout, patch: built.append(('patch', checkout)))
    monkeypatch.setattr(release, 'build_bundle', lambda checkout, backend, destination: built.append(('build', backend)) or destination / 'lib')
    checkout = tmp_path / 'llama.cpp'
    checkout.mkdir()
    git = lambda *args: subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false', *args],
        cwd=checkout, capture_output=True, text=True, check=True).stdout.strip()
    git('init', '-q')
    git('commit', '-q', '--allow-empty', '-m', 'not the pin')
    with pytest.raises(RuntimeError, match='not the pinned llama.cpp'): release.smoke_bundle(release.native.backends_for()[0], checkout, tmp_path / 'out')
    monkeypatch.setattr(release, 'PIN', git('rev-parse', 'HEAD'))
    with pytest.raises(RuntimeError, match='does not ship a vax bundle'): release.smoke_bundle('vax', checkout, tmp_path / 'out')
    assert built == []
    backend = release.native.backends_for()[0]
    assert release.smoke_bundle(backend, checkout, tmp_path / 'out') == (tmp_path / 'out').resolve() / 'lib'
    assert built == [('patch', checkout.resolve()), ('build', backend)]


def action_references():
    """Every `uses:` line in the workflows and composite actions, as raw text so its comment counts."""
    files = sorted((ROOT / '.github/workflows').glob('*.yml')) + sorted((ROOT / '.github/actions').rglob('action.yml'))
    return [(path.relative_to(ROOT).as_posix(), match) for path in files
        for match in re.finditer(r'^\s*(?:- )?uses:\s*(\S+)(.*)$', path.read_text(encoding='utf-8'), re.M)]


def test_every_action_is_pinned_to_a_commit_with_its_release_in_a_comment():
    # A tag moves (a compromised or rewritten action would run with the release's secrets); a commit does not.
    references = action_references()
    assert len(references) >= 20
    commits = {}
    for path, match in references:
        reference, rest = match.group(1), match.group(2)
        if reference.startswith('./'): continue  # this repository's own composite action
        pinned = re.fullmatch(r'([\w.-]+/[\w.-]+)((?:/[\w.-]+)*)@([0-9a-f]{40})', reference)
        assert pinned and re.fullmatch(r'\s+# v\d+(\.\d+)*', rest), f'{path}: {match.group(0).strip()} is not pinned to a commit with its tag in a comment'
        commits.setdefault(pinned.group(1), set()).add((pinned.group(3), rest.strip()))
    assert all(len(pins) == 1 for pins in commits.values()), {repository: pins for repository, pins in commits.items() if len(pins) > 1}
    assert {'actions/checkout', 'actions/setup-python', 'actions/setup-node', 'actions/cache', 'Jimver/cuda-toolkit'} <= set(commits)


def test_release_checks_each_vulkan_sdk_download_against_lunargs_sha256_before_using_it():
    job = workflow('release.yml')['jobs']['package']
    env = job['env']
    assert re.fullmatch(r'\d+(\.\d+){3}', str(env['VULKAN_SDK_VERSION']))
    assert re.fullmatch(r'[0-9a-f]{64}', env['VULKAN_SDK_LINUX_SHA256']) and re.fullmatch(r'[0-9a-f]{64}', env['VULKAN_SDK_WINDOWS_SHA256'])
    text = (ROOT / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert text.count(str(env['VULKAN_SDK_VERSION'])) == 1  # one version, used through the variable
    by_name = steps(job)
    linux = by_name['Linux Vulkan SDK shader compiler']['run']
    assert '$VULKAN_SDK_VERSION' in linux and 'curl --fail' in linux
    assert linux.index('curl ') < linux.index('echo "$VULKAN_SDK_LINUX_SHA256  ') < linux.index('sha256sum -c -') < linux.index('tar -xJf')
    windows = by_name['Windows Vulkan SDK']
    script = windows['run']
    assert windows['shell'] == 'pwsh' and '${env:VULKAN_SDK_VERSION}' in script
    assert script.index('Invoke-WebRequest') < script.index('Get-FileHash -Algorithm SHA256') < script.index('${env:VULKAN_SDK_WINDOWS_SHA256}') < script.index('Start-Process')
    assert re.search(r'-ne \$\{env:VULKAN_SDK_WINDOWS_SHA256\}\) \{ throw ', script)
    assert '-Wait -PassThru' in script and re.search(r'\.ExitCode -ne 0\) \{ throw ', script)  # -Wait alone ignores a failed install
