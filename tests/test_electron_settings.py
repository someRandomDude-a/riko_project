"""What Electron and the backend must read alike: the rules and provider visibility Settings returns (the renderer evaluates
them, electron/src/settings_model.mjs), and the YAML Electron reads at launch (electron/launch_config.cjs)."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from process.app_core.configuration import schema
from process.app_core.configuration.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'electron' / 'src' / 'settings_contract.json'


def test_the_renderers_rule_and_visibility_cases_are_the_backends():
    """electron/src/settings_model.test.mjs runs these cases through ruleErrors and fieldRelevant; here the backend's own
    Rule.broken and visible_when judge them, so the two languages cannot drift apart. When a rule changes, regenerate
    'rules' in the fixture from schema.rules() and add a case for it."""
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert contract['rules'] == schema.rules(), 'electron/src/settings_contract.json is stale: copy schema.rules() into it'
    rules = [rule for section in schema.sections() for rule in section.rules]
    for case in contract['rule_cases']:
        assert sorted({rule.path for rule in rules if rule.broken(case['inputs'])}) == case['errors'], case['inputs']
    for path, visible_when in contract['visible_when'].items():
        assert schema.field(path, None).get('visible_when') == visible_when, path
    for provider, visible in contract['visible'].items():
        assert [path for path in contract['visible_when'] if schema.visible(schema.setting(path), {'runtime.provider': provider})] == visible, provider


SAMPLES = ('yes', 'Yes', 'NO', 'on', 'Off', 'y', 'n', 'Y', 'true', 'FALSE', '"yes"', "'off'", '~', 'null', '017', '09', '0x1F', '0b101', '1:30',
    '0:30', '1:30.5', '1_000', '42', '-0', '1e3', '1.0e+3', '1.0e3', '1.5', '.5', '-.5', '1.', '2024-1-5', 'Riko', '[on, off, y]', '{a: yes, b: n}')


def node_reads(function, document):
    script = f"process.stdout.write(JSON.stringify(require('./launch_config.cjs').{function}(require('fs').readFileSync(0,'utf8'))))"
    return json.loads(subprocess.run(['node', '-e', script], input=document, cwd=ROOT / 'electron', capture_output=True, text=True, check=True, timeout=60).stdout)


@pytest.mark.skipif(not shutil.which('node') or not (ROOT / 'electron' / 'node_modules' / 'yaml').is_dir(), reason='needs node and the Electron dependencies')
def test_electron_reads_the_launch_settings_as_pyyaml_does():
    """Regression: Electron parsed character_config.yaml as YAML 1.2, where desktop.debug: yes, a shortcut set to off or
    sovits_ping_config.auto_start: no are text while the backend's PyYAML reads booleans, and a repeated key failed the
    launch the backend survives."""
    document = ''.join(f'v{index}: {sample}\n' for index, sample in enumerate(SAMPLES))
    assert node_reads('parseConfig', document) == yaml.safe_load(document)
    for document in ('a: 1\na: 2\n', 'base: &b {x: 1}\nm:\n  <<: *b\n  y: 2\n'):
        assert node_reads('parseConfig', document) == yaml.safe_load(document)
    document = 'desktop:\n  debug: on\n  shortcuts:\n    popup: off\npresets:\n  default:\n    name: Yes\nsovits_ping_config:\n  auto_start: no\n'
    parsed = yaml.safe_load(document)
    assert node_reads('launchSettings', document) == {'debug': parsed['desktop']['debug'], 'shortcuts': parsed['desktop']['shortcuts'],
        'name': str(parsed['presets']['default']['name']), 'sovits': parsed['sovits_ping_config']}


@pytest.mark.parametrize('document', ['presets:\n  default:\n    name: Mika\n', 'character_name: Aoi\n', 'character_name: Aoi\npresets:\n  default:\n    name: 42\n'])
@pytest.mark.skipif(not shutil.which('node') or not (ROOT / 'electron' / 'node_modules' / 'yaml').is_dir(), reason='needs node and the Electron dependencies')
def test_electron_names_the_companion_as_load_config_does(tmp_path, document):
    path = tmp_path / 'character_config.yaml'; path.write_text(document, encoding='utf-8')
    assert node_reads('launchSettings', document)['name'] == str(load_config(path).character_name)
