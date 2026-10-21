"""Collaborators are passed in, never patched on: ChatService, DesktopState and RuntimeConfig have __slots__, so setting an
attribute their class does not declare fails at once, and no code reads the names that used to be patched on through a
getattr or hasattr default (which would hide a missing collaborator instead of failing)."""
import ast

import pytest

from process.app_core.configuration.config import RuntimeConfig
from process.app_core.conversation.chat import ChatService
from process.app_core.desktop.state import DesktopState
from process.app_core.inference.kv_budget import pool_capacity
from test_private_access import CODE

# What factory.py and SessionManager used to set on these objects; now ChatDeps, TurnContext, DesktopServices or fields. For
# DesktopState also the lock and fields desktop_server and tests wrote directly, now its components' (desktop/state.py).
RETIRED = {
    ChatService: ['action_controller', 'task_store', 'task_mcp', 'initiative_provider', 'context_limit', 'turn_origin',
                  'runtime_context', 'memory_runtime_context', 'emotion_playback_managed', 'history_file'],
    DesktopState: ['action_controller', 'media_resolver', 'effects_directory', 'effect_library', 'avatar_motion', '_lock', 'actions', 'displays',
                   'avatar_geometry', 'tool_activity', 'whiteboard', 'mic_enabled', 'audio_enabled', 'sleep_mode'],
    RuntimeConfig: ['initiative_context_tokens', 'n_ctx_initiative'],
}
DECLARED_READS = {'action_controller', 'task_store', 'task_mcp', 'initiative_provider', 'context_limit', 'turn_origin',
    'runtime_context', 'memory_runtime_context', 'emotion_playback_managed', 'media_resolver', 'effects_directory',
    'effect_library', 'avatar_motion', 'playback_active', 'initiative_n_ctx', 'initiative_max_output_tokens', 'reflection_n_ctx'}


def instances():
    return {ChatService: ChatService(None, system_prompt=''), DesktopState: DesktopState(), RuntimeConfig: RuntimeConfig()}


@pytest.mark.parametrize('cls', list(RETIRED), ids=lambda cls: cls.__name__)
def test_setting_an_undeclared_attribute_fails(cls):
    target = instances()[cls]
    for name in RETIRED[cls]:
        with pytest.raises(AttributeError): setattr(target, name, object())
    assert not hasattr(target, '__dict__')


def test_no_code_reads_a_collaborator_through_a_getattr_or_hasattr_default():
    found = []
    for path in [*sorted((CODE / 'process' / 'app_core').rglob('*.py')), *sorted(CODE.glob('*.py'))]:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'getattr', 'hasattr'} and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value in DECLARED_READS):
                found.append(f'{path.relative_to(CODE).as_posix()}:{node.lineno} {ast.unparse(node)}')
    assert not found, 'Read the declared attribute (ChatDeps, TurnContext, DesktopServices, RuntimeConfig) directly:\n' + '\n'.join(found)


def test_runtime_config_declares_the_background_budgets():
    runtime = RuntimeConfig(provider='llama_cpp', n_ctx=8192, parallel_slots=2)
    assert (runtime.initiative_n_ctx, runtime.initiative_max_output_tokens, runtime.reflection_n_ctx) == (4096, 1024, 4096)
    assert pool_capacity(runtime) == 8192 + 4096  # a RuntimeConfig not made by load_config sizes the pool from its own fields
    runtime.reflection_n_ctx = 6144
    assert pool_capacity(runtime) == 8192 + 6144
