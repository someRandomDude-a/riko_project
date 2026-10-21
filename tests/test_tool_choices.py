from types import SimpleNamespace
import threading

import pytest

from process.app_core.tools.choices import ChoiceResolver
from process.app_core.tools.registry import ToolRegistry
from process.app_core.desktop.state import DesktopState
from process.app_core.desktop.tools import DesktopServices, EffectTool, AvatarWindowTool
from process.app_core.desktop.effects import EffectLibrary
from process.app_core.desktop.media import resolve_media


def test_choice_correction_is_finite_abstaining_and_cannot_repair_paths():
    resolver = ChoiceResolver(lambda *args:{'choice':'choice-1','confidence':.95})
    try:
        assert resolver.choose('gesture','name',' WAVe ',['nod','wave']) == 'wave'
        assert resolver.choose('gesture','name','greet', ['nod','wave']) == 'wave'
        assert resolver.choose('effect','name','../../private.mp4',['stars.mp4']) is None
        assert resolver.choose('effect','name','C:\\private.mp4',['stars.mp4']) is None
    finally: resolver.close()


def test_low_confidence_or_timed_out_julia_does_not_guess_or_spawn_more_jobs():
    gate = threading.Event()
    calls = []
    def select(*args): calls.append(True);gate.wait();return {'choice':'choice-0','confidence':1}
    resolver = ChoiceResolver(select,timeout=.01)
    try:
        assert resolver.choose('effect','name','unrelated',['rain.mp4']) is None
        assert resolver.choose('effect','name','something else',['rain.mp4']) is None
        assert calls == [True]
    finally: gate.set();resolver.close()


def test_effect_catalog_relative_asset_is_resolved_before_exact_approval(tmp_path):
    state = DesktopState()
    directory = tmp_path/'effects'/'greenscreens'
    directory.mkdir(parents=True)
    asset = directory/'stars.mp4';asset.write_bytes(b'video')
    services = DesktopServices.of(state, effect_library=EffectLibrary(directory, rules=[]), media_resolver=lambda value:resolve_media(tmp_path,value))
    registry = ToolRegistry()
    registry.choice_resolver = ChoiceResolver()
    registry.register_local(EffectTool(services))
    approved=[]
    registry.approvals=SimpleNamespace(authorize=lambda name,args,*rest,**keys:approved.append(args.copy()) or False,close=lambda:None)
    try:
        result = registry.execute('visual_effect',{'action':'play','asset':'stars.mp4'})
        assert result.is_error and approved[0]['asset'] == str(asset)
        assert state.snapshot()['effect'] is None
        result = registry.execute('visual_effect',{'action':'play','asset':'../stars.mp4'})
        assert result.is_error and len(approved)==1
    finally:registry.close()


def test_move_avatar_uses_walk_cycle_by_default_and_preserves_explicit_teleport():
    state = DesktopState()
    walked=[]
    motion=SimpleNamespace(walk_to=lambda x,y:walked.append((x,y)) or SimpleNamespace(id='walk'),stop_movement=lambda:None)
    tool=AvatarWindowTool(DesktopServices.of(state, avatar_motion=motion))
    before=state.snapshot()['avatar_geometry']['x']
    assert 'Walk queued' in tool.execute(x=200,y=100)
    assert walked==[(200,100)] and state.snapshot()['avatar_geometry']['x']==before
    tool.execute(x=200,y=100,walk=False)
    assert state.snapshot()['avatar_geometry']['x']==200
