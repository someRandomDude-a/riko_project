import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from process.app_core.persistence.conversation_store import ConversationStore
from process.app_core.desktop.state import DesktopState
from process.app_core.events.bus import RuntimeEvent
from process.app_core.runtime.lifecycle import close_bounded
from process.app_core.conversation.messages import ChatMessage
from process.app_core.tools.registry import RegisteredTool, ToolRegistry
from process.app_core.runtime.workers import DaemonExecutor


def test_archive_pages_sessions_and_interrupted_text_survive_restart(tmp_path):
    path=tmp_path/'conversation.sqlite3'
    store=ConversationStore(path, 'fake', [ChatMessage('user','legacy')])
    old_session=store.session_id
    for index in range(3):
        turn=f'turn-{index}'
        store.observe(RuntimeEvent('chat.input', {'text':f'Question {index}'}, turn_id=turn))
        store.observe(RuntimeEvent('chat.delta', {'text':'one two three'}, turn_id=turn))
        store.observe(RuntimeEvent('chat.interrupted', {'text':'one two three','offset':3}, turn_id=turn))
        store.observe(RuntimeEvent('chat.cancelled', {}, turn_id=turn))
    page=store.page(limit=2)
    assert len(page['messages'])==2 and page['has_more']
    assert page['messages'][-1]['cutoff']==3
    assert page['messages'][-1]['text']=='one two three'
    older=store.page(page['before'],2)
    assert not set(m['id'] for m in page['messages']) & set(m['id'] for m in older['messages'])
    store.close()
    restored=ConversationStore(path,'fake')
    assert len(restored.page(limit=100)['messages'])==7
    assert restored.page()['sessions'][old_session]['outcome']=='stopped'
    restored.close()


def test_crash_recovery_marks_partial_messages_incomplete(tmp_path):
    store=ConversationStore(tmp_path/'history.sqlite3')
    store.observe(RuntimeEvent('chat.delta', {'text':'partial'}, turn_id='turn'))
    store._flush();store.stopped.set()
    store.db.close() # Simulate process death without graceful close.
    restored=ConversationStore(tmp_path/'history.sqlite3')
    message=restored.page()['messages'][0]
    assert message['status']=='abandoned' and message['interrupted']
    restored.close()


def test_interjection_annotations_remain_backend_metadata_and_merge_for_display(tmp_path):
    store=ConversationStore(tmp_path/'annotated.sqlite3')
    try:
        for text,start,end in [('one',1,2),('two',2.5,3)]:
            store.observe(RuntimeEvent('chat.interjection',{'text':'[speaking over you] '+text,'display_text':text,'system_label':'Speaking over you','offset':0,'started_at':start,'ended_at':end,'debounce_seconds':1},turn_id='turn'))
        item=store.page()['messages'][0]['interjections'][0]
        assert item['display_text']=='one two' and item['system_label']=='Speaking over you'
        assert item['text']=='[speaking over you] one two'
    finally:store.close()


def test_board_content_pages_bounds_and_geometry_survive_restart(tmp_path):
    path=tmp_path/'whiteboard.json'
    board=DesktopState();board.configure_board_store(path)
    board.board_page('new_page')
    command=board.add_whiteboard('text',{'text':'# Formula\n$y=x^2$'})
    board.surface_result('whiteboard',command,'rendered',bounds={'x':40,'y':40,'width':436,'height':240})
    board.update_geometry('whiteboard',width=1000)
    restored=DesktopState();restored.configure_board_store(path)
    assert restored.whiteboard_page=='page-2'
    assert restored.whiteboard[0].bounds['height']==240
    assert restored.whiteboard[0].status=='queued' # Renderer must confirm again.
    assert restored.whiteboard_geometry['width']==1000
    restored.clear_whiteboard()
    third=DesktopState();third.configure_board_store(path)
    assert third.whiteboard==[]


def test_corrupt_board_preserved_and_recovery_file_resumes(tmp_path):
    path=tmp_path/'whiteboard.json';path.write_text('damaged')
    board=DesktopState();board.configure_board_store(path)
    board.add_whiteboard('text',{'text':'Recovered board'})
    assert path.read_text()=='damaged'
    restored=DesktopState();restored.configure_board_store(path)
    assert restored.whiteboard[0].payload['text']=='Recovered board'


def test_daemon_worker_shutdown_cancels_queued_jobs():
    executor=DaemonExecutor(max_workers=1)
    release=threading.Event();started=threading.Event()
    def block(): started.set();release.wait(1)
    first=executor.submit(block);assert started.wait(1)
    second=executor.submit(lambda: 'should never run')
    executor.shutdown()
    assert second.cancelled()
    release.set();first.result(timeout=1)
    with pytest.raises(RuntimeError): executor.submit(lambda:None)


def test_hung_worker_does_not_keep_python_process_alive():
    script='''import time, threading
from process.app_core.runtime.workers import DaemonExecutor
started=threading.Event()
def stuck():
 started.set()
 time.sleep(60)
executor=DaemonExecutor()
executor.submit(stuck)
assert started.wait(1)
print('exiting',flush=True)
'''
    result=subprocess.run([sys.executable,'-c',script],env={**os.environ,'PYTHONPATH':str(Path('Code').resolve())},
        capture_output=True,text=True,timeout=3)
    assert result.returncode==0 and 'exiting' in result.stdout


def test_cleanup_deadline_does_not_wait_for_stuck_resource():
    release=threading.Event()
    before=time.monotonic()
    close_bounded(SimpleNamespace(close=lambda:release.wait(2)),timeout=.02)
    assert time.monotonic()-before<.5
    release.set()


def test_factory_rolls_back_resources_after_partial_construction_failure(tmp_path,monkeypatch):
    from process.app_core import factory
    from process.app_core.configuration.config import AppConfig
    closed=[]
    monkeypatch.setattr(factory,'create_provider',lambda config:SimpleNamespace(close=lambda:closed.append('provider')))
    monkeypatch.setattr(factory.ToolRegistry,'from_config',lambda config:SimpleNamespace(
        register_mcp=lambda client:None, close=lambda:closed.append('registry')))
    def fail(*args,**kwargs): raise RuntimeError('memory failure')
    monkeypatch.setattr(factory,'MemoryStore',fail)
    # Keep one shared desktop state for the factory's effect-directory lookups.
    state=SimpleNamespace();monkeypatch.setattr(factory,'get_desktop_state',lambda:state)
    config=AppConfig(root=tmp_path)
    with pytest.raises(RuntimeError,match='memory failure'): factory.create_chat_service(config)
    assert closed==['registry','provider']


def test_timed_out_shared_tool_blocks_duplicate_retry():
    release=threading.Event()
    registry=ToolRegistry(timeout_seconds=.02)
    registry.tools['slow']=RegisteredTool('slow','',{},lambda args:release.wait(2))
    try:
        assert registry.execute('slow',{}).is_error
        assert 'retry blocked' in registry.execute('slow',{}).content
    finally: release.set();registry.close()


def test_isolated_tool_is_killed_at_deadline(tmp_path,monkeypatch):
    (tmp_path/'slow_tool.py').write_text('''from pathlib import Path
import time
class Tool:
 def __init__(self,config,context): self.path=Path(config['path'])
 def execute(self,**args):
  self.path.write_text('started')
  time.sleep(30)
  self.path.write_text('late side effect')
''')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path))
    marker=tmp_path/'marker'
    registry=ToolRegistry(timeout_seconds=1)
    registry.tools['isolated']=RegisteredTool('isolated','',{},None,isolated={'module':'slow_tool','class':'Tool','config':{'path':str(marker)}})
    try:
        result=registry.execute('isolated',{})
        assert result.is_error and 'terminated' in result.content
        assert marker.read_text()=='started'
        assert not registry._processes
    finally: registry.close()
