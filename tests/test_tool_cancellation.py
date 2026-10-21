"""Stop ends a running tool call at once, so the turn (and its turn lock) is released without waiting for the tool deadline."""
import json
import sys
import threading
import time

from process.app_core.tools.mcp import untrusted_output
from process.app_core.tools.registry import RegisteredTool, StdioMCPClient, ToolRegistry


def when(condition, then, timeout=30):
    """Run then() once condition() holds, from another thread (what a Stop click does while execute blocks)."""
    def watch():
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline: time.sleep(.01)
        then()
    threading.Thread(target=watch, daemon=True).start()


def test_stop_kills_the_isolated_worker_and_frees_the_tool(tmp_path, monkeypatch):
    (tmp_path / 'slow_tool.py').write_text('''from pathlib import Path
import time
class Tool:
 def __init__(self,config,context): self.path=Path(config['path'])
 def execute(self,**args):
  self.path.write_text('started')
  time.sleep(60)
  self.path.write_text('late side effect')
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    marker, stop = tmp_path / 'marker', threading.Event()
    registry = ToolRegistry(timeout_seconds=60)
    registry.tools['isolated'] = RegisteredTool('isolated', '', {}, None, isolated={'module': 'slow_tool', 'class': 'Tool', 'config': {'path': str(marker)}})
    try:
        when(marker.exists, stop.set)
        started = time.monotonic()
        result = registry.execute('isolated', {}, cancelled=stop.is_set)
        assert time.monotonic() - started < 30
        assert result.is_error and 'cancelled' in result.content and 'terminated' in result.content
        assert not registry.workers.processes and registry._running['isolated'].done()  # killed, so a retry is not blocked
    finally: registry.close()


def test_stop_returns_from_an_in_process_tool_and_blocks_its_duplicate_until_it_finishes():
    release, started, stop = threading.Event(), threading.Event(), threading.Event()
    registry = ToolRegistry(timeout_seconds=60)
    registry.tools['slow'] = RegisteredTool('slow', '', {}, lambda args: started.set() or release.wait(30) and 'late')
    try:
        when(started.is_set, stop.set)
        result = registry.execute('slow', {}, cancelled=stop.is_set)
        assert result.is_error and 'cancelled' in result.content and 'may still finish' in result.content
        assert 'retry blocked' in registry.execute('slow', {}).content
    finally: release.set(); registry.close()


def test_stop_cancels_a_stdio_mcp_request_and_the_server_keeps_answering(tmp_path):
    # A server that answers 'slow' only after the client cancels it (late, as the spec allows), then answers 'fast' normally.
    log, server = tmp_path / 'log.jsonl', tmp_path / 'server.py'
    server.write_text(f'''import json, sys
log = open({str(log)!r}, 'a', encoding='utf-8')
slow = None
for line in sys.stdin:
    message = json.loads(line)
    log.write(line); log.flush()
    reply = lambda id, result: print(json.dumps({{'jsonrpc': '2.0', 'id': id, 'result': result}}), flush=True)
    if message.get('method') == 'initialize': reply(message['id'], {{}})
    elif message.get('method') == 'tools/list': reply(message['id'], {{'tools': [{{'name': 'slow'}}, {{'name': 'fast'}}]}})
    elif message.get('method') == 'tools/call' and message['params']['name'] == 'slow': slow = message['id']
    elif message.get('method') == 'notifications/cancelled': reply(slow, {{'content': [{{'type': 'text', 'text': 'late'}}]}})
    elif message.get('method') == 'tools/call': reply(message['id'], {{'content': [{{'type': 'text', 'text': 'fast answer'}}]}})
''', encoding='utf-8')
    def logged():  # whole lines only: the server may be mid-write
        try: return [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines(keepends=True) if line.endswith('\n')]
        except OSError: return []
    client = StdioMCPClient(sys.executable, [str(server)])
    registry, stop = ToolRegistry(timeout_seconds=60), threading.Event()
    registry.register_mcp(client, source='mcp:test')
    try:
        when(lambda: any(m.get('params', {}).get('name') == 'slow' for m in logged()), stop.set)
        started = time.monotonic()
        result = registry.execute('slow', {}, cancelled=stop.is_set)
        assert time.monotonic() - started < 30  # not the 30 s MCP deadline
        assert result.is_error and 'cancelled' in result.content and 'asked to stop' in result.content
        call = next(m for m in logged() if m.get('params', {}).get('name') == 'slow')
        assert any(m.get('method') == 'notifications/cancelled' and m['params']['requestId'] == call['id'] for m in logged())
        assert registry._running['slow'].done()
        answer = registry.execute('fast', {})  # the late 'slow' reply is skipped, not mistaken for this one
        assert not answer.is_error and answer.content == untrusted_output('test', 'fast', 'fast answer')  # framed: a server's text
        assert client.process.poll() is None  # the server was not restarted
    finally: registry.close()
