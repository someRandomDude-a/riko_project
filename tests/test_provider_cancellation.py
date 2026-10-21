"""Stop reaches OpenAI-compatible servers at once: while reasoning, while streaming tool-call arguments, while silent, and
before any header."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest

from process.app_core.configuration.config import RuntimeConfig
from process.app_core.kernel.messages import ChatMessage
from process.app_core.kernel.cancellation import BackgroundPreempted
from process.app_core.inference.providers import OpenAIProvider


def chunk(content=None, reasoning=None, finish=None, arguments=None):
    delta = {**({'content': content} if content else {}), **({'reasoning_content': reasoning} if reasoning else {}),
        **({'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': arguments}}]} if arguments else {})}
    return {'id': 'c', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'test', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


COMPLETED = {'type': 'response.completed', 'sequence_number': 3, 'response': {'id': 'resp_1', 'object': 'response', 'created_at': 0, 'model': 'test',
    'status': 'completed', 'parallel_tool_calls': False, 'tool_choice': 'auto', 'tools': [], 'output': [{'type': 'message', 'id': 'm', 'role': 'assistant',
    'status': 'completed', 'content': [{'type': 'output_text', 'text': 'Hello world', 'annotations': []}]}]}}


class FakeServer:
    """Speaks streamed /v1/responses and /v1/chat/completions. stall: None; 'prefill' (silent before any header, like a
    server loading or prefilling); 'stream' (silent after its first event); 'reasoning' (reasoning deltas, no text);
    'tool_call' (tool-call argument deltas, no text)."""
    def __init__(self, stall=None):
        self.stall, self.streaming, self.disconnected = stall, threading.Event(), threading.Event()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def gone(self, seconds):
                self.connection.settimeout(seconds)
                try: return self.connection.recv(1, socket.MSG_PEEK) == b''
                except socket.timeout: return False
                except OSError: return True
            def stalled(self, send=None):  # until the client goes away (True) or the test ends the stall (False)
                deadline = time.monotonic() + 60
                while owner.stall and time.monotonic() < deadline:
                    if send: send()
                    owner.streaming.set()
                    if self.gone(.02): owner.disconnected.set(); return True
                return False
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                chat, stall = self.path.endswith('/chat/completions'), owner.stall
                self.close_connection = True
                if stall == 'prefill' and self.stalled(): return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                def send(event):
                    data = json.dumps(event)
                    self.wfile.write((f'data: {data}\n\n' if chat else f'event: {event["type"]}\ndata: {data}\n\n').encode()); self.wfile.flush()
                try:
                    if stall == 'stream':
                        send(chunk(reasoning='Hmm') if chat else {'type': 'response.created', 'sequence_number': 0,
                            'response': {**COMPLETED['response'], 'status': 'in_progress', 'output': []}})
                        if self.stalled(): return
                    if stall == 'reasoning' and self.stalled(lambda: send(chunk(reasoning='think ') if chat else {'type': 'response.reasoning_summary_text.delta',
                            'delta': 'think ', 'item_id': 'r', 'output_index': 0, 'summary_index': 0, 'sequence_number': 1})): return
                    if stall == 'tool_call' and self.stalled(lambda: send(chunk(arguments='{"q') if chat else {'type': 'response.function_call_arguments.delta',
                            'delta': '{"q', 'item_id': 'fc', 'output_index': 0, 'sequence_number': 1})): return
                    if chat:
                        send(chunk(content='Hello world')); send(chunk(finish='stop')); self.wfile.write(b'data: [DONE]\n\n')
                    else:
                        send({'type': 'response.output_text.delta', 'delta': 'Hello world', 'item_id': 'm', 'output_index': 0, 'content_index': 0,
                            'sequence_number': 2, 'logprobs': []})
                        send(COMPLETED)
                    self.wfile.flush()
                except OSError: owner.disconnected.set()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.stall = None
        self.server.shutdown(); self.server.server_close()


@pytest.mark.parametrize('api_mode', ['responses', 'chat_completions'])
@pytest.mark.parametrize('stall', ['reasoning', 'tool_call', 'stream', 'prefill'])
def test_stop_ends_a_streamed_reply_at_once_and_the_provider_keeps_working(api_mode, stall):
    server = FakeServer(stall)
    # request_timeout_seconds stays 120 and the server stalls for 60 s: Stop must not wait for either.
    provider = OpenAIProvider(RuntimeConfig(provider='openai_compatible', base_url=server.url, api_key='test', model='test', api_mode=api_mode))
    stop, reasoning = threading.Event(), []
    threading.Thread(target=lambda: server.streaming.wait(30) and stop.set(), daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(BackgroundPreempted):
            provider.generate([ChatMessage('user', 'Hi')], on_delta=lambda _: None, on_reasoning=reasoning.append, cancelled=stop.is_set)
        assert time.monotonic() - started < 20
        if stall != 'prefill': assert server.disconnected.wait(10)  # the server stops generating too
        server.stall = None
        assert provider.generate([ChatMessage('user', 'Again')], on_delta=lambda _: None).message.content == 'Hello world'
    finally: provider.close(); server.close()


def test_a_reply_that_arrives_after_stop_is_hung_up_by_its_own_request_thread():
    from process.app_core.inference.providers import cancellable_stream
    release, closed = threading.Event(), threading.Event()
    class Stream:
        closes = 0
        def __iter__(self): yield 'event'
        def close(self): self.closes += 1; closed.set()
    late, stop = Stream(), threading.Event()
    stop.set()
    with pytest.raises(BackgroundPreempted, match='before the server answered'):
        with cancellable_stream(lambda: release.wait(10) and late, stop.is_set): pass
    release.set()  # the server answers only now
    assert closed.wait(10) and late.closes == 1
    with cancellable_stream(Stream, lambda: False) as events: assert list(events) == ['event']  # uncancelled: streams normally
