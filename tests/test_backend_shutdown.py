import socket
import threading
import time
import urllib.request

import uvicorn
from fastapi import FastAPI

import run_server


def test_shutdown_stops_the_turn_before_uvicorn_drains_requests():
    """A reply still generating must end at shutdown, not hold its request through uvicorn's drain."""
    cancel, order, result = threading.Event(), [], {}
    app = FastAPI()
    @app.post('/chat')
    def chat():
        order.append('chat')
        return {'cancelled': cancel.wait(10)}
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(8)
    port = listener.getsockname()[1]
    server = run_server.Server(uvicorn.Config(app, log_level='warning', timeout_graceful_shutdown=10),
        lambda: (order.append('stop'), cancel.set()))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline: time.sleep(.01)
    request = urllib.request.Request(f'http://127.0.0.1:{port}/chat', method='POST')
    client = threading.Thread(target=lambda: result.update(body=urllib.request.urlopen(request, timeout=10).read()))
    client.start()
    while 'chat' not in order and time.monotonic() < deadline: time.sleep(.01)
    started = time.monotonic()
    server.should_exit = True
    thread.join(10); client.join(10)
    assert not thread.is_alive() and time.monotonic() - started < 3  # not the 10 s drain
    assert order == ['chat', 'stop'] and result['body'] == b'{"cancelled":true}'
