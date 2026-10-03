import logging
from types import SimpleNamespace

import pytest

import run_server


@pytest.mark.parametrize('debug', [False, True])
def test_application_debug_does_not_enable_websocket_packet_dump(monkeypatch, debug):
    calls, levels = [], {}
    monkeypatch.setattr(run_server.os, 'chdir', lambda path: None)
    monkeypatch.setattr(run_server, 'load_config', lambda: SimpleNamespace(raw={'desktop':{'debug':debug}}))
    monkeypatch.setattr(run_server, 'logging', SimpleNamespace(INFO=logging.INFO, DEBUG=logging.DEBUG,
        basicConfig=lambda **kwargs: None,
        getLogger=lambda name: SimpleNamespace(setLevel=lambda level: levels.update({name:level}))))
    monkeypatch.setattr(run_server.uvicorn, 'run', lambda *args, **kwargs: calls.append(kwargs))
    run_server.main()
    assert calls[0]['log_level'] == 'info'
    assert levels['process.app_core'] == (logging.DEBUG if debug else logging.INFO)
    assert levels['desktop_server'] == levels['process.app_core']
