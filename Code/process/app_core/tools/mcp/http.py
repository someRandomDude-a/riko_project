"""MCP over HTTP: one JSON-RPC POST per request, which a cancelled call cannot interrupt."""
from __future__ import annotations

import json
import urllib.request


class HTTPMCPClient:
    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.url, self.headers, self.counter = url, headers or {}, 0
        self._initialize()

    def _request(self, method, params=None):
        self.counter += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params or {}}).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(request, timeout=30) as response: payload = json.loads(response.read())
        if "error" in payload: raise RuntimeError(payload["error"])
        return payload.get("result", {})

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "riko", "version": "0.1"}})
    def list_tools(self): return self._request("tools/list").get("tools", [])
    def call(self, name, arguments): return self._request("tools/call", {"name": name, "arguments": arguments})
    def close(self): pass
