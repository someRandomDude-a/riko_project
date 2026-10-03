"""Standalone newline-delimited JSON-RPC task MCP server (stdio)."""
import argparse
import json
from pathlib import Path
import sys

from process.app_core.persistence.tasks import TaskMCP, TaskStore


def dispatch(request, server):
    method = request.get('method')
    if method == 'initialize':
        version = request.get('params', {}).get('protocolVersion', '2024-11-05')
        if version not in {'2024-11-05', '2025-03-26', '2025-06-18'}: version = '2024-11-05'
        return {'protocolVersion': version, 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'riko-tasks', 'version': '1.0'}}
    if method == 'ping': return {}
    if method == 'tools/list': return {'tools': server.list_tools()}
    if method == 'tools/call':
        params = request.get('params', {})
        return server.call(params['name'], params.get('arguments', {}))
    raise ValueError('Unknown method')


def main():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, default=Path(__file__).resolve().parent.parent / 'persistent_memories' / 'tasks.sqlite3')
    args = parser.parse_args()
    server = TaskMCP(TaskStore(args.store), actor='external_mcp')
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            print(json.dumps({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid JSON'}}), flush=True)
            continue
        if not isinstance(request, dict): continue
        if 'id' not in request: continue # Notifications never receive responses.
        try:
            result = dispatch(request, server)
            response = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
        except (ValueError, KeyError, TypeError) as exc:
            response = {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32601 if str(exc) == 'Unknown method' else -32602, 'message': str(exc)}}
        except Exception as exc:
            response = {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32603, 'message': str(exc)}}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
