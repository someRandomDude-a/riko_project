"""Isolated tools run each call in a disposable worker process (tools/worker.py, --tool-worker when frozen) that imports
only the tool's own module and is killed at its deadline, on Stop or when the registry closes."""
from __future__ import annotations

from concurrent.futures import TimeoutError
import json
import logging
import subprocess
import threading

from .tool import ToolCancelled
from ..kernel.code_paths import CodePaths

logger = logging.getLogger(__name__)


def worker_command(): return CodePaths.current().tool_command()  # tools/worker.py, or the frozen backend's --tool-worker


class IsolatedWorkers:
    def __init__(self):
        self.lock = threading.Lock()
        self.closed = False
        self.processes = {}  # live worker -> its call's stop event

    def run(self, request, arguments, stop, timeout):
        """The tool's result, after running request ({module, class, config, context}) on arguments in a new worker."""
        # stderr carries whatever the tool's libraries print, in the ANSI code page on Windows: never fail decoding it.
        process = subprocess.Popen(worker_command(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        with self.lock:
            if self.closed or stop.is_set():
                process.kill(); process.wait()
                raise RuntimeError('Tool registry closed') if self.closed else ToolCancelled('Tool call cancelled')
            self.processes[process] = stop
        try:
            try:
                output, errors = process.communicate(json.dumps({**request, 'arguments': arguments}), timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill(); process.communicate()
                raise TimeoutError('Isolated tool terminated at deadline')
            if stop.is_set(): raise ToolCancelled('Tool call cancelled')  # killed by cancel()
            if not output.strip():  # the worker itself failed to start (a frozen build missing a module, say)
                logger.warning('Tool worker exited with code %s and no result:\n%s', process.returncode, errors[-4000:])
                raise RuntimeError(f'Tool worker exited with code {process.returncode} and no result')
            payload = json.loads(output.splitlines()[-1])
            if 'error' in payload: raise RuntimeError(payload['error'])
            return payload['result']
        finally:
            with self.lock: self.processes.pop(process, None)

    def cancel(self, stop):
        """Set stop and kill the workers of its call: one starting now sees stop under the lock and kills itself."""
        with self.lock:
            stop.set()
            workers = [process for process, owner in self.processes.items() if owner is stop]
        for process in workers:
            if process.poll() is None: process.kill()

    def close(self):
        with self.lock:
            self.closed = True
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None: process.kill()
