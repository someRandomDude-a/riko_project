"""Bounded daemon workers: hung native calls must not own interpreter shutdown."""
from concurrent.futures import Future
import queue
import threading


class DaemonExecutor:
    def __init__(self, max_workers=1, thread_name_prefix='worker', max_pending=128):
        self.jobs = queue.Queue(maxsize=max_pending)
        self.lock = threading.Lock()
        self.closed = False
        self.threads = []
        for index in range(max_workers):
            thread = threading.Thread(target=self._run, daemon=True, name=f'{thread_name_prefix}-{index}')
            thread.start()
            self.threads.append(thread)

    def submit(self, fn, *args, **kwargs):
        future = Future()
        with self.lock:
            if self.closed: raise RuntimeError('Worker is closed')
            try: self.jobs.put_nowait((future, fn, args, kwargs))
            except queue.Full: raise RuntimeError('Worker backlog is full')
        return future

    def _run(self):
        while True:
            try: future, fn, args, kwargs = self.jobs.get(timeout=.1)
            except queue.Empty:
                if self.closed: return
                continue
            try:
                if future.set_running_or_notify_cancel():
                    try: future.set_result(fn(*args, **kwargs))
                    except BaseException as exc: future.set_exception(exc)
            finally: self.jobs.task_done()

    def shutdown(self, wait=False, cancel_futures=True):
        with self.lock:
            self.closed = True
            if cancel_futures:
                while True:
                    try: future, *_ = self.jobs.get_nowait()
                    except queue.Empty: break
                    future.cancel(); self.jobs.task_done()
        if wait:
            for thread in self.threads: thread.join(timeout=.5)
