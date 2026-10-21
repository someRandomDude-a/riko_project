import logging
import threading

logger = logging.getLogger(__name__)


def run_bounded(call, timeout, label):
    """Run call on a daemon thread and wait at most timeout seconds; a hung call is abandoned, never joined."""
    done = threading.Event()
    def run():
        try: call()
        except Exception: logger.exception('Resource cleanup failed: %s', label)
        finally: done.set()
    threading.Thread(target=run, daemon=True, name='resource-cleanup').start()
    finished = done.wait(timeout)
    if not finished: logger.warning('Cleanup deadline exceeded: %s', label)
    return finished


def close_bounded(resource, timeout=1):
    close = getattr(resource, 'close', None)
    return run_bounded(close, timeout, type(resource).__name__) if close else True
