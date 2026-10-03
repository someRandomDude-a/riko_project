import logging
import threading

logger = logging.getLogger(__name__)


def close_bounded(resource, timeout=1):
    close = getattr(resource, 'close', None)
    if not close: return
    done = threading.Event()
    def run():
        try: close()
        except Exception: logger.exception('Resource cleanup failed: %s', type(resource).__name__)
        finally: done.set()
    threading.Thread(target=run, daemon=True, name='resource-cleanup').start()
    if not done.wait(timeout): logger.warning('Cleanup deadline exceeded: %s', type(resource).__name__)
