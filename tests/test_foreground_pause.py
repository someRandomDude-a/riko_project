import threading
import time

from process.app_core.inference.llama_server import SlotScheduler


def test_foreground_preempts_background_and_blocks_new_work_until_turn_ends():
    scheduler=SlotScheduler(3,pause_background=True)
    with scheduler.lease('reflection') as (slot,stop):
        scheduler.set_foreground(True)
        assert stop.is_set()
    entered=threading.Event()
    def job():
        with scheduler.lease('initiative'):entered.set()
    thread=threading.Thread(target=job);thread.start()
    try:
        assert not entered.wait(.1)
        with scheduler.lease('live'): assert not entered.is_set()
        assert not entered.wait(.1) # Tool waits between live inference calls still block background.
        scheduler.set_foreground(False)
        assert entered.wait(1)
    finally:scheduler.close();thread.join(1)


def test_pause_toggle_allows_parallel_background_work_when_disabled():
    scheduler=SlotScheduler(2,pause_background=True)
    scheduler.set_foreground(True)
    scheduler.set_pause_background(False)
    with scheduler.lease('live'):
        with scheduler.lease('reflection') as (_,stop):assert not stop.is_set()
    scheduler.close()
