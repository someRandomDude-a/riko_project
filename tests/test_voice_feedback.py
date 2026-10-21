from types import SimpleNamespace

from process.app_core.audio.wake_word import WakeWord
from process.app_core.events.bus import event_bus
from process.app_core.audio.voice_segments import VoiceSegments, Segment
from process.app_core.kernel.audio_config import audio_sections
from process.app_core.configuration.paths import DataPaths


def test_activation_and_expiry_publish_status_without_polling(tmp_path,monkeypatch):
    now=[100.0];timers=[];events=[]
    class Timer:
        def __init__(self,delay,callback):self.delay,self.callback,self.cancelled=delay,callback,False;timers.append(self)
        def start(self):pass
        def cancel(self):self.cancelled=True
    monkeypatch.setattr('process.app_core.audio.wake_word.threading.Timer',Timer)
    monkeypatch.setattr('process.app_core.audio.wake_word.time.monotonic',lambda:now[0])
    config=SimpleNamespace(root=tmp_path,paths=DataPaths.at(tmp_path),character_name='Riko',**audio_sections({'voice':{'mode':'manual','follow_up_seconds':10}}))
    wake=WakeWord(config)
    off=event_bus.subscribe(events.append)
    try:
        wake.activate()
        assert [event.type for event in events]==['voice.activated','voice.wake_status']
        assert events[-1].payload['active'] and timers[-1].delay==10
        now[0]=111
        timers[-1].callback()
        assert events[-2].type=='voice.wake_status' and not events[-2].payload['active']
        assert events[-1].type=='voice.waiting'
        wake.response_finished();timer=timers[-1];wake.responding()
        count=len(events);timer.callback()
        assert len(events)==count and timer.cancelled
    finally:off();wake.close()


def test_continuous_speech_emits_rolling_partial_windows_without_committing_them():
    emitted=[]
    segmenter=VoiceSegments(emitted.append,lambda:None,lambda *args:None,partial_interval=2)
    for index in range(130):segmenter.feed(b'\1\0'*512,True,index*.032)
    assert len(emitted)==2 and all(segment.provisional and not segment.final for segment in emitted)
    assert len(emitted[1].pcm)>len(emitted[0].pcm)
    assert len(segmenter.audio)==130 # Provisional decoding did not consume/commit the original.
    for index in range(35):segmenter.feed(b'\0\0'*512,False,(130+index)*.032)
    assert emitted[-1].final and not emitted[-1].provisional


def test_partial_asr_revisions_replace_text_and_never_dispatch_a_reply_early(monkeypatch):
    import queue
    import sys
    import threading
    from process.app_core.audio.voice_input import VoiceInput
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=object))
    voice=VoiceInput.__new__(VoiceInput)
    voice.closed=threading.Event();voice.jobs=queue.Queue()
    voice._partial_lock=threading.Lock();voice._partial_pending=set();voice._parts={}
    outputs=iter(['hel','hello','hello there'])
    voice.session=SimpleNamespace(wake=SimpleNamespace(calibrating=False,testing=False),config=SimpleNamespace(raw={},**audio_sections({})),transcribe=lambda pcm,**options:next(outputs))
    replies=[]
    def submit(fn,text,segment):replies.append(text);voice.closed.set()
    voice.responses=SimpleNamespace(submit=submit)
    for provisional,final in [(True,False),(True,False),(False,True)]:
        voice.jobs.put(Segment('utterance',b'\1\0'*512,None,0,1,1,final,provisional))
    events=[];off=event_bus.subscribe(lambda event:events.append(event) if event.type=='voice.transcript' else None)
    try:
        voice._asr()
        assert [event.payload['text'] for event in events]==['hel','hello','hello there']
        assert replies==['hello there'] and not voice._parts
    finally:off();voice.closed.set()
