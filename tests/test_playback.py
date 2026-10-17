import threading

from fps_stats import FpsStats
from playback import PlaybackClock, run_playback
from tests.fakes import FakeTime


class ListSource:
    """Sorgente finta: restituisce (indice, bytes) da una lista di frame."""

    def __init__(self, frames):
        self.frames = frames
        self.pos = 0
        self.restarts = 0

    def next_frame(self):
        if self.pos >= len(self.frames):
            return None
        i = self.pos
        self.pos += 1
        return i, self.frames[i]

    def restart(self):
        self.pos = 0
        self.restarts += 1


class RecordingDisplay:
    """Display finto: registra gli indici disegnati e gli overlay."""

    def __init__(self, ft=None, advance=0.0, on_show=None):
        self.ft = ft
        self.advance = advance
        self.on_show = on_show
        self.drawn = []
        self.overlays = []

    def show(self, frame, overlay=None):
        self.drawn.append(int.from_bytes(frame, "big"))
        self.overlays.append(overlay)
        if self.ft is not None and self.advance:
            self.ft.advance(self.advance)
        if self.on_show:
            self.on_show(len(self.drawn))


class FakeAudio:
    def __init__(self):
        self.times = []

    def update_video_time(self, t):
        self.times.append(t)


def make(n):
    return [i.to_bytes(2, "big") for i in range(n)]


def make_clock(fps, ft):
    return PlaybackClock(fps, now=ft.now, sleep=ft.sleep)


def test_plays_all_in_order_without_loop():
    ft = FakeTime()
    d = RecordingDisplay()
    n = run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, clock=make_clock(10, ft))
    assert d.drawn == list(range(10))
    assert n == 10


def test_skips_late_frames():
    ft = FakeTime()
    d = RecordingDisplay(ft, advance=0.25)
    run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, clock=make_clock(10, ft))
    assert all(a < b for a, b in zip(d.drawn, d.drawn[1:]))
    assert len(d.drawn) < 10
    assert d.drawn[-1] == 9


def test_loop_restarts_until_stopped():
    ft = FakeTime()
    stop = threading.Event()
    d = RecordingDisplay(on_show=lambda k: stop.set() if k == 25 else None)
    run_playback(ListSource(make(10)), 10, d, stop, loop=True, clock=make_clock(10, ft))
    assert d.drawn == list(range(10)) + list(range(10)) + list(range(5))


def test_empty_source_with_loop_terminates():
    assert run_playback(ListSource([]), 10, RecordingDisplay(), threading.Event(), loop=True) == 0


def test_audio_receives_video_time():
    ft = FakeTime()
    a = FakeAudio()
    run_playback(ListSource(make(10)), 10, RecordingDisplay(), threading.Event(), loop=False,
                 audio=a, clock=make_clock(10, ft))
    assert a.times == [i / 10 for i in range(10)]


def test_stall_rebases_instead_of_skipping():
    ft = FakeTime()
    clock = make_clock(10, ft)
    clock.start()
    ft.advance(5.0)  # stallo all'avvio
    d = RecordingDisplay()
    run_playback(ListSource(make(3)), 10, d, threading.Event(), loop=False, clock=clock)
    assert d.drawn == [0, 1, 2]
    assert clock.lateness(2) < 1.0  # orologio riancorato


def test_overlay_every_five_drawn_frames():
    ft = FakeTime()
    d = RecordingDisplay()
    run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False,
                 stats=FpsStats(10), clock=make_clock(10, ft))
    with_overlay = [i for i, o in enumerate(d.overlays) if o]
    assert with_overlay == [0, 5]
    assert "Frame 5" in d.overlays[5][0]


class SlowSource(ListSource):
    """Sorgente lenta: ogni estrazione consuma tempo finto."""

    def __init__(self, frames, ft, cost):
        super().__init__(frames)
        self.ft = ft
        self.cost = cost

    def next_frame(self):
        self.ft.advance(self.cost)
        return super().next_frame()


def test_slow_source_draws_every_frame():
    ft = FakeTime()
    d = RecordingDisplay()
    run_playback(SlowSource(make(10), ft, 0.15), 10, d, threading.Event(), loop=False,
                 clock=make_clock(10, ft))
    assert d.drawn == list(range(10))


def test_single_frame_loop_bounded_by_stop():
    ft = FakeTime()
    stop = threading.Event()
    d = RecordingDisplay(on_show=lambda k: stop.set() if k == 7 else None)
    run_playback(ListSource(make(1)), 10, d, stop, loop=True, clock=make_clock(10, ft))
    assert d.drawn == [0] * 7
