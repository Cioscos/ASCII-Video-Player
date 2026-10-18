import threading

import pytest

from fps_stats import FpsStats
from playback import (PAUSE_TEXT, ROTATE_FAILED_TEXT, SEEK_UNAVAILABLE_TEXT, SEEK_WAIT_TEXT, FrameDecimator,
                      PlaybackClock, run_playback, seek_target)
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
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1

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
        self.paused_calls = []
        self.seeks = []
        self.events = []  # sequenza ordinata di tutte le chiamate

    def update_video_time(self, t):
        self.times.append(t)
        self.events.append(("time", t))

    def set_paused(self, paused):
        self.paused_calls.append(paused)
        self.events.append(("paused", paused))

    def seek(self, t):
        self.seeks.append(t)
        self.events.append(("seek", t))


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


class FakeKeys:
    """Tastiera finta: `script` indicizzato per numero (da 1) di chiamate a poll."""

    def __init__(self, script=None, after_pause=(), ft=None):
        self.script = dict(script or {})
        self.after_pause = list(after_pause)
        self.ft = ft
        self.polls = 0

    def poll(self):
        self.polls += 1
        return self.script.get(self.polls)

    def read(self, timeout):
        if self.ft is not None:
            self.ft.advance(timeout)  # simula l'attesa del timeout
        return self.after_pause.pop(0) if self.after_pause else None


class SeekableListSource(ListSource):
    """ListSource con seek, come CachedFrameSource."""

    def seek(self, index):
        self.pos = index

    def __len__(self):
        return len(self.frames)


def test_q_quits():
    ft = FakeTime()
    d = RecordingDisplay()
    n = run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False,
                     clock=make_clock(10, ft), controls=FakeKeys({3: "q"}))
    assert d.drawn == [0, 1]
    assert n == 2


def test_esc_quits_during_pause():
    ft = FakeTime()
    d = RecordingDisplay()
    a = FakeAudio()
    n = run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, audio=a,
                     clock=make_clock(10, ft), controls=FakeKeys({2: "space"}, after_pause=[None, "esc"]))
    assert d.drawn == [0, 0]
    assert n == 1
    assert a.paused_calls == [True]


def test_pause_and_resume_draws_no_extra_frames():
    ft = FakeTime()
    d = RecordingDisplay()
    a = FakeAudio()
    keys = FakeKeys({3: "space"}, after_pause=[None, None, "space"], ft=ft)
    n = run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, audio=a,
                     clock=make_clock(10, ft), controls=keys)
    assert d.drawn == [0, 1, 1, 1] + list(range(2, 10))
    assert n == 10  # i ridisegni di pausa/ripresa non contano
    assert a.paused_calls == [True, False]
    assert PAUSE_TEXT in d.overlays[2]
    assert d.overlays[3] == []  # ripresa: riga di stato rimossa
    assert d.overlays[4] is None


def test_pause_overlay_follows_stats_lines():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({3: "space"}, after_pause=["space"])
    run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, stats=FpsStats(10),
                 clock=make_clock(10, ft), controls=keys)
    stats_lines = d.overlays[0]
    assert d.overlays[2][-1] == PAUSE_TEXT
    assert len(d.overlays[2]) == len(stats_lines) + 1
    assert len(d.overlays[3]) == len(stats_lines) and PAUSE_TEXT not in d.overlays[3]


def test_seek_forward_and_back_on_seekable_source():
    ft = FakeTime()
    d = RecordingDisplay()
    a = FakeAudio()
    keys = FakeKeys({11: "right", 12: "left"})
    run_playback(SeekableListSource(make(100)), 10, d, threading.Event(), loop=False, audio=a,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn[:13] == list(range(10)) + [59, 9, 10]
    assert a.seeks == [5.9, 0.9]


def test_seek_long_and_before_start():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({4: "down", 6: "up"})
    run_playback(SeekableListSource(make(400)), 10, d, threading.Event(), loop=False,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn[:7] == [0, 1, 2, 0, 1, 301, 302]


def test_seek_clamps_at_end_without_loop():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({7: "up"})
    run_playback(SeekableListSource(make(20)), 10, d, threading.Event(), loop=False,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn == [0, 1, 2, 3, 4, 5, 19]


def test_seek_wraps_to_start_with_loop():
    ft = FakeTime()
    stop = threading.Event()
    d = RecordingDisplay(on_show=lambda k: stop.set() if k == 9 else None)
    keys = FakeKeys({7: "up"})
    run_playback(SeekableListSource(make(20)), 10, d, stop, loop=True,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn == [0, 1, 2, 3, 4, 5, 0, 1, 2]


def test_seek_unavailable_shows_status_then_clears():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({3: "right"})
    run_playback(ListSource(make(40)), 10, d, threading.Event(), loop=False,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn == list(range(40))  # nessun seek
    assert d.overlays[2] == [SEEK_UNAVAILABLE_TEXT]
    cleared = d.overlays.index([])
    assert 20 <= cleared <= 23  # dopo circa 2 s di tempo finto (frame a 10 fps)
    assert all(o is None for o in d.overlays[3:cleared])
    assert all(o is None for o in d.overlays[cleared + 1:])


def test_seek_unavailable_status_kept_in_stats_refresh():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({3: "left"})
    run_playback(ListSource(make(40)), 10, d, threading.Event(), loop=False, stats=FpsStats(10),
                 clock=make_clock(10, ft), controls=keys)
    stats_len = len(d.overlays[0])
    assert d.overlays[2][-1] == SEEK_UNAVAILABLE_TEXT and len(d.overlays[2]) == stats_len + 1
    assert d.overlays[5][-1] == SEEK_UNAVAILABLE_TEXT  # aggiornamento stats con stato attivo
    cleared = next(i for i, o in enumerate(d.overlays) if i > 2 and o and SEEK_UNAVAILABLE_TEXT not in o)
    assert 20 <= cleared <= 23


def test_seek_target_rules():
    assert seek_target(5, -10, 20, False) == 0 and seek_target(15, 10, 20, False) == 19
    assert seek_target(15, 10, 20, True) == 0 and seek_target(3, 4, 20, True) == 7


def test_pause_updates_audio_video_time_consistently():
    ft = FakeTime()
    a = FakeAudio()
    keys = FakeKeys({3: "space"}, after_pause=[None, None, "space"], ft=ft)
    run_playback(ListSource(make(10)), 10, RecordingDisplay(), threading.Event(), loop=False, audio=a,
                 clock=make_clock(10, ft), controls=keys)
    resume = a.events.index(("paused", False))
    assert a.events[resume - 1] == ("paused", True)  # nessun update durante la pausa
    assert a.events[resume + 1] == ("time", 0.2)
    assert a.times == [i / 10 for i in range(10)]


def test_seek_then_audio_time_matches_target():
    ft = FakeTime()
    a = FakeAudio()
    keys = FakeKeys({4: "right"})
    run_playback(SeekableListSource(make(100)), 10, RecordingDisplay(), threading.Event(), loop=False,
                 audio=a, clock=make_clock(10, ft), controls=keys)
    i = a.events.index(("seek", 5.2))
    assert a.events[i + 1] == ("time", 5.2)


def test_pause_time_not_recorded_in_fps_stats():
    ft = FakeTime()
    stats = FpsStats(10)
    keys = FakeKeys({3: "space"}, after_pause=[None] * 200 + ["space"], ft=ft)  # 10 s di pausa
    run_playback(ListSource(make(10)), 10, RecordingDisplay(), threading.Event(), loop=False,
                 stats=stats, clock=make_clock(10, ft), controls=keys)
    assert ft.now() > 10.0
    assert len(stats._times) == 8  # 10 frame disegnati, intervallo a cavallo della pausa escluso
    assert max(stats._times) < 1.0


class RotatableListSource(SeekableListSource):
    """SeekableListSource con rotazione, come RotatingSource."""

    def __init__(self, frames, result=True, slow_seek=False, ft=None, advance=0.0):
        super().__init__(frames)
        self.rotation = 0
        self.rotate_calls = []
        self.result = result
        self.slow_seek = slow_seek
        self.ft = ft
        self.advance = advance

    def rotate(self, step):
        self.rotate_calls.append(step)
        if self.ft is not None:
            self.ft.advance(self.advance)
        return self.result


def _run_rotate(key, **kw):
    ft = FakeTime()
    d = RecordingDisplay()
    a = FakeAudio()
    src = RotatableListSource(make(10), ft=ft, **kw)
    run_playback(src, 10, d, threading.Event(), loop=False, audio=a, clock=make_clock(10, ft),
                 controls=FakeKeys({3: key}))
    return src, d, a


def test_r_rotates_clockwise_and_redraws():
    src, d, _ = _run_rotate("r")
    assert src.rotate_calls == [1]
    assert d.overlays[2] == ["Rotazione 90°…"]
    assert d.invalidations == 1
    assert d.overlays[3] == []  # overlay ridisegnato senza la riga di attesa


def test_shift_r_rotates_counterclockwise():
    src, d, _ = _run_rotate("R")
    assert src.rotate_calls == [-1]
    assert d.overlays[2] == ["Rotazione 270°…"]


def test_rotation_pauses_and_resyncs_audio():
    _, _, a = _run_rotate("r")
    events = [e for e in a.events if e[0] != "time"]
    assert events == [("paused", True), ("seek", 0.2), ("paused", False)]


def test_rotation_failure_shows_status():
    _, d, _ = _run_rotate("r", result=False)
    assert any(o and ROTATE_FAILED_TEXT in o for o in d.overlays[3:])


def test_rotation_interval_not_in_stats():
    ft = FakeTime()
    stats = FpsStats(10)
    src = RotatableListSource(make(10), ft=ft, advance=2.0)
    run_playback(src, 10, RecordingDisplay(), threading.Event(), loop=False, stats=stats,
                 clock=make_clock(10, ft), controls=FakeKeys({3: "r"}))
    assert ft.now() > 2.0
    assert stats._times and max(stats._times) < 1.0


def test_slow_seek_shows_wait_and_pauses_audio():
    ft = FakeTime()
    d = RecordingDisplay()
    a = FakeAudio()
    src = RotatableListSource(make(100), slow_seek=True)
    run_playback(src, 10, d, threading.Event(), loop=False, audio=a, clock=make_clock(10, ft),
                 controls=FakeKeys({3: "right"}))
    assert d.overlays[2] == [SEEK_WAIT_TEXT]
    events = [e for e in a.events if e[0] != "time"]
    assert events == [("paused", True), ("seek", 5.1), ("paused", False)]
    assert d.invalidations == 0


def test_r_ignored_without_rotate():
    ft = FakeTime()
    d = RecordingDisplay()
    run_playback(ListSource(make(10)), 10, d, threading.Event(), loop=False, clock=make_clock(10, ft),
                 controls=FakeKeys({3: "r"}))
    assert d.drawn == list(range(10))
    assert d.invalidations == 0


@pytest.mark.parametrize("src,target", [(24.0, 12), (30.0, 24), (60.0, 25), (25.0, 60), (float("nan"), 30)])
def test_source_index_inverts_should_keep(src, target):
    d = FrameDecimator(src, target)
    kept = [s for s in range(300) if d.should_keep(s)]
    assert [d.source_index(i) for i in range(len(kept))] == kept


class UnknownLengthSource(SeekableListSource):
    """Sorgente con seek ma numero di frame sconosciuto (len 0)."""

    def __len__(self):
        return 0


def test_seek_target_unknown_total_has_no_upper_bound():
    assert seek_target(10, 50, 0, False) == 60 and seek_target(10, 50, 0, True) == 60
    assert seek_target(3, -50, 0, True) == 0 and seek_target(3, -50, -1, False) == 0


def test_seek_with_unknown_length_moves_forward_and_clamps_at_zero():
    ft = FakeTime()
    d = RecordingDisplay()
    keys = FakeKeys({4: "right", 6: "down"})
    run_playback(UnknownLengthSource(make(100)), 10, d, threading.Event(), loop=False,
                 clock=make_clock(10, ft), controls=keys)
    assert d.drawn[:7] == [0, 1, 2, 52, 53, 0, 1]
