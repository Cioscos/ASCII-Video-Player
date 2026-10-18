import io

import cv2
import numpy as np
import pytest

from ascii_converter import make_converter
from pipeline import VideoPipeline
from playback import run_playback
from terminal_display import TerminalDisplay
from tests.conftest import nth_kept_frame


def test_no_loop_plays_every_decimated_frame_and_exits(tiny_video):
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 4, " .:@", loop_video=False); p.start()
    try:
        drawn = run_playback(p.frame_source(), p.effective_fps, TerminalDisplay(io.BytesIO()), p.should_stop, loop=False)
    finally:
        p.stop()
    assert drawn == 24


def test_stop_is_idempotent_and_ends_source(tiny_video):
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 4, " .:@", loop_video=True); p.start()
    source = p.frame_source()
    assert source.next_frame() is not None
    p.stop()
    p.stop()
    assert source.next_frame() is None


def test_source_ends_when_reader_crashes(tiny_video):
    import threading
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 4, " .:@", loop_video=True); p.start()
    try:
        source = p.frame_source()
        assert source.next_frame() is not None
        p.reader_process.kill()
        p.reader_process.join(timeout=5)
        done = threading.Event()

        def drain():
            while source.next_frame() is not None:
                pass
            done.set()

        threading.Thread(target=drain, daemon=True).start()
        assert done.wait(timeout=10), "next_frame() bloccato dopo il crash del reader"
        assert p.should_stop.is_set()
    finally:
        p.stop()


def test_loop_restarts_decimated_indices(tiny_video):
    # 48 frame a 24 fps decimati a 12 fps -> 24 frame per giro
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 4, " .:@", loop_video=True); p.start()
    try:
        source = p.frame_source()
        indices = [source.next_frame()[0] for _ in range(30)]
    finally:
        p.stop()
    assert indices == list(range(24)) + list(range(6))


def test_partial_last_batch_then_marker(tiny_video):
    # batch_size=5 su 24 frame decimati: l'ultimo batch ha 4 frame, poi il marker
    p = VideoPipeline(tiny_video, 32, 8, 24.0, 12, 5, " .:@", loop_video=False); p.start()
    try:
        drawn = run_playback(p.frame_source(), p.effective_fps, TerminalDisplay(io.BytesIO()), p.should_stop, loop=False)
    finally:
        p.stop()
    assert drawn == 24


def test_half_truecolor_pipeline_matches_converter(tiny_video):
    p = VideoPipeline(tiny_video, 16, 6, 24.0, 12, 4, "half", loop_video=False, truecolor=True); p.start()
    try:
        first = p.frame_source().next_frame()
    finally:
        p.stop()
    conv = make_converter(16, 6, "half", truecolor=True)
    assert first[0] == 0 and first[1] == conv.convert_frame(nth_kept_frame(tiny_video, 0))


@pytest.fixture
def marked_video(tmp_path):
    """Video MJPG 24 fps, 48 frame 64x48: blu = i*5 % 256, quadrato bianco 16x12 in alto a sinistra."""
    path = str(tmp_path / "marked.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 48))
    assert writer.isOpened()
    for i in range(48):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (i * 5 % 256, 0, 0)
        frame[:12, :16] = 255
        writer.write(frame)
    writer.release()
    return path


def run_reader(monkeypatch, video, width, height, queue_cls=None, stop=None, **kw):
    """Esegue frame_reader_process nel processo di test e restituisce [(indice, ndarray)]."""
    import queue, signal, threading
    from pipeline import END_OF_VIDEO_MARKER, frame_reader_process
    monkeypatch.setattr(signal, "signal", lambda *a: None)  # non disattivare Ctrl+C di pytest
    q = (queue_cls or queue.Queue)()
    stop = stop or threading.Event()
    frame_reader_process(video, width, height, 24.0, 12, q, stop, 100, kw.pop("loop", False), **kw)
    out = []
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item == END_OF_VIDEO_MARKER:
            break
        out.extend(item)
    return out


def test_reader_rotates_clockwise_90(monkeypatch, marked_video):
    frames = run_reader(monkeypatch, marked_video, 24, 32, rotation=90)
    img = frames[0][1]
    assert img.shape[:2] == (32, 24)
    assert img[1, -2].min() > 200 and img[-2, 1].max() < 60   # marcatore in alto a destra


def test_reader_rotates_180_and_270(monkeypatch, marked_video):
    assert run_reader(monkeypatch, marked_video, 32, 24, rotation=180)[0][1][-2, -2].min() > 200
    img = run_reader(monkeypatch, marked_video, 24, 32, rotation=270)[0][1]
    assert img.shape[:2] == (32, 24) and img[-2, 1].min() > 200   # in basso a sinistra


def test_reader_starts_at_index(monkeypatch, marked_video):
    frames = run_reader(monkeypatch, marked_video, 32, 24, start_index=10)
    assert [i for i, _ in frames] == list(range(10, 24))
    expected = nth_kept_frame(marked_video, 10)
    assert abs(int(frames[0][1][20, 28, 0]) - int(expected[0, 63, 0])) < 12


def test_reader_start_past_end(monkeypatch, marked_video):
    assert run_reader(monkeypatch, marked_video, 32, 24, start_index=500) == []


def test_reader_start_past_end_with_loop_restarts(monkeypatch, marked_video):
    import queue
    import threading
    stop = threading.Event()

    class StoppingQueue(queue.Queue):
        def put(self, item, *a, **k):
            super().put(item, *a, **k)
            stop.set()

        def cancel_join_thread(self):
            pass

    frames = run_reader(monkeypatch, marked_video, 32, 24, queue_cls=StoppingQueue, stop=stop,
                        loop=True, start_index=500)
    assert frames and frames[0][0] == 0


def test_half_rotated_pipeline_rows(tiny_video):
    p = VideoPipeline(tiny_video, 20, 16, 24.0, 12, 4, "half", loop_video=False, rotation=90); p.start()
    try:
        idx, frame = p.frame_source().next_frame()
    finally:
        p.stop()
    assert idx == 0 and frame.count(b"\n") == 16 - 1


def test_next_frame_timeout_raises():
    import logging
    import queue
    import threading
    from types import SimpleNamespace
    from pipeline import _PipelineFrameSource
    fake = SimpleNamespace(ascii_queue=queue.Queue(), should_stop=threading.Event(),
                           workers_alive=lambda: True, logger=logging.getLogger("fake"))
    with pytest.raises(TimeoutError):
        _PipelineFrameSource(fake).next_frame(timeout=0.3)


def _reader_batches(monkeypatch, video, loop=False, stop_after=None, **kw):
    """Esegue frame_reader_process (batch_size 10) e restituisce gli elementi grezzi della coda."""
    import queue, signal, threading
    from pipeline import frame_reader_process
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    stop = threading.Event()
    items = []

    class RecordingQueue(queue.Queue):
        def put(self, item, *a, **k):
            items.append(item)
            if stop_after is not None and len(items) >= stop_after:
                stop.set()

        def cancel_join_thread(self):
            pass

    frame_reader_process(video, 32, 24, 24.0, 12, RecordingQueue(), stop, 10, loop, **kw)
    return items


def test_reader_first_batch_has_one_frame(monkeypatch, marked_video):
    from pipeline import END_OF_VIDEO_MARKER
    items = _reader_batches(monkeypatch, marked_video)
    assert [len(b) for b in items[:-1]] == [1, 10, 10, 3]  # 24 frame decimati
    assert items[-1] == END_OF_VIDEO_MARKER
    assert [i for b in items[:-1] for i, _ in b] == list(range(24))


def test_reader_first_batch_after_seek_has_one_frame(monkeypatch, marked_video):
    items = _reader_batches(monkeypatch, marked_video, start_index=10)
    assert [len(b) for b in items[:-1]] == [1, 10, 3]


def test_reader_loop_restart_uses_full_batches(monkeypatch, marked_video):
    items = _reader_batches(monkeypatch, marked_video, loop=True, stop_after=7)
    # primo giro: 1 + 10 + 10 + 3; secondo giro (dopo il riavvio): batch interi
    assert [len(b) for b in items[:7]] == [1, 10, 10, 3, 10, 10, 4]
