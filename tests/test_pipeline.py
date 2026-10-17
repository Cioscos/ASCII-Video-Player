import io

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
