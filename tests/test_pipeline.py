import io

from pipeline import VideoPipeline
from playback import run_playback
from terminal_display import TerminalDisplay


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
