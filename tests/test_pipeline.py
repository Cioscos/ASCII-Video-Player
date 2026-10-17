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
