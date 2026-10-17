import numpy as np
import pytest

from audio_player import AudioPlayer


def test_callback_wraps_at_end():
    p = AudioPlayer("x")
    p.load_array(np.arange(10, dtype=np.float64).reshape(10, 1).repeat(2, 1), 10)
    p.audio_time = 0.8
    out = np.zeros((4, 2), np.float32)
    assert p._audio_callback(out, 4, None, None) is None
    assert out[:, 0].tolist() == [8, 9, 0, 1] and not p.should_stop.is_set()


def test_resync_applied_before_copy():
    p = AudioPlayer("x")
    p.load_array(np.arange(100.0).reshape(100, 1).repeat(2, 1), 10)
    p.update_video_time(5.0)
    out = np.zeros((3, 2), np.float32)
    p._audio_callback(out, 3, None, None)
    assert out[:, 0].tolist() == [50, 51, 52] and p.audio_time == pytest.approx(5.3)


def test_downmix_and_dtype():
    p = AudioPlayer("x")
    p.load_array(np.ones((5, 6)), 10)
    assert p.audio_array.shape == (5, 2) and p.audio_array.dtype == np.float32


def test_video_without_audio_track(tiny_video):
    assert AudioPlayer(tiny_video).initialize() is False
