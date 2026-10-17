import pytest

from playback import FrameDecimator


def kept(src, tgt, n):
    d = FrameDecimator(src, tgt)
    return [i for i in range(n) if d.should_keep(i)]


def test_30_to_24_keeps_4_of_5_uniformly():
    k = kept(30, 24, 300)
    assert len(k) == 240 and max(b - a for a, b in zip(k, k[1:])) == 2


def test_target_above_source_is_capped():
    d = FrameDecimator(30, 60)
    assert d.effective_fps == 30 and len(kept(30, 60, 90)) == 90


def test_25_to_10():
    k = kept(25, 10, 250)
    assert len(k) == 100 and max(b - a for a, b in zip(k, k[1:])) == 3


def test_none_target_keeps_all():
    assert FrameDecimator(25, None).effective_fps == 25 and len(kept(25, None, 50)) == 50


@pytest.mark.parametrize("bad", [0, float("nan")])
def test_invalid_source_fps_keeps_all(bad):
    d = FrameDecimator(bad, 20)
    assert d.effective_fps == 20 and len(kept(bad, 20, 40)) == 40
