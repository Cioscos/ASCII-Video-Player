from ascii_converter import auto_width, compute_ascii_height
from rotation import ROTATIONS, rotated_dimensions


def test_rotations_cycle():
    assert ROTATIONS == (0, 90, 180, 270)


def test_auto_width_uses_rotated_source():
    for rot in (0, 180):
        w = auto_width(1920, 1080, 200, 60, 2)
        assert rotated_dimensions(1920, 1080, rot, None, 200, 60, 2) == (w, compute_ascii_height(1920, 1080, w))
    for rot in (90, 270):
        w = auto_width(1080, 1920, 200, 60, 2)
        assert rotated_dimensions(1920, 1080, rot, None, 200, 60, 2) == (w, compute_ascii_height(1080, 1920, w))


def test_explicit_width_kept_at_0_and_180():
    assert rotated_dimensions(1920, 1080, 180, 300, 200, 60, 2) == (300, compute_ascii_height(1920, 1080, 300))


def test_explicit_width_reduced_at_90_to_fit_height():
    w = auto_width(1080, 1920, 200, 60, 2)
    assert w < 300
    assert rotated_dimensions(1920, 1080, 90, 300, 200, 60, 2) == (w, compute_ascii_height(1080, 1920, w))


def test_explicit_small_width_kept_at_270():
    assert rotated_dimensions(1920, 1080, 270, 10, 200, 60, 2) == (10, compute_ascii_height(1080, 1920, 10))
