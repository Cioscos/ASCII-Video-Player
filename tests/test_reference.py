import cv2
import pytest

from main import EXTENDED
from precomputer import FramePrecomputer
from tests.reference import parse_ansi_grid, reference_grid


@pytest.mark.parametrize("palette", [" .:-=+*#%@", EXTENDED, "@", "box"])
def test_legacy_precomputer_matches_reference(random_frame, palette):
    pre = FramePrecomputer.__new__(FramePrecomputer)  # evita la creazione di directory
    pre.width, pre.ascii_palette = 120, palette
    if palette == "box":
        out = pre._convert_frame_to_box_ascii(random_frame)
    else:
        out = pre._convert_frame_to_ascii(random_frame)
    assert parse_ansi_grid(out) == reference_grid(random_frame, 120, palette)


def test_tiny_video_fixture(tiny_video):
    cap = cv2.VideoCapture(tiny_video)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 48
    cap.release()
