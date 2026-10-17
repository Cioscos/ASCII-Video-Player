import cv2
import numpy as np
import pytest


@pytest.fixture
def random_frame():
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(90, 160, 3), dtype=np.uint8)


@pytest.fixture
def tiny_video(tmp_path):
    """Video MJPG 24 fps, 48 frame 64x48, frame i a colore pieno (i*5 % 256, 100, 200)."""
    path = str(tmp_path / "tiny.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 48))
    assert writer.isOpened()
    for i in range(48):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (i * 5 % 256, 100, 200)
        writer.write(frame)
    writer.release()
    return path
