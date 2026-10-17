import os

import cv2
import pytest

from ascii_converter import AsciiConverter
from playback import FrameDecimator
from precomputer import CachedFrames, CachedFrameSource, cache_dir_for, precompute


def nth_kept_frame(video, n):
    """n-esimo frame sorgente tenuto da FrameDecimator(24.0, 12)."""
    decimator = FrameDecimator(24.0, 12)
    cap = cv2.VideoCapture(video)
    try:
        source_index = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            assert ok, "video terminato prima del frame richiesto"
            keep = decimator.should_keep(source_index)
            source_index += 1
            if not keep:
                continue
            if kept == n:
                return frame
            kept += 1
    finally:
        cap.release()


def test_roundtrip(tiny_video, tmp_path):
    d = precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    c = CachedFrames.open(d)
    conv = AsciiConverter(32, 8, " .:@")
    try:
        assert len(c) == 24 and c.frame(3) == conv.convert_frame(nth_kept_frame(tiny_video, 3))
        assert c.metadata["effective_fps"] == 12.0 and c.metadata["frames"] == 24
    finally:
        c.close()


def test_key_changes(tiny_video, tmp_path):
    base = cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))
    assert base == cache_dir_for(tiny_video, 32, 8, 12.0, " .:@", str(tmp_path))
    assert base != cache_dir_for(tiny_video, 32, 8, 12, "box", str(tmp_path))
    assert base != cache_dir_for(tiny_video, 32, 8, 6, " .:@", str(tmp_path))
    assert base != cache_dir_for(tiny_video, 32, 9, 12, " .:@", str(tmp_path))
    os.utime(tiny_video, ns=(1, 1))
    assert base != cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))


def test_precompute_uses_cache_dir_for(tiny_video, tmp_path):
    d = precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    assert d == cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))


def test_interrupted_precompute_not_reused(tiny_video, tmp_path, monkeypatch):   # Review Focus 4
    calls = iter(range(100))

    def boom(self, r):
        if next(calls) == 5:
            raise KeyboardInterrupt
        return b"x"

    monkeypatch.setattr(AsciiConverter, "convert", boom)
    with pytest.raises(KeyboardInterrupt):
        precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    d = cache_dir_for(tiny_video, 32, 8, 12, " .:@", str(tmp_path))
    assert CachedFrames.open(d) is None and not os.path.exists(os.path.join(d, "frames.bin.tmp"))
    assert not os.path.exists(os.path.join(d, "frames.bin"))
    assert not os.path.exists(os.path.join(d, "index.npy"))


def test_recompute_removes_stale_metadata_on_failure(tiny_video, tmp_path, monkeypatch):
    d = precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)

    def boom(self, r):
        raise RuntimeError("errore di conversione")

    monkeypatch.setattr(AsciiConverter, "convert", boom)
    with pytest.raises(RuntimeError):
        precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0)
    assert CachedFrames.open(d) is None


def test_multiprocess_matches_inline(tiny_video, tmp_path):
    inline = CachedFrames.open(precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path / "a"), processes=0))
    multi = CachedFrames.open(precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path / "b"), processes=2))
    try:
        assert len(inline) == len(multi) == 24
        assert all(inline.frame(i) == multi.frame(i) for i in range(24))
    finally:
        inline.close()
        multi.close()


def test_max_frames_limits_cache(tiny_video, tmp_path):
    c = CachedFrames.open(precompute(tiny_video, 32, 8, 24.0, 12, "box", str(tmp_path), processes=0, max_frames=5))
    try:
        assert len(c) == 5
    finally:
        c.close()


def test_unreadable_video_returns_none(tmp_path):
    bogus = tmp_path / "nonvideo.avi"
    bogus.write_bytes(b"non e' un video")
    assert precompute(str(bogus), 32, 8, 24.0, 12, " .:@", str(tmp_path / "c"), processes=0) is None


def test_source_iterates_and_restarts(tiny_video, tmp_path):
    c = CachedFrames.open(precompute(tiny_video, 32, 8, 24.0, 12, " .:@", str(tmp_path), processes=0, max_frames=3))
    try:
        src = CachedFrameSource(c)
        items = [src.next_frame() for _ in range(4)]
        assert [i for i, _ in items[:3]] == [0, 1, 2] and items[3] is None
        assert items[1][1] == c.frame(1)
        src.restart()
        assert src.next_frame() == (0, c.frame(0))
    finally:
        c.close()


def test_open_missing_metadata_returns_none(tmp_path):
    assert CachedFrames.open(str(tmp_path / "inesistente")) is None
