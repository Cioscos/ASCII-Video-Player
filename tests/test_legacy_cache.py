import os

from precomputer import FramePrecomputer


def test_all_frames_written_before_completed(tiny_video, tmp_path):
    p = FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=str(tmp_path / "c"))
    md = p.precompute_frames()
    assert md["completed"] and md["processed_frames"] == 24
    assert all((tmp_path / "c" / f"frame_{i:06d}.txt").exists() for i in range(24))


def test_stale_metadata_is_recomputed(tiny_video, tmp_path):
    out = str(tmp_path / "c")
    FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out).precompute_frames()
    os.remove(os.path.join(out, "frame_000005.txt"))
    p = FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out)
    assert not p.is_cache_valid()
    p.precompute_frames()
    assert os.path.exists(os.path.join(out, "frame_000005.txt"))


def test_palette_invalidates_cache(tiny_video, tmp_path):
    out = str(tmp_path / "c")
    FramePrecomputer(tiny_video, 32, 12, " .:@", output_dir=out).precompute_frames()
    assert not FramePrecomputer(tiny_video, 32, 12, "box", output_dir=out).is_cache_valid()
