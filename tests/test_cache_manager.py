import json
import os

import pytest

from ascii_converter import CONVERTER_VERSION
from cache_manager import enforce_limit, list_caches, parse_size
from precomputer import CachedFrames, precompute


def make_cache(root, name, size, mtime, video="a.mp4", version=CONVERTER_VERSION):
    d = root / name
    d.mkdir()
    (d / "frames.bin").write_bytes(b"x" * size)
    meta = d / "metadata.json"
    meta.write_text(json.dumps({
        "effective_fps": 10.0, "frames": 1, "width": 80, "height": 40, "palette": "abc",
        "truecolor": False, "video_path": f"/v/{video}", "converter_version": version}))
    os.utime(meta, (mtime, mtime))
    return str(d)


def test_parse_size():
    assert parse_size("0") == 0 and parse_size("512") == 512
    assert parse_size("2G") == 2 * 1024**3 and parse_size("5m") == 5 * 1024**2
    for bad in ("abc", "", "-1", "1.5G", "G"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_list_sorted_and_obsolete(tmp_path):
    make_cache(tmp_path, "old", 10, 1000, video="old.mp4")
    make_cache(tmp_path, "new", 20, 2000, video="new.mp4")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "frame_000000.txt").write_text("x")
    (legacy / "metadata.pkl").write_bytes(b"y")
    os.utime(legacy, (100, 100))
    make_cache(tmp_path, "stale", 5, 500, version=CONVERTER_VERSION + 99)

    entries = list_caches(str(tmp_path))
    by_name = {os.path.basename(e.path): e for e in entries}
    assert [e.video_name for e in entries[:2]] == ["new.mp4", "old.mp4"]
    assert by_name["new"].size_bytes > 20 and not by_name["new"].obsolete
    assert (by_name["new"].width, by_name["new"].palette) == (80, "abc")
    assert by_name["legacy"].obsolete and by_name["legacy"].width is None
    assert by_name["legacy"].video_name == "legacy"
    assert by_name["stale"].obsolete
    assert list_caches(str(tmp_path / "nope")) == []


def test_enforce_limit_deletes_least_recent(tmp_path):
    a = make_cache(tmp_path, "a", 100, 1000)
    b = make_cache(tmp_path, "b", 100, 2000)
    c = make_cache(tmp_path, "c", 100, 3000)
    total = sum(e.size_bytes for e in list_caches(str(tmp_path)))
    removed = enforce_limit(str(tmp_path), total - 1, keep=set())
    assert removed == [a]
    assert not os.path.exists(a) and os.path.exists(b) and os.path.exists(c)


def test_enforce_limit_keeps_new_even_if_too_big(tmp_path):
    big = make_cache(tmp_path, "big", 500, 1000)  # la più vecchia ma in keep
    x = make_cache(tmp_path, "x", 100, 2000)
    y = make_cache(tmp_path, "y", 100, 3000)
    removed = enforce_limit(str(tmp_path), 200, keep={big})
    assert sorted(removed) == sorted([x, y])
    assert os.path.exists(big)


def test_limit_zero_disables(tmp_path):
    a = make_cache(tmp_path, "a", 100, 1000)
    assert enforce_limit(str(tmp_path), 0, keep=set()) == []
    assert os.path.exists(a)


def test_open_updates_last_used(tiny_video, tmp_path):
    root = str(tmp_path / "cache")
    cache_dir = precompute(tiny_video, 20, 10, 24, 24, "abc", cache_root=root)
    meta = os.path.join(cache_dir, "metadata.json")
    os.utime(meta, (1000, 1000))
    cached = CachedFrames.open(cache_dir)
    try:
        assert cached is not None
        assert os.path.getmtime(meta) > 1000
    finally:
        cached.close()
