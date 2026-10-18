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


# --- RotatingSource -------------------------------------------------------

from rotation import RotatingSource


class FakeSource:
    def __init__(self, pipe):
        self.pipe = pipe
        self.k = 0

    def next_frame(self, timeout=None):
        if self.pipe.fail or self.pipe.stopped:
            return None
        i = self.pipe.start_index + self.k
        self.k += 1
        return i, b"r%d-%d" % (self.pipe.rotation, i)


class FakePipeline:
    def __init__(self, rotation, start_index, fail=False):
        self.rotation = rotation
        self.start_index = start_index
        self.fail = fail
        self.started = False
        self.stopped = False
        self._source = FakeSource(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def frame_source(self):
        return self._source


class FakeCache:
    def __init__(self, n=100):
        self.n = n
        self.pos = 0
        self.seeks = []

    def next_frame(self):
        if self.pos >= self.n:
            return None
        i = self.pos
        self.pos += 1
        return i, b"c-%d" % i

    def restart(self):
        self.pos = 0

    def seek(self, index):
        self.seeks.append(index)
        self.pos = index

    def __len__(self):
        return self.n


class Factory:
    def __init__(self, fail_at=()):
        self.made = []
        self.fail_at = fail_at

    def __call__(self, rotation, start_index):
        p = FakePipeline(rotation, start_index, fail=rotation in self.fail_at)
        self.made.append(p)
        return p


def test_standard_starts_pipeline_at_zero():
    f = Factory()
    src = RotatingSource(f, 50)
    src.start()
    assert [(p.rotation, p.start_index, p.started) for p in f.made] == [(0, 0, True)]
    assert src.next_frame() == (0, b"r0-0")
    assert src.slow_seek is True
    assert len(src) == 50


def test_rotate_from_cache_to_pipeline_and_back():
    f = Factory()
    cache = FakeCache()
    src = RotatingSource(f, len(cache), cache=cache)
    src.start()
    assert f.made == []
    assert src.slow_seek is False
    for _ in range(3):
        src.next_frame()
    assert src.rotate(+1) is True
    assert src.rotation == 90
    assert (f.made[0].rotation, f.made[0].start_index) == (90, 3)
    assert src.next_frame() == (3, b"r90-3")
    assert src.next_frame() == (4, b"r90-4")
    assert src.rotate(-1) is True
    assert src.rotation == 0
    assert f.made[0].stopped
    assert cache.seeks == [5]
    assert src.next_frame() == (5, b"c-5")


def test_rotate_cycles_and_stops_intermediate_pipelines():
    f = Factory()
    src = RotatingSource(f, 50)
    src.start()
    angles = []
    for _ in range(4):
        assert src.rotate(+1) is True
        angles.append(src.rotation)
    assert angles == [90, 180, 270, 0]
    assert all(p.stopped for p in f.made[:-1])
    assert not f.made[-1].stopped


def test_rotate_counterclockwise_from_zero_is_270():
    src = RotatingSource(Factory(), 50)
    src.start()
    assert src.rotate(-1) is True
    assert src.rotation == 270


def test_rotate_failure_restores_previous_cache():
    f = Factory(fail_at=(90,))
    cache = FakeCache()
    src = RotatingSource(f, len(cache), cache=cache)
    src.start()
    for _ in range(2):
        src.next_frame()
    assert src.rotate(+1) is False
    assert src.rotation == 0
    assert f.made[0].stopped
    assert src.next_frame() == (2, b"c-2")


def test_rotate_failure_restores_pipeline_without_cache():
    f = Factory(fail_at=(90,))
    src = RotatingSource(f, 50)
    src.start()
    src.next_frame()
    src.next_frame()
    assert src.rotate(+1) is False
    assert src.rotation == 0
    restored = f.made[-1]
    assert (restored.rotation, restored.start_index, restored.started) == (0, 2, True)
    assert not restored.stopped
    assert src.next_frame() == (2, b"r0-2")


def test_rotate_failure_on_exception_restores_previous():
    def make(rotation, start):
        if rotation == 90:
            raise RuntimeError("boom")
        return FakePipeline(rotation, start)

    cache = FakeCache()
    src = RotatingSource(make, len(cache), cache=cache)
    src.start()
    assert src.rotate(+1) is False
    assert src.rotation == 0


def test_seek_while_rotated_restarts_pipeline():
    f = Factory()
    cache = FakeCache()
    src = RotatingSource(f, len(cache), cache=cache)
    src.start()
    src.rotate(+1)
    src.seek(20)
    assert (f.made[-1].rotation, f.made[-1].start_index) == (90, 20)
    assert f.made[0].stopped
    assert src.slow_seek is True
    assert src.next_frame() == (20, b"r90-20")


def test_back_to_cache_after_seek_while_rotated():
    f = Factory()
    cache = FakeCache()
    src = RotatingSource(f, len(cache), cache=cache)
    src.start()
    src.rotate(+1)
    src.seek(20)
    assert src.rotate(-1) is True
    assert cache.seeks[-1] == 20
    assert src.next_frame() == (20, b"c-20")


def test_seek_with_cache_active_delegates():
    cache = FakeCache()
    src = RotatingSource(Factory(), len(cache), cache=cache)
    src.start()
    src.seek(7)
    assert src.next_frame() == (7, b"c-7")


def test_restart_delegates_to_cache():
    cache = FakeCache()
    src = RotatingSource(Factory(), len(cache), cache=cache)
    src.start()
    src.next_frame()
    src.restart()
    assert src.next_frame() == (0, b"c-0")


def test_close_stops_current_pipeline_and_is_idempotent():
    f = Factory()
    src = RotatingSource(f, 50)
    src.start()
    src.close()
    src.close()
    assert f.made[0].stopped
