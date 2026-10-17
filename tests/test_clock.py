import pytest

from playback import PlaybackClock
from tests.fakes import FakeTime


def test_wait_for_sleeps_until_absolute_deadline():
    t = FakeTime()
    c = PlaybackClock(10, t.now, t.sleep)
    c.start()
    t.advance(0.03)
    c.wait_for(1)
    assert t.value == pytest.approx(0.1)


def test_no_drift_when_work_takes_time():
    t = FakeTime()
    c = PlaybackClock(10, t.now, t.sleep)
    c.start()
    for n in range(1, 11):
        t.advance(0.05)
        c.wait_for(n)
    assert t.value == pytest.approx(1.0)


def test_lateness_and_rebase():
    t = FakeTime()
    c = PlaybackClock(10, t.now, t.sleep)
    c.start()
    t.advance(0.35)
    assert c.lateness(2) == pytest.approx(0.15)
    c.rebase(2)
    assert c.lateness(2) == pytest.approx(0.0)


def test_wait_for_autostarts():
    t = FakeTime()
    t.advance(5.0)
    c = PlaybackClock(10, t.now, t.sleep)
    c.wait_for(1)
    assert t.value == pytest.approx(5.1)
