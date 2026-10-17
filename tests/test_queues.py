import queue
import threading

from pipeline import put_until_stopped


def test_put_waits_for_space():
    q = queue.Queue(maxsize=1)
    q.put("a")
    stop = threading.Event()
    threading.Timer(0.2, q.get).start()
    assert put_until_stopped(q, "b", stop) is True and q.get() == "b"


def test_put_gives_up_when_stopped():
    q = queue.Queue(maxsize=1)
    q.put("a")
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert put_until_stopped(q, "b", stop) is False
