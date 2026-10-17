import io

from keyboard_input import KeyReader, decode_keys


def test_windows_arrows_and_keys():
    assert decode_keys("\xe0H\x00P\xe0K\xe0M\r\x1b \x08q", windows=True) == \
        ["up", "down", "left", "right", "enter", "esc", "space", "backspace", "q"]


def test_posix_arrows_and_keys():
    assert decode_keys("\x1b[A\x1b[B\x1b[C\x1b[D\n\x7f \x1bq", windows=False) == \
        ["up", "down", "right", "left", "enter", "backspace", "space", "esc", "q"]


def test_windows_unknown_prefix_dropped():
    assert decode_keys("\x00;q", windows=True) == ["q"]


def test_posix_unknown_csi_does_not_leak():
    assert decode_keys("\x1b[15~q\x1b[1;5Ax", windows=False) == ["q", "up", "x"]


def test_inactive_when_not_a_tty():
    with KeyReader(io.StringIO("q")) as k:
        assert not k.active and k.poll() is None and k.read(0.01) is None


def test_inactive_without_isatty():
    class Senza:
        pass
    with KeyReader(Senza()) as k:
        assert not k.active and k.poll() is None


def test_windows_a_grave_literal():
    assert decode_keys("\xe0", windows=True) == ["\xe0"]
    assert decode_keys("\xe0q", windows=True) == ["\xe0", "q"]
    assert decode_keys("\xe0H", windows=True) == ["up"]


def _reader_attivo(monkeypatch, blocchi):
    k = KeyReader(io.StringIO())
    k.active = True
    k._windows = True
    it = iter(blocchi)
    monkeypatch.setattr(k, "_read_raw_windows", lambda: next(it, ""))
    return k


def test_poll_returns_queued_keys_one_at_a_time(monkeypatch):
    k = _reader_attivo(monkeypatch, ["ab"])
    assert [k.poll(), k.poll(), k.poll()] == ["a", "b", None]


def test_read_returns_key_and_times_out(monkeypatch):
    k = _reader_attivo(monkeypatch, ["", "", "x"])
    assert k.read(1.0) == "x"
    assert k.read(0.05) is None
