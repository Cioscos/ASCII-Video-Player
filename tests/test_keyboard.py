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
