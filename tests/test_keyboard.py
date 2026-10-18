import io
import json
import os
import subprocess
import sys

import pytest

from keyboard_input import KeyReader, decode_keys, records_to_chars


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


def test_windows_extended_keys_marked_are_dropped():
    m = "￾"
    # Home, End, Canc, PgUp, PgDn, Ins, F11: nessun tasto; freccia mappata
    assert decode_keys(m + "àG" + m + "àO" + m + "àS" + m + "àI"
                       + m + "àQ" + m + "àR" + m + "à", windows=True) == []
    assert decode_keys(m + "àHq" + m + "àSx", windows=True) == ["up", "q", "x"]


def test_windows_lone_prefix_stays_a_grave():
    assert decode_keys("à", windows=True) == ["à"]


def test_records_to_chars_arrows_never_a_grave():
    # (premuto, virtual key, carattere, ripetizioni): ↓ ha carattere nullo, "à" ha vk 0
    events = [(True, 0x28, "\x00", 1), (False, 0x28, "\x00", 1), (True, 0, "à", 1),
              (True, 0x26, "\x00", 2), (True, 0x2E, "\x00", 1), (True, 0x51, "q", 1)]
    assert decode_keys(records_to_chars(events), windows=True) == ["down", "à", "up", "up", "q"]


def test_records_to_chars_a_grave_then_key_keeps_both():
    events = [(True, 0, "à", 1), (True, 0x51, "q", 1)]
    assert decode_keys(records_to_chars(events), windows=True) == ["à", "q"]


@pytest.mark.skipif(os.name != "nt", reason="richiede una console Windows")
def test_windows_real_console_arrows_and_a_grave(tmp_path):
    # Console vera: con msvcrt.getwch le frecce arrivavano come "à" + lettera;
    # il tasto premuto prima dell'avvio del lettore viene scartato
    out = tmp_path / "keys.json"
    probe = os.path.join(os.path.dirname(__file__), "console_keys_probe.py")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, probe, str(out), root],
                   creationflags=subprocess.CREATE_NEW_CONSOLE, timeout=30, check=True)
    assert json.loads(out.read_text(encoding="utf-8"))["keys"] == ["down", "left", "à", "à", "q", "enter"]
