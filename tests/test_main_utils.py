import sys
import pytest

import main
from utils import configure_process_logging


def test_standard_palette():
    assert len(main.PALETTES["standard"]) == 42 and main.PALETTES["standard"][0] == " "


def test_custom_palette_keeps_leading_space(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(" .░▓█\r\n", encoding="utf-8")
    assert main.load_custom_palette(str(f)) == " .░▓█"


def test_custom_palette_empty(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        main.load_custom_palette(str(f))


def test_process_logging_no_duplicates():
    a = configure_process_logging("T")
    b = configure_process_logging("T")
    assert a is b and len(a.handlers) == 2 and a.propagate is False


def test_warn_if_frame_too_tall(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(main, "get_terminal_size", lambda: (80, 30))
    log = logging.getLogger("test_main")
    assert main.warn_if_frame_too_tall(29, log) is False
    with caplog.at_level(logging.WARNING, logger="test_main"):
        assert main.warn_if_frame_too_tall(30, log) is True
    assert "supera l'altezza del terminale" in caplog.text


def test_shutdown_restores_first_and_survives_interrupt():
    calls = []

    class FakeDisplay:
        def restore(self):
            calls.append("restore")

    def interrupted():
        calls.append("stop")
        raise KeyboardInterrupt

    import logging
    main._shutdown(FakeDisplay(), logging.getLogger("test_main"), interrupted, None, lambda: calls.append("close"))
    assert calls == ["restore", "stop", "close"]


def test_cli_accepts_half_and_truecolor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "v.mp4", "80", "--palette", "half", "--truecolor"])
    a = main.parse_arguments()
    assert a.palette == "half" and a.truecolor


def test_open_controls_returns_key_reader():
    from keyboard_input import KeyReader
    assert isinstance(main._open_controls(), KeyReader)


def test_no_args_without_tty_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert main.main([]) == 1 and "Uso:" in capsys.readouterr().out


def test_cli_args_go_to_play(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "play", lambda a: calls.append(a) or 0)
    assert main.main(["v.mp4", "80"]) == 0 and calls[0].width == 80


def test_menu_play_then_quit(monkeypatch, tmp_path):
    import json
    import keyboard_input
    import menu

    monkeypatch.chdir(tmp_path)
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "a.mp4"
    video.write_bytes(b"x")
    settings = menu.Settings(video_path=str(video))
    (tmp_path / "settings.json").write_text(json.dumps(settings.__dict__), encoding="utf-8")

    keys = iter(["enter", "esc"])

    class FakeKeyReader:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, timeout):
            return next(keys)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(keyboard_input, "KeyReader", FakeKeyReader)
    calls = []
    monkeypatch.setattr(main, "play", lambda a: calls.append(a) or 0)
    assert main.main([]) == 0
    assert len(calls) == 1 and calls[0].video_path == str(video)
    assert (tmp_path / "settings.json").exists()
