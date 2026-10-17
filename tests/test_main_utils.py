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
