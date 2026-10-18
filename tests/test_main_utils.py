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


def _run_menu_with_keys(monkeypatch, tmp_path, key_list, play):
    import json
    import keyboard_input
    import menu

    monkeypatch.chdir(tmp_path)
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "a.mp4"
    video.write_bytes(b"x")
    settings = menu.Settings(video_path=str(video))
    (tmp_path / "settings.json").write_text(json.dumps(settings.__dict__), encoding="utf-8")
    keys = iter(key_list)

    class FakeKeyReader:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, timeout):
            return next(keys)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(keyboard_input, "KeyReader", FakeKeyReader)
    monkeypatch.setattr(main, "play", play)
    return settings, video


def test_menu_play_then_quit(monkeypatch, tmp_path):
    import menu
    events = []
    settings, video = _run_menu_with_keys(monkeypatch, tmp_path, ["enter", "esc"],
                                          lambda a: events.append(("play", a)) or 0)
    real_save = menu.save_settings
    monkeypatch.setattr(menu, "save_settings",
                        lambda s, path: (events.append(("save", s)), real_save(s, path))[1])
    assert main.main([]) == 0
    assert [e[0] for e in events] == ["save", "play"]
    assert events[0][1] == settings
    assert events[1][1] == main.parse_arguments(settings.to_argv())


def test_menu_play_error_code_sets_message(monkeypatch, tmp_path, capsys):
    _run_menu_with_keys(monkeypatch, tmp_path, ["enter", "esc"], lambda a: 1)
    assert main.main([]) == 0
    assert "Riproduzione terminata con errori" in capsys.readouterr().out


def test_menu_play_exception_sets_message(monkeypatch, tmp_path, capsys):
    def boom(a):
        raise RuntimeError("guasto")
    _run_menu_with_keys(monkeypatch, tmp_path, ["enter", "esc"], boom)
    assert main.main([]) == 0
    assert "guasto" in capsys.readouterr().out


def test_menu_keyboard_interrupt_in_play_returns_to_menu(monkeypatch, tmp_path):
    calls = []

    def interrupted(a):
        calls.append(a)
        raise KeyboardInterrupt
    _run_menu_with_keys(monkeypatch, tmp_path, ["enter", "esc"], interrupted)
    assert main.main([]) == 0
    assert len(calls) == 1


def test_menu_frame_overwrites_in_place_without_clearing():
    # \033[2J su Windows Terminal sposta lo schermo nello scrollback: il menu si sovrascrive
    frame = main.menu_frame("Pixellator\n> Riproduci")
    assert "\033[2J" not in frame
    assert frame == "\033[HPixellator\033[K\n> Riproduci\033[K\033[J"


def test_decimated_total_matches_should_keep():
    from playback import FrameDecimator
    d = FrameDecimator(60.0, 25)
    assert main.decimated_total(1000, 60.0, 25) == sum(d.should_keep(i) for i in range(1000))
    assert main.decimated_total(0, 60.0, 25) == 0
    assert main.decimated_total(-5, 60.0, 25) == 0


@pytest.mark.parametrize("src,target", [(23.976, 12), (29.97, 24), (60.0, 25), (24.0, 12), (30.0, 60),
                                        (float("nan"), 30), (0.0, 30)])
@pytest.mark.parametrize("n", [1, 2, 3, 47, 48, 1001, 7200])
def test_decimated_total_closed_form_matches_should_keep(src, target, n):
    from playback import FrameDecimator
    d = FrameDecimator(src, target)
    assert main.decimated_total(n, src, target) == sum(d.should_keep(i) for i in range(n))


def test_decimated_total_is_constant_time(monkeypatch):
    from playback import FrameDecimator
    monkeypatch.setattr(FrameDecimator, "should_keep", lambda *a: (_ for _ in ()).throw(AssertionError("O(n)")))
    assert main.decimated_total(10 ** 9, 60.0, 25) > 0


def test_controls_hint_lists_rotation_in_both_modes():
    assert main.CONTROLS_HINT == main.CONTROLS_HINT_HP
    assert "r/R ruota" in main.CONTROLS_HINT and "←/→" in main.CONTROLS_HINT


def test_pipeline_factory_uses_rotated_dimensions(monkeypatch):
    from rotation import rotated_dimensions
    monkeypatch.setattr(main, "get_terminal_size", lambda: (200, 60))
    args = main.parse_arguments(["v.mp4"])
    factory = main.make_pipeline_factory(args, 1920, 1080, 30.0, "abc")
    pipeline = factory(90, 7)
    assert (pipeline.width, pipeline.height) == rotated_dimensions(1920, 1080, 90, None, 200, 60, 2)
    assert pipeline.rotation == 90
    assert pipeline.start_index == 7
