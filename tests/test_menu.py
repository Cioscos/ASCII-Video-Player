import json
import os
import sys

import main
from menu import (PALETTE_CHOICES, MenuState, Settings, list_videos, load_settings, render_menu,
                  save_settings)


def main_parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return main.parse_arguments()


def press(state, *keys):
    action = None
    for key in keys:
        action = state.handle_key(key)
    return action


def make_cache(root, name, size=10, mtime=1000, version=None):
    from ascii_converter import CONVERTER_VERSION
    d = root / name
    d.mkdir(parents=True)
    (d / "frames.bin").write_bytes(b"x" * size)
    meta = d / "metadata.json"
    meta.write_text(json.dumps({
        "effective_fps": 10.0, "frames": 1, "width": 80, "height": 40, "palette": "half",
        "truecolor": False, "video_path": f"/v/{name}.mp4",
        "converter_version": CONVERTER_VERSION if version is None else version}))
    os.utime(meta, (mtime, mtime))
    return d


def test_to_argv_roundtrip(monkeypatch):
    s = Settings(video_path="v.mp4", width=None, fps=30, palette="half", truecolor=True, loop=False,
                 cache_limit="5G")
    a = main_parse(s.to_argv(), monkeypatch)
    assert a.video_path == "v.mp4" and a.width is None and a.fps == 30 and a.palette == "half"
    assert a.truecolor and a.no_loop and a.cache_limit == "5G"
    assert not a.audio and not a.high_performance and not a.log_fps and a.custom_palette is None


def test_to_argv_all_flags(monkeypatch):
    s = Settings(video_path="v.mp4", width=120, fps=5, palette="custom", custom_palette="p.txt",
                 high_performance=True, audio=True, log_fps=True)
    a = main_parse(s.to_argv(), monkeypatch)
    assert a.width == 120 and a.palette == "custom" and a.custom_palette == "p.txt"
    assert a.high_performance and a.audio and a.log_fps and not a.no_loop and not a.truecolor
    assert "--custom-palette" not in Settings(video_path="v.mp4", custom_palette="p.txt").to_argv()


def test_load_missing_settings(tmp_path):
    assert load_settings(str(tmp_path / "manca.json")) == (Settings(), None)


def test_load_corrupted_settings(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{nope", encoding="utf-8")
    s, msg = load_settings(str(p))
    assert s == Settings() and msg


def test_load_settings_with_wrong_types(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"fps": "tanti", "palette": "boh"}), encoding="utf-8")
    s, msg = load_settings(str(p))
    assert s == Settings() and msg


def test_save_and_load(tmp_path):
    p = str(tmp_path / "settings.json")
    s = Settings(video_path="a b.mp4", width=200, fps=24, palette="custom", custom_palette="p.txt",
                 truecolor=True, high_performance=True, audio=True, loop=False, log_fps=True,
                 cache_limit="500M")
    save_settings(s, p)
    assert load_settings(p) == (s, None)


def test_list_videos(tmp_path):
    for name in ("b.MP4", "a.mkv", "c.webm", "note.txt", "d.Avi", "e.mov"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "dir.mp4").mkdir()
    videos = list_videos(str(tmp_path))
    assert [os.path.basename(v) for v in videos] == ["a.mkv", "b.MP4", "c.webm", "d.Avi", "e.mov"]
    assert videos[0] == os.path.join(str(tmp_path), "a.mkv")
    assert list_videos(str(tmp_path / "manca")) == []


def test_empty_videos_folder_shows_hint_and_path_entry(tmp_path):
    st = MenuState(Settings(), list_videos(str(tmp_path / "manca")))
    st.handle_key("down")
    st.handle_key("enter")
    screen = render_menu(st, 80, 24)
    assert "Nessun video in videos/" in screen and "Inserisci percorso" in screen


def test_select_video_from_list(tmp_path):
    st = MenuState(Settings(), ["videos/a.mp4", "videos/b.mp4"])
    press(st, "down", "enter", "down", "enter")
    assert st.screen == "main" and st.settings.video_path == "videos/b.mp4"


def test_enter_path_manually():
    st = MenuState(Settings(), ["videos/a.mp4"])
    press(st, "down", "enter", "down", "enter")
    assert st.screen == "edit"
    press(st, "x", "space", "y", "z", "backspace", ".", "m", "p", "4", "enter")
    assert st.screen == "main" and st.settings.video_path == "x y.mp4"


def test_play_requires_existing_video(tmp_path):
    st = MenuState(Settings(), [])
    assert st.handle_key("enter") is None and st.message == "Seleziona prima un video"
    missing = str(tmp_path / "manca.mp4")
    st.settings.video_path = missing
    assert st.handle_key("enter") is None and st.message == f"Il file video '{missing}' non esiste"
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    st.settings.video_path = str(video)
    assert st.handle_key("enter") == "play"


def test_message_cleared_on_next_key():
    st = MenuState(Settings(), [])
    st.handle_key("enter")
    assert st.message
    st.handle_key("down")
    assert st.message is None


def settings_state(tmp_path):
    st = MenuState(Settings(), [], settings_path=str(tmp_path / "settings.json"))
    press(st, "down", "down", "enter")
    assert st.screen == "settings"
    return st


def test_settings_palette_cycles_and_width_auto(tmp_path):
    st = settings_state(tmp_path)
    assert "auto" in render_menu(st, 80, 24)
    press(st, "down", "down")  # palette
    seen = []
    for _ in PALETTE_CHOICES:
        st.handle_key("right")
        seen.append(st.settings.palette)
    assert seen == PALETTE_CHOICES[2:] + PALETTE_CHOICES[:2]
    st.handle_key("left")
    assert st.settings.palette == "basic"
    press(st, "up", "up", "enter")  # larghezza
    assert st.screen == "edit"
    press(st, "1", "2", "0", "enter")
    assert st.screen == "settings" and st.settings.width == 120
    press(st, "enter", "backspace", "backspace", "backspace", "enter")
    assert st.settings.width is None and "auto" in render_menu(st, 80, 24)


def test_settings_fps_and_bools(tmp_path):
    st = settings_state(tmp_path)
    st.handle_key("down")  # fps
    press(st, "right", "right")
    assert st.settings.fps == 12
    for _ in range(20):
        st.handle_key("left")
    assert st.settings.fps == 1
    press(st, "down", "down", "down", "right")  # truecolor
    assert st.settings.truecolor is True
    st.handle_key("left")
    assert st.settings.truecolor is False
    press(st, "down", "down", "down", "right")  # loop
    assert st.settings.loop is False


def test_settings_saved_on_esc(tmp_path):
    st = settings_state(tmp_path)
    press(st, "down", "right", "esc")
    assert st.screen == "main"
    assert load_settings(str(tmp_path / "settings.json"))[0].fps == 11


def test_settings_save_error_goes_to_message(tmp_path):
    st = MenuState(Settings(), [], settings_path=str(tmp_path))  # una directory: open fallisce
    press(st, "down", "down", "enter", "esc")
    assert st.screen == "main" and st.message


def test_edit_invalid_number_keeps_editing(tmp_path):
    st = settings_state(tmp_path)
    press(st, "down", "enter")  # fps
    press(st, "backspace", "backspace", "a", "enter")
    assert st.screen == "edit" and st.message
    press(st, "backspace", "0", "enter")
    assert st.screen == "edit" and st.message and st.settings.fps == 10
    press(st, "esc")
    assert st.screen == "settings" and st.settings.fps == 10


def test_edit_cache_limit_and_custom_path(tmp_path):
    st = settings_state(tmp_path)
    for _ in range(9):
        st.handle_key("down")  # limite cache (ultima voce)
    st.handle_key("enter")
    press(st, "backspace", "backspace", "x", "enter")
    assert st.screen == "edit" and st.message and st.settings.cache_limit == "2G"
    press(st, "backspace", "5", "0", "0", "M", "enter")
    assert st.screen == "settings" and st.settings.cache_limit == "500M"
    press(st, "up", "up", "up", "up", "up", "up", "enter")  # percorso custom
    press(st, "p", ".", "t", "x", "t", "enter")
    assert st.settings.custom_palette == "p.txt"


def test_cache_delete_with_confirmation(tmp_path):
    root = tmp_path / "cached_frames"
    a = make_cache(root, "a", mtime=2000)
    b = make_cache(root, "b", mtime=1000)
    st = MenuState(Settings(), [], cache_root=str(root))
    press(st, "down", "down", "down", "enter")
    assert st.screen == "cache"
    screen = render_menu(st, 100, 24)
    assert "a.mp4" in screen and "b.mp4" in screen and "1970-01-01" in screen
    press(st, "enter")
    assert st.screen == "confirm" and "Cancellare a.mp4? (s/n)" in render_menu(st, 100, 24)
    press(st, "n")
    assert st.screen == "cache" and a.exists()
    press(st, "down", "enter", "s")
    assert st.screen == "cache" and not b.exists() and a.exists()
    assert "b.mp4" not in render_menu(st, 100, 24)


def test_cache_delete_obsolete(tmp_path):
    root = tmp_path / "cached_frames"
    keep = make_cache(root, "ok", mtime=2000)
    old = make_cache(root, "old", mtime=1000, version=-1)
    st = MenuState(Settings(), [], cache_root=str(root))
    press(st, "down", "down", "down", "enter")
    assert "[obsoleta]" in render_menu(st, 100, 24)
    press(st, "o")
    assert st.screen == "confirm"
    press(st, "esc")
    assert old.exists()
    press(st, "o", "s")
    assert not old.exists() and keep.exists()


def test_cache_empty(tmp_path):
    st = MenuState(Settings(), [], cache_root=str(tmp_path / "manca"))
    press(st, "down", "down", "down", "enter", "enter")
    assert st.screen == "cache" and "Nessuna cache" in render_menu(st, 80, 24)


def test_render_fits_and_scrolls():
    videos = [f"videos/v{i:02d}.mp4" for i in range(50)]
    st = MenuState(Settings(), videos)
    press(st, "down", "enter")
    for _ in range(40):
        st.handle_key("down")
    screen = render_menu(st, 20, 10)
    lines = screen.split("\n")
    assert len(lines) <= 10 and all(len(line) <= 20 for line in lines)
    assert any(line.startswith("> ") and "v40.mp4" in line for line in lines)


def test_esc_on_main_quits():
    assert MenuState(Settings(), []).handle_key("esc") == "quit"


def test_quit_item():
    st = MenuState(Settings(), [])
    assert press(st, "down", "down", "down", "down", "enter") == "quit"
