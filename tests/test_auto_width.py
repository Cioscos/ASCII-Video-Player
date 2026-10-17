import sys

import main
from ascii_converter import auto_width, compute_ascii_height


def test_limited_by_columns():
    assert auto_width(1920, 1080, 100, 200, 2) == 100


def test_limited_by_rows():
    w = auto_width(1920, 1080, 400, 50, 2)
    assert compute_ascii_height(1920, 1080, w) + 2 <= 50 < compute_ascii_height(1920, 1080, w + 1) + 2


def test_limited_by_source_width():
    assert auto_width(64, 48, 300, 300, 2) == 64


def test_tiny_terminal_minimum_one():
    assert auto_width(1920, 1080, 10, 1, 6) == 1


def test_reserved_rows():
    assert main.reserved_rows(True) == 6
    assert main.reserved_rows(False) == 2


def test_cli_width_optional(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "v.mp4"])
    assert main.parse_arguments().width is None
