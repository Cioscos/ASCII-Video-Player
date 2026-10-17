import numpy as np
import pytest

from ascii_converter import AsciiConverter, compute_ascii_height, make_converter, pixel_rows
from main import EXTENDED
from tests.reference import parse_ansi_cells, parse_ansi_grid, reference_cells, reference_grid


@pytest.mark.parametrize("palette", [" .:-=+*#%@", EXTENDED, "@", "box", "░▒▓█"])
def test_matches_reference(random_frame, palette):
    h = compute_ascii_height(160, 90, 120)
    out = AsciiConverter(120, h, palette).convert_frame(random_frame)
    assert parse_ansi_grid(out.decode("utf-8")) == reference_grid(random_frame, 120, palette)


def test_height_one():
    assert compute_ascii_height(1000, 10, 50) == 1
    assert len(AsciiConverter(50, 1, "@").convert_frame(np.zeros((10, 1000, 3), np.uint8)).split(b"\n")) == 1


def test_color_emitted_only_on_change():
    out = AsciiConverter(40, 3, "@").convert(np.full((3, 40, 3), 200, np.uint8))
    assert out.count(b"\033[38;5;") == 3


def test_two_color_rows_emit_two_escapes():
    f = np.zeros((4, 40, 3), np.uint8)
    f[:, 20:] = 255
    assert AsciiConverter(40, 4, "@").convert(f).count(b"[38;5;") == 8


def test_long_custom_palette_matches_reference(random_frame):
    palette = "".join(chr(0x100 + i) for i in range(300))  # oltre 255 caratteri
    h = compute_ascii_height(160, 90, 120)
    out = AsciiConverter(120, h, palette).convert_frame(random_frame)
    assert parse_ansi_grid(out.decode("utf-8")) == reference_grid(random_frame, 120, palette)


def test_empty_palette_raises():
    with pytest.raises(ValueError):
        AsciiConverter(10, 2, "")


def test_half_matches_reference(random_frame):
    h = compute_ascii_height(160, 90, 120)
    out = make_converter(120, h, "half").convert_frame(random_frame)
    assert parse_ansi_cells(out) == reference_cells(random_frame, 120, "half")


def test_half_doubles_pixel_rows():
    assert pixel_rows(10, "half") == 20 and pixel_rows(10, "@") == 10
    assert make_converter(40, 10, "half").resize(np.zeros((90, 160, 3), np.uint8)).shape == (20, 40, 3)


def test_half_run_length_on_pair():
    f = np.zeros((8, 40, 3), np.uint8)
    f[1::2, 20:] = 255  # cambia solo il pixel sotto, da meta' riga
    out = make_converter(40, 4, "half").convert(f)
    assert out.count(b"\033[38;5;") == 8 and out.count("▀".encode()) == 160


def test_make_converter_default_is_ascii_converter():
    assert type(make_converter(40, 4, "@")) is AsciiConverter


def test_reference_cells_agrees_with_legacy_reference(random_frame):
    legacy = reference_grid(random_frame, 120, EXTENDED)
    assert [[(c, f) for c, f, _ in row] for row in reference_cells(random_frame, 120, EXTENDED)] == legacy
