import io
import random

from terminal_display import TerminalDisplay
from tests.vt import VirtualTerminal

N_ROWS = 20


def make_frame(seed: int) -> bytes:
    """Frame deterministico di 20 righe; le righe pari restano uguali tra frame consecutivi."""
    rows = []
    for r in range(N_ROWS):
        rng = random.Random(r * 1000 + (seed if r % 2 else 0))
        text = "".join(rng.choice("abc#@.") for _ in range(30))
        rows.append(f"\033[38;5;{16 + r}m{text}\033[0m")
    return "\n".join(rows).encode("utf-8")


def test_delta_equals_full_redraw():
    frames = [make_frame(seed) for seed in range(5)]
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    for f in frames:
        d.show(f)
        vt.feed(buf.getvalue())
        buf.seek(0)
        buf.truncate()
        full = VirtualTerminal()
        full.feed(b"\033[2J\033[H" + f)
        assert vt.screen() == full.screen()


def test_identical_frame_writes_almost_nothing():
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    f = make_frame(0)
    d.show(f)
    n = len(buf.getvalue())
    d.show(f)
    assert len(buf.getvalue()) - n < 16
    assert len(buf.getvalue()) == n  # per design: nessuna scrittura


def test_overlay_lines_cleared():
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    f = make_frame(0)
    d.show(f, ["AAAAAA"])
    d.show(f, ["B"])
    vt.feed(buf.getvalue())
    assert vt.screen()[N_ROWS + 1].rstrip() == "B"


def test_overlay_shorter_clears_old_lines_and_persists():
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    f = make_frame(0)
    d.show(f, ["L1", "L2", "L3"])
    d.show(f, ["X"])
    d.show(make_frame(1))  # overlay None: resta quello precedente
    vt.feed(buf.getvalue())
    s = vt.screen()
    assert s[N_ROWS + 1].rstrip() == "X"
    assert s[N_ROWS + 2].strip() == "" and s[N_ROWS + 3].strip() == ""


def test_overlay_survives_full_redraw():
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    f = make_frame(0)
    d.show(f, ["HUD"])
    d.invalidate()
    d.show(f)
    vt.feed(buf.getvalue())
    assert vt.screen()[N_ROWS + 1].rstrip() == "HUD"


def test_row_count_change_redraws_fully():
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    d.show(make_frame(0))
    short = b"\n".join(make_frame(1).split(b"\n")[:10])
    d.show(short)
    vt.feed(buf.getvalue())
    full = VirtualTerminal()
    full.feed(b"\033[2J\033[H" + short)
    assert vt.screen() == full.screen()


def test_restore_shows_cursor():
    buf = io.BytesIO()
    TerminalDisplay(buf).restore()
    assert buf.getvalue().endswith(b"\033[0m\033[?25h")


def _variant(base: bytes, changed: set) -> bytes:
    """Sostituisce le righe indicate con righe di pari larghezza prese dalla riga successiva."""
    rows = base.split(b"\n")
    other = make_frame(99).split(b"\n")
    for r in changed:
        rows[r] = other[(r + 1) % N_ROWS]  # riga vicina: colore e testo diversi
    return b"\n".join(rows)


def test_delta_multi_row_runs_and_edges():
    base = make_frame(0)
    frames = [
        base,
        _variant(base, set(range(3, 9))),  # run di 6 righe adiacenti
        _variant(base, {0, N_ROWS - 1}),  # prima e ultima riga
        _variant(base, {0, 1, 2, N_ROWS - 2, N_ROWS - 1}),
        base,  # ritorno a righe invariate
        base,
    ]
    buf = io.BytesIO()
    d = TerminalDisplay(buf)
    vt = VirtualTerminal()
    for f in frames:
        d.show(f)
        vt.feed(buf.getvalue())
        buf.seek(0)
        buf.truncate()
        full = VirtualTerminal()
        full.feed(b"\033[2J\033[H" + f)
        assert vt.screen() == full.screen()
