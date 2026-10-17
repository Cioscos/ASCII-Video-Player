import sys
from typing import BinaryIO, Optional


class TerminalDisplay:
    """
    Scrive i frame ASCII sul terminale ridisegnando solo le righe cambiate.

    Il primo frame, il frame dopo invalidate() e quello con un numero di righe
    diverso producono un ridisegno completo; negli altri casi vengono riscritte
    solo le sequenze di righe consecutive modificate. L'overlay (statistiche FPS)
    sta sotto il frame e persiste finche' non viene sostituito.
    """

    def __init__(self, out: Optional[BinaryIO] = None):
        self._out = out if out is not None else sys.stdout.buffer
        self._rows: Optional[list] = None
        self._overlay: list = []

    def invalidate(self) -> None:
        """Forza un ridisegno completo al prossimo show()."""
        self._rows = None

    def show(self, frame: bytes, overlay: Optional[list] = None) -> None:
        """Mostra il frame; overlay=None lascia invariato l'overlay precedente."""
        rows = frame.split(b"\n")
        parts = []
        old_overlay_len = len(self._overlay)
        overlay_changed = overlay is not None
        if overlay_changed:
            self._overlay = list(overlay)

        if self._rows is None or len(rows) != len(self._rows):
            parts.append(b"\033[?25l\033[2J\033[H" + frame)
            overlay_changed = True
            old_overlay_len = 0  # lo schermo e' stato cancellato
        else:
            old = self._rows
            r = 0
            n = len(rows)
            while r < n:
                if rows[r] == old[r]:
                    r += 1
                    continue
                start = r
                while r < n and rows[r] != old[r]:
                    r += 1
                parts.append(b"\033[%d;1H" % (start + 1) + b"\n".join(rows[start:r]))
        self._rows = rows

        if overlay_changed:
            base = len(rows) + 2
            for i, line in enumerate(self._overlay):
                parts.append(b"\033[%d;1H" % (base + i) + line.encode("utf-8") + b"\033[K")
            for i in range(len(self._overlay), old_overlay_len):
                parts.append(b"\033[%d;1H\033[K" % (base + i))

        if parts:
            self._out.write(b"".join(parts))
            self._out.flush()

    def restore(self) -> None:
        """Ripristina colori e cursore."""
        self._out.write(b"\033[0m\033[?25h")
        self._out.flush()
