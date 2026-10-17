"""Terminale virtuale minimale per i test: interpreta solo le sequenze usate da TerminalDisplay."""
import re

_CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")


class VirtualTerminal:
    def __init__(self, rows: int = 200, cols: int = 400):
        self.rows = rows
        self.cols = cols
        self._grid = [[" "] * cols for _ in range(rows)]
        self._r = 0
        self._c = 0

    def feed(self, data: bytes) -> None:
        text = data.decode("utf-8")
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                m = _CSI.match(text, i)
                if not m:
                    raise ValueError(f"sequenza non supportata in posizione {i}")
                self._csi(m.group(1), m.group(2))
                i = m.end()
                continue
            if ch == "\n":
                self._r = min(self._r + 1, self.rows - 1)
                self._c = 0
            elif ch == "\r":
                self._c = 0
            else:
                if self._c < self.cols:
                    self._grid[self._r][self._c] = ch
                    self._c += 1
            i += 1

    def _csi(self, params: str, final: str) -> None:
        if final == "m" or params.startswith("?"):
            return  # colori e modalita' privata (cursore): ignorati
        if final == "J" and params == "2":
            self._grid = [[" "] * self.cols for _ in range(self.rows)]
        elif final == "H":
            parts = [p for p in params.split(";") if p]
            r = int(parts[0]) if parts else 1
            c = int(parts[1]) if len(parts) > 1 else 1
            self._r = min(r - 1, self.rows - 1)
            self._c = min(c - 1, self.cols - 1)
        elif final == "K":
            for k in range(self._c, self.cols):
                self._grid[self._r][k] = " "
        else:
            raise ValueError(f"CSI non supportata: {params}{final}")

    @property
    def cursor(self) -> tuple[int, int]:
        """Posizione del cursore (riga, colonna), 0-based."""
        return self._r, self._c

    def screen(self) -> list[str]:
        return ["".join(row) for row in self._grid]
