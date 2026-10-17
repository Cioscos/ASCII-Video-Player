"""Convertitore frame -> ASCII ANSI 256 colori, vettorizzato, con run-length del colore."""
import cv2
import numpy as np

CHAR_ASPECT_CORRECTION = 2.25
CONVERTER_VERSION = 2

_BLOCK_CHAR = '█'


def compute_ascii_height(src_w: int, src_h: int, width: int) -> int:
    """Altezza in righe ASCII per un frame sorgente src_w x src_h (minimo 1)."""
    return max(1, int(src_h / src_w * width / CHAR_ASPECT_CORRECTION))


class AsciiConverter:
    """Converte frame BGR in byte UTF-8 con sequenze ANSI; il colore e' emesso solo al cambio."""

    def __init__(self, width: int, height: int, palette: str):
        self.width = width
        self.height = height
        self.palette = palette
        chars = [_BLOCK_CHAR] if palette == 'box' else list(palette)
        n = len(chars)
        if n == 0:
            raise ValueError("La palette non puo' essere vuota")
        # intp: le palette custom possono superare 255 caratteri
        self.char_lut = np.array([int(g / 255.0 * (n - 1)) for g in range(256)], dtype=np.intp)
        self.full_tok = np.empty((216, n), dtype=object)
        for color in range(16, 232):
            for k, ch in enumerate(chars):
                self.full_tok[color - 16, k] = f"\033[38;5;{color}m{ch}".encode("utf-8")
        self.short_tok = np.empty(n, dtype=object)
        for k, ch in enumerate(chars):
            self.short_tok[k] = ch.encode("utf-8")
        self.row_end = b"\033[0m\n"

    def resize(self, frame):
        """Ridimensiona il frame alla griglia ASCII (nearest neighbour)."""
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    def convert(self, resized) -> bytes:
        """Converte un frame gia' ridimensionato (H, W, 3 BGR); nessun '\\n' finale."""
        h, w = resized.shape[:2]
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        b = resized[:, :, 0].astype(np.int16)
        g = resized[:, :, 1].astype(np.int16)
        r = resized[:, :, 2].astype(np.int16)
        color = 16 + 36 * np.minimum(5, r // 43) + 6 * np.minimum(5, g // 43) + np.minimum(5, b // 43)
        k = self.char_lut[gray]
        change = np.ones(color.shape, dtype=bool)
        change[:, 1:] = color[:, 1:] != color[:, :-1]
        cells = np.where(change, self.full_tok[color - 16, k], self.short_tok[k])
        grid = np.empty((h, w + 1), dtype=object)
        grid[:, :w] = cells
        grid[:, w] = self.row_end
        return b"".join(grid.ravel().tolist())[:-1]

    def convert_frame(self, frame) -> bytes:
        """Ridimensiona e converte un frame sorgente."""
        return self.convert(self.resize(frame))
