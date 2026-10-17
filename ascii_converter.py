"""Convertitore frame -> ASCII ANSI 256 colori, vettorizzato, con run-length del colore."""
import cv2
import numpy as np

CHAR_ASPECT_CORRECTION = 2.25
CONVERTER_VERSION = 3

_BLOCK_CHAR = '█'


def compute_ascii_height(src_w: int, src_h: int, width: int) -> int:
    """Altezza in righe ASCII per un frame sorgente src_w x src_h (minimo 1)."""
    return max(1, int(src_h / src_w * width / CHAR_ASPECT_CORRECTION))


def auto_width(src_w: int, src_h: int, term_cols: int, term_rows: int, reserved_rows: int) -> int:
    """Larghezza massima che entra nel terminale (colonne e righe) senza superare la sorgente (minimo 1)."""
    lo, hi = 1, max(1, min(term_cols, src_w))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if compute_ascii_height(src_w, src_h, mid) + reserved_rows <= term_rows:
            lo = mid
        else:
            hi = mid - 1
    return lo


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


HALF_CHAR = '▀'


def pixel_rows(height: int, palette: str) -> int:
    """Righe di pixel necessarie per `height` righe di terminale (2 per riga con 'half')."""
    return 2 * height if palette == 'half' else height


def _color256(resized):
    """Codice colore 256 (16..231) per ogni pixel di un frame BGR."""
    b = resized[:, :, 0].astype(np.int16)
    g = resized[:, :, 1].astype(np.int16)
    r = resized[:, :, 2].astype(np.int16)
    return 16 + 36 * np.minimum(5, r // 43) + 6 * np.minimum(5, g // 43) + np.minimum(5, b // 43)


class HalfBlockConverter:
    """Converte frame BGR in mezzi blocchi '▀': colore testo = pixel sopra, sfondo = pixel sotto."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.palette = 'half'
        half = HALF_CHAR.encode("utf-8")
        self.full_tok = np.empty((216, 216), dtype=object)
        for top in range(16, 232):
            for bot in range(16, 232):
                self.full_tok[top - 16, bot - 16] = f"\033[38;5;{top};48;5;{bot}m{HALF_CHAR}".encode("utf-8")
        self.short_tok = half
        self.row_end = b"\033[0m\n"

    def resize(self, frame):
        """Ridimensiona il frame a width x (2 * height) pixel (nearest neighbour)."""
        return cv2.resize(frame, (self.width, 2 * self.height), interpolation=cv2.INTER_NEAREST)

    def convert(self, resized) -> bytes:
        """Converte un frame gia' ridimensionato (2*H, W, 3 BGR); nessun '\n' finale."""
        color = _color256(resized)
        top = color[0::2]
        bot = color[1::2]
        h, w = top.shape
        change = np.ones(top.shape, dtype=bool)
        change[:, 1:] = (top[:, 1:] != top[:, :-1]) | (bot[:, 1:] != bot[:, :-1])
        cells = np.where(change, self.full_tok[top - 16, bot - 16], self.short_tok)
        grid = np.empty((h, w + 1), dtype=object)
        grid[:, :w] = cells
        grid[:, w] = self.row_end
        return b"".join(grid.ravel().tolist())[:-1]

    def convert_frame(self, frame) -> bytes:
        """Ridimensiona e converte un frame sorgente."""
        return self.convert(self.resize(frame))


_NUM_TOK = np.array([str(v).encode() for v in range(256)], dtype=object)


def _rgb_key(resized):
    """Chiave intera univoca per il colore RGB esatto di ogni pixel (confronto sui 3 canali)."""
    b = resized[:, :, 0].astype(np.int32)
    g = resized[:, :, 1].astype(np.int32)
    r = resized[:, :, 2].astype(np.int32)
    return (r << 16) | (g << 8) | b


def _join_grid(cells, row_end):
    """Unisce la griglia di celle (bytes) in un unico frame, senza '\\n' finale."""
    h, w = cells.shape
    grid = np.empty((h, w + 1), dtype=object)
    grid[:, :w] = cells
    grid[:, w] = row_end
    return b"".join(grid.ravel().tolist())[:-1]


class TrueColorConverter:
    """Come AsciiConverter ma con colore 24 bit (ESC[38;2;R;G;Bm), emesso solo al cambio."""

    def __init__(self, width: int, height: int, palette: str):
        self.width = width
        self.height = height
        self.palette = palette
        chars = [_BLOCK_CHAR] if palette == 'box' else list(palette)
        n = len(chars)
        if n == 0:
            raise ValueError("La palette non puo' essere vuota")
        self.char_lut = np.array([int(g / 255.0 * (n - 1)) for g in range(256)], dtype=np.intp)
        self.char_tok = np.empty(n, dtype=object)
        for k, ch in enumerate(chars):
            self.char_tok[k] = ch.encode("utf-8")
        self.row_end = b"\033[0m\n"

    def resize(self, frame):
        """Ridimensiona il frame alla griglia ASCII (nearest neighbour)."""
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    def convert(self, resized) -> bytes:
        """Converte un frame gia' ridimensionato (H, W, 3 BGR); nessun '\\n' finale."""
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        k = self.char_lut[gray]
        key = _rgb_key(resized)
        change = np.ones(key.shape, dtype=bool)
        change[:, 1:] = key[:, 1:] != key[:, :-1]
        cells = self.char_tok[k]
        idx = np.flatnonzero(change.ravel())
        flat = resized.reshape(-1, 3)[idx]
        full = (b"\033[38;2;" + _NUM_TOK[flat[:, 2]] + b";" + _NUM_TOK[flat[:, 1]] + b";"
                + _NUM_TOK[flat[:, 0]] + b"m" + cells.ravel()[idx])
        cells.ravel()[idx] = full
        return _join_grid(cells, self.row_end)

    def convert_frame(self, frame) -> bytes:
        """Ridimensiona e converte un frame sorgente."""
        return self.convert(self.resize(frame))


class TrueColorHalfConverter:
    """Mezzi blocchi '▀' a colore 24 bit: testo = pixel sopra, sfondo = pixel sotto."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.palette = 'half'
        self.half_tok = HALF_CHAR.encode("utf-8")
        self.row_end = b"\033[0m\n"

    def resize(self, frame):
        """Ridimensiona il frame a width x (2 * height) pixel (nearest neighbour)."""
        return cv2.resize(frame, (self.width, 2 * self.height), interpolation=cv2.INTER_NEAREST)

    def convert(self, resized) -> bytes:
        """Converte un frame gia' ridimensionato (2*H, W, 3 BGR); nessun '\\n' finale."""
        key = _rgb_key(resized)
        top_key = key[0::2]
        bot_key = key[1::2]
        h, w = top_key.shape
        change = np.ones(top_key.shape, dtype=bool)
        change[:, 1:] = (top_key[:, 1:] != top_key[:, :-1]) | (bot_key[:, 1:] != bot_key[:, :-1])
        cells = np.full((h, w), self.half_tok, dtype=object)
        idx = np.flatnonzero(change.ravel())
        top = resized[0::2].reshape(-1, 3)[idx]
        bot = resized[1::2].reshape(-1, 3)[idx]
        n = _NUM_TOK
        full = (b"\033[38;2;" + n[top[:, 2]] + b";" + n[top[:, 1]] + b";" + n[top[:, 0]]
                + b";48;2;" + n[bot[:, 2]] + b";" + n[bot[:, 1]] + b";" + n[bot[:, 0]]
                + b"m" + self.half_tok)
        cells.ravel()[idx] = full
        return _join_grid(cells, self.row_end)

    def convert_frame(self, frame) -> bytes:
        """Ridimensiona e converte un frame sorgente."""
        return self.convert(self.resize(frame))


def make_converter(width: int, height: int, palette: str, truecolor: bool = False):
    """Crea il converter adatto: 'half' -> mezzi blocchi, altrimenti caratteri/'box'; truecolor a 24 bit."""
    if truecolor:
        if palette == 'half':
            return TrueColorHalfConverter(width, height)
        return TrueColorConverter(width, height, palette)
    if palette == 'half':
        return HalfBlockConverter(width, height)
    return AsciiConverter(width, height, palette)
