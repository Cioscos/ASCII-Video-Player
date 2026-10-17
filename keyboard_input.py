"""Lettore di tasti non bloccante per il terminale (Windows e POSIX)."""
import os
import sys
import time
from collections import deque

_WIN_PREFIXES = ("\x00", "\xe0")
# Marcatore privato inserito da _read_raw_windows davanti a una coppia prefisso+codice
# arrivata insieme: indica con certezza un tasto esteso (non la lettera "à")
_EXT_MARK = "￾"
_WIN_ARROWS = {"H": "up", "P": "down", "K": "left", "M": "right"}
_POSIX_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}
_SIMPLE = {
    "\r": "enter",
    "\n": "enter",
    "\x1b": "esc",
    "\x08": "backspace",
    "\x7f": "backspace",
    " ": "space",
}


def decode_keys(chars: str, windows: bool) -> list[str]:
    """Decodifica caratteri grezzi in nomi di tasti (funzione pura).

    Le sequenze sconosciute dopo un prefisso (tasti funzione, ecc.) vengono scartate.
    """
    keys: list[str] = []
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if windows and c == _EXT_MARK:
            # Tasto esteso certo: frecce mappate, gli altri codici (Home, Canc, F11...) scartati
            code = chars[i + 2] if i + 2 < n else ""
            if code in _WIN_ARROWS:
                keys.append(_WIN_ARROWS[code])
            i += 3
        elif windows and c in _WIN_PREFIXES:
            if i + 1 < n and chars[i + 1] in _WIN_ARROWS:
                keys.append(_WIN_ARROWS[chars[i + 1]])
                i += 2
            elif c == "\x00":
                i += 2  # tasto speciale sconosciuto: scarta prefisso e codice
            else:
                keys.append(c)  # \xe0 da solo e' la lettera "\u00e0"
                i += 1
        elif not windows and c == "\x1b" and i + 1 < n and chars[i + 1] in "[O":
            # Sequenza CSI/SS3: parametri, poi un byte finale (0x40-0x7e)
            j = i + 2
            while j < n and not "@" <= chars[j] <= "~":
                j += 1
            if j < n and chars[j] in _POSIX_ARROWS:
                keys.append(_POSIX_ARROWS[chars[j]])  # anche con parametri, es. \x1b[1;5A
            i = j + 1
        else:
            keys.append(_SIMPLE.get(c, c))
            i += 1
    return keys


class KeyReader:
    """Legge tasti senza bloccare; inattivo se lo stream non e' un terminale."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._windows = os.name == "nt"
        self._queue: deque[str] = deque()
        self._old_attrs = None
        self._fd = None
        self.active = False

    def __enter__(self):
        isatty = getattr(self._stream, "isatty", None)
        try:
            self.active = bool(isatty()) if isatty else False
        except Exception:
            self.active = False
        if self.active and not self._windows:
            try:
                import termios
                import tty
                self._fd = self._stream.fileno()
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                termios.tcflush(self._fd, termios.TCIFLUSH)  # scarta i tasti gia' digitati
            except Exception:
                self._old_attrs = None
                self.active = False
        elif self.active:
            self._drain_windows()
        return self

    def _drain_windows(self) -> None:
        """Scarta i tasti digitati prima dell'avvio (es. durante la pre-elaborazione)."""
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        if self._old_attrs is not None:
            import termios
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            finally:
                self._old_attrs = None
        self.active = False
        return False

    def _read_raw_windows(self) -> str:
        import msvcrt
        chars = ""
        while msvcrt.kbhit():
            c = msvcrt.getwch()
            chars += c
            if c in _WIN_PREFIXES and msvcrt.kbhit():
                # Seconda parte arrivata subito: e' un tasto esteso, non la lettera "à"
                chars = chars[:-1] + _EXT_MARK + c + msvcrt.getwch()
        return chars

    def _read_raw_posix(self, timeout: float) -> str:
        import select
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return ""
        return os.read(self._fd, 64).decode("utf-8", errors="ignore")

    def _fill(self, timeout: float) -> None:
        if self._windows:
            raw = self._read_raw_windows()
        else:
            raw = self._read_raw_posix(timeout)
        self._queue.extend(decode_keys(raw, self._windows))

    def poll(self) -> "str | None":
        """Restituisce il prossimo tasto disponibile, o None subito."""
        if not self.active:
            return None
        if not self._queue:
            self._fill(0)
        return self._queue.popleft() if self._queue else None

    def read(self, timeout: float) -> "str | None":
        """Attende un tasto fino a `timeout` secondi; None se scade."""
        if not self.active:
            return None
        deadline = time.monotonic() + timeout
        while True:
            if self._queue:
                return self._queue.popleft()
            remaining = deadline - time.monotonic()
            if self._windows:
                self._fill(0)
                if self._queue:
                    continue
                if remaining <= 0:
                    return None
                time.sleep(min(0.01, remaining))
            else:
                self._fill(max(remaining, 0))
                if not self._queue and remaining <= 0:
                    return None
