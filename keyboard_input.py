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
# Virtual key code delle frecce → lettera del codice esteso (VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN)
_VK_ARROWS = {0x25: "K", 0x26: "H", 0x27: "M", 0x28: "P"}
_STD_INPUT_HANDLE = -10
_KEY_EVENT = 1
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


def records_to_chars(events) -> str:
    """Converte eventi tastiera della console Windows in caratteri grezzi per decode_keys.

    Args:
        events: sequenza di (premuto, virtual key code, carattere, ripetizioni)

    Le frecce diventano coppie estese marcate (mai confuse con la lettera "à");
    gli altri tasti senza carattere (Home, Canc, F1...) e i rilasci vengono scartati.
    """
    chars = ""
    for key_down, vk, char, repeat in events:
        if not key_down:
            continue
        if vk in _VK_ARROWS:
            chars += (_EXT_MARK + "\xe0" + _VK_ARROWS[vk]) * max(repeat, 1)
        elif char and char != "\x00":
            chars += char * max(repeat, 1)
    return chars


def _console_input():
    """Funzioni e handle della console di input Windows (ctypes), o None se non disponibile."""
    import ctypes
    from ctypes import wintypes

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]

    class _Event(ctypes.Union):
        # KEY_EVENT_RECORD è il membro più grande che ci serve; il padding copre gli altri tipi
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("_pad", ctypes.c_byte * 16)]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _Event)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # istanza propria: tipi dichiarati qui
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.FlushConsoleInputBuffer.argtypes = [wintypes.HANDLE]
    kernel32.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.ReadConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(INPUT_RECORD),
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE & 0xFFFFFFFF)
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return None  # stdin non è una console (es. pipe di un terminale emulato)
    return ctypes, wintypes, kernel32, handle, INPUT_RECORD


class KeyReader:
    """Legge tasti senza bloccare; inattivo se lo stream non e' un terminale."""

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._windows = os.name == "nt"
        self._queue: deque[str] = deque()
        self._old_attrs = None
        self._fd = None
        self._console = None
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
                try:
                    termios.tcflush(self._fd, termios.TCIFLUSH)  # scarta i tasti gia' digitati
                except Exception:
                    pass  # il terminale va comunque ripristinato in __exit__
            except Exception:
                self._old_attrs = None
                self.active = False
        elif self.active:
            try:
                self._console = _console_input()
            except Exception:
                self._console = None
            if self._console is None:
                self.active = False
            else:
                # Scarta i tasti digitati prima dell'avvio (es. durante la pre-elaborazione)
                self._console[2].FlushConsoleInputBuffer(self._console[3])
        return self

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
        # ReadConsoleInputW distingue le frecce (virtual key) dalla lettera "à": con
        # msvcrt.getwch entrambe iniziano con "à" e kbhit non vede il secondo carattere
        ctypes, wintypes, kernel32, handle, INPUT_RECORD = self._console
        pending = wintypes.DWORD()
        if not kernel32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(pending)) or not pending.value:
            return ""
        records = (INPUT_RECORD * pending.value)()
        read = wintypes.DWORD()
        if not kernel32.ReadConsoleInputW(handle, records, pending.value, ctypes.byref(read)):
            return ""
        events = []
        for record in records[:read.value]:
            if record.EventType == _KEY_EVENT:  # eventi mouse, focus e ridimensionamento ignorati
                k = record.Event.KeyEvent
                events.append((k.bKeyDown, k.wVirtualKeyCode, k.uChar, k.wRepeatCount))
        return records_to_chars(events)

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
