"""Processo figlio di test: inietta tasti veri nella propria console e li legge con KeyReader."""
import ctypes
import json
import sys
from ctypes import wintypes

sys.path.insert(0, sys.argv[2])
from keyboard_input import KeyReader  # noqa: E402

NUL = chr(0)


class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                ("uChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [("EventType", wintypes.WORD), ("Event", KEY_EVENT_RECORD)]


def inject(handle, keys):
    """Scrive pressione e rilascio di ogni (vk, scan, carattere) nel buffer della console."""
    records = (INPUT_RECORD * (2 * len(keys)))()
    for i, (vk, scan, ch) in enumerate(keys):
        for j, down in enumerate((1, 0)):
            records[2 * i + j].EventType = 1  # KEY_EVENT
            records[2 * i + j].Event = KEY_EVENT_RECORD(down, 1, vk, scan, ch, 0x100 if ch == NUL else 0)
    written = wintypes.DWORD()
    ctypes.windll.kernel32.WriteConsoleInputW(handle, records, len(records), ctypes.byref(written))


handle = ctypes.windll.kernel32.GetStdHandle(-10)
keys = []
inject(handle, [(0x58, 0x2D, "x")])  # premuto prima dell'avvio: va scartato
with KeyReader() as reader:
    # Un tasto alla volta, come premuto da una persona; più "à" seguita subito da "q"
    for batch in ([(0x28, 0x50, NUL)],      # freccia giù
                  [(0x25, 0x4B, NUL)],      # freccia sinistra
                  [(0x2E, 0x53, NUL)],      # Canc (scartato)
                  [(0, 0, "à")],
                  [(0, 0, "à"), (0x51, 0x10, "q")],
                  [(0x0D, 0x1C, "\r")]):
        inject(handle, batch)
        while (key := reader.read(0.2)) is not None:
            keys.append(key)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"keys": keys}, f)
