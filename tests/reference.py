"""Riferimento di caratterizzazione: griglia (carattere, colore) attesa per un frame."""
import cv2
import numpy as np

CHAR_ASPECT = 2.25


def reference_grid(frame_bgr, width, palette):
    """Ritorna la griglia di celle (carattere, codice colore 256) per il frame."""
    src_h, src_w = frame_bgr.shape[:2]
    height = max(1, int(src_h / src_w * width / CHAR_ASPECT))
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            b, g, r = (int(v) for v in resized[y, x])
            color = 16 + 36 * min(5, r // 43) + 6 * min(5, g // 43) + min(5, b // 43)
            if palette == 'box':
                char = '█'
            else:
                char = palette[int(gray[y, x] / 255.0 * (len(palette) - 1))]
            row.append((char, color))
        grid.append(row)
    return grid


def parse_ansi_grid(data):
    """Decodifica l'output ANSI in una griglia di celle (carattere, colore corrente)."""
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    grid = []
    for line in data.split('\n'):
        row = []
        color = None
        i = 0
        while i < len(line):
            if line.startswith('\033[0m', i):
                i += 4
            elif line.startswith('\033[38;5;', i):
                end = line.index('m', i)
                color = int(line[i + 7:end])
                i = end + 1
            else:
                row.append((line[i], color))
                i += 1
        grid.append(row)
    return grid


def _color256(r, g, b):
    return 16 + 36 * min(5, r // 43) + 6 * min(5, g // 43) + min(5, b // 43)


def reference_cells(frame_bgr, width, palette, truecolor=False):
    """Griglia di celle (carattere, colore testo, colore sfondo); sfondo None se non 'half'."""
    src_h, src_w = frame_bgr.shape[:2]
    height = max(1, int(src_h / src_w * width / CHAR_ASPECT))

    def col(px):
        b, g, r = (int(v) for v in px)
        return (r, g, b) if truecolor else _color256(r, g, b)

    if palette == 'half':
        resized = cv2.resize(frame_bgr, (width, 2 * height), interpolation=cv2.INTER_NEAREST)
        return [[('▀', col(resized[2 * y, x]), col(resized[2 * y + 1, x])) for x in range(width)]
                for y in range(height)]
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if palette == 'box':
                char = '█'
            else:
                char = palette[int(gray[y, x] / 255.0 * (len(palette) - 1))]
            row.append((char, col(resized[y, x]), None))
        grid.append(row)
    return grid


def parse_ansi_cells(data):
    """Decodifica l'output ANSI in celle (carattere, colore testo, colore sfondo)."""
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    grid = []
    for line in data.split('\n'):
        row = []
        fg = bg = None
        i = 0
        while i < len(line):
            if line[i] == '\033' and line[i + 1] == '[':
                end = line.index('m', i)
                params = line[i + 2:end].split(';')
                j = 0
                while j < len(params):
                    p = params[j]
                    if p in ('0', ''):
                        fg = bg = None
                        j += 1
                    elif p in ('38', '48'):
                        if params[j + 1] == '5':
                            value, j = int(params[j + 2]), j + 3
                        else:
                            value, j = tuple(int(v) for v in params[j + 2:j + 5]), j + 5
                        if p == '38':
                            fg = value
                        else:
                            bg = value
                    else:
                        raise ValueError(f"Parametro ANSI non supportato: {p}")
                i = end + 1
            else:
                row.append((line[i], fg, bg))
                i += 1
        grid.append(row)
    return grid
