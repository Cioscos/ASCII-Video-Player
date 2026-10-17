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
