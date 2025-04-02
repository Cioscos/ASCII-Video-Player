"""
Funzioni ottimizzate per convertire frame video in ASCII art colorata.
"""
import cv2
import numpy as np

# Pre-calcola la palette ASCII
ASCII_CHARS = "@%#*+=-:. "
ASCII_CHARS_LEN = len(ASCII_CHARS)
ASCII_INDICES = np.arange(0, 256)  # Mappatura da luminosità (0-255) a indice ASCII

def resize_frame(frame, width):
    """
    Ridimensiona un frame mantenendo l'aspect ratio.

    Parametri:
        frame (numpy.ndarray): Frame da ridimensionare.
        width (int): Larghezza desiderata in caratteri.

    Ritorna:
        numpy.ndarray: Frame ridimensionato.
        int: Altezza calcolata basata sull'aspect ratio.
    """
    height, original_width, _ = frame.shape
    aspect_ratio = height / original_width
    # Il terminale ha un rapporto altezza/larghezza carattere di circa 2:1,
    # quindi moltiplichiamo per 0.5 per ottenere un aspect ratio corretto
    new_height = int(width * aspect_ratio * 0.5)

    # Usa INTER_AREA per downscaling (più veloce e migliore per ridurre dimensioni)
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA), new_height


def frame_to_ascii(frame, width, color_cache=None):
    """
    Converte un frame in una rappresentazione ASCII colorata con cache dei colori.
    """
    # Inizializza la cache se non esiste
    if color_cache is None:
        color_cache = {}

    # Ridimensiona il frame
    resized_frame, height = resize_frame(frame, width)

    # Pre-allocazione di memoria per le stringhe di output
    rows = []
    last_color = None

    # Calcola la luminosità per l'intero frame in una volta sola
    luminosity = np.dot(resized_frame[..., :3], [0.0722, 0.7152, 0.2126])

    # Mappa la luminosità agli indici della palette ASCII
    ascii_indices = (luminosity / 255 * (ASCII_CHARS_LEN - 1)).astype(int)

    # Costruisci ciascuna riga in modo ottimizzato
    for y in range(height):
        row_chars = []
        for x in range(width):
            # Ottieni i valori BGR per il pixel corrente
            b, g, r = resized_frame[y, x]

            # Quantizza i colori (riduci la profondità di colore)
            r = (r // 5) * 5
            g = (g // 5) * 5
            b = (b // 5) * 5

            # Ottieni il carattere ASCII corrispondente alla luminosità
            char_idx = ascii_indices[y, x]
            char = ASCII_CHARS[char_idx]

            # Ottimizzazione: cambia il colore solo se diverso dall'ultimo usato
            color = (r, g, b)
            if color != last_color:
                # Usa la cache per le sequenze di colore
                if color not in color_cache:
                    color_cache[color] = f"\033[38;2;{r};{g};{b}m"

                row_chars.append(color_cache[color] + char)
                last_color = color
            else:
                row_chars.append(char)

        # Unisci tutti i caratteri della riga in una sola stringa
        rows.append("".join(row_chars))

    # Unisci tutte le righe con newline e reset del colore alla fine di ogni riga
    return "\n".join(r + "\033[0m" for r in rows), height

def batch_process_frames(frames_batch, width):
    """
    Processa un batch di frame convertendoli in ASCII.

    Parametri:
        frames_batch (list): Lista di frame da convertire.
        width (int): Larghezza desiderata in caratteri.

    Ritorna:
        list: Lista di frame ASCII elaborati.
    """
    # Pre-allocazione dell'array di output
    ascii_frames = []

    # Controllo di sicurezza per evitare operazioni inutili
    if not frames_batch:
        return ascii_frames

    # Ottimizzazione: elabora i frame in batch
    for frame in frames_batch:
        if frame is not None:
            ascii_frame, _ = frame_to_ascii(frame, width)
            ascii_frames.append(ascii_frame)

    return ascii_frames
