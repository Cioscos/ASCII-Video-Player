"""
Funzioni per convertire frame video in ASCII art colorata.
"""
import cv2

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
    return cv2.resize(frame, (width, new_height)), new_height

def frame_to_ascii(frame, width):
    """
    Converte un frame in una rappresentazione ASCII colorata.

    Parametri:
        frame (numpy.ndarray): Frame da convertire.
        width (int): Larghezza desiderata in caratteri.

    Ritorna:
        str: Rappresentazione ASCII colorata del frame.
        int: Altezza del frame ASCII.
    """
    # Ridimensiona il frame
    resized_frame, height = resize_frame(frame, width)

    # Palette di caratteri ASCII ordinata per intensità (dal più scuro al più chiaro)
    ascii_chars = "@%#*+=-:. "

    # Crea una matrice per memorizzare gli indici dei caratteri e i colori
    ascii_matrix = []

    for y in range(height):
        ascii_row = []
        for x in range(width):
            # Ottieni i valori BGR per il pixel corrente
            b, g, r = resized_frame[y, x]

            # Calcola la luminosità (0-255)
            brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b

            # Mappa la luminosità a un indice nella palette ASCII
            ascii_index = int(brightness / 255 * (len(ascii_chars) - 1))
            ascii_char = ascii_chars[ascii_index]

            # Salva carattere e colore
            ascii_row.append((ascii_char, (r, g, b)))

        ascii_matrix.append(ascii_row)

    # Genera la stringa ASCII colorata
    colored_ascii = ""
    for row in ascii_matrix:
        for char, (r, g, b) in row:
            # Sequenza ANSI per impostare il colore (formato: \033[38;2;R;G;Bm)
            colored_ascii += f"\033[38;2;{r};{g};{b}m{char}"
        colored_ascii += "\033[0m\n"  # Reset colore e nuova riga

    return colored_ascii, height

def batch_process_frames(frames_batch, width):
    """
    Processa un batch di frame convertendoli in ASCII.

    Parametri:
        frames_batch (list): Lista di frame da convertire.
        width (int): Larghezza desiderata in caratteri.

    Ritorna:
        list: Lista di frame ASCII elaborati.
    """
    ascii_frames = []
    for frame in frames_batch:
        if frame is not None:
            ascii_frame, _ = frame_to_ascii(frame, width)
            ascii_frames.append(ascii_frame)
    return ascii_frames
