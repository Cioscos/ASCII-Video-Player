"""
Funzioni per la gestione del terminale e la visualizzazione di frame ASCII.
"""
import sys

# CONSTANTS
HIDE_CURSOR = "\033[?25l"
CLEAR_SCREEN = "\033[2J\033[H"
SHOW_CURSOR = "\033[?25h"

def generate_calibration_frame(width, height):
    """
    Genera un frame ASCII di calibrazione tutto bianco, con un bordo e una croce centrale.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.

    Ritorna:
        str: Stringa ASCII con il frame bianco, bordi e croce centrale.
    """
    # Caratteri
    BORDER_CHAR = "#"
    CROSS_CHAR = "+"
    WHITE_CHAR = "█"  # Blocchi pieni per simulare un frame bianco

    # Crea una matrice di caratteri bianchi
    ascii_frame = [[WHITE_CHAR] * width for _ in range(height)]

    # Disegna i bordi del rettangolo
    for x in range(width):
        ascii_frame[0][x] = BORDER_CHAR  # Riga superiore
        ascii_frame[-1][x] = BORDER_CHAR  # Riga inferiore

    for y in range(height):
        ascii_frame[y][0] = BORDER_CHAR  # Colonna sinistra
        ascii_frame[y][-1] = BORDER_CHAR  # Colonna destra

    # Disegna la croce centrale
    center_y, center_x = height // 2, width // 2
    for x in range(width):
        ascii_frame[center_y][x] = CROSS_CHAR  # Linea orizzontale
    for y in range(height):
        ascii_frame[y][center_x] = CROSS_CHAR  # Linea verticale

    # Converti la matrice in una stringa
    ascii_string = "\n".join("".join(row) for row in ascii_frame)
    return ascii_string


def render_calibration_frame(width, height):
    """
    Mostra un frame di calibrazione nel terminale e attende che l'utente prema ENTER.
    Dopo l'input, il terminale viene svuotato e il buffer di scorrimento viene rimosso.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.
    """
    # Genera il frame di calibrazione
    calibration_frame = generate_calibration_frame(width, height)

    # Escape sequences per il terminale
    RESET_TERMINAL = "\033c"  # Reset completo del terminale

    # Pulisce lo schermo e nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(calibration_frame + "\n")
    sys.stdout.write("\n[INFO] Regola la dimensione del terminale e premi ENTER per iniziare...\n")
    sys.stdout.flush()

    # Attendi l'input dell'utente
    input()

    # Resetta completamente il terminale (rimuove lo scrollback buffer)
    sys.stdout.write(RESET_TERMINAL)
    sys.stdout.flush()

    # Ripristina il cursore
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def clear_terminal():
    """
    Pulisce lo schermo del terminale.

    Utilizza sequenze di escape ANSI per pulire il terminale e posizionare
    il cursore nell'angolo superiore sinistro.
    """
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.flush()


def hide_cursor():
    """
    Nasconde il cursore del terminale.

    Utilizza sequenze di escape ANSI per nascondere il cursore del terminale.
    """
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()


def show_cursor():
    """
    Mostra il cursore del terminale.

    Utilizza sequenze di escape ANSI per mostrare il cursore del terminale.
    """
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def print_frame(frame_str):
    """
    Stampa un frame ASCII nel terminale, aggiornando solo le parti modificate
    con un algoritmo ottimizzato per sequenze di caratteri.

    Parametri:
        frame_str (str): Stringa ASCII da visualizzare.
        previous_frame (str): Frame precedente per confronto.

    Ritorna:
        str: Il frame corrente per uso futuro.
    """
    sys.stdout.write("\033[H")

    # Scrive l'intero frame in un'unica operazione
    sys.stdout.write(frame_str)
    sys.stdout.flush()

    return frame_str
