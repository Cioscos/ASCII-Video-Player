"""
Funzioni ottimizzate per la gestione del terminale e la visualizzazione di frame ASCII.
"""
import sys
import time

# CONSTANTS
HIDE_CURSOR = "\033[?25l"
CLEAR_SCREEN = "\033[2J\033[H"
SHOW_CURSOR = "\033[?25h"
RESET_TERMINAL = "\033c"  # Reset completo del terminale
MOVE_TO_HOME = "\033[H"   # Sposta il cursore in alto a sinistra

# Cache per l'ultimo frame visualizzato
last_frame_cache = None
last_frame_time = 0

def generate_calibration_frame(width, height):
    """
    Genera un frame ASCII di calibrazione tutto bianco, con un bordo e una croce centrale.

    Args:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.

    Returns:
        str: Stringa ASCII con il frame bianco, bordi e croce centrale.
    """
    # Caratteri
    BORDER_CHAR = "#"
    CROSS_CHAR = "+"
    WHITE_CHAR = "█"  # Blocchi pieni per simulare un frame bianco

    # Crea una matrice di caratteri bianchi
    # Ottimizzazione: pre-allocazione delle stringhe di riga
    rows = [BORDER_CHAR * width]

    # Riga superiore (bordo)

    # Righe intermedie
    for y in range(1, height-1):
        if y == height // 2:
            # Riga centrale con croce
            row = BORDER_CHAR + CROSS_CHAR * (width-2) + BORDER_CHAR
        else:
            # Riga normale con bordi
            row = BORDER_CHAR + WHITE_CHAR * (width-2) + BORDER_CHAR
        rows.append(row)

    # Riga inferiore (bordo)
    if height > 1:
        rows.append(BORDER_CHAR * width)

    # Sovrascrivere la croce verticale
    center_x = width // 2
    for y in range(1, height-1):
        if y != height // 2:  # Salta la riga centrale (già con croce)
            row_list = list(rows[y])
            row_list[center_x] = CROSS_CHAR
            rows[y] = "".join(row_list)

    return "\n".join(rows)

def render_calibration_frame(width, height):
    """
    Mostra un frame di calibrazione nel terminale e attende che l'utente prema ENTER.
    Dopo l'input, il terminale viene svuotato e il buffer di scorrimento viene rimosso.

    Args:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.
    """
    # Genera il frame di calibrazione
    calibration_frame = generate_calibration_frame(width, height)

    # Pulisce lo schermo e nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(calibration_frame + "\n")
    sys.stdout.write("\n[INFO] Regola la dimensione del terminale e premi ENTER per iniziare...\n")
    sys.stdout.flush()

    # Attendi l'input dell'utente
    try:
        input()
    except KeyboardInterrupt:
        # Gestisci CTRL+C durante la calibrazione
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write("\n\nCalibrazione interrotta.\n")
        sys.stdout.flush()
        sys.exit(1)

    # Resetta completamente il terminale (rimuove lo scrollback buffer)
    sys.stdout.write(RESET_TERMINAL)
    sys.stdout.flush()

    # Ripristina il cursore (sarà poi nascosto di nuovo all'avvio della pipeline)
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
    Stampa un frame ASCII nel terminale, con throttling intelligente per alte performance.

    Args:
        frame_str (str): Stringa ASCII da visualizzare.

    Returns:
        str: Il frame corrente per uso futuro.
    """
    global last_frame_cache, last_frame_time

    current_time = time.time()

    # Throttling intelligente - limita il frame rate se stiamo inviando
    # troppi frame al terminale (può causare flickering o rallentamenti)
    min_frame_interval = 1.0 / 60.0  # Max 60 FPS al terminale

    if current_time - last_frame_time < min_frame_interval:
        # Troppe chiamate ravvicinate, salta questo frame per mantenere performance
        return frame_str

    # Aggiorna il timestamp dell'ultimo frame
    last_frame_time = current_time

    # Ottimizzazione: muovi il cursore in alto a sinistra invece di pulire lo schermo
    sys.stdout.write(MOVE_TO_HOME)

    # Scrive l'intero frame in un'unica operazione
    sys.stdout.write(frame_str)
    sys.stdout.flush()

    # Aggiorna la cache per il prossimo confronto
    last_frame_cache = frame_str

    return frame_str
