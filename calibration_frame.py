import logging
import sys

from utils import get_terminal_size

# Sequenze di escape ANSI per il controllo del terminale
HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'
CLEAR_SCREEN = '\033[2J\033[H'
RESET_TERMINAL = '\033[!p\033[2J\033[H'  # Reset completo del terminale

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

    # Assicura che le dimensioni siano almeno 3x3
    width = max(3, width)
    height = max(3, height)

    # Crea una matrice di caratteri bianchi
    # Ottimizzazione: pre-allocazione delle stringhe di riga
    rows = [BORDER_CHAR * width]

    # Riga superiore (bordo)

    # Righe intermedie
    for y in range(1, height - 1):
        if y == height // 2:
            # Riga centrale con croce
            row = BORDER_CHAR + CROSS_CHAR * (width - 2) + BORDER_CHAR
        else:
            # Riga normale con bordi
            row = BORDER_CHAR + WHITE_CHAR * (width - 2) + BORDER_CHAR
        rows.append(row)

    # Riga inferiore (bordo)
    if height > 1:
        rows.append(BORDER_CHAR * width)

    # Sovrascrivere la croce verticale
    center_x = width // 2
    for y in range(1, height - 1):
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
    # Log delle dimensioni effettive del frame di calibrazione
    logging.info(f"Generazione frame di calibrazione: {width}x{height}")

    # Ottieni dimensioni terminale
    term_width, term_height = get_terminal_size()

    # Genera il frame di calibrazione
    calibration_frame = generate_calibration_frame(width, height)

    # Conta e log delle dimensioni effettive
    lines = calibration_frame.split('\n')
    actual_height = len(lines)
    actual_width = max(len(line) for line in lines)
    logging.info(f"Dimensioni effettive frame di calibrazione: {actual_width}x{actual_height}")

    # Pulisce lo schermo e nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(calibration_frame + "\n")
    sys.stdout.write("\n[INFO] Dimensioni frame: " + str(width) + "x" + str(height))
    sys.stdout.write("\n[INFO] Dimensioni terminale attuale: " + str(term_width) + "x" + str(term_height))
    sys.stdout.write(
        "\n\n[CALIBRAZIONE] Regola la dimensione dei caratteri del terminale finche' il frame entra completamente")
    sys.stdout.write("\n[CALIBRAZIONE] Dovresti vedere un rettangolo completo con una croce al centro")
    sys.stdout.write("\n[CALIBRAZIONE] Premi ENTER quando il frame e' visualizzato correttamente")
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
