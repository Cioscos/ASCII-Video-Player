import argparse
import logging
import signal
import sys
import time
import os
import cv2

from converter import FrameConverter
from renderer import AsciiRenderer
from pipeline import VideoPipeline
from utils import setup_logging, get_terminal_size

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


def main():
    """
    Funzione principale che inizializza e gestisce l'applicazione ASCII video.
    Gestisce il parsing degli argomenti, la configurazione del logging e
    l'avvio della pipeline di elaborazione video.

    Returns:
        int: Codice di uscita del programma (0 per successo, 1 per errore)
    """
    # Parsing degli argomenti della riga di comando
    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using a parallel pipeline with separate conversion and rendering."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, default=100, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--verbose", action="store_true", help="Show all log messages in the terminal")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--palette", type=str, choices=["basic", "standard", "extended"], default="standard",
                        help="ASCII character palette to use: basic (10 chars), standard (42 chars), extended (70 chars)")
    # Nuovo parametro per controllare il loop del video
    parser.add_argument("--no-loop", action="store_true", help="Disable video looping (stop when video ends)")

    args = parser.parse_args()

    # Verifica che il file video esista
    if not os.path.isfile(args.video_path):
        print(f"Errore: Il file video '{args.video_path}' non esiste.")
        return 1

    # Configura il sistema di logging
    try:
        import io
        import locale
        # Imposta la codifica UTF-8 per evitare problemi con caratteri speciali
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

        # Usa il livello INFO per il terminale se verbose è attivo, altrimenti WARNING
        console_level = logging.INFO if args.verbose else logging.WARNING
        logger = setup_logging(args.log_fps, args.log_performance, console_level=console_level)
        logger.info(f"Codifica terminale: {locale.getpreferredencoding()}")

    except Exception as e:
        # In caso di errore nella configurazione della codifica, usa una configurazione più semplice
        logger = setup_logging(args.log_fps, args.log_performance)
        logger.warning(f"Impossibile configurare la codifica UTF-8: {e}")

    # Apertura video per ottenere le dimensioni iniziali
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logger.error(f"Errore: Impossibile aprire il file video '{args.video_path}'.")
        return 1

    ret, frame = cap.read()

    # Ottieni informazioni sul video
    fps_originale = cap.get(cv2.CAP_PROP_FPS)
    frame_totali = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Video originale: {args.video_path}, {frame.shape[1]}x{frame.shape[0]}, {fps_originale} FPS, {frame_totali} frames")

    cap.release()
    if not ret:
        logger.error("Errore: impossibile leggere il primo frame.")
        return 1

    height, width_frame = frame.shape[:2]
    aspect_ratio = height / width_frame

    # Ottieni le dimensioni del terminale
    term_width, term_height = get_terminal_size()
    logger.info(f"Dimensioni terminale: {term_width}x{term_height}")

    # Calcola la larghezza dell'output ASCII
    # Usa il valore specificato dall'utente senza limitazioni
    term_width, term_height = get_terminal_size()
    logger.info(f"Dimensioni terminale: {term_width}x{term_height}")

    # Usiamo il valore specificato dall'utente senza avvisi
    # Il frame di calibrazione serve proprio per regolare la dimensione dei caratteri
    width = args.width
    logger.info(f"Utilizzando larghezza ASCII specificata: {width}")

    # Calcola l'altezza proporzionale mantenendo le proporzioni originali del video
    # I caratteri ASCII non sono quadrati, ma più alti che larghi.
    # Usiamo un fattore di correzione di circa 2.0-2.5 per compensare
    height, width_frame = frame.shape[:2]
    aspect_ratio = height / width_frame
    char_aspect_correction = 2.25  # Fattore di correzione per i caratteri ASCII
    new_height = int(aspect_ratio * width / char_aspect_correction)

    logger.info(
        f"Dimensioni output ASCII: {width}x{new_height} (aspect ratio video: {aspect_ratio:.2f}, corretta: {aspect_ratio / char_aspect_correction:.2f})")

    # Mostra il frame di calibrazione
    try:
        render_calibration_frame(width, new_height)
        logger.info("Calibrazione completata con successo")
    except Exception as e:
        logger.error(f"Errore durante la calibrazione: {e}")

    # Nascondi il cursore prima di avviare la pipeline
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    # Ottieni la palette di caratteri ASCII in base alla scelta dell'utente
    palette_map = {
        "basic": "@%#*+=-:. ",
        "standard": "@%#*+=-:. WM#oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[] ",
        "extended": "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~i!lI;:,\"^`'. "
    }
    ascii_palette = palette_map[args.palette]
    logger.info(f"Utilizzando palette con {len(ascii_palette)} caratteri: {ascii_palette[:10]}...")

    # Inizializza la pipeline
    pipeline = VideoPipeline(
        args.video_path,
        width,
        args.fps,
        args.batch_size,
        args.log_fps,
        ascii_palette=ascii_palette,
        loop_video=not args.no_loop  # Utilizziamo il nuovo parametro
    )

    # Gestione dei segnali per una chiusura pulita
    def signal_handler(sig, frame):
        logger.info("Segnale di interruzione ricevuto")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Avvia la pipeline
        pipeline.start()

        # Mostra istruzioni all'utente
        print("\nASCII Video Player avviato!")
        print("Premi Ctrl+C per terminare\n")
        if not args.no_loop:
            print("Il video si ripeterà automaticamente alla fine")
        else:
            print("Il video terminerà automaticamente alla fine")

        # Attendi che l'utente interrompa l'esecuzione o che il video finisca
        while not pipeline.should_stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera")
    finally:
        # Ferma la pipeline
        pipeline.stop()

        # Mostra il cursore alla fine dell'esecuzione
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
