import argparse
import logging
import signal
import sys
import time
import os
import cv2

from calibration_frame import render_calibration_frame
from pipeline import VideoPipeline
from utils import setup_logging, get_terminal_size

HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'


def main():
    """
    Funzione principale dell'applicazione ASCII video.

    Gestisce il parsing degli argomenti da riga di comando, la configurazione del logging,
    l'avvio della pipeline di elaborazione video e la gestione del ciclo di vita
    dell'applicazione.

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
    parser.add_argument("--log_performance", action="store_true",
                        help="Enable logging of conversion and rendering performance")
    parser.add_argument("--verbose", action="store_true", help="Show all log messages in the terminal")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--palette", type=str, choices=["basic", "standard", "extended", "box", "custom"],
                        default="standard",
                        help="ASCII character palette to use: basic (10 chars), standard (42 chars), extended (70 chars)")
    parser.add_argument("--custom-palette", type=str,
                        help="Path to a custom palette file. It can be a normal .txt file.")
    parser.add_argument("--no-loop", action="store_true", help="Disable video looping (stop when video ends)")
    parser.add_argument("--audio", action="store_true", help="Enable audio playback")

    args = parser.parse_args()

    # Verifica che il file video esista
    if not os.path.isfile(args.video_path):
        print(f"Errore: Il file video '{args.video_path}' non esiste.")
        return 1

    # Verifica delle dipendenze per l'audio
    if args.audio:
        try:
            import sounddevice
            from moviepy import AudioFileClip
            print("Dipendenze audio verificate con successo.")
        except ImportError as e:
            print(f"Errore: impossibile abilitare l'audio. Mancano le dipendenze necessarie: {e}")
            print("Per abilitare l'audio, installa: pip install sounddevice moviepy")
            print("Continuo senza audio...")
            args.audio = False

    # Configura il sistema di logging
    try:
        import io
        import locale
        # Imposta la codifica UTF-8 per evitare problemi con caratteri speciali
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

        # Livello INFO per terminale se verbose è attivo, altrimenti WARNING
        console_level = logging.INFO if args.verbose else logging.WARNING
        logger = setup_logging(args.log_fps, args.log_performance, console_level=console_level)

        logger.info(f"Codifica terminale: {locale.getpreferredencoding()}")

        # Controlla se il terminale supporta UTF-8
        if 'utf-8' not in locale.getpreferredencoding().lower():
            logger.warning(
                "Terminale non supporta UTF-8. Alcuni caratteri potrebbero non essere visualizzati correttamente.")

    except Exception as e:
        # Configurazione più semplice in caso di errore
        console_level = logging.INFO if args.verbose else logging.WARNING
        logger = setup_logging(args.log_fps, args.log_performance, console_level=console_level)
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

    # Suggerimento FPS per audio
    if args.audio and args.fps is None:
        target_fps = min(30, int(fps_originale))  # Limita a max 30 FPS
        logger.info(
            f"Audio abilitato senza FPS target specificati. Usando {target_fps} FPS per una migliore sincronizzazione.")
        args.fps = target_fps

    # Ottieni le dimensioni del terminale
    term_width, term_height = get_terminal_size()
    logger.info(f"Dimensioni terminale: {term_width}x{term_height}")

    # Usa larghezza specificata dall'utente
    width = args.width
    logger.info(f"Utilizzando larghezza ASCII specificata: {width}")

    # Calcola l'altezza proporzionale
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

    # Controllo palette personalizzata
    if args.custom_palette and args.palette != 'custom':
        logger.warning(
            f"La custom palette non sara' utilizzata perché è stata utilizzata la palette {args.palette} e non 'custom'")

    # Seleziona la palette ASCII
    if args.palette != 'box':
        if args.palette == 'custom':
            with open(args.custom_palette, 'r') as f:
                ascii_palette = f.read().strip()
        else:
            palette_map = {
                "basic": " .:-=+*#%@",
                "standard": " ][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao#MW .:-=+*#%@",
                "extended": " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"
            }
            ascii_palette = palette_map[args.palette]
            logger.info(f"Utilizzando palette con {len(ascii_palette)} caratteri: {ascii_palette[:10]}...")
    else:
        logger.info(f"Utilizzando metodo palette box")
        ascii_palette = 'box'

    # Inizializza la pipeline
    pipeline = VideoPipeline(
        args.video_path,
        width,
        args.fps,
        args.batch_size,
        args.log_performance,
        args.log_fps,
        ascii_palette=ascii_palette,
        loop_video=not args.no_loop,
        enable_audio=args.audio
    )

    # Gestione dei segnali per una chiusura pulita
    def signal_handler(sig, frame):
        """
        Gestisce i segnali di interruzione per chiudere correttamente l'applicazione.

        Args:
            sig: Segnale ricevuto
            frame: Frame corrente
        """
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

        if args.audio:
            print("Audio abilitato")

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