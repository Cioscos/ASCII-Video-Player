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


def parse_arguments():
    """
    Analizza gli argomenti della riga di comando.

    Returns:
        argparse.Namespace: Gli argomenti parsati
    """
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
    parser.add_argument("--high-performance", action="store_true",
                        help="Pre-elabora tutti i frame su disco per migliorare le performance")
    parser.add_argument("--palette", type=str, choices=["basic", "standard", "extended", "box", "custom"],
                        default="standard",
                        help="ASCII character palette to use: basic (10 chars), standard (42 chars), extended (70 chars)")
    parser.add_argument("--custom-palette", type=str,
                        help="Path to a custom palette file. It can be a normal .txt file.")
    parser.add_argument("--no-loop", action="store_true", help="Disable video looping (stop when video ends)")
    parser.add_argument("--audio", action="store_true", help="Enable audio playback")

    return parser.parse_args()


def setup_environment(args):
    """
    Configura l'ambiente di esecuzione, verifica dipendenze e configura il logging.

    Args:
        args (argparse.Namespace): Gli argomenti parsati

    Returns:
        tuple: (logger, has_audio, has_high_performance)
    """
    # Verifica dipendenze audio
    has_audio = False
    if args.audio:
        try:
            import sounddevice
            from moviepy import AudioFileClip
            print("Dipendenze audio verificate con successo.")
            has_audio = True
        except ImportError as e:
            print(f"Errore: impossibile abilitare l'audio. Mancano le dipendenze necessarie: {e}")
            print("Per abilitare l'audio, installa: pip install sounddevice moviepy")
            print("Continuo senza audio...")

    # Verifica dipendenze high-performance
    has_high_performance = False
    if args.high_performance:
        try:
            from precomputer import FramePrecomputer, PrecomputedFramePlayer
            has_high_performance = True
        except ImportError as e:
            print(f"Impossibile utilizzare modalità high-performance: {e}")
            print("Per usare questa funzionalità, installa: pip install tqdm")

    # Configura logging
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

    return logger, has_audio, has_high_performance


def get_video_info(video_path, logger):
    """
    Ottiene le informazioni di base sul video.

    Args:
        video_path (str): Percorso del file video
        logger (logging.Logger): Logger configurato

    Returns:
        tuple: (frame, fps_originale, frame_totali) o (None, None, None) in caso di errore
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Errore: Impossibile aprire il file video '{video_path}'.")
        return None, None, None

    ret, frame = cap.read()
    if not ret:
        logger.error("Errore: impossibile leggere il primo frame.")
        cap.release()
        return None, None, None

    fps_originale = cap.get(cv2.CAP_PROP_FPS)
    frame_totali = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Video originale: {video_path}, {frame.shape[1]}x{frame.shape[0]}, {fps_originale} FPS, {frame_totali} frames")

    cap.release()
    return frame, fps_originale, frame_totali


def get_ascii_palette(args, logger):
    """
    Determina la palette ASCII da utilizzare.

    Args:
        args (argparse.Namespace): Gli argomenti parsati
        logger (logging.Logger): Logger configurato

    Returns:
        str: La palette ASCII da utilizzare
    """
    if args.palette != 'box':
        if args.palette == 'custom':
            if not args.custom_palette:
                logger.warning("È stata specificata 'custom' ma nessun file di palette. Uso 'standard'.")
                args.palette = 'standard'
            else:
                try:
                    with open(args.custom_palette, 'r') as f:
                        ascii_palette = f.read().strip()
                    return ascii_palette
                except Exception as e:
                    logger.error(f"Errore nella lettura della palette personalizzata: {e}")
                    args.palette = 'standard'

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

    return ascii_palette


def run_high_performance_mode(args, video_info, ascii_dimensions, ascii_palette, logger):
    """
    Esegue il video in modalità high-performance.

    Args:
        args (argparse.Namespace): Gli argomenti parsati
        video_info (tuple): (frame, fps_originale, frame_totali)
        ascii_dimensions (tuple): (width, height)
        ascii_palette (str): Palette ASCII da utilizzare
        logger (logging.Logger): Logger configurato

    Returns:
        int: Codice di uscita (0 per successo, altro valore per errore)
    """
    from precomputer import FramePrecomputer, PrecomputedFramePlayer

    width, height = ascii_dimensions
    logger.info("Modalità high-performance attivata: pre-elaborazione frame")

    # Inizializza precomputer
    precomputer = FramePrecomputer(
        args.video_path,
        width,
        args.fps,
        ascii_palette,
        batch_size=args.batch_size
    )

    # Verifica cache esistente
    if not precomputer.is_cache_valid():
        print("\nPreparazione video in modalità high-performance...")
        print("Questo processo potrebbe richiedere tempo ma migliorerà le performance.")
        metadata = precomputer.precompute_frames()
        if not metadata:
            logger.error("Pre-elaborazione fallita, utilizzo pipeline standard")
            return run_standard_mode(args, video_info, ascii_dimensions, ascii_palette, logger)
    else:
        print("\nUtilizzo cache esistente in modalità high-performance")
        metadata = precomputer.metadata

    # Inizializza player
    player = PrecomputedFramePlayer(
        metadata,
        args.fps,
        not args.no_loop,
        args.audio,
        args.log_fps
    )

    # Nascondi cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        # Avvia player
        player.start()

        # Mostra istruzioni
        print("\nASCII Video Player avviato in modalità high-performance!")
        print("Premi Ctrl+C per terminare\n")
        if not args.no_loop:
            print("Il video si ripeterà automaticamente alla fine")
        else:
            print("Il video terminerà automaticamente alla fine")
        if args.audio:
            print("Audio abilitato")

        # Attendi termine
        while not player.should_stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera")
    finally:
        # Ferma player
        player.stop()

        # Mostra cursore
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    return 0


def run_standard_mode(args, video_info, ascii_dimensions, ascii_palette, logger):
    """
    Esegue il video in modalità standard con pipeline.

    Args:
        args (argparse.Namespace): Gli argomenti parsati
        video_info (tuple): (frame, fps_originale, frame_totali)
        ascii_dimensions (tuple): (width, height)
        ascii_palette (str): Palette ASCII da utilizzare
        logger (logging.Logger): Logger configurato

    Returns:
        int: Codice di uscita (0 per successo, altro valore per errore)
    """
    width, height = ascii_dimensions

    # Inizializza pipeline
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

    # Gestione segnali
    def signal_handler(sig, frame):
        logger.info("Segnale di interruzione ricevuto")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Avvia pipeline
        pipeline.start()

        # Mostra istruzioni
        print("\nASCII Video Player avviato!")
        print("Premi Ctrl+C per terminare\n")
        if not args.no_loop:
            print("Il video si ripeterà automaticamente alla fine")
        else:
            print("Il video terminerà automaticamente alla fine")
        if args.audio:
            print("Audio abilitato")

        # Attendi termine
        while not pipeline.should_stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera")
    finally:
        # Ferma pipeline
        pipeline.stop()

        # Mostra cursore
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    return 0


def main():
    """
    Funzione principale dell'applicazione ASCII video.

    Returns:
        int: Codice di uscita del programma (0 per successo, 1 per errore)
    """
    # Parse degli argomenti
    args = parse_arguments()

    # Verifica file video
    if not os.path.isfile(args.video_path):
        print(f"Errore: Il file video '{args.video_path}' non esiste.")
        return 1

    # Setup ambiente e logging
    logger, has_audio, has_high_performance = setup_environment(args)
    args.audio = args.audio and has_audio
    args.high_performance = args.high_performance and has_high_performance

    # Ottieni info video
    frame, fps_originale, frame_totali = get_video_info(args.video_path, logger)
    if frame is None:
        return 1

    # Suggerimento FPS per audio
    if args.audio and args.fps is None:
        target_fps = min(30, int(fps_originale))  # Limita a max 30 FPS
        logger.info(
            f"Audio abilitato senza FPS target specificati. Usando {target_fps} FPS per una migliore sincronizzazione.")
        args.fps = target_fps

    # Calcola dimensioni ASCII
    term_width, term_height = get_terminal_size()
    logger.info(f"Dimensioni terminale: {term_width}x{term_height}")

    width = args.width
    height, width_frame = frame.shape[:2]
    aspect_ratio = height / width_frame
    char_aspect_correction = 2.25  # Fattore di correzione per i caratteri ASCII
    new_height = int(aspect_ratio * width / char_aspect_correction)

    logger.info(
        f"Dimensioni output ASCII: {width}x{new_height} (aspect ratio video: {aspect_ratio:.2f}, corretta: {aspect_ratio / char_aspect_correction:.2f})")

    # Mostra frame di calibrazione
    try:
        render_calibration_frame(width, new_height)
        logger.info("Calibrazione completata con successo")
    except Exception as e:
        logger.error(f"Errore durante la calibrazione: {e}")

    # Nascondi cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    # Ottieni palette ASCII
    ascii_palette = get_ascii_palette(args, logger)

    # Esegui in modalità appropriata
    if args.high_performance:
        return run_high_performance_mode(
            args,
            (frame, fps_originale, frame_totali),
            (width, new_height),
            ascii_palette,
            logger
        )
    else:
        return run_standard_mode(
            args,
            (frame, fps_originale, frame_totali),
            (width, new_height),
            ascii_palette,
            logger
        )


if __name__ == "__main__":
    sys.exit(main())
