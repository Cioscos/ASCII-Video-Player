import argparse
import logging
import os
import sys
import threading

import cv2

from ascii_converter import compute_ascii_height
from calibration_frame import render_calibration_frame
from pipeline import VideoPipeline
from utils import setup_logging, get_terminal_size

HIDE_CURSOR = '\033[?25l'
# Ripristino colori e cursore (come TerminalDisplay.restore, idempotente)
RESET_AND_SHOW_CURSOR = '\033[0m\033[?25h'

# Palette 'extended' (67 caratteri), esposta a livello di modulo per i test
EXTENDED = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"

# Palette 'standard' (42 caratteri): sottocampionamento uniforme di 'extended'
STANDARD = ' \'`",;l!~+-][{)(\\/frxuvzYUCL0Zmqdbha*MW8%@'

PALETTES = {
    "basic": " .:-=+*#%@",
    "standard": STANDARD,
    "extended": EXTENDED,
}


def load_custom_palette(path):
    """
    Legge una palette personalizzata da file (UTF-8).

    Rimuove solo i caratteri di fine riga, preservando spazi iniziali e finali.

    Args:
        path (str): Percorso del file di palette

    Returns:
        str: La palette letta

    Raises:
        ValueError: Se la palette risultante e' vuota
    """
    with open(path, 'r', encoding='utf-8') as f:
        palette = f.read().replace('\r', '').replace('\n', '')
    if not palette:
        raise ValueError(f"La palette personalizzata '{path}' e' vuota")
    return palette


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
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true",
                        help="Enable logging of conversion and rendering performance")
    parser.add_argument("--verbose", action="store_true", help="Show all log messages in the terminal")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--high-performance", action="store_true",
                        help="Pre-elabora tutti i frame su disco per migliorare le performance")
    parser.add_argument("--palette", type=str, choices=["basic", "standard", "extended", "box", "half", "custom"],
                        default="standard",
                        help="ASCII character palette to use: basic (10 chars), standard (42 chars), extended (67 chars), box (colored blocks), half (half blocks, double vertical resolution)")
    parser.add_argument("--custom-palette", type=str,
                        help="Path to a custom palette file. It can be a normal .txt file.")
    parser.add_argument("--no-loop", action="store_true", help="Disable video looping (stop when video ends)")
    parser.add_argument("--truecolor", action="store_true",
                        help="Use 24-bit colors instead of the 256-color palette")
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
            import precomputer  # richiede tqdm
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
    if args.palette == 'half':
        logger.info("Utilizzando metodo palette half (mezzi blocchi)")
        return 'half'

    if args.palette == 'box':
        logger.info("Utilizzando metodo palette box")
        return 'box'

    if args.palette == 'custom':
        if not args.custom_palette:
            logger.warning("È stata specificata 'custom' ma nessun file di palette. Uso 'standard'.")
        else:
            try:
                ascii_palette = load_custom_palette(args.custom_palette)
                logger.info(f"Utilizzando palette personalizzata con {len(ascii_palette)} caratteri")
                return ascii_palette
            except Exception as e:
                logger.error(f"Errore nella lettura della palette personalizzata: {e}. Uso 'standard'.")
        args.palette = 'standard'

    ascii_palette = PALETTES[args.palette]
    logger.info(f"Utilizzando palette con {len(ascii_palette)} caratteri: {ascii_palette[:10]}...")
    return ascii_palette


def run_high_performance_mode(args, video_info, ascii_dimensions, ascii_palette, logger):
    """
    Esegue il video in modalità high-performance: cache binaria pre-elaborata letta via mmap.

    Se la cache valida non esiste viene creata con precompute; in caso di errore
    (o video non leggibile) si ripiega sulla modalità standard. Ctrl+C durante la
    pre-elaborazione termina senza fallback.

    Args:
        args (argparse.Namespace): Gli argomenti parsati
        video_info (tuple): (frame, fps_originale, frame_totali)
        ascii_dimensions (tuple): (width, height)
        ascii_palette (str): Palette ASCII da utilizzare
        logger (logging.Logger): Logger configurato

    Returns:
        int: Codice di uscita (0 per successo, altro valore per errore)
    """
    from fps_stats import FpsStats
    from playback import FrameDecimator, run_playback
    from precomputer import CachedFrames, CachedFrameSource, cache_dir_for, precompute
    from terminal_display import TerminalDisplay

    width, height = ascii_dimensions
    source_fps = video_info[1]
    display = TerminalDisplay()
    logger.info("Modalità high-performance attivata")

    effective_fps = FrameDecimator(source_fps, args.fps).effective_fps
    cache_dir = cache_dir_for(args.video_path, width, height, effective_fps, ascii_palette,
                              truecolor=args.truecolor)
    cached = CachedFrames.open(cache_dir)
    if cached is not None:
        logger.info(f"Cache esistente riutilizzata: {cache_dir}")
        print("\nUtilizzo cache esistente in modalità high-performance")
    else:
        print("\nPreparazione video in modalità high-performance...")
        print("Questo processo potrebbe richiedere tempo ma migliorerà le performance.")
        sys.stdout.flush()
        try:
            cache_dir = precompute(args.video_path, width, height, source_fps, args.fps, ascii_palette,
                                 truecolor=args.truecolor)
        except KeyboardInterrupt:
            logger.info("Pre-elaborazione interrotta da tastiera")
            display.restore()
            return 0
        except Exception as e:
            logger.exception(f"Pre-elaborazione fallita: {e}. Utilizzo modalità standard")
            return run_standard_mode(args, video_info, ascii_dimensions, ascii_palette, logger)
        cached = CachedFrames.open(cache_dir) if cache_dir is not None else None
        if cached is None:
            logger.error("Pre-elaborazione non riuscita, utilizzo modalità standard")
            return run_standard_mode(args, video_info, ascii_dimensions, ascii_palette, logger)

    fps = cached.metadata["effective_fps"]
    stats = FpsStats(fps) if args.log_fps else None
    audio = None
    exit_code = 0

    try:
        print("\nASCII Video Player avviato in modalità high-performance!")
        print("Premi Ctrl+C per terminare\n")
        if not args.no_loop:
            print("Il video si ripeterà automaticamente alla fine")
        else:
            print("Il video terminerà automaticamente alla fine")
        sys.stdout.flush()

        if args.audio:
            audio = _start_audio(args.video_path, fps, logger)

        drawn = run_playback(CachedFrameSource(cached), fps, display, threading.Event(),
                             loop=not args.no_loop, stats=stats, audio=audio)
        logger.info(f"Riproduzione terminata, frame mostrati: {drawn}")
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera")
    except Exception as e:
        logger.exception(f"Errore durante la riproduzione: {e}")
        exit_code = 1
    finally:
        _shutdown(display, logger, audio.stop if audio is not None else None, cached.close)

    if stats is not None:
        logger.info(stats.summary())
    return exit_code


def _shutdown(display, logger, *cleanups):
    """
    Ripristina subito il terminale, poi esegue le operazioni di chiusura.

    Il terminale viene ripristinato per primo, così un secondo Ctrl+C durante la
    chiusura non lascia il cursore nascosto; un KeyboardInterrupt in una chiusura
    non impedisce l'esecuzione delle successive.

    Args:
        display (TerminalDisplay): Display da ripristinare
        logger (logging.Logger): Logger configurato
        *cleanups: Funzioni di chiusura senza argomenti (None viene ignorato)
    """
    display.restore()
    for cleanup in cleanups:
        if cleanup is None:
            continue
        try:
            cleanup()
        except KeyboardInterrupt:
            logger.info("Interruzione da tastiera durante la chiusura, proseguo")


def warn_if_frame_too_tall(height, logger):
    """
    Avvisa se il frame ASCII (più una riga) non entra nell'altezza del terminale.

    Args:
        height (int): Altezza del frame ASCII in righe
        logger (logging.Logger): Logger configurato

    Returns:
        bool: True se il frame è troppo alto per il terminale
    """
    _, term_rows = get_terminal_size()
    if height + 1 > term_rows:
        logger.warning(
            f"Il frame ASCII ({height} righe) supera l'altezza del terminale ({term_rows} righe): "
            f"l'immagine risulterà distorta. Riduci la larghezza o la dimensione del font")
        return True
    return False


def _start_audio(video_path, fps, logger):
    """
    Avvia l'audio del video; in caso di problemi continua senza audio.

    Returns:
        AudioPlayer | None: Il player avviato, oppure None
    """
    try:
        from audio_player import AudioPlayer
        player = AudioPlayer(video_path, fps)
        if player.initialize() and player.start():
            logger.info("Riproduzione audio avviata")
            return player
        logger.warning("Audio non disponibile, continuo senza audio")
    except Exception as e:
        logger.warning(f"Impossibile avviare l'audio: {e}. Continuo senza audio")
    return None


def run_standard_mode(args, video_info, ascii_dimensions, ascii_palette, logger):
    """
    Esegue il video in modalità standard: pipeline multi-processo e riproduzione nel thread principale.

    Args:
        args (argparse.Namespace): Gli argomenti parsati
        video_info (tuple): (frame, fps_originale, frame_totali)
        ascii_dimensions (tuple): (width, height)
        ascii_palette (str): Palette ASCII da utilizzare
        logger (logging.Logger): Logger configurato

    Returns:
        int: Codice di uscita (0 per successo, altro valore per errore)
    """
    from fps_stats import FpsStats
    from playback import run_playback
    from terminal_display import TerminalDisplay

    width, height = ascii_dimensions
    pipeline = VideoPipeline(
        args.video_path, width, height, video_info[1], args.fps, args.batch_size, ascii_palette,
        loop_video=not args.no_loop, log_performance=args.log_performance, truecolor=args.truecolor
    )
    display = TerminalDisplay()
    stats = FpsStats(pipeline.effective_fps) if args.log_fps else None
    audio = None
    exit_code = 0

    try:
        pipeline.start()

        print("\nASCII Video Player avviato!")
        print("Premi Ctrl+C per terminare\n")
        if not args.no_loop:
            print("Il video si ripeterà automaticamente alla fine")
        else:
            print("Il video terminerà automaticamente alla fine")
        sys.stdout.flush()

        if args.audio:
            audio = _start_audio(args.video_path, pipeline.effective_fps, logger)

        drawn = run_playback(pipeline.frame_source(), pipeline.effective_fps, display, pipeline.should_stop,
                             loop=False, stats=stats, audio=audio)
        logger.info(f"Riproduzione terminata, frame mostrati: {drawn}")
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera")
    except Exception as e:
        logger.exception(f"Errore durante la riproduzione: {e}")
        exit_code = 1
    finally:
        _shutdown(display, logger, pipeline.stop, audio.stop if audio is not None else None)

    if stats is not None:
        logger.info(stats.summary())
    return exit_code


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

    # Calcola dimensioni ASCII
    term_width, term_height = get_terminal_size()
    logger.info(f"Dimensioni terminale: {term_width}x{term_height}")

    width = args.width
    src_h, src_w = frame.shape[:2]
    new_height = compute_ascii_height(src_w, src_h, width)

    logger.info(f"Dimensioni output ASCII: {width}x{new_height} (aspect ratio video: {src_h / src_w:.2f})")

    # Mostra frame di calibrazione
    try:
        render_calibration_frame(width, new_height)
        logger.info("Calibrazione completata con successo")
    except Exception as e:
        logger.error(f"Errore durante la calibrazione: {e}")

    # Il terminale può essere stato ridimensionato durante la calibrazione
    warn_if_frame_too_tall(new_height, logger)

    try:
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
    finally:
        # Ripristina colori e cursore anche se un'eccezione precede i try delle modalità
        sys.stdout.write(RESET_AND_SHOW_CURSOR)
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
