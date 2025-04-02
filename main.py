import argparse
import sys
import os
import signal
import logging

import cv2

from parallel_video_pipeline import ParallelVideoPipeline
from terminal import render_calibration_frame, show_cursor
from video_pipeline import VideoPipeline
from utils import estimate_height, setup_logging

# Variabile globale per la pipeline
pipeline = None


def signal_handler(sig, frame):
    """
    Gestore dei segnali per intercettare CTRL+C.

    Parametri:
        sig: Segnale ricevuto.
        frame: Frame corrente.
    """
    global pipeline
    logger = logging.getLogger('ascii_video')

    logger.info("Segnale di interruzione ricevuto (CTRL+C)")
    if pipeline:
        pipeline.stop_requested = True
        pipeline.running = False

    # Assicurati che il cursore sia visibile
    show_cursor()
    print("\nInterruzione in corso...")


def main():
    """
    Funzione principale dell'applicazione.

    Analizza gli argomenti da linea di comando, verifica il file video,
    mostra il frame di calibrazione e avvia la pipeline video.
    """
    global pipeline

    # Inizializza il logger principale
    logger = setup_logging()

    # Registra il gestore di segnali per CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Parsing degli argomenti
    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using a parallel pipeline with separate conversion and rendering."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, default=100, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true",
                        help="Enable logging of conversion and rendering performance")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")

    args = parser.parse_args()

    logger.info(f"Avvio dell'applicazione con parametri: {args}")

    # Verifica che il file video esista
    if not os.path.isfile(args.video_path):
        logger.error(f"Il file video '{args.video_path}' non esiste.")
        print(f"Errore: Il file video '{args.video_path}' non esiste.")
        sys.exit(1)

    try:
        # Apertura video per ottenere le dimensioni iniziali
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            logger.error("Impossibile aprire il video.")
            print("Errore: impossibile aprire il video.")
            sys.exit(1)

        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error("Impossibile leggere il primo frame.")
            print("Errore: impossibile leggere il primo frame.")
            sys.exit(1)

        height, width_frame = frame.shape[:2]
        aspect_ratio = height / width_frame
        new_height = int(aspect_ratio * args.width * 0.45)  # Altezza adattata

        logger.info(f"Dimensioni video originali: {width_frame}x{height}, Aspect ratio: {aspect_ratio:.4f}")
        logger.info(f"Dimensioni dell'output ASCII: {args.width}x{new_height}")

        # Mostra il frame di calibrazione
        logger.info("Mostrando il frame di calibrazione...")
        render_calibration_frame(args.width, new_height)
        logger.info("Calibrazione completata, avvio rendering...")

        # Crea e avvia la pipeline video
        if args.parallel:
            pipeline = ParallelVideoPipeline(
                video_path=args.video_path,
                width=args.width,
                fps=args.fps,
                log_fps=args.log_fps,
                log_performance=args.log_performance,
                batch_size=args.batch_size,
                num_processes=None,  # Auto-determina il numero di processi
                use_cache=True
            )
        else:
            pipeline = VideoPipeline(
                video_path=args.video_path,
                width=args.width,
                fps=args.fps,
                log_fps=args.log_fps,
                log_performance=args.log_performance,
                batch_size=args.batch_size,
                logger=logger
            )

        # Avvia la pipeline
        pipeline.start()

    except KeyboardInterrupt:
        logger.info("Interruzione manuale dell'applicazione.")
        # Il signal_handler si occuperà di gestire l'interruzione
        sys.exit(0)
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}", exc_info=True)
        # Assicurati che il cursore sia visibile
        show_cursor()
        print(f"\nErrore durante l'esecuzione: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

