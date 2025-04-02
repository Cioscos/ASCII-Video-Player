import argparse
import sys
import os
import signal
import logging

import cv2

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

    args = parser.parse_args()

    logger.info(f"Avvio dell'applicazione con parametri: {args}")

    # Verifica che il file video esista
    if not os.path.isfile(args.video_path):
        logger.error(f"Il file video '{args.video_path}' non esiste.")
        print(f"Errore: Il file video '{args.video_path}' non esiste.")
        sys.exit(1)

    try:
        # Leggi le dimensioni del video per determinare l'aspect ratio
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            logger.error(f"Il file video '{args.video_path}' non esiste o non può essere aperto.")
            print(f"Errore: Il file video '{args.video_path}' non esiste o non può essere aperto.")
            sys.exit(1)

        # Ottieni le dimensioni del video
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()  # Rilascia le risorse

        # Calcola l'aspect ratio del video (altezza/larghezza)
        video_aspect_ratio = video_height / video_width if video_width > 0 else 9 / 16
        logger.info(f"Dimensioni video: {video_width}x{video_height}, Aspect ratio: {video_aspect_ratio:.4f}")

        # Stima l'altezza basata sulla larghezza e sull'aspect ratio del video
        estimated_height = estimate_height(args.width, video_aspect_ratio)
        logger.info(f"Dimensioni dell'output ASCII stimate: {args.width}x{estimated_height}")

        # Mostra il frame di calibrazione
        logger.info("Mostrando il frame di calibrazione...")
        render_calibration_frame(args.width, estimated_height)
        logger.info("Calibrazione completata, avvio rendering...")

        # Crea e avvia la pipeline video
        pipeline = VideoPipeline(
            video_path=args.video_path,
            width=args.width,
            fps=args.fps,
            log_fps=args.log_fps,
            log_performance=args.log_performance,
            batch_size=args.batch_size
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

