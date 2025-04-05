import argparse
import sys
import os
import signal
import logging
import traceback

import cv2

from parallel_video_pipeline import ParallelVideoPipeline
from robust_logging import safe_log, shutdown_logging, setup_logging
from terminal_old import render_calibration_frame, show_cursor
from video_pipeline import VideoPipeline

# Variabile globale per la pipeline
pipeline = None


def signal_handler(sig, frame):
    """
    Gestore dei segnali per intercettare CTRL+C.

    Parametri:
        sig (int): Segnale ricevuto.
        frame (frame): Frame corrente.
    """
    global pipeline
    logger = logging.getLogger('ascii_video')

    try:
        safe_log(logger, logging.INFO, "Segnale di interruzione ricevuto (CTRL+C)")
        if pipeline:
            pipeline.stop_requested = True
            pipeline.running = False

        # Chiudi correttamente il logger
        shutdown_logging(logger)

        # Assicurati che il cursore sia visibile
        show_cursor()
        print("\nInterruzione in corso...")
    except Exception as e:
        print(f"Errore durante la gestione del segnale: {e}")
        # Assicurati che il cursore sia visibile anche in caso di errore
        show_cursor()


def main():
    """
    Funzione principale dell'applicazione.

    Analizza gli argomenti da linea di comando, verifica il file video,
    mostra il frame di calibrazione e avvia la pipeline video.
    """
    global pipeline

    try:
        # Inizializza il logger principale con la nuova implementazione
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
        parser.add_argument("--processes", type=int, default=None,
                            help="Number of processes to use for parallel processing (default: auto)")

        args = parser.parse_args()

        safe_log(logger, logging.INFO, f"Avvio dell'applicazione con parametri: {args}")

        # Verifica che il file video esista
        if not os.path.isfile(args.video_path):
            safe_log(logger, logging.ERROR, f"Il file video '{args.video_path}' non esiste.")
            print(f"Errore: Il file video '{args.video_path}' non esiste.")
            sys.exit(1)

        try:
            # Apertura video per ottenere le dimensioni iniziali
            cap = cv2.VideoCapture(args.video_path)
            if not cap.isOpened():
                safe_log(logger, logging.ERROR, "Impossibile aprire il video.")
                print("Errore: impossibile aprire il video.")
                sys.exit(1)

            ret, frame = cap.read()
            cap.release()
            if not ret:
                safe_log(logger, logging.ERROR, "Impossibile leggere il primo frame.")
                print("Errore: impossibile leggere il primo frame.")
                sys.exit(1)

            height, width_frame = frame.shape[:2]
            aspect_ratio = height / width_frame
            new_height = int(aspect_ratio * args.width * 0.45)  # Altezza adattata

            safe_log(logger, logging.INFO,
                     f"Dimensioni video originali: {width_frame}x{height}, Aspect ratio: {aspect_ratio:.4f}")
            safe_log(logger, logging.INFO, f"Dimensioni dell'output ASCII: {args.width}x{new_height}")

            # Mostra il frame di calibrazione
            safe_log(logger, logging.INFO, "Mostrando il frame di calibrazione...")
            render_calibration_frame(args.width, new_height)
            safe_log(logger, logging.INFO, "Calibrazione completata, avvio rendering...")

            # Regolazioni di performance automatiche
            if args.batch_size < 1:
                # Auto-regola il batch size in base alla larghezza se non specificato
                auto_batch_size = max(1, min(10, args.width // 50))
                safe_log(logger, logging.INFO, f"Batch size auto-regolato a {auto_batch_size}")
                args.batch_size = auto_batch_size

            # Crea e avvia la pipeline video
            if args.parallel:
                # Imposta il metodo di avvio per processi multipli
                if sys.platform == 'win32':
                    # Su Windows è necessario proteggere l'entry point
                    import multiprocessing as mp
                    mp.set_start_method('spawn', force=True)

                pipeline = ParallelVideoPipeline(
                    video_path=args.video_path,
                    width=args.width,
                    fps=args.fps,
                    log_fps=args.log_fps,
                    log_performance=args.log_performance,
                    batch_size=args.batch_size,
                    num_processes=args.processes,  # Auto-determina se None
                    use_cache=True,
                    logger=logger
                )
                safe_log(logger, logging.INFO, f"Avvio in modalità parallela con {pipeline.num_processes} processi")
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
                safe_log(logger, logging.INFO, "Avvio in modalità singolo processo")

            # Avvia la pipeline
            pipeline.start()

        except KeyboardInterrupt:
            safe_log(logger, logging.INFO, "Interruzione manuale dell'applicazione.")
            # Il signal_handler si occuperà di gestire l'interruzione
            sys.exit(0)
        except Exception as e:
            safe_log(logger, logging.ERROR, f"Errore durante l'esecuzione: {e}", exc_info=True)
            # Assicurati che il cursore sia visibile
            show_cursor()
            print(f"\nErrore durante l'esecuzione: {e}")
            sys.exit(1)

    except Exception as e:
        # Catch-all per qualsiasi errore nell'inizializzazione
        print(f"Errore critico durante l'inizializzazione: {e}")
        traceback.print_exc()
        show_cursor()  # Assicurati sempre che il cursore sia visibile
        sys.exit(1)

    finally:
        # Pulizia finale
        if 'logger' in locals():
            try:
                shutdown_logging(logger)
            except:
                pass

        show_cursor()  # Assicurati sempre che il cursore sia visibile



if __name__ == "__main__":
    main()
