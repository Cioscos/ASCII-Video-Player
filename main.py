import queue
from typing import List

import cv2
import numpy as np
import argparse
import time
import logging
import threading
import sys
import shutil
from numba import jit
from multiprocessing import Process, Queue, Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool  # Pool basato su thread (utile su Windows)

# Configurazione del logging per salvare i tempi (in ms) di conversione e rendering.
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

# Stringa dei caratteri ASCII ordinati per densità (dal più chiaro al più scuro).
ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"
# Precompone una Look-Up Table (LUT) per mappare un valore di luminosità (0-255) in un carattere ASCII.
ASCII_LUT = np.array(list(ASCII_CHARS))


def frame_to_ascii(frame_data):
    """
    Converte un frame video in una stringa ASCII a colori usando operazioni vettorializzate.

    Parametri:
        frame_data (tuple): (frame, new_width) dove:
            - frame: array NumPy contenente il frame (in formato BGR).
            - new_width (int): larghezza desiderata dell'output ASCII.

    Ritorna:
        str: stringa contenente il frame convertito in ASCII con escape ANSI.
    """
    frame, new_width = frame_data
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    # Ridimensiona il frame (operazione eseguita in C)
    resized = cv2.resize(frame, (new_width, new_height))
    # Converte il frame da BGR a RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Calcola la luminosità come media dei canali RGB
    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    # Mappa la luminosità in un indice per la LUT
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = ASCII_LUT[char_indices]

    # Converte ciascun canale RGB in stringa
    r_str = np.char.mod("%d", rgb_frame[:, :, 0])
    g_str = np.char.mod("%d", rgb_frame[:, :, 1])
    b_str = np.char.mod("%d", rgb_frame[:, :, 2])

    # Costruisce la stringa ANSI per ciascun pixel:
    # Formato: "\033[38;2;{r};{g};{b}m{char}\033[0m"
    prefix = "\033[38;2;"
    mid = "m"
    suffix = "\033[0m"
    ansi = np.char.add(prefix, r_str)
    ansi = np.char.add(ansi, ";")
    ansi = np.char.add(ansi, g_str)
    ansi = np.char.add(ansi, ";")
    ansi = np.char.add(ansi, b_str)
    ansi = np.char.add(ansi, mid)
    ansi = np.char.add(ansi, ascii_chars)
    ansi = np.char.add(ansi, suffix)

    # Unisce le righe per ottenere la stringa finale
    ascii_str = "\n".join("".join(row) for row in ansi)
    return ascii_str


def extract_frames(video_path, raw_queue, fps):
    """
    Estrae i frame da un video e li inserisce nella coda raw_queue.

    Parametri:
        video_path (str): percorso del file video.
        raw_queue (Queue): coda per passare i frame grezzi.
        fps (int): numero di frame al secondo da estrarre.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return
    frame_time = 1 / fps
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_queue.put(frame)
            time.sleep(frame_time)
    except KeyboardInterrupt:
        print("\n[!] Interruzione nell'estrazione dei frame.")
    finally:
        cap.release()
        raw_queue.put(None)  # Segnala la fine dello stream


def convert_frames(raw_queue, ascii_queue, pool, batch_size, new_width, stop_event):
    """
    Legge i frame grezzi dalla raw_queue, li elabora in batch con un pool parallelo
    e mette nella coda ascii_queue una tupla (ascii_frame, conversion_time_ms).

    Parametri:
        raw_queue (Queue): coda dei frame grezzi.
        ascii_queue (Queue): coda dei frame convertiti in ASCII insieme al tempo di conversione.
        pool (Pool): pool di processi o thread per la conversione.
        batch_size (int): numero di frame da elaborare insieme.
        new_width (int): larghezza desiderata per l'output ASCII.
        stop_event (threading.Event): evento per terminare il ciclo.
        log_metrics (bool): se True, abilita logging dettagliato (non utilizzato qui).
    """
    frame_count = 0
    while not stop_event.is_set():
        batch = []
        for _ in range(batch_size):
            try:
                frame = raw_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                stop_event.set()
                break
            batch.append((frame, new_width))

        if not batch:
            continue

        conv_start = time.time()
        try:
            ascii_frames = pool.map(frame_to_ascii, batch, chunksize=len(batch))
        except Exception as e:
            print("Errore durante la conversione dei frame:", e)
            break
        conv_end = time.time()
        conversion_time_ms = ((conv_end - conv_start) * 1000) / len(batch)

        # Invia nella coda per ogni frame una tupla (ascii_frame, conversion_time_ms)
        for af in ascii_frames:
            ascii_queue.put((af, conversion_time_ms))
            frame_count += 1


@jit(nopython=True)
def fast_diff_lines(new_lines: List[str], old_lines: List[str], max_lines: int) -> List[int]:
    """
    Computazione accelerata delle differenze tra righe mediante Numba JIT.

    Parametri:
        new_lines (List[str]): Lista delle nuove righe.
        old_lines (List[str]): Lista delle righe precedenti.
        max_lines (int): Numero massimo di righe da confrontare.

    Ritorna:
        List[int]: Indici delle righe che risultano differenti.
    """
    diff_indices = []  # Lista per salvare gli indici delle righe modificate
    n_new = len(new_lines)
    n_old = len(old_lines)
    for i in range(max_lines):
        # Se entrambi gli indici sono fuori dai limiti, passa al prossimo ciclo
        if i >= n_new and i >= n_old:
            continue

        new_line = new_lines[i] if i < n_new else ""
        old_line = old_lines[i] if i < n_old else ""

        # Se le lunghezze sono differenti, la riga va aggiornata
        if len(new_line) != len(old_line):
            diff_indices.append(i)
            continue

        # Confronto carattere per carattere
        for j in range(len(new_line)):
            if j >= len(old_line) or new_line[j] != old_line[j]:
                diff_indices.append(i)
                break

    return diff_indices


def render_frames_sys_partial(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Reads ASCII frames from ascii_queue and partially updates the terminal display
    using sys.stdout.write with JIT-accelerated diff calculation.
    Only modified lines are rewritten, avoiding clearing the entire screen.
    If terminal resizing is detected, a full reset is forced to clear any residual characters.

    Parameters:
        ascii_queue (Queue): Queue of ASCII frames, containing tuples (ascii_frame, conversion_time_ms).
        stop_event (threading.Event): Event to terminate the loop.
        log_fps (bool): If True, logs the number of display updates per second.
        log_performance (bool): If True, logs the time spent on the entire rendering process.
    """
    prev_frame_lines = None
    prev_terminal_size = shutil.get_terminal_size()
    max_line_length = prev_terminal_size.columns
    frame_counter = 0
    fps_count = 0
    fps_start = time.time()

    # Pre-compile escape sequences for better performance
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J\033[H"
    GOTO_FORMAT = "\033[{};1H"

    # Buffer for output
    output_buffer = []

    # Hide cursor to avoid flickering
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        while not stop_event.is_set():
            # Use a non-blocking get with shorter timeout
            try:
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.01)
            except queue.Empty:
                # Breve pausa per ridurre il consumo CPU in attesa di nuovi frame
                time.sleep(0.001)
                continue

            # Start measuring rendering time here, including all logic
            if log_performance:
                rendering_start = time.time()

            # Split once and reuse the result
            frame_lines = ascii_frame.split("\n")

            # Check if terminal has been resized
            current_terminal_size = shutil.get_terminal_size()
            terminal_resized = current_terminal_size != prev_terminal_size

            if terminal_resized:
                sys.stdout.write(CLEAR_SCREEN)
                sys.stdout.flush()
                prev_frame_lines = None
                prev_terminal_size = current_terminal_size

            # Clear the buffer for this frame
            output_buffer.clear()

            # Full frame if no previous frame or terminal resized
            if prev_frame_lines is None:
                output_buffer.append(ascii_frame)
                diff_indices = list(range(len(frame_lines)))  # All lines are different
            else:
                # Calculate differences using JIT-compiled function
                max_lines = max(len(frame_lines), len(prev_frame_lines))
                diff_indices = fast_diff_lines(frame_lines, prev_frame_lines, max_lines)

                # Build output buffer based on diff results
                for i in diff_indices:
                    if i < len(frame_lines):
                        new_line = frame_lines[i]
                        # Add padding if new line is shorter than old one
                        if i < len(prev_frame_lines):
                            old_line = prev_frame_lines[i]
                            if len(new_line) < len(old_line):
                                new_line += " " * (len(old_line) - len(new_line))

                        # Add to buffer
                        output_buffer.append(f"{GOTO_FORMAT.format(i + 1)}{new_line}")

            # Join once and write once
            if output_buffer:
                sys.stdout.write("".join(output_buffer))
                sys.stdout.flush()

            # Store current frame lines for next comparison
            prev_frame_lines = frame_lines

            # End performance measurement after all rendering logic
            if log_performance:
                rendering_end = time.time()
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms, "
                    f"Changed lines: {len(diff_indices)}"
                )

            frame_counter += 1
            fps_count += 1

            # Update FPS counter once per second
            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (sys partial): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now
    finally:
        # Restore cursor visibility
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def main():
    """
    Funzione principale che configura il parsing degli argomenti, crea le code e i thread/processi
    necessari alla pipeline (estrazione, conversione e rendering) e avvia il rendering in tempo reale
    tramite sys.stdout.write con aggiornamenti parziali.
    """
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using a parallel pipeline with separate conversion and sys-based partial rendering."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true",
                        help="Enable logging of conversion and rendering performance")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--use_threads", action="store_true",
                        help="Use thread pool instead of multiprocessing pool (utile su Windows)")
    args = parser.parse_args()

    # Creazione delle code: una per i frame grezzi e una per i frame ASCII.
    raw_queue = Queue(maxsize=args.fps)
    ascii_queue = Queue(maxsize=args.fps)

    # Avvia il processo di estrazione dei frame.
    extractor_process = Process(target=extract_frames, args=(args.video_path, raw_queue, args.fps))
    extractor_process.start()

    # Crea il pool di conversione (processi o thread).
    if args.use_threads:
        pool = ThreadPool(processes=cpu_count())
    else:
        pool = Pool(processes=cpu_count())

    stop_event = threading.Event()

    # Avvia il thread di conversione.
    converter_thread = threading.Thread(
        target=convert_frames,
        args=(raw_queue, ascii_queue, pool, args.batch_size, args.width, stop_event),
        daemon=True
    )
    converter_thread.start()

    # Avvia il thread di rendering basato su sys con aggiornamenti parziali.
    renderer_thread = threading.Thread(
        target=render_frames_sys_partial,
        args=(ascii_queue, stop_event, args.log_fps, args.log_performance),
        daemon=True
    )
    renderer_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # Cleanup: terminazione ordinata di thread e processi.
    stop_event.set()
    renderer_thread.join(timeout=1)
    converter_thread.join(timeout=1)
    pool.close()
    pool.join()
    extractor_process.terminate()
    extractor_process.join()
    raw_queue.close()
    raw_queue.cancel_join_thread()
    ascii_queue.close()
    ascii_queue.cancel_join_thread()
    print("[✔] Terminazione completata.")


if __name__ == '__main__':
    main()
