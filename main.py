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
import curses
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
    Utilizza sequenze ANSI per colorare ciascun carattere.

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


def frame_to_ascii_curses(frame_data):
    """
    Converte un frame video in una stringa ASCII senza sequenze ANSI, adatto al rendering con curses.
    Questa funzione non aggiunge informazioni sul colore, semplificando la visualizzazione in curses.

    Parametri:
        frame_data (tuple): (frame, new_width) dove:
            - frame: array NumPy contenente il frame (in formato BGR).
            - new_width (int): larghezza desiderata dell'output ASCII.

    Ritorna:
        str: stringa contenente il frame convertito in ASCII.
    """
    frame, new_width = frame_data
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    resized = cv2.resize(frame, (new_width, new_height))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = ASCII_LUT[char_indices]

    # Costruisce la stringa ASCII senza escape di colore
    ascii_str = "\n".join("".join(row) for row in ascii_chars)
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


def convert_frames(raw_queue, ascii_queue, pool, batch_size, new_width, stop_event, conversion_function):
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
        conversion_function (callable): funzione da utilizzare per convertire un frame in ASCII.
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
            ascii_frames = pool.map(conversion_function, batch, chunksize=len(batch))
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
    Legge i frame ASCII dalla ascii_queue e aggiorna parzialmente il display
    del terminale utilizzando sys.stdout.write con calcolo accelerato delle differenze.
    Vengono riscritte solo le righe modificate.

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII, contenente tuple (ascii_frame, conversion_time_ms).
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga il numero di aggiornamenti al secondo.
        log_performance (bool): se True, logga il tempo impiegato nel rendering.
    """
    prev_frame_lines = None
    prev_terminal_size = shutil.get_terminal_size()
    frame_counter = 0
    fps_count = 0
    fps_start = time.time()

    # Sequenze di escape precompilate
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J\033[H"
    GOTO_FORMAT = "\033[{};1H"

    output_buffer = []

    # Nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        while not stop_event.is_set():
            try:
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.01)
            except queue.Empty:
                time.sleep(0.001)
                continue

            if log_performance:
                rendering_start = time.time()

            frame_lines = ascii_frame.split("\n")
            current_terminal_size = shutil.get_terminal_size()
            terminal_resized = current_terminal_size != prev_terminal_size

            if terminal_resized:
                sys.stdout.write(CLEAR_SCREEN)
                sys.stdout.flush()
                prev_frame_lines = None
                prev_terminal_size = current_terminal_size

            output_buffer.clear()

            if prev_frame_lines is None:
                output_buffer.append(ascii_frame)
            else:
                max_lines = max(len(frame_lines), len(prev_frame_lines))
                diff_indices = fast_diff_lines(frame_lines, prev_frame_lines, max_lines)
                for i in diff_indices:
                    if i < len(frame_lines):
                        new_line = frame_lines[i]
                        if i < len(prev_frame_lines):
                            old_line = prev_frame_lines[i]
                            if len(new_line) < len(old_line):
                                new_line += " " * (len(old_line) - len(new_line))
                        output_buffer.append(f"{GOTO_FORMAT.format(i + 1)}{new_line}")

            if output_buffer:
                sys.stdout.write("".join(output_buffer))
                sys.stdout.flush()

            prev_frame_lines = frame_lines

            if log_performance:
                rendering_end = time.time()
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms, "
                    f"Changed lines: {len(output_buffer)}"
                )

            frame_counter += 1
            fps_count += 1

            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (sys partial): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now
    finally:
        try:
            term_height = shutil.get_terminal_size().lines
        except Exception:
            term_height = 25
        sys.stdout.write("\n" * term_height)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def render_frames_curses(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Renderizza i frame ASCII utilizzando la libreria curses.
    Questa funzione si occupa di aggiornare lo schermo curses con i nuovi frame.
    Premendo il tasto 'q' l'utente può interrompere il rendering.

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII, contenente tuple (ascii_frame, conversion_time_ms).
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga il numero di aggiornamenti al secondo.
        log_performance (bool): se True, logga il tempo impiegato nel rendering.
    """
    def curses_loop(stdscr):
        curses.curs_set(0)  # Nasconde il cursore
        stdscr.nodelay(True)  # Input non bloccante
        frame_counter = 0
        fps_count = 0
        fps_start = time.time()

        while not stop_event.is_set():
            try:
                ascii_frame, conversion_time_ms = ascii_queue.get(timeout=0.01)
            except queue.Empty:
                time.sleep(0.001)
                continue

            if log_performance:
                rendering_start = time.time()

            frame_lines = ascii_frame.split("\n")
            max_y, max_x = stdscr.getmaxyx()
            for i, line in enumerate(frame_lines):
                if i >= max_y:
                    break
                try:
                    stdscr.addstr(i, 0, line[:max_x-1])
                except curses.error:
                    pass

            stdscr.refresh()

            if log_performance:
                rendering_end = time.time()
                total_rendering_time_ms = (rendering_end - rendering_start) * 1000
                logging.info(
                    f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, "
                    f"Total Rendering: {total_rendering_time_ms:.2f} ms"
                )

            frame_counter += 1
            fps_count += 1

            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (curses): {fps_count / elapsed:.1f}")
                fps_count = 0
                fps_start = now

            try:
                key = stdscr.getch()
                if key == ord('q'):
                    stop_event.set()
            except Exception:
                pass

    curses.wrapper(curses_loop)


def generate_calibration_frame(width, height):
    """
    Genera un frame ASCII di calibrazione tutto bianco, con un bordo e una croce centrale.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.

    Ritorna:
        str: Stringa ASCII con il frame bianco, bordi e croce centrale.
    """
    # Caratteri
    BORDER_CHAR = "#"
    CROSS_CHAR = "+"
    WHITE_CHAR = "█"  # Blocchi pieni per simulare un frame bianco

    # Crea una matrice di caratteri bianchi
    ascii_frame = [[WHITE_CHAR] * width for _ in range(height)]

    # Disegna i bordi del rettangolo
    for x in range(width):
        ascii_frame[0][x] = BORDER_CHAR  # Riga superiore
        ascii_frame[-1][x] = BORDER_CHAR  # Riga inferiore

    for y in range(height):
        ascii_frame[y][0] = BORDER_CHAR  # Colonna sinistra
        ascii_frame[y][-1] = BORDER_CHAR  # Colonna destra

    # Disegna la croce centrale
    center_y, center_x = height // 2, width // 2
    for x in range(width):
        ascii_frame[center_y][x] = CROSS_CHAR  # Linea orizzontale
    for y in range(height):
        ascii_frame[y][center_x] = CROSS_CHAR  # Linea verticale

    # Converti la matrice in una stringa
    ascii_string = "\n".join("".join(row) for row in ascii_frame)
    return ascii_string


def render_calibration_frame(width, height):
    """
    Mostra un frame di calibrazione nel terminale e attende che l'utente prema ENTER.
    Dopo l'input, il terminale viene svuotato e il buffer di scorrimento viene rimosso.

    Parametri:
        width (int): Larghezza dell'output ASCII.
        height (int): Altezza dell'output ASCII.
    """
    # Genera il frame di calibrazione
    calibration_frame = generate_calibration_frame(width, height)

    # Escape sequences per il terminale
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    CLEAR_SCREEN = "\033[2J\033[H"
    RESET_TERMINAL = "\033c"  # Reset completo del terminale

    # Pulisce lo schermo e nasconde il cursore
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(calibration_frame + "\n")
    sys.stdout.write("\n[INFO] Regola la dimensione del terminale e premi ENTER per iniziare...\n")
    sys.stdout.flush()

    # Attendi l'input dell'utente
    input()

    # Resetta completamente il terminale (rimuove lo scrollback buffer)
    sys.stdout.write(RESET_TERMINAL)
    sys.stdout.flush()

    # Ripristina il cursore
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def main():
    """
    Funzione principale che configura il parsing degli argomenti, crea le code e i thread/processi
    necessari alla pipeline (estrazione, conversione e rendering) e avvia il rendering in tempo reale.
    In base alla flag --use_curses, il rendering avviene tramite curses oppure con il metodo standard (sys).
    """
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using a parallel pipeline with separate conversion and rendering."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true", help="Enable logging of conversion and rendering performance")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing frames (default: 1)")
    parser.add_argument("--use_threads", action="store_true", help="Use thread pool instead of multiprocessing pool (utile su Windows)")
    parser.add_argument("--use_curses", action="store_true", help="Use curses library for rendering instead of sys-based partial rendering")
    args = parser.parse_args()

    # Apertura video per ottenere le dimensioni iniziali
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Errore: impossibile leggere il primo frame.")
        return

    height, width_frame = frame.shape[:2]
    aspect_ratio = height / width_frame
    new_height = int(aspect_ratio * args.width * 0.45)  # Altezza adattata

    render_calibration_frame(args.width, new_height)

    raw_queue = Queue(maxsize=args.fps)
    ascii_queue = Queue(maxsize=args.fps)

    extractor_process = Process(target=extract_frames, args=(args.video_path, raw_queue, args.fps))
    extractor_process.start()

    if args.use_threads:
        pool = ThreadPool(processes=cpu_count())
    else:
        pool = Pool(processes=cpu_count())

    stop_event = threading.Event()

    # Se si usa curses, si usa una conversione senza sequenze ANSI
    if args.use_curses:
        conversion_function = frame_to_ascii_curses
        renderer_function = render_frames_curses
    else:
        conversion_function = frame_to_ascii
        renderer_function = render_frames_sys_partial

    converter_thread = threading.Thread(
        target=convert_frames,
        args=(raw_queue, ascii_queue, pool, args.batch_size, args.width, stop_event, conversion_function),
        daemon=True
    )
    converter_thread.start()

    renderer_thread = threading.Thread(
        target=renderer_function,
        args=(ascii_queue, stop_event, args.log_fps, args.log_performance),
        daemon=True
    )
    renderer_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

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
