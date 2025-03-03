import queue
import cv2
import numpy as np
import argparse
import time
import logging
import threading
import sys
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


def render_frames_sys_partial(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Legge i frame ASCII dalla ascii_queue e aggiorna parzialmente il display nel terminale
    utilizzando sys.stdout.write. Solo le righe modificate vengono riscritte, evitando di cancellare
    l'intero schermo. Inoltre, per ogni frame viene loggato in una sola riga il tempo di conversione (in ms)
    e il tempo di rendering (in ms).

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII, contenente tuple (ascii_frame, conversion_time_ms).
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga il numero di aggiornamenti del display al secondo.
        log_performance (bool): se True, logga il tempo impiegato per la stampa.
    """
    prev_frame_lines = None  # Memorizza il frame precedente diviso in righe
    frame_counter = 0
    fps_count = 0
    fps_start = time.time()

    # Nasconde il cursore per evitare che il suo lampeggiamento disturbi l'output
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        while not stop_event.is_set():
            try:
                # Estrae dalla coda una tupla (ascii_frame, conversion_time_ms)
                ascii_item = ascii_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            ascii_frame, conversion_time_ms = ascii_item
            frame_lines = ascii_frame.split("\n")
            diff_string = ""

            if prev_frame_lines is None:
                # Primo frame: aggiorna l'intero schermo
                diff_string = ascii_frame
            else:
                # Calcola il numero massimo di righe tra frame corrente e precedente
                max_lines = max(len(frame_lines), len(prev_frame_lines))
                for i in range(max_lines):
                    new_line = frame_lines[i] if i < len(frame_lines) else ""
                    old_line = prev_frame_lines[i] if i < len(prev_frame_lines) else ""
                    if new_line != old_line:
                        # Posiziona il cursore sulla riga (i+1) e scrive la riga modificata.
                        # Se la nuova riga è più corta, aggiunge spazi per cancellare residui.
                        padding = " " * (len(old_line) - len(new_line)) if len(old_line) > len(new_line) else ""
                        diff_string += f"\033[{i + 1};1H" + new_line + padding
            prev_frame_lines = frame_lines

            print_start = time.time()
            sys.stdout.write(diff_string)
            sys.stdout.flush()
            print_end = time.time()
            rendering_time_ms = (print_end - print_start) * 1000

            # Logga in una sola riga i tempi di conversione e rendering per questo frame
            logging.info(
                f"Frame {frame_counter} - Conversion: {conversion_time_ms:.2f} ms, Rendering: {rendering_time_ms:.2f} ms")
            frame_counter += 1

            fps_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                if log_fps:
                    logging.info(f"[LOG] FPS display (sys partial): {fps_count}")
                fps_count = 0
                fps_start = now
    finally:
        # Ripristina la visualizzazione del cursore al termine
        sys.stdout.write("\033[?25h")
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
