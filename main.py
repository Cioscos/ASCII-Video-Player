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

# Importa prompt_toolkit solo se necessario per il rendering
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings

# Configurazione del logging per salvare i tempi (in ms) di conversione e stampa.
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

# Stringa dei caratteri ASCII ordinati per densità (dal più chiaro al più scuro)
ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"
# Precompone una Look-Up Table (LUT) per mappare un valore di luminosità (0-255) in un carattere ASCII.
ASCII_LUT = np.array(list(ASCII_CHARS))

# Variabile globale per memorizzare il tempo medio di rendering (in secondi)
latest_print_time = 0.0
# Valore massimo di batch size (puoi modificarlo in base alle tue esigenze)
MAX_BATCH_SIZE = 30


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

    # Ridimensiona il frame (operazione molto performante in C)
    resized = cv2.resize(frame, (new_width, new_height))
    # Converte da BGR a RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Calcola la luminosità come media dei canali RGB
    brightness = np.mean(rgb_frame, axis=2).astype(np.uint8)
    # Mappa la luminosità in un indice per la LUT: [0,255] -> [0, len(ASCII_CHARS)-1]
    char_indices = (brightness.astype(np.uint16) * (len(ASCII_CHARS) - 1) // 255).astype(np.uint8)
    ascii_chars = ASCII_LUT[char_indices]

    # Converte ciascun canale RGB in stringa
    r_str = np.char.mod("%d", rgb_frame[:, :, 0])
    g_str = np.char.mod("%d", rgb_frame[:, :, 1])
    b_str = np.char.mod("%d", rgb_frame[:, :, 2])

    # Costruisce la stringa ANSI per ciascun pixel:
    # Formato ANSI: "\033[38;2;{r};{g};{b}m{char}\033[0m"
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

    # Concatena le righe per ottenere la stringa finale
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


def convert_frames(raw_queue, ascii_queue, pool, initial_batch_size, new_width, stop_event, target_fps,
                   log_metrics=False):
    """
    Legge i frame grezzi dalla raw_queue, li elabora in batch con un pool parallelo,
    adatta dinamicamente il batch size in base alle performance e mette i frame ASCII
    risultanti nella ascii_queue.

    Parametri:
        raw_queue (Queue): coda dei frame grezzi.
        ascii_queue (Queue): coda dei frame convertiti in ASCII.
        pool (Pool): pool di processi o thread per la conversione.
        initial_batch_size (int): batch size iniziale.
        new_width (int): larghezza desiderata per l'output ASCII.
        stop_event (threading.Event): evento per terminare il ciclo.
        target_fps (int): FPS target per il rendering.
        log_metrics (bool): se True, logga i tempi di conversione per frame.
    """
    global latest_print_time
    frame_count = 0
    # Imposta il batch size corrente come quello iniziale
    current_batch_size = initial_batch_size
    # Calcola il periodo target (in secondi) per ogni frame visualizzato
    target_period = 1.0 / target_fps

    while not stop_event.is_set():
        batch = []
        # Acquisizione dinamica del batch in base al batch size corrente
        for _ in range(current_batch_size):
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

        # Esecuzione della conversione in batch e misurazione del tempo
        conv_start = time.time()
        try:
            ascii_frames = pool.map(frame_to_ascii, batch, chunksize=len(batch))
        except Exception as e:
            print("Errore durante la conversione dei frame:", e)
            break
        conv_end = time.time()
        batch_conv_time = conv_end - conv_start  # tempo in secondi per il batch
        avg_conv_time = batch_conv_time / len(batch)  # tempo medio di conversione per frame (in secondi)

        if log_metrics:
            conversion_time_ms = (batch_conv_time * 1000) / len(batch)
            for _ in ascii_frames:
                logging.info(f"Frame {frame_count} - Conversione: {conversion_time_ms:.2f} ms")
                frame_count += 1

        # Inserisce ogni frame ASCII nella coda per il rendering.
        for af in ascii_frames:
            ascii_queue.put(af)

        # Calcolo dinamico del nuovo batch size.
        # L'idea è di scegliere N tale che: (N * avg_conv_time + latest_print_time) <= target_period
        # Se il rendering impiega troppo tempo, si preferisce un batch size minore per aggiornare il display più frequentemente.
        new_bs = int((target_period - latest_print_time) / avg_conv_time) if avg_conv_time > 0 else current_batch_size
        # Assicura che il batch size sia almeno 1 e non superi MAX_BATCH_SIZE
        new_bs = max(1, min(new_bs, MAX_BATCH_SIZE))
        if new_bs != current_batch_size:
            logging.info(f"[AUTO-TUNING] Batch size aggiornato da {current_batch_size} a {new_bs}")
            current_batch_size = new_bs


def render_frames_ptk(ascii_queue, app, control, stop_event, log_fps=False, log_performance=False):
    """
    Legge i frame ASCII dalla ascii_queue e aggiorna il display utilizzando prompt_toolkit.
    Misura e logga anche il tempo impiegato per la stampa.

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII.
        app (Application): applicazione prompt_toolkit.
        control (FormattedTextControl): controllo che mostra il testo.
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga il numero di aggiornamenti del display al secondo.
        log_performance (bool): se True, logga il tempo impiegato per la stampa.
    """
    global latest_print_time
    fps_count = 0
    fps_start = time.time()
    while not stop_event.is_set():
        try:
            ascii_frame = ascii_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        print_start = time.time()
        control.text = ANSI(ascii_frame)
        app.invalidate()
        print_end = time.time()
        printing_time = print_end - print_start  # in secondi
        latest_print_time = printing_time  # aggiornamento della variabile globale per il tuning

        if log_performance:
            logging.info(f"Printing: {printing_time * 1000:.2f} ms")

        fps_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[LOG] FPS display: {fps_count}")
            fps_count = 0
            fps_start = now


def render_frames_sys(ascii_queue, stop_event, log_fps=False, log_performance=False):
    """
    Legge i frame ASCII dalla ascii_queue e li stampa sul terminale utilizzando sys.stdout.write.
    Cancella lo schermo ad ogni aggiornamento e logga il tempo impiegato per la stampa.

    Parametri:
        ascii_queue (Queue): coda dei frame ASCII.
        stop_event (threading.Event): evento per terminare il ciclo.
        log_fps (bool): se True, logga gli aggiornamenti del display al secondo.
        log_performance (bool): se True, logga il tempo impiegato per la stampa.
    """
    global latest_print_time
    fps_count = 0
    fps_start = time.time()
    while not stop_event.is_set():
        try:
            ascii_frame = ascii_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        print_start = time.time()
        sys.stdout.write("\033[2J\033[H" + ascii_frame)  # cancella lo schermo e posiziona il cursore in alto a sinistra
        sys.stdout.flush()
        print_end = time.time()
        printing_time = print_end - print_start  # in secondi
        latest_print_time = printing_time

        if log_performance:
            logging.info(f"Printing: {printing_time * 1000:.2f} ms")
        fps_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[LOG] FPS display: {fps_count}")
            fps_count = 0
            fps_start = now


def main():
    """
    Funzione principale che configura il parsing degli argomenti, crea le code e i thread/processi
    necessari alla pipeline (estrazione, conversione e rendering) e avvia il rendering (con prompt_toolkit
    oppure tramite sys.stdout.write) in base alla flag --use_sys.

    Il batch size viene adattato dinamicamente in base agli FPS target e alle performance della macchina.
    """
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(
        description="Real-time ASCII video with auto-tuning batch size based on rendering performance."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of display FPS")
    parser.add_argument("--log_performance", action="store_true",
                        help="Enable logging of conversion and printing performance")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Initial batch size for processing frames (default: 1)")
    parser.add_argument("--use_threads", action="store_true",
                        help="Use thread pool instead of multiprocessing pool (utile su Windows)")
    parser.add_argument("--use_sys", action="store_true",
                        help="Use sys.stdout.write for rendering instead of prompt_toolkit")
    args = parser.parse_args()

    # Code per le queue: una per i frame grezzi e una per i frame ASCII.
    raw_queue = Queue(maxsize=args.fps)
    ascii_queue = Queue(maxsize=args.fps)

    # Avvia il processo di estrazione dei frame.
    extractor_process = Process(target=extract_frames, args=(args.video_path, raw_queue, args.fps))
    extractor_process.start()

    # Crea il pool per la conversione (processi o thread).
    if args.use_threads:
        pool = ThreadPool(processes=cpu_count())
    else:
        pool = Pool(processes=cpu_count())

    stop_event = threading.Event()

    # Avvia il thread di conversione, che adatta dinamicamente il batch size.
    converter_thread = threading.Thread(
        target=convert_frames,
        args=(raw_queue, ascii_queue, pool, args.batch_size, args.width, stop_event, args.fps, args.log_performance),
        daemon=True
    )
    converter_thread.start()

    # Seleziona la modalità di rendering in base alla flag --use_sys.
    if args.use_sys:
        print("[INFO] Rendering con sys.stdout.write attivato. Premere Ctrl+C per uscire.")
        renderer_thread = threading.Thread(
            target=render_frames_sys,
            args=(ascii_queue, stop_event, args.log_fps, args.log_performance),
            daemon=True
        )
        renderer_thread.start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    else:
        kb = KeyBindings()

        @kb.add("q")
        def exit_(event):
            event.app.exit()

        @kb.add("c-c")
        def exit_ctrl_c(event):
            event.app.exit()

        control = __import__('prompt_toolkit').layout.controls.FormattedTextControl(text=ANSI("Loading..."))
        root_container = Window(content=control)
        layout = Layout(root_container)
        app = Application(layout=layout, key_bindings=kb, full_screen=True)

        renderer_thread = threading.Thread(
            target=render_frames_ptk,
            args=(ascii_queue, app, control, stop_event, args.log_fps, args.log_performance),
            daemon=True
        )
        renderer_thread.start()
        try:
            app.run()
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
