import queue
import cv2
import numpy as np
import argparse
import time
import logging
import threading
from PIL import Image
from multiprocessing import Process, Queue, Pool, cpu_count
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl

# Configura il logging: salverà i tempi in ms per conversione e stampa.
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"


def frame_to_ascii(frame_data):
    """
    Converte un frame in ASCII a colori.
    frame_data: tupla (frame, new_width)
    Restituisce una stringa con il frame in ASCII (con escape ANSI).
    """
    frame, new_width = frame_data
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)
    image = image.resize((new_width, new_height))
    pixels = np.array(image)  # Shape: (new_height, new_width, 3)

    # Calcola la luminosità (media dei canali RGB)
    brightness = np.mean(pixels, axis=2).astype(int)
    # Mappa la luminosità in indici del set di caratteri
    char_indices = (brightness * (len(ASCII_CHARS) - 1) // 255).astype(int)
    ascii_chars = np.vectorize(lambda x: ASCII_CHARS[x])(char_indices)

    # Funzione per applicare il colore ANSI a un carattere
    def colorize(r, g, b, char):
        return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

    vectorized_colorize = np.vectorize(colorize)
    color_ascii = vectorized_colorize(
        pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], ascii_chars
    )

    ascii_str = "\n".join("".join(row) for row in color_ascii)
    return ascii_str


def extract_frames(video_path, frame_queue, fps):
    """
    Estrae i frame dal video e li inserisce nella coda.
    In caso di fine video o interruzione, inserisce None per segnalare la fine.
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
            frame_queue.put(frame)
            time.sleep(frame_time)
    except KeyboardInterrupt:
        print("\n[!] Interruzione nel processo di estrazione frame.")
    finally:
        cap.release()
        frame_queue.put(None)  # Segnala la fine


def update_frames(frame_queue, new_width, app, control, pool, stop_event, batch_size=2, log_fps=False, log_metrics=False):
    """
    Legge un batch di frame dalla coda, converte i frame in ASCII in parallelo e aggiorna il display.
    Misura il tempo medio (in ms) di conversione per frame e il tempo per aggiornare il display.
    batch_size: numero di frame processati insieme (puoi sperimentare con 1, 2 o 3).
    """
    frame_count = 0
    fps_count = 0
    fps_start = time.time()

    while not stop_event.is_set():
        frames = []
        for _ in range(batch_size):
            try:
                frame = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                stop_event.set()
                break
            frames.append((frame, new_width))

        if not frames:
            continue

        # Misura il tempo di conversione in batch (in ms per frame)
        conv_start = time.time()
        try:
            async_result = pool.map_async(frame_to_ascii, frames, chunksize=1)
            ascii_frames = async_result.get(timeout=1)
        except Exception as e:
            break
        conv_end = time.time()
        conversion_time_ms = ((conv_end - conv_start) * 1000) / len(frames)

        # Misura il tempo di aggiornamento del display (stampa)
        print_start = time.time()
        # Per aggiornare il display usiamo l'ultimo frame del batch
        control.text = ANSI(ascii_frames[-1])
        app.invalidate()
        print_end = time.time()
        printing_time_ms = (print_end - print_start) * 1000

        # Logga i tempi medi per ogni frame del batch
        if log_metrics:
            for _ in ascii_frames:
                logging.info(
                    f"Frame {frame_count} - Conversione: {conversion_time_ms:.2f} ms - Stampa: {printing_time_ms:.2f} ms")
                frame_count += 1
                fps_count += 1

        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[LOG] FPS effettivi: {fps_count}")
            fps_count = 0
            fps_start = now


def main():
    parser = argparse.ArgumentParser(
        description="Real-time ASCII video using prompt_toolkit and multiprocessing."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of actual FPS")
    parser.add_argument("--log_performance", action="store_true", help="Enable logging of performance metrics")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of processes to use (default: 1)")
    args = parser.parse_args()

    # Imposta la coda: maxsize = fps (o puoi sperimentare un valore leggermente maggiore)
    frame_queue = Queue(maxsize=args.fps)

    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps))
    extractor_process.start()

    pool = Pool(processes=cpu_count())

    # Definisci i key bindings: 'q' o Ctrl+C per uscire.
    kb = KeyBindings()

    @kb.add("q")
    def exit_(event):
        event.app.exit()

    @kb.add("c-c")
    def exit_ctrl_c(event):
        event.app.exit()

    control = FormattedTextControl(text=ANSI("Loading..."))
    root_container = Window(content=control)
    layout = Layout(root_container)

    app = Application(layout=layout, key_bindings=kb, full_screen=True)

    # Flag per la terminazione pulita del thread di aggiornamento
    stop_event = threading.Event()

    updater_thread = threading.Thread(
        target=update_frames,
        args=(frame_queue, args.width, app, control, pool, stop_event, args.batch_size, args.log_fps, args.log_performance),
        daemon=True
    )
    updater_thread.start()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        updater_thread.join(timeout=1)
        pool.close()
        pool.join()
        extractor_process.terminate()
        extractor_process.join()
        frame_queue.close()
        frame_queue.cancel_join_thread()
        print("[✔] Terminazione completata.")


if __name__ == '__main__':
    main()
