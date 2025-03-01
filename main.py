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

# Configura il logging
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"


def frame_to_ascii(frame_data):
    """Converte un frame in ASCII a colori usando operazioni vettoriali con NumPy.
       frame_data è una tupla (frame, new_width)."""
    frame, new_width = frame_data
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)
    image = image.resize((new_width, new_height))
    pixels = np.array(image)  # (new_height, new_width, 3)

    # Calcola la luminosità (media dei canali RGB)
    brightness = np.mean(pixels, axis=2).astype(int)
    # Mappa la luminosità a indici del set di caratteri
    char_indices = (brightness * (len(ASCII_CHARS) - 1) // 255).astype(int)
    ascii_chars = np.vectorize(lambda x: ASCII_CHARS[x])(char_indices)

    # Funzione per applicare il colore ANSI
    def colorize(r, g, b, char):
        return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

    vectorized_colorize = np.vectorize(colorize)
    color_ascii = vectorized_colorize(
        pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], ascii_chars
    )

    ascii_str = "\n".join("".join(row) for row in color_ascii)
    return ascii_str


def extract_frames(video_path, frame_queue, fps):
    """Estrae i frame dal video e li mette nella coda."""
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


def update_frames(frame_queue, new_width, app, control, log_fps, pool, stop_event, batch_size=2):
    """
    Thread di aggiornamento:
      - Legge un batch di frame dalla coda.
      - Converte i frame in ASCII in parallelo tramite il pool.
      - Aggiorna il display.
    Il batch_size può essere ridotto (es. a 1 o 2) per aggiornamenti più fluidi.
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

        try:
            # Utilizza map_async per evitare blocchi lunghi e per poter gestire il caso di terminazione
            async_result = pool.map_async(frame_to_ascii, frames, chunksize=1)
            ascii_frames = async_result.get(timeout=1)
        except Exception as e:
            # Se il pool non è in esecuzione o c'è un timeout, esce dal loop
            break

        # Per ogni frame convertito, aggiorna il display (qui mostriamo l'ultimo frame del batch)
        # Se vuoi aggiornare per ogni frame, potresti ciclare e aggiungere un breve delay.
        if ascii_frames:
            control.text = ANSI(ascii_frames[-1])
            app.invalidate()

        for _ in ascii_frames:
            logging.info(f"Frame {frame_count} - Conversione completata")
            frame_count += 1
            fps_count += 1

        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[LOG] FPS effettivi: {fps_count}")
            fps_count = 0
            fps_start = now


def main():
    parser = argparse.ArgumentParser(description="Real-time ASCII video using prompt_toolkit and multiprocessing.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of actual FPS")
    args = parser.parse_args()

    # Imposta la coda con maxsize uguale agli fps
    frame_queue = Queue(maxsize=args.fps)

    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps))
    extractor_process.start()

    pool = Pool(processes=cpu_count())  # Usa tutti i core disponibili

    # Definisci key bindings per uscire con 'q' o Ctrl+C
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

    # Flag di terminazione per il thread di aggiornamento
    stop_event = threading.Event()

    updater_thread = threading.Thread(
        target=update_frames,
        args=(frame_queue, args.width, app, control, args.log_fps, pool, stop_event),
        daemon=True
    )
    updater_thread.start()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()  # Segnala al thread di uscire
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
