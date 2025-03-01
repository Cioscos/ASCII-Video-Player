import queue
import cv2
import numpy as np
import argparse
import time
import logging
import threading
from PIL import Image
from multiprocessing import Process, Queue
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import FormattedTextControl

# Configura il logging su file
logging.basicConfig(
    filename="ascii_video.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"


def frame_to_ascii(frame, new_width):
    """Converte un frame in ASCII a colori usando operazioni vettoriali con NumPy."""
    # Converti da BGR a RGB e crea una PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)
    image = image.resize((new_width, new_height))
    pixels = np.array(image)  # Shape: (new_height, new_width, 3)

    # Calcola la luminosità (media dei canali RGB)
    brightness = np.mean(pixels, axis=2).astype(int)
    # Mappa la luminosità a indici del set di caratteri
    char_indices = (brightness * (len(ASCII_CHARS) - 1) // 255).astype(int)
    ascii_chars = np.vectorize(lambda x: ASCII_CHARS[x])(char_indices)

    # Funzione per applicare il colore ANSI a un carattere
    def colorize(r, g, b, char):
        return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

    # Applica la colorizzazione in maniera vettoriale
    vectorized_colorize = np.vectorize(colorize)
    color_ascii = vectorized_colorize(
        pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], ascii_chars
    )

    # Combina le righe in una singola stringa
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


def update_frames(frame_queue, new_width, app, control, log_fps):
    """Thread di background che legge i frame, li converte in ASCII e aggiorna il display."""
    frame_count = 0
    fps_count = 0
    fps_start = time.time()
    while True:
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if frame is None:
            break
        conv_start = time.time()
        ascii_frame = frame_to_ascii(frame, new_width)
        conv_end = time.time()
        conversion_time = conv_end - conv_start

        # Aggiorna il controllo con il testo ANSI
        control.text = ANSI(ascii_frame)
        app.invalidate()  # Forza il redraw

        logging.info(f"Frame {frame_count} - Tempo conversione: {conversion_time:.4f} sec")
        frame_count += 1
        fps_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            if log_fps:
                logging.info(f"[LOG] FPS effettivi: {fps_count}")
            fps_count = 0
            fps_start = now


def main():
    parser = argparse.ArgumentParser(description="Real-time ASCII video using prompt_toolkit.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for extraction (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of actual FPS")
    args = parser.parse_args()

    frame_queue = Queue(maxsize=args.fps)
    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps))
    extractor_process.start()

    # Definisci i key bindings
    kb = KeyBindings()

    @kb.add("q")
    def exit_(event):
        event.app.exit()

    @kb.add("c-c")
    def exit_ctrl_c(event):
        event.app.exit()

    # Imposta il controllo per visualizzare il testo ANSI
    control = FormattedTextControl(text=ANSI("Loading..."))
    root_container = Window(content=control)
    layout = Layout(root_container)

    app = Application(layout=layout, key_bindings=kb, full_screen=True)

    # Avvia il thread per aggiornare i frame
    updater_thread = threading.Thread(
        target=update_frames,
        args=(frame_queue, args.width, app, control, args.log_fps),
        daemon=True
    )
    updater_thread.start()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        extractor_process.terminate()
        extractor_process.join()
        frame_queue.close()
        frame_queue.cancel_join_thread()
        print("[✔] Terminazione completata.")


if __name__ == '__main__':
    main()
