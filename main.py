import cv2
import numpy as np
import argparse
import time
import signal
from PIL import Image
from multiprocessing import Process, Queue


ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"


def rgb_to_ansi(r, g, b):
    """ Restituisce un codice ANSI per i colori """
    return f"\033[38;2;{r};{g};{b}m"


def frame_to_ascii(frame, new_width):
    """ Converte un frame OpenCV in ASCII colorato """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    image = image.resize((new_width, new_height))
    pixels = np.array(image)

    ascii_frame = ""
    for row in pixels:
        for r, g, b in row:
            r, g, b = int(r), int(g), int(b)
            brightness = (r + g + b) // 3
            char = ASCII_CHARS[brightness * (len(ASCII_CHARS) - 1) // 255]
            ascii_frame += f"{rgb_to_ansi(r, g, b)}{char}"
        ascii_frame += "\033[0m\n"

    return ascii_frame


def extract_frames(video_path, frame_queue, fps):
    """ Estrae i frame dal video e li mette nella coda """
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
        print("\n[!] Interruzione ricevuta: terminazione del processo di estrazione frame.")
    finally:
        cap.release()
        frame_queue.put(None)  # Segnala la fine


def process_frames(frame_queue, new_width):
    """ Prende i frame dalla coda e li converte in ASCII """
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)  # Previene blocco infinito
            except queue.Empty:
                continue

            if frame is None:
                break

            ascii_frame = frame_to_ascii(frame, new_width)
            print("\033[H\033[J", end="")  # Pulisce lo schermo
            print(ascii_frame)
    except KeyboardInterrupt:
        print("\n[!] Interruzione ricevuta: terminazione del processo di elaborazione frame.")


def main():
    parser = argparse.ArgumentParser(description="Convert a video to ASCII art in real-time.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    args = parser.parse_args()
    frame_queue = Queue(maxsize=10)

    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps))
    processor_process = Process(target=process_frames, args=(frame_queue, args.width))

    extractor_process.start()
    processor_process.start()

    try:
        # Invece di join() bloccanti, controlliamo periodicamente lo stato dei processi.
        while extractor_process.is_alive() or processor_process.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[!] Interruzione rilevata! Terminazione in corso...")
        # Termina forzatamente i processi
        extractor_process.terminate()
        processor_process.terminate()
    finally:
        extractor_process.join()
        processor_process.join()
        frame_queue.close()
        frame_queue.cancel_join_thread()
        print("[✔] Terminazione completata.")

if __name__ == '__main__':
    main()
