import queue
import cv2
import numpy as np
import argparse
import time
import logging
import sys
from PIL import Image
from multiprocessing import Process, Queue

# Configura il logging su file
logging.basicConfig(filename="ascii_video.log", level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')


ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@"


def frame_to_ascii(frame, new_width):
    """Converte un frame in ASCII a colori in modo più efficiente con NumPy, evitando cicli nidificati."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.5)

    image = image.resize((new_width, new_height))
    pixels = np.array(image)

    # Calcoliamo la luminosità in parallelo con NumPy
    brightness = np.mean(pixels, axis=2).astype(int)

    # Convertiamo la luminosità in caratteri ASCII in modo vettoriale
    char_indices = (brightness * (len(ASCII_CHARS) - 1) // 255).astype(int)
    ascii_chars = np.vectorize(lambda x: ASCII_CHARS[x])(char_indices)

    # Funzione vettorializzata per applicare il colore ANSI senza cicli annidati
    def colorize(r, g, b, char):
        return f"\033[38;2;{r};{g};{b}m{char}\033[0m"

    # Applichiamo i colori ANSI a tutta la matrice in un'unica operazione
    vectorized_colorize = np.vectorize(colorize)
    color_ascii = vectorized_colorize(pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], ascii_chars)

    # Combiniamo le righe in una stringa unica per l'output
    ascii_str = "\n".join("".join(row) for row in color_ascii)

    return ascii_str


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


def process_frames(frame_queue, new_width, log_fps):
    frame_count = 0  # Contatore globale dei frame
    fps_frame_count = 0  # Contatore per calcolare gli FPS
    start_time = time.time()

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if frame is None:
                break

            # Tempo di inizio conversione
            conversion_start_time = time.time()

            ascii_frame = frame_to_ascii(frame, new_width)

            # Tempo di fine conversione
            conversion_end_time = time.time()
            conversion_time = conversion_end_time - conversion_start_time

            # Tempo di inizio stampa
            print_start_time = time.time()

            # print("\033[H\033[J", end="")  # Pulisce lo schermo
            # print(ascii_frame)

            # Stampa ottimizzata
            sys.stdout.write("\033[H\033[J" + ascii_frame + "\n")
            sys.stdout.flush()  # Forza la stampa immediata senza buffering

            # Tempo di fine stampa
            print_end_time = time.time()
            print_time = print_end_time - print_start_time

            # Log del tempo di elaborazione nel file
            logging.info(f"Frame {frame_count} - Tempo conversione: {conversion_time:.4f} sec - Tempo stampa: {print_time:.4f} sec")

            # Incrementiamo il contatore dei frame totali
            frame_count += 1
            fps_frame_count += 1

            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= 1.0:  # Stampa ogni secondo
                if log_fps:
                    print(f"\033[92m[LOG] FPS EFFETTIVI: {fps_frame_count}\033[0m")
                fps_frame_count = 0  # Reset del contatore FPS
                start_time = current_time

    except KeyboardInterrupt:
        print("\n[!] Interruzione ricevuta: terminazione del processo di elaborazione frame.")


def main():
    parser = argparse.ArgumentParser(description="Convert a video to ASCII art in real-time.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("width", type=int, help="Width of the ASCII output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--log_fps", action="store_true", help="Enable logging of actual FPS")

    args = parser.parse_args()
    frame_queue = Queue(maxsize=10)

    extractor_process = Process(target=extract_frames, args=(args.video_path, frame_queue, args.fps))
    processor_process = Process(target=process_frames, args=(frame_queue, args.width, args.log_fps))

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
