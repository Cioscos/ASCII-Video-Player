import cv2
import time
import queue
import threading
import logging
import os
import sys
import numpy as np
import multiprocessing

"""
File con funzioni standalone per i processi multiprocessing
"""


def frame_reader_process(video_path, frame_queue, should_stop, target_fps, batch_size):
    """
    Funzione standalone per il processo di lettura frame.

    Args:
        video_path (str): Percorso del file video
        frame_queue (multiprocessing.Queue): Coda per i frame video
        should_stop (multiprocessing.Event): Flag per la terminazione
        target_fps (int): FPS target per l'estrazione dei frame
        batch_size (int): Numero di frame da processare in batch
    """
    # Configura logging locale per questo processo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Reader - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    logger = logging.getLogger("Reader")

    try:
        logger.info("Avvio processo di lettura frame")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Impossibile aprire il video: {video_path}")
            should_stop.set()
            return

        # Ottieni informazioni sul video
        original_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video aperto: {video_path}, FPS: {original_fps}, Frames totali: {total_frames}")

        # Calcola il ritardo necessario per rispettare il target FPS
        if target_fps and target_fps < original_fps:
            frame_delay = 1.0 / target_fps
        else:
            frame_delay = 1.0 / original_fps

        # Leggi i frame in batch
        batch = []
        last_read_time = time.time()
        frame_count = 0

        while not should_stop.is_set():
            if target_fps:
                # Rispetta il target FPS
                current_time = time.time()
                elapsed = current_time - last_read_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            # Leggi il frame
            success, frame = video.read()
            if not success:
                # Fine del video, riavvia o termina
                logger.info("Fine del video raggiunta")
                # Invia gli ultimi frame in batch se presenti
                if batch:
                    try:
                        frame_queue.put(batch, block=True, timeout=1)
                    except queue.Full:
                        pass
                # Riavvia il video
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                batch = []
                continue

            batch.append(frame)
            frame_count += 1

            # Quando il batch è completo, invialo alla coda
            if len(batch) >= batch_size:
                try:
                    frame_queue.put(batch, block=True, timeout=1)
                    batch = []
                    last_read_time = time.time()
                except queue.Full:
                    # Se la coda è piena, riduce la dimensione del batch per adattarsi
                    time.sleep(0.1)
                    reduced_batch = batch[:batch_size // 2]
                    try:
                        frame_queue.put(reduced_batch, block=False)
                        batch = batch[batch_size // 2:]
                        last_read_time = time.time()
                    except queue.Full:
                        # Se ancora piena, salta alcuni frame
                        batch = batch[1:]
                        logger.warning("Coda frame piena, saltando frame")
    except Exception as e:
        logger.error(f"Errore nel processo di lettura frame: {e}")
    finally:
        video.release()
        logger.info("Processo di lettura frame terminato")


def frame_converter_process(width, frame_queue, ascii_queue, should_stop, ascii_palette=None):
    """
    Funzione standalone per il processo di conversione frame.

    Args:
        width (int): Larghezza dell'output ASCII
        frame_queue (multiprocessing.Queue): Coda per i frame video
        ascii_queue (multiprocessing.Queue): Coda per i frame ASCII
        should_stop (multiprocessing.Event): Flag per la terminazione
        ascii_palette (str, optional): Stringa di caratteri ASCII da usare per la conversione
    """
    # Configura logging locale per questo processo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Converter - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    logger = logging.getLogger("Converter")

    try:
        logger.info(f"Avvio processo di conversione frame con larghezza={width}")

        # Caratteri ASCII ordinati per intensità (dal più scuro al più chiaro)
        # Se fornita una palette personalizzata, usala
        if ascii_palette:
            ascii_chars = np.array(list(ascii_palette))
            logger.info(f"Utilizzo palette ASCII personalizzata con {len(ascii_chars)} caratteri")
        else:
            # Set predefinito di caratteri ASCII
            ascii_chars = np.array(list("@%#*+=-:. "))
            logger.info(f"Utilizzo palette ASCII predefinita con {len(ascii_chars)} caratteri")

        # Cache per le mappature di luminosità
        luminance_cache = {}

        # Lookup table per i codici colore - precalcolata per velocizzare il rendering
        color_lookup = np.zeros((6, 6, 6), dtype=np.int16)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_lookup[r, g, b] = 16 + 36 * r + 6 * g + b

        height_scale = None
        last_shape = None

        # Fattore di correzione per le proporzioni dei caratteri ASCII
        char_aspect_correction = 2.25

        def convert_frame_to_ascii_color(frame):
            """
            Converte un singolo frame in ASCII art colorata con vettorizzazione ottimizzata.
            """
            nonlocal height_scale, last_shape

            # Ridimensiona il frame alla larghezza desiderata
            height, width_frame, _ = frame.shape

            # Calcola l'altezza proporzionale alla larghezza desiderata
            # con correzione per l'aspect ratio dei caratteri ASCII
            if height_scale is None or last_shape != (height, width_frame):
                height_scale = width / width_frame / char_aspect_correction
                last_shape = (height, width_frame)
                logger.info(
                    f"Frame originale: {width_frame}x{height}, ridimensionato a: {width}x{int(height * height_scale)}")

            new_height = int(height * height_scale)
            if new_height < 1:
                new_height = 1

            # Usa INTER_NEAREST per una conversione più veloce rispetto a INTER_LINEAR
            resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_NEAREST)

            # Estrai i canali di colore in modo più efficiente
            b = resized[:, :, 0]
            g = resized[:, :, 1]
            r = resized[:, :, 2]

            # Calcola la luminosità (scala di grigi) per scegliere il carattere
            # Y = 0.299R + 0.587G + 0.114B (pesi ottimali per la percezione umana)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Normalizza i valori di grigio e mappali all'indice nell'array ascii_chars
            # Ottimizzazione: manteniamo tutto in array numpy il più a lungo possibile
            indices = (gray / 255 * (len(ascii_chars) - 1)).astype(np.int_)

            # Precalcola gli indici di colore per migliorare le prestazioni
            r_idx = np.minimum(5, r // 43).astype(np.int_)  # 255 / 6 ≈ 43
            g_idx = np.minimum(5, g // 43).astype(np.int_)
            b_idx = np.minimum(5, b // 43).astype(np.int_)

            # Crea le stringhe ASCII con codici colore ANSI
            rows = []
            for y in range(new_height):
                row = []
                for x in range(width):
                    # Ottieni il carattere ASCII in base alla luminosità
                    char = ascii_chars[indices[y, x]]

                    # Usa la lookup table per il codice colore (più veloce)
                    color_code = color_lookup[r_idx[y, x], g_idx[y, x], b_idx[y, x]]

                    # Sequenza ANSI per impostare il colore
                    colored_char = f"\033[38;5;{color_code}m{char}"
                    row.append(colored_char)

                # Resetta il colore alla fine della riga
                rows.append(''.join(row) + "\033[0m")

            return '\n'.join(rows)

        while not should_stop.is_set():
            try:
                # Ottieni un batch di frame dalla coda
                batch = frame_queue.get(block=True, timeout=1)

                # Misura il tempo di conversione
                start_time = time.time()

                # Converti i frame in ASCII colorati
                ascii_frames = [convert_frame_to_ascii_color(frame) for frame in batch]

                conversion_time = time.time() - start_time
                logger.debug(f"Tempo di conversione batch: {conversion_time:.4f}s")

                # Invia i frame ASCII alla coda di rendering
                try:
                    ascii_queue.put(ascii_frames, block=True, timeout=1)
                except queue.Full:
                    # Se la coda è piena, riduci il batch
                    if len(ascii_frames) > 1:
                        try:
                            ascii_queue.put(ascii_frames[:len(ascii_frames) // 2], block=False)
                            logger.warning("Coda ASCII piena, ridotto batch")
                        except queue.Full:
                            logger.warning("Impossibile inviare frame ASCII, coda piena")
                    else:
                        logger.warning("Impossibile inviare frame ASCII, coda piena")

            except queue.Empty:
                # Nessun frame disponibile, aspetta
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Errore nel processo di conversione: {e}")
    except Exception as e:
        logger.error(f"Errore generale nel processo di conversione: {e}")
    finally:
        logger.info("Processo di conversione frame terminato")


class VideoPipeline:
    """
    Classe che gestisce la pipeline parallela per la lettura, conversione e rendering dei frame video.

    Attributes:
        video_path (str): Percorso del file video
        width (int): Larghezza dell'output ASCII
        target_fps (int): FPS target per l'estrazione dei frame
        batch_size (int): Numero di frame da processare in batch
        log_performance (bool): Se True, registra le informazioni sulle prestazioni
        log_fps (bool): Se True, registra le informazioni sugli FPS
    """

    def __init__(self, video_path, width, target_fps=None, batch_size=1,
                 log_performance=False, log_fps=False, ascii_palette=None):
        """
        Inizializza la pipeline video.

        Args:
            video_path (str): Percorso del file video
            width (int): Larghezza dell'output ASCII
            target_fps (int, optional): FPS target per l'estrazione dei frame.
                                        Se None, estrae alla massima velocità possibile.
            batch_size (int): Numero di frame da processare in batch
            log_performance (bool): Se True, registra le informazioni sulle prestazioni
            log_fps (bool): Se True, registra le informazioni sugli FPS
            ascii_palette (str, optional): Stringa di caratteri ASCII da usare per la conversione
        """
        self.video_path = video_path
        self.width = width
        self.target_fps = target_fps
        self.batch_size = batch_size
        self.log_performance = log_performance
        self.log_fps = log_fps
        self.ascii_palette = ascii_palette

        # Coda di comunicazione tra processi
        # Usiamo un maxsize per evitare di sovraccaricare la memoria
        queue_size = max(10, batch_size * 3)
        self.frame_queue = multiprocessing.Queue(maxsize=queue_size)
        self.ascii_queue = multiprocessing.Queue(maxsize=queue_size)

        # Flag per la terminazione
        self.should_stop = multiprocessing.Event()

        # Statistiche di performance
        self.frame_times = []

        # Logger
        self.logger = logging.getLogger('VideoPipeline')
        self.logger.info(
            f"Inizializzata pipeline con video={video_path}, width={width}, batch_size={batch_size}, fps={target_fps}")

        # Attributi per i processi
        self.reader_process = None
        self.converter_process = None
        self.renderer_thread = None

    def _frame_renderer_thread(self):
        """
        Thread che renderizza i frame ASCII sul terminale.
        """
        self.logger.info("Avvio thread di rendering frame")

        # Determina il metodo di pulizia dello schermo in base al sistema operativo
        if os.name == 'nt':  # Windows
            clear_command = 'cls'
        else:  # Unix/Linux/MacOS
            clear_command = 'clear'

        last_render_time = None
        frame_times = []

        try:
            while not self.should_stop.is_set():
                try:
                    # Ottieni un batch di frame ASCII dalla coda
                    ascii_frames = self.ascii_queue.get(block=True, timeout=1)

                    # Renderizza ciascun frame
                    for ascii_frame in ascii_frames:
                        if self.should_stop.is_set():
                            break

                        now = time.time()

                        # Calcola il tempo trascorso dall'ultimo rendering
                        if last_render_time is not None:
                            elapsed = now - last_render_time
                            # Se abbiamo un target FPS, aspetta il tempo necessario
                            if self.target_fps and elapsed < 1.0 / self.target_fps:
                                time.sleep(1.0 / self.target_fps - elapsed)
                                now = time.time()

                        # Pulisci lo schermo
                        os.system(clear_command)

                        # Stampa il frame usando sys.stdout.write invece di print
                        sys.stdout.write(ascii_frame)
                        sys.stdout.flush()

                        render_time = time.time() - now
                        last_render_time = now

                        # Registra le informazioni sugli FPS
                        if self.log_fps:
                            frame_times.append(render_time)
                            if len(frame_times) > 100:
                                frame_times.pop(0)

                except queue.Empty:
                    # Nessun frame disponibile, aspetta
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Errore nel thread di rendering: {e}")
        finally:
            # Resetta il colore del terminale alla fine
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()

            # Salva le statistiche FPS per essere usate dal metodo stop()
            self.frame_times = frame_times
            self.logger.info("Thread di rendering frame terminato")

    def start(self):
        """
        Avvia la pipeline video.
        """
        self.logger.info("Avvio pipeline video")

        # Avvia il processo di lettura frame
        self.reader_process = multiprocessing.Process(
            target=frame_reader_process,
            args=(self.video_path, self.frame_queue, self.should_stop, self.target_fps, self.batch_size),
            daemon=True
        )
        self.reader_process.start()

        # Avvia il processo di conversione frame
        self.converter_process = multiprocessing.Process(
            target=frame_converter_process,
            args=(self.width, self.frame_queue, self.ascii_queue, self.should_stop, self.ascii_palette),
            daemon=True
        )
        self.converter_process.start()

        # Avvia il thread di rendering (nel processo principale)
        self.renderer_thread = threading.Thread(
            target=self._frame_renderer_thread,
            daemon=True
        )
        self.renderer_thread.start()

    def stop(self):
        """
        Ferma la pipeline video.
        """
        self.logger.info("Arresto pipeline video")
        self.should_stop.set()

        # Attendi la terminazione dei processi
        if hasattr(self, 'reader_process') and self.reader_process and self.reader_process.is_alive():
            self.reader_process.join(timeout=2)
            if self.reader_process.is_alive():
                self.reader_process.terminate()

        if hasattr(self, 'converter_process') and self.converter_process and self.converter_process.is_alive():
            self.converter_process.join(timeout=2)
            if self.converter_process.is_alive():
                self.converter_process.terminate()

        # Attendi la terminazione del thread
        if hasattr(self, 'renderer_thread') and self.renderer_thread and self.renderer_thread.is_alive():
            self.renderer_thread.join(timeout=2)

        # Stampa le statistiche FPS
        if self.log_fps and self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            min_time = min(self.frame_times)
            max_time = max(self.frame_times)

            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            min_fps = 1.0 / max_time if max_time > 0 else 0
            max_fps = 1.0 / min_time if min_time > 0 else 0

            self.logger.info(f"FPS medio: {avg_fps:.2f}, Min: {min_fps:.2f}, Max: {max_fps:.2f}")
