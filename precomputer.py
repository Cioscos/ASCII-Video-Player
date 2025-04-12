import os
import sys

import cv2
import pickle
import logging
import threading
import time
import queue
import numpy as np
from tqdm import tqdm


class FramePrecomputer:
    """
    Classe che gestisce la pre-elaborazione e il caching dei frame video.

    Estrae tutti i frame dal video, li converte in ASCII e li salva su disco
    per ottimizzare le performance durante la riproduzione.

    Attributes:
        video_path (str): Percorso del file video
        width (int): Larghezza dell'output ASCII
        target_fps (float): FPS target per l'estrazione dei frame
        ascii_palette (str): Caratteri ASCII da utilizzare per la conversione
        output_dir (str): Cartella dove salvare i frame elaborati
        cache_file (str): Percorso del file di cache con i metadati
        logger (logging.Logger): Logger per la registrazione delle operazioni
    """

    def __init__(self, video_path, width, target_fps=None, ascii_palette=None,
                 output_dir=None, batch_size=10):
        """
        Inizializza il precomputer per generare e salvare i frame ASCII.

        Args:
            video_path (str): Percorso del file video da processare
            width (int): Larghezza dell'output ASCII
            target_fps (float, optional): FPS target per l'estrazione. Se None, usa l'FPS originale
            ascii_palette (str, optional): Caratteri da usare per la conversione ASCII
            output_dir (str, optional): Directory per salvare i frame. Se None, usa './cached_frames'
            batch_size (int): Numero di frame da processare in parallelo
        """
        self.video_path = video_path
        self.width = width
        self.target_fps = target_fps
        self.ascii_palette = ascii_palette
        self.batch_size = batch_size

        # Crea un nome univoco basato sul file e sui parametri
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        cache_name = f"{video_name}_w{width}_fps{target_fps or 'orig'}"

        # Directory per la cache
        self.output_dir = output_dir or os.path.join("cached_frames", cache_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # File di metadati
        self.cache_file = os.path.join(self.output_dir, "metadata.pkl")

        # Setup logging
        self.logger = logging.getLogger('FramePrecomputer')
        self.metadata = {}

    def _extract_video_info(self):
        """
        Estrae le informazioni di base dal video come FPS, durata, dimensioni.

        Returns:
            dict: Dizionario con le informazioni del video
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.logger.error(f"Impossibile aprire il video: {self.video_path}")
            return None

        # Estrai informazioni di base
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calcola durata
        duration = frame_count / fps if fps > 0 else 0

        # FPS target effettivo
        target_fps = self.target_fps or fps

        # Numero effettivo di frame da estrarre
        if self.target_fps and self.target_fps < fps:
            # Se il target FPS è inferiore, estraiamo meno frame
            frame_step = fps / target_fps
            actual_frames = int(frame_count / frame_step)
        else:
            frame_step = 1
            actual_frames = frame_count

        cap.release()

        return {
            "original_fps": fps,
            "target_fps": target_fps,
            "frame_count": frame_count,
            "actual_frames": actual_frames,
            "frame_step": frame_step,
            "width": width,
            "height": height,
            "duration": duration,
            "ascii_width": self.width,
            "video_path": self.video_path
        }

    def _convert_frame_to_ascii(self, frame):
        """
        Converte un singolo frame in rappresentazione ASCII.

        Args:
            frame (numpy.ndarray): Frame video da convertire

        Returns:
            str: Rappresentazione ASCII del frame
        """
        height, width_frame, _ = frame.shape

        # Calcola l'altezza proporzionale
        char_aspect_correction = 2.25
        aspect_ratio = height / width_frame
        new_height = int(aspect_ratio * self.width / char_aspect_correction)

        # Ridimensiona il frame
        resized = cv2.resize(frame, (self.width, new_height), interpolation=cv2.INTER_NEAREST)

        # Converti in scala di grigi
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Estrai canali di colore
        b = resized[:, :, 0]
        g = resized[:, :, 1]
        r = resized[:, :, 2]

        # Calcola indici colore
        r_idx = np.minimum(5, r // 43).astype(np.int_)
        g_idx = np.minimum(5, g // 43).astype(np.int_)
        b_idx = np.minimum(5, b // 43).astype(np.int_)

        # Prepara la palette ASCII
        if self.ascii_palette and self.ascii_palette != 'box':
            ascii_chars = np.array(list(self.ascii_palette))
        else:
            # Usa blocchi Unicode se scelto 'box'
            if self.ascii_palette == 'box':
                return self._convert_frame_to_box_ascii(frame)
            # Altrimenti usa la palette di default
            ascii_chars = np.array(list(" .:+*=#%@"))

        # Normalizza i valori di grigio per la palette
        indices = (gray / 255.0 * (len(ascii_chars) - 1)).astype(np.int_)

        # Pre-calcola la lookup table colori
        color_lookup = np.zeros((6, 6, 6), dtype=np.int16)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_lookup[r, g, b] = 16 + 36 * r + 6 * g + b

        # Cache delle sequenze ANSI
        color_sequences = {}
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_code = color_lookup[r, g, b]
                    color_sequences[color_code] = f"\033[38;5;{color_code}m"

        RESET_SEQ = "\033[0m"

        # Costruisci le righe
        rows = []
        for y in range(new_height):
            row_chars = []
            for x in range(self.width):
                char = ascii_chars[indices[y, x]]
                color_code = color_lookup[r_idx[y, x], g_idx[y, x], b_idx[y, x]]
                row_chars.append(color_sequences[color_code] + char)
            rows.append(''.join(row_chars) + RESET_SEQ)

        return '\n'.join(rows)

    def _convert_frame_to_box_ascii(self, frame):
        """
        Converte un frame in blocchi colorati Unicode.

        Args:
            frame (numpy.ndarray): Frame video da convertire

        Returns:
            str: Rappresentazione con blocchi Unicode colorati
        """
        height, width_frame, _ = frame.shape

        # Calcola l'altezza proporzionale
        char_aspect_correction = 2.25
        aspect_ratio = height / width_frame
        new_height = int(aspect_ratio * self.width / char_aspect_correction)

        # Ridimensiona il frame
        resized = cv2.resize(frame, (self.width, new_height), interpolation=cv2.INTER_NEAREST)

        # Estrai canali di colore
        b = resized[:, :, 0]
        g = resized[:, :, 1]
        r = resized[:, :, 2]

        # Calcola indici colore
        r_idx = np.minimum(5, r // 43).astype(np.int_)
        g_idx = np.minimum(5, g // 43).astype(np.int_)
        b_idx = np.minimum(5, b // 43).astype(np.int_)

        # Pre-calcola la lookup table colori
        color_lookup = np.zeros((6, 6, 6), dtype=np.int16)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_lookup[r, g, b] = 16 + 36 * r + 6 * g + b

        # Cache delle sequenze ANSI
        color_sequences = {}
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_code = color_lookup[r, g, b]
                    color_sequences[color_code] = f"\033[38;5;{color_code}m"

        RESET_SEQ = "\033[0m"
        block_char = '█'

        # Costruisci le righe
        rows = []
        for y in range(new_height):
            row_chars = []
            for x in range(self.width):
                color_code = color_lookup[r_idx[y, x], g_idx[y, x], b_idx[y, x]]
                row_chars.append(color_sequences[color_code] + block_char)
            rows.append(''.join(row_chars) + RESET_SEQ)

        return '\n'.join(rows)

    def precompute_frames(self):
        """
        Estrae, converte e salva tutti i frame del video.

        Utilizza un approccio parallelizzato per l'elaborazione dei frame
        e mostra una barra di avanzamento.

        Returns:
            dict: Metadati del processo di precomputing
        """
        # Verifica se la cache esiste già
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.logger.info(f"Caricati metadati dalla cache esistente: {self.output_dir}")
                return self.metadata
            except Exception as e:
                self.logger.warning(f"Impossibile caricare la cache esistente: {e}")

        # Estrai informazioni video
        video_info = self._extract_video_info()
        if not video_info:
            return None

        self.logger.info(f"Avvio pre-elaborazione di {video_info['actual_frames']} frame...")

        # Apri il video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.logger.error(f"Impossibile aprire il video: {self.video_path}")
            return None

        # Prepara metadati
        self.metadata = {
            **video_info,
            "frames_dir": self.output_dir,
            "processed_frames": 0,
            "completed": False,
            "creation_time": time.time()
        }

        # Coda per l'elaborazione parallela
        frame_queue = queue.Queue(maxsize=self.batch_size * 2)
        result_queue = queue.Queue()

        # Flag per il completamento
        processing_complete = threading.Event()

        # Thread worker per la conversione
        def worker():
            while not processing_complete.is_set() or not frame_queue.empty():
                try:
                    frame_data = frame_queue.get(timeout=0.5)
                    if frame_data is None:
                        break

                    frame_index, frame = frame_data
                    ascii_frame = self._convert_frame_to_ascii(frame)

                    # Salva il frame
                    frame_file = os.path.join(self.output_dir, f"frame_{frame_index:06d}.txt")
                    with open(frame_file, 'w', encoding='utf-8') as f:
                        f.write(ascii_frame)

                    result_queue.put(frame_index)
                    frame_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Errore nell'elaborazione del frame: {e}")
                    frame_queue.task_done()

        # Avvia i worker threads
        num_workers = min(os.cpu_count(), 4)  # Limita il numero di thread
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            workers.append(t)

        try:
            # Setup barra di progresso
            with tqdm(total=video_info["actual_frames"], desc="Pre-elaborazione frame") as pbar:
                frame_idx = 0
                processed_count = 0

                # Loop principale di estrazione
                while processed_count < video_info["actual_frames"] and cap.isOpened():
                    # Leggi il frame
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Se abbiamo un frame step, verifica se questo frame va processato
                    if frame_idx % video_info["frame_step"] < 1:
                        # Aggiungi il frame alla coda
                        frame_queue.put((processed_count, frame))
                        processed_count += 1

                    frame_idx += 1

                    # Aggiorna la barra di progresso con i frame elaborati
                    while not result_queue.empty():
                        result_queue.get()
                        pbar.update(1)

            # Aspetta che tutti i frame siano processati
            processing_complete.set()
            for t in workers:
                t.join(timeout=1.0)

            # Aggiorna i metadati
            self.metadata["processed_frames"] = processed_count
            self.metadata["completed"] = True

            # Salva i metadati
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.metadata, f)

            self.logger.info(f"Pre-elaborazione completata: {processed_count} frame salvati in {self.output_dir}")
            return self.metadata

        except Exception as e:
            self.logger.error(f"Errore durante la pre-elaborazione: {e}")
            return None
        finally:
            cap.release()

    def is_cache_valid(self):
        """
        Verifica se esiste una cache valida per questo video.

        Returns:
            bool: True se la cache è valida, False altrimenti
        """
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, 'rb') as f:
                metadata = pickle.load(f)

            # Verifica base: la cache è completa?
            if not metadata.get("completed", False):
                return False

            # Verifica che i parametri corrispondano
            if (metadata.get("video_path") != self.video_path or
                    metadata.get("ascii_width") != self.width or
                    (self.target_fps and metadata.get("target_fps") != self.target_fps)):
                return False

            # Verifica che esistano i file dei frame
            frame_count = metadata.get("processed_frames", 0)
            if frame_count <= 0:
                return False

            # Controlla alcuni frame a campione
            sample_indices = [0, frame_count // 2, frame_count - 1]
            for idx in sample_indices:
                frame_file = os.path.join(self.output_dir, f"frame_{idx:06d}.txt")
                if not os.path.exists(frame_file):
                    return False

            self.metadata = metadata
            return True
        except Exception:
            return False

class PrecomputedFramePlayer:
    """
    Classe per la riproduzione di frame ASCII precomputati.

    Carica e visualizza i frame ASCII pre-elaborati dal disco
    con sincronizzazione temporale precisa.

    Attributes:
        metadata (dict): Metadati dei frame precomputati
        target_fps (float): FPS target per la riproduzione
        loop_video (bool): Se True, riavvia il video quando raggiunge la fine
        should_stop (threading.Event): Flag per la terminazione della riproduzione
        current_frame (int): Indice del frame corrente
        audio_player (AudioPlayer): Player per la riproduzione sincronizzata dell'audio
    """

    def __init__(self, metadata, target_fps=None, loop_video=True,
                 enable_audio=False, log_fps=False):
        """
        Inizializza il player dei frame precomputati.

        Args:
            metadata (dict): Metadati della cache dei frame
            target_fps (float, optional): FPS target per la riproduzione.
                                         Se None, usa l'FPS dai metadati
            loop_video (bool): Se True, riavvia il video quando raggiunge la fine
            enable_audio (bool): Se True, riproduce l'audio del video
            log_fps (bool): Se True, registra le statistiche sugli FPS
        """
        self.metadata = metadata
        self.frames_dir = metadata["frames_dir"]
        self.target_fps = target_fps or metadata["target_fps"]
        self.total_frames = metadata["processed_frames"]
        self.loop_video = loop_video
        self.enable_audio = enable_audio
        self.log_fps = log_fps

        # Flag di controllo
        self.should_stop = threading.Event()
        self.current_frame = 0

        # Inizializzazione audio se abilitato
        self.audio_player = None
        if enable_audio:
            try:
                from audio_player import AudioPlayer
                self.audio_player = AudioPlayer(metadata["video_path"], self.target_fps)
                self.video_duration = metadata["duration"]
            except ImportError as e:
                logging.error(f"Impossibile inizializzare l'audio: {e}")
                self.enable_audio = False

        # Statistiche FPS
        self.frame_times = []
        self.logger = logging.getLogger('PrecomputedFramePlayer')

    def start(self):
        """
        Avvia la riproduzione dei frame precomputati.

        Inizializza l'audio (se abilitato) e avvia il thread di riproduzione.

        Returns:
            bool: True se l'avvio è avvenuto con successo, False altrimenti
        """
        self.logger.info(f"Avvio riproduzione di {self.total_frames} frame precomputati")

        # Inizializza audio
        if self.enable_audio and self.audio_player:
            if self.audio_player.initialize():
                self.audio_player.start()
                self.logger.info("Riproduzione audio avviata")
            else:
                self.logger.warning("Inizializzazione audio fallita")
                self.enable_audio = False

        # Avvia il thread di riproduzione
        self.playback_thread = threading.Thread(
            target=self._playback_thread_func,
            daemon=True
        )
        self.playback_thread.start()
        return True

    def _playback_thread_func(self):
        """
        Thread principale di riproduzione dei frame.

        Legge i frame precomputati dal disco e li visualizza
        rispettando il timing e gestendo la sincronizzazione audio.
        """
        from terminal_output_buffer import TerminalOutputBuffer
        import sys
        import time

        output_buffer = TerminalOutputBuffer(sys.stdout)

        # Sequenze ANSI
        CURSOR_HOME = '\033[H'
        CLEAR_SCREEN = '\033[2J'

        # Statistiche FPS
        last_frame_time = None
        frame_times = []

        try:
            # Pulisci lo schermo all'inizio
            output_buffer.write(CLEAR_SCREEN)
            output_buffer.flush()

            # Loop di riproduzione
            while not self.should_stop.is_set():
                frame_start_time = time.time()

                # Carica il frame corrente
                frame_file = os.path.join(self.frames_dir, f"frame_{self.current_frame:06d}.txt")

                try:
                    with open(frame_file, 'r', encoding='utf-8') as f:
                        ascii_frame = f.read()
                except FileNotFoundError:
                    self.logger.warning(f"Frame {self.current_frame} non trovato: {frame_file}")
                    # Passa al frame successivo
                    self.current_frame = (self.current_frame + 1) % self.total_frames
                    if self.current_frame == 0 and not self.loop_video:
                        break
                    continue

                # Calcola timing
                frame_duration = 1.0 / self.target_fps

                # Sincronizzazione audio
                if self.enable_audio and self.audio_player:
                    video_time = (self.current_frame / self.total_frames) * self.video_duration
                    self.audio_player.update_video_time(video_time)

                # Rendering frame
                output_buffer.write(CURSOR_HOME)
                output_buffer.write(ascii_frame)

                # Aggiungi statistiche FPS se richiesto
                if self.log_fps and last_frame_time is not None:
                    fps = 1.0 / (frame_start_time - last_frame_time)
                    frame_times.append(frame_start_time - last_frame_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    output_buffer.write(f"\n\nFPS: {avg_fps:.1f}")

                output_buffer.flush()

                # Aggiorna timestamp per statistiche
                last_frame_time = frame_start_time

                # Calcola tempo di elaborazione e attendi se necessario
                process_time = time.time() - frame_start_time
                sleep_time = frame_duration - process_time

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Passa al frame successivo
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    if not self.loop_video:
                        break
                    self.current_frame = 0

            # Alla fine, salva le statistiche FPS
            self.frame_times = frame_times

        except Exception as e:
            self.logger.error(f"Errore durante la riproduzione: {e}")
        finally:
            self.logger.info("Thread di riproduzione terminato")

    def stop(self):
        """
        Ferma la riproduzione e rilascia le risorse.

        Implementa una chiusura pulita di tutti i componenti del player.
        """
        self.logger.info("Arresto riproduzione")

        # Imposta flag di terminazione
        self.should_stop.set()

        # Ferma l'audio
        if self.enable_audio and hasattr(self, 'audio_player') and self.audio_player:
            try:
                self.audio_player.stop()
            except Exception as e:
                self.logger.error(f"Errore durante l'arresto dell'audio: {e}")

        # Attendi la terminazione del thread
        if hasattr(self, 'playback_thread') and self.playback_thread:
            self.playback_thread.join(timeout=0.5)

        # Stampa statistiche FPS
        if self.log_fps and self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            self.logger.info(f"FPS medio: {avg_fps:.2f}")

        # Mostra il cursore
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
