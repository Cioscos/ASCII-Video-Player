import cv2
import time
import queue
import threading
import logging
import multiprocessing

# Marker per segnalare la fine del video
END_OF_VIDEO_MARKER = "END_OF_VIDEO"


def frame_reader_process(video_path, frame_queue, should_stop, target_fps, batch_size, loop_video=True):
    """
    Processo dedicato alla lettura dei frame dal video.

    Estrae frame dal video a una velocità controllata e li invia alla coda
    per la conversione. Gestisce anche il loop del video se richiesto.

    Args:
        video_path (str): Percorso del file video
        frame_queue (multiprocessing.Queue): Coda per i frame video
        should_stop (multiprocessing.Event): Flag per la terminazione
        target_fps (int): FPS target per l'estrazione dei frame
        batch_size (int): Numero di frame da processare in batch
        loop_video (bool): Se True, riavvia il video quando raggiunge la fine
    """
    # Configura logging locale per questo processo
    from utils import configure_process_logging
    logger = configure_process_logging("Reader", console_level=logging.WARNING)

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
                # Fine del video
                logger.info("Fine del video raggiunta")

                # Invia gli ultimi frame in batch se presenti
                if batch:
                    try:
                        frame_queue.put(batch, block=True, timeout=1)
                    except queue.Full:
                        pass

                # Se loop_video è False, invia il marker di fine video e termina
                if not loop_video:
                    logger.info("Invio marker di fine video")
                    try:
                        frame_queue.put(END_OF_VIDEO_MARKER, block=True, timeout=1)
                    except queue.Full:
                        pass
                    logger.info("Loop disabilitato, terminazione del processo di lettura")
                    break

                # Altrimenti riavvia il video
                logger.info("Riavvio del video...")
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
                    # Gestione adattiva di code piene
                    time.sleep(0.1)
                    reduced_batch = batch[:batch_size // 2]
                    try:
                        frame_queue.put(reduced_batch, block=False)
                        batch = batch[batch_size // 2:]
                        last_read_time = time.time()
                    except queue.Full:
                        # Salta alcuni frame se la coda è ancora piena
                        batch = batch[1:]
                        logger.warning("Coda frame piena, saltando frame")
    except Exception as e:
        logger.error(f"Errore nel processo di lettura frame: {e}")
    finally:
        video.release()
        logger.info("Processo di lettura frame terminato")


def frame_converter_process(width, frame_queue, ascii_queue, should_stop, ascii_palette=None):
    """
    Processo dedicato alla conversione dei frame in ASCII.

    Implementa ottimizzazioni per l'elaborazione di batch e utilizza NumPy per
    conversioni vettorizzate ad alte prestazioni.

    Args:
        width (int): Larghezza dell'output ASCII
        frame_queue (multiprocessing.Queue): Coda per i frame video
        ascii_queue (multiprocessing.Queue): Coda per i frame ASCII convertiti
        should_stop (multiprocessing.Event): Flag per la terminazione
        ascii_palette (str, optional): Stringa di caratteri ASCII da usare per la conversione
    """
    # Configura logging locale per questo processo
    from utils import configure_process_logging
    import cv2
    import time
    import queue
    import numpy as np

    logger = configure_process_logging("Converter", console_level=logging.WARNING)

    box_palette = False

    try:
        logger.info(f"Avvio processo di conversione frame con larghezza={width}")

        # Prepara la palette di caratteri ASCII
        if ascii_palette:
            if ascii_palette == 'box':
                box_palette = True
            else:
                ascii_chars = np.array(list(ascii_palette))
                logger.info(f"Utilizzo palette ASCII personalizzata con {len(ascii_chars)} caratteri")
        else:
            # Set predefinito di caratteri ASCII (spazio è il più scuro)
            ascii_chars = np.array(list(" .:+*=#%@"))
            logger.info(f"Utilizzo palette ASCII predefinita con {len(ascii_chars)} caratteri")

        # Pre-calcolo della lookup table colori (ottimizzazione)
        color_lookup = np.zeros((6, 6, 6), dtype=np.int16)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_lookup[r, g, b] = 16 + 36 * r + 6 * g + b

        # Pre-calcolo delle sequenze ANSI per ogni codice colore
        color_sequences = {}
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    color_code = color_lookup[r, g, b]
                    color_sequences[color_code] = f"\033[38;5;{color_code}m"

        # Cache per sequenze ANSI comuni
        RESET_SEQ = "\033[0m"

        height_scale = None
        last_shape = None

        # Fattore di correzione per le proporzioni dei caratteri ASCII
        char_aspect_correction = 2.25

        def convert_frame_to_ascii_color(frame):
            """
            Converte un frame in ASCII con colori.

            Versione ottimizzata con allocazioni di memoria minime e operazioni vettorizzate.

            Args:
                frame (numpy.ndarray): Frame video da convertire

            Returns:
                str: Rappresentazione ASCII colorata del frame
            """
            nonlocal height_scale, last_shape

            # Ridimensiona il frame alla larghezza desiderata
            height, width_frame, _ = frame.shape

            # Calcola l'altezza proporzionale con correzione aspect ratio
            if height_scale is None or last_shape != (height, width_frame):
                height_scale = width / width_frame / char_aspect_correction
                last_shape = (height, width_frame)
                logger.info(
                    f"Frame originale: {width_frame}x{height}, ridimensionato a: {width}x{int(height * height_scale)}")

            new_height = int(height * height_scale)
            if new_height < 1:
                new_height = 1

            # Usa INTER_NEAREST per velocità e meno artefatti in ASCII
            resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_NEAREST)

            # Calcola direttamente la luminosità con cvtColor
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Estrai canali di colore come viste
            b = resized[:, :, 0]
            g = resized[:, :, 1]
            r = resized[:, :, 2]

            # Calcola gli indici di colore in un'unica operazione vettorizzata
            r_idx = np.minimum(5, r // 43).astype(np.int_)  # 255 / 6 ≈ 43
            g_idx = np.minimum(5, g // 43).astype(np.int_)
            b_idx = np.minimum(5, b // 43).astype(np.int_)

            # Normalizza i valori di grigio per la mappatura dei caratteri
            indices = (gray / 255.0 * (len(ascii_chars) - 1)).astype(np.int_)

            # Costruisci le righe con liste e join (efficiente)
            rows = []
            for y in range(new_height):
                row_chars = []
                for x in range(width):
                    # Ottieni il carattere ASCII in base alla luminosità
                    char = ascii_chars[indices[y, x]]

                    # Usa i valori precalcolati per il codice colore
                    color_code = color_lookup[r_idx[y, x], g_idx[y, x], b_idx[y, x]]

                    # Usa le sequenze colore precalcolate
                    row_chars.append(color_sequences[color_code] + char)

                # Unisci l'intera riga con reset alla fine
                rows.append(''.join(row_chars) + RESET_SEQ)

            # Unisci tutte le righe in un'unica stringa
            return '\n'.join(rows)

        def convert_frame_to_ascii_color_blocks(frame):
            """
            Converte un frame in blocchi colorati Unicode.

            Versione specifica per la modalità 'box' che usa caratteri blocco.

            Args:
                frame (numpy.ndarray): Frame video da convertire

            Returns:
                str: Rappresentazione con blocchi Unicode colorati
            """
            nonlocal height_scale, last_shape

            # Ridimensiona il frame alla larghezza desiderata
            height, width_frame, _ = frame.shape

            # Calcola l'altezza proporzionale
            if height_scale is None or last_shape != (height, width_frame):
                height_scale = width / width_frame / char_aspect_correction
                last_shape = (height, width_frame)
                logger.info(
                    f"Frame originale: {width_frame}x{height}, ridimensionato a: {width}x{int(height * height_scale)}")

            new_height = int(height * height_scale)
            if new_height < 1:
                new_height = 1

            # Resize ottimizzato per blocchi
            resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_NEAREST)

            # Estrai i canali come viste
            b = resized[:, :, 0]
            g = resized[:, :, 1]
            r = resized[:, :, 2]

            # Calcola indici colore in operazione vettorizzata
            r_idx = np.minimum(5, r // 43).astype(np.int_)
            g_idx = np.minimum(5, g // 43).astype(np.int_)
            b_idx = np.minimum(5, b // 43).astype(np.int_)

            # Carattere blocco Unicode
            block_char = '█'

            # Costruzione efficiente dell'output
            rows = []
            for y in range(new_height):
                row_chars = []
                for x in range(width):
                    # Accedi direttamente alla lookup table
                    color_code = color_lookup[r_idx[y, x], g_idx[y, x], b_idx[y, x]]
                    row_chars.append(color_sequences[color_code] + block_char)

                # Unisci con reset
                rows.append(''.join(row_chars) + RESET_SEQ)

            # Output finale
            return '\n'.join(rows)

        # Tracciamento performance
        conversion_times = []
        max_times_to_track = 50  # Campioni per media mobile

        # Loop principale di conversione
        while not should_stop.is_set():
            try:
                # Ottieni un batch di frame dalla coda
                batch = frame_queue.get(block=True, timeout=1)

                # Controlla se è il marker di fine video
                if batch == END_OF_VIDEO_MARKER:
                    logger.info("Ricevuto marker di fine video")
                    # Passa il marker al renderer
                    ascii_queue.put(END_OF_VIDEO_MARKER, block=True, timeout=1)
                    break

                # Misura il tempo di conversione
                start_time = time.time()

                # Seleziona funzione di conversione appropriata
                convert_function = convert_frame_to_ascii_color_blocks if box_palette else convert_frame_to_ascii_color

                # Gestione adattiva di carico basata sullo stato della coda
                queue_ratio = ascii_queue.qsize() / ascii_queue._maxsize if hasattr(ascii_queue,
                                                                                    '_maxsize') and ascii_queue._maxsize else 0

                if queue_ratio > 0.8 and len(batch) > 2:
                    # Converti solo parte del batch se la coda è quasi piena
                    process_batch = batch[:len(batch) // 2]
                    logger.debug(
                        f"Coda ASCII quasi piena ({queue_ratio:.1%}), processando batch ridotto: {len(process_batch)}/{len(batch)}")
                else:
                    process_batch = batch

                # Converti i frame in ASCII
                ascii_frames = [convert_function(frame) for frame in process_batch]

                conversion_time = time.time() - start_time

                # Tracciamento performance
                conversion_times.append(conversion_time)
                if len(conversion_times) > max_times_to_track:
                    conversion_times.pop(0)

                # Log periodico dei tempi medi
                if len(conversion_times) % 10 == 0:
                    avg_time = sum(conversion_times) / len(conversion_times)
                    frames_per_sec = len(process_batch) / conversion_time
                    logger.debug(f"Tempo medio conversione: {avg_time:.4f}s, {frames_per_sec:.1f} frame/s")

                # Invia i frame ASCII alla coda di rendering
                try:
                    ascii_queue.put(ascii_frames, block=True, timeout=0.5)
                except queue.Full:
                    # Gestione adattiva overflow coda
                    if len(ascii_frames) > 1:
                        reduced_size = max(1, len(ascii_frames) // 2)
                        try:
                            # Prova con batch dimezzato
                            ascii_queue.put(ascii_frames[:reduced_size], block=False)
                            logger.warning(f"Coda ASCII piena, ridotto batch a {reduced_size}")
                        except queue.Full:
                            # Prova con singolo frame
                            try:
                                ascii_queue.put([ascii_frames[0]], block=False)
                                logger.warning("Coda ancora piena, inviando un singolo frame")
                            except queue.Full:
                                logger.warning("Impossibile inviare frame, coda completamente piena")
                    else:
                        logger.warning("Impossibile inviare frame ASCII, coda piena")

            except queue.Empty:
                # Nessun frame disponibile
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Errore nel processo di conversione: {e}")
    except Exception as e:
        logger.error(f"Errore generale nel processo di conversione: {e}")
    finally:
        logger.info("Processo di conversione frame terminato")


class VideoPipeline:
    """
    Classe che gestisce la pipeline parallela per l'elaborazione video ASCII.

    Coordina i processi di lettura frame, conversione e rendering attraverso
    code di comunicazione, implementando una soluzione efficiente e parallela.

    Attributes:
        video_path (str): Percorso del file video
        width (int): Larghezza dell'output ASCII
        target_fps (int): FPS target per l'estrazione dei frame
        batch_size (int): Numero di frame da processare in batch
        log_performance (bool): Se True, registra le informazioni sulle prestazioni
        log_fps (bool): Se True, registra le informazioni sugli FPS
        loop_video (bool): Se True, riavvia il video quando raggiunge la fine
        enable_audio (bool): Se True, riproduce l'audio del video
    """

    def __init__(self, video_path, width, target_fps=None, batch_size=1,
                 log_performance=False, log_fps=False, ascii_palette=None, loop_video=True,
                 enable_audio=False):
        """
        Inizializza la pipeline video.

        Args:
            video_path (str): Percorso del file video
            width (int): Larghezza dell'output ASCII
            target_fps (int, optional): FPS target per l'estrazione dei frame
            batch_size (int): Numero di frame da processare in batch
            log_performance (bool): Se True, registra le informazioni sulle prestazioni
            log_fps (bool): Se True, registra le informazioni sugli FPS
            ascii_palette (str, optional): Stringa di caratteri ASCII da usare per la conversione
            loop_video (bool): Se True, riavvia il video quando raggiunge la fine
            enable_audio (bool): Se True, riproduce l'audio del video
        """
        self.video_path = video_path
        self.width = width
        self.target_fps = target_fps
        self.batch_size = batch_size
        self.log_performance = log_performance
        self.log_fps = log_fps
        self.ascii_palette = ascii_palette
        self.loop_video = loop_video
        self.enable_audio = enable_audio

        # Flag per indicare che il video è finito
        self.video_finished = multiprocessing.Event()

        # Code di comunicazione tra processi
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
            f"Inizializzata pipeline con video={video_path}, width={width}, batch_size={batch_size}, "
            f"fps={target_fps}, loop={loop_video}, audio={enable_audio}")

        # Attributi per i processi
        self.reader_process = None
        self.converter_process = None
        self.renderer_thread = None

        # Attributi per l'audio
        self.audio_player = None
        self.video_duration = None
        self.current_frame = 0
        self.total_frames = 0

        # Inizializzazione audio se abilitato
        if self.enable_audio:
            try:
                from audio_player import AudioPlayer
                self.audio_player = AudioPlayer(video_path, target_fps)
                self.logger.info("Player audio inizializzato")
            except ImportError as e:
                self.logger.error(f"Impossibile inizializzare l'audio: {e}")
                self.enable_audio = False

    def _frame_renderer_thread(self):
        """
        Thread che renderizza i frame ASCII sul terminale.

        Gestisce la visualizzazione del video, le statistiche FPS e la sincronizzazione audio.
        Implementa anche la generazione di grafici per il monitoraggio delle prestazioni.
        """
        # Importazioni e setup
        from utils import configure_process_logging
        from terminal_output_buffer import TerminalOutputBuffer
        import sys
        import time
        import queue

        renderer_logger = configure_process_logging("Renderer", console_level=logging.WARNING)

        renderer_logger.info("Avvio thread di rendering frame")
        self.logger.info("Thread di rendering avviato")

        # Buffer di output ottimizzato
        output_buffer = TerminalOutputBuffer(sys.stdout, max_buffer_size=512 * 1024)

        # Sequenze ANSI per controllo terminale
        CURSOR_HOME = '\033[H'  # Sposta il cursore all'inizio
        CLEAR_SCREEN = '\033[2J'  # Pulisce lo schermo
        BOLD = '\033[1m'  # Testo in grassetto
        RESET = '\033[0m'  # Reset stile
        GREEN = '\033[32m'  # Testo verde
        YELLOW = '\033[33m'  # Testo giallo
        RED = '\033[31m'  # Testo rosso
        BLUE = '\033[34m'  # Testo blu

        last_frame_time = None  # Tempo dell'ultimo frame visualizzato
        frame_times = []  # Intervalli tra frame consecutivi
        frame_count = 0  # Contatore per aggiornamenti periodici
        stats_update_interval = 5  # Aggiorna le statistiche ogni X frame
        max_graph_points = 50  # Numero massimo di punti nel grafico
        first_frame = True

        # Buffer per il grafico degli ultimi frame times
        recent_frame_times = []

        def create_frame_time_graph(recent_times, width=30, target_fps=None):
            """
            Crea un grafico ASCII che mostra l'andamento dei tempi di frame.

            Visualizza i tempi di rendering con gestione intelligente degli outlier
            e colori per indicare le performance.

            Args:
                recent_times: Lista di tempi di frame in secondi
                width: Larghezza massima del grafico
                target_fps: FPS target (se disponibile)

            Returns:
                str: Grafico ASCII dei tempi di frame
            """
            if not recent_times or len(recent_times) < 2:
                return "Raccolta dati per il grafico..."

            # Converti in millisecondi per leggibilità
            times_ms = [t * 1000 for t in recent_times]

            # Target time in ms
            target_time_ms = (1000 / target_fps) if target_fps else None

            # Valori min/max originali
            original_min = min(times_ms)
            original_max = max(times_ms)

            # Rilevamento outlier con MAD
            times_sorted = sorted(times_ms)
            median = times_sorted[len(times_sorted) // 2]
            mad = sum(abs(x - median) for x in times_ms) / len(times_ms)

            # Outlier: valori che deviano più di 3.5 MAD
            threshold = 3.5 * mad
            outliers = [t for t in times_ms if abs(t - median) > threshold]
            has_outliers = len(outliers) > 0

            # Filtra outlier per visualizzazione
            filtered_times = times_ms
            if has_outliers:
                filtered_times = [t for t in times_ms if abs(t - median) <= threshold]
                if not filtered_times:
                    filtered_times = times_sorted[:int(len(times_ms) * 0.8)]

            # Calcola min/max per grafico filtrato
            min_time = min(filtered_times)
            max_time = max(filtered_times)

            # Assicura intervallo minimo
            range_time = max_time - min_time
            if range_time < 0.5:
                mid_value = (max_time + min_time) / 2
                min_time = mid_value - 0.5
                max_time = mid_value + 0.5

            # Caratteri blocchi per grafico
            block_chars = " ▁▂▃▄▅▆▇█"

            # Intestazione
            if has_outliers:
                header = f"{BOLD}Frame Time (ms){RESET}: min={original_min:.1f} max={original_max:.1f}"
                if target_time_ms:
                    header += f" target={target_time_ms:.1f}"
                header += f" {RED}[outliers rilevati]{RESET}"
            else:
                header = f"{BOLD}Frame Time (ms){RESET}: min={min_time:.1f} max={max_time:.1f}"
                if target_time_ms:
                    header += f" target={target_time_ms:.1f}"

            # Limita punti dati alla larghezza
            if len(times_ms) > width:
                stride = len(times_ms) // width
                samples = [times_ms[i] for i in range(0, len(times_ms), stride)][:width]
            else:
                samples = times_ms[-width:]

            # Genera grafico a blocchi
            graph_line = ""
            for time_ms in samples:
                # Verifica se outlier
                is_outlier = abs(time_ms - median) > threshold

                if is_outlier:
                    # Outlier: blocchi rossi
                    graph_line += f"{RED}█{RESET}"
                else:
                    # Normalizza valore
                    normalized = max(0, min(1, (time_ms - min_time) / (max_time - min_time)))
                    block_index = min(int(normalized * (len(block_chars) - 1)), len(block_chars) - 1)

                    # Colore basato su target
                    if target_time_ms is not None:
                        if time_ms < target_time_ms * 0.85:  # Molto veloce
                            color = BLUE
                        elif time_ms < target_time_ms * 1.1:  # Vicino al target
                            color = GREEN
                        elif time_ms < target_time_ms * 1.5:  # Leggermente lento
                            color = YELLOW
                        else:  # Molto lento
                            color = RED
                    else:
                        color = GREEN

                    # Blocco colorato
                    graph_line += f"{color}{block_chars[block_index]}{RESET}"

            # Etichette asse
            if min_time < 10:
                min_label = f"{min_time:.1f}ms"
            else:
                min_label = f"{min_time:.0f}ms"

            if max_time < 10:
                max_label = f"{max_time:.1f}ms"
            else:
                max_label = f"{max_time:.0f}ms"

            # Allinea etichette
            spaces = width - len(min_label) - len(max_label)
            if spaces < 1:
                spaces = 1

            # Scala in basso
            scale = f"{min_label}{' ' * spaces}{max_label}"

            # Info outlier
            if has_outliers and original_max > max_time:
                scale += f" [{RED}max: {original_max:.1f}ms{RESET}]"

            # Grafico completo
            return f"{header}\n{graph_line}\n{scale}"

        def create_fps_stats(frame_times, target_fps=None):
            """
            Genera statistiche FPS formattate per la visualizzazione.

            Args:
                frame_times: Lista dei tempi di frame
                target_fps: FPS target (se disponibile)

            Returns:
                str: Stringa formattata con statistiche FPS
            """
            if not frame_times:
                return "Calcolando FPS..."

            # Calcola FPS dalle ultime misurazioni
            recent_times = frame_times[-10:]  # Ultimi 10 frame
            avg_time = sum(recent_times) / len(recent_times)
            current_fps = 1.0 / avg_time

            # Calcola deviazione standard per stabilità
            if len(recent_times) > 1:
                variance = sum((t - avg_time) ** 2 for t in recent_times) / len(recent_times)
                std_dev = variance ** 0.5
                stability = 1 - min(1, (std_dev / avg_time))  # 0=instabile, 1=stabile
            else:
                stability = 1

            # Colore in base al target
            if target_fps:
                fps_ratio = current_fps / target_fps
                if fps_ratio > 0.95 and fps_ratio < 1.05:  # Entro 5% del target
                    fps_color = GREEN
                elif fps_ratio > 0.8:  # Almeno 80% del target
                    fps_color = YELLOW
                else:
                    fps_color = RED
            else:
                fps_color = GREEN

            # Barra stabilità
            stability_bar_length = int(stability * 10)
            stability_bar = f"[{'|' * stability_bar_length}{' ' * (10 - stability_bar_length)}]"

            # Statistiche formattate
            stats = (
                f"{BOLD}FPS: {fps_color}{current_fps:.1f}{RESET} "
                f"| Stabilità: {stability_bar} "
            )

            if target_fps:
                stats += f"| Target: {target_fps} FPS "
                stats += f"| Ratio: {fps_ratio:.2f} "

            stats += f"| Frame {frame_count}"

            return stats

        try:
            while not self.should_stop.is_set():
                try:
                    # Ottieni batch di frame ASCII
                    try:
                        ascii_frames = self.ascii_queue.get(block=True, timeout=0.5)
                    except queue.Empty:
                        # Controllo fine video
                        if hasattr(self,
                                   'video_finished') and self.video_finished.is_set() and self.ascii_queue.empty():
                            renderer_logger.info("Video finito e coda ASCII vuota, terminazione del renderer")
                            break
                        continue

                    # Controllo marker fine video
                    if ascii_frames == END_OF_VIDEO_MARKER:
                        renderer_logger.info("Ricevuto marker di fine video, terminazione del renderer")
                        self.logger.info("Video terminato, chiusura thread renderer")
                        self.should_stop.set()
                        break

                    # Renderizza ciascun frame
                    for ascii_frame in ascii_frames:
                        if self.should_stop.is_set():
                            break

                        # Incrementa contatore frame
                        frame_count += 1
                        if hasattr(self, 'current_frame'):
                            self.current_frame = frame_count

                        # Tempo corrente
                        current_time = time.time()

                        # Gestione timing con target FPS
                        if last_frame_time is not None:
                            elapsed = current_time - last_frame_time
                            if self.target_fps and elapsed < 1.0 / self.target_fps:
                                sleep_time = 1.0 / self.target_fps - elapsed
                                if sleep_time > 0.001:  # Solo se > 1ms
                                    time.sleep(sleep_time)

                        # Pulizia schermo solo al primo frame
                        if first_frame:
                            output_buffer.write(CLEAR_SCREEN)
                            first_frame = False

                        # Sincronizzazione audio-video
                        if hasattr(self, 'enable_audio') and self.enable_audio and \
                                hasattr(self, 'video_duration') and self.video_duration and \
                                hasattr(self, 'total_frames') and self.total_frames:
                            # Calcola tempo attuale in base ai frame
                            video_time = (self.current_frame / self.total_frames) * self.video_duration

                            # Aggiorna player audio
                            if hasattr(self, 'audio_player') and self.audio_player:
                                self.audio_player.update_video_time(video_time)

                        # Preparazione statistiche FPS
                        fps_display = ""
                        if self.log_fps:
                            if frame_count % stats_update_interval == 0 and frame_times:
                                # Statistiche FPS
                                fps_stats = create_fps_stats(frame_times, self.target_fps)

                                # Grafico tempi frame
                                frame_time_graph = create_frame_time_graph(recent_frame_times,
                                                                           width=50,
                                                                           target_fps=self.target_fps)

                                # Visualizzazione completa
                                fps_display = f"\n\n{fps_stats}\n{frame_time_graph}"

                        # Rendering frame
                        output_buffer.write(CURSOR_HOME)
                        output_buffer.write(ascii_frame)

                        # Aggiungi statistiche FPS
                        if fps_display:
                            output_buffer.write(fps_display)

                        # Flush per visualizzazione
                        output_buffer.flush()

                        # Tempo dopo rendering
                        frame_end_time = time.time()

                        # Calcola intervallo tra frame
                        if last_frame_time is not None:
                            frame_interval = frame_end_time - last_frame_time

                            # Registra per statistiche
                            if self.log_fps:
                                frame_times.append(frame_interval)
                                if len(frame_times) > 100:
                                    frame_times.pop(0)

                                # Aggiorna buffer grafico
                                recent_frame_times.append(frame_interval)
                                if len(recent_frame_times) > max_graph_points:
                                    recent_frame_times.pop(0)

                        # Aggiorna timestamp
                        last_frame_time = frame_end_time

                except Exception as e:
                    renderer_logger.error(f"Errore nel thread di rendering: {e}")
                    self.logger.error(f"Errore nel thread di rendering: {e}")
        finally:
            # Reset colore terminale
            output_buffer.write("\033[0m\n")
            output_buffer.flush()

            # Salva statistiche
            self.frame_times = frame_times
            renderer_logger.info("Thread di rendering frame terminato")
            self.logger.info("Thread di rendering frame terminato")

    def start(self):
        """
        Avvia la pipeline video e tutti i suoi componenti.

        Inizializza l'audio se abilitato, avvia i processi di lettura e conversione
        frame, e il thread di rendering.
        """
        self.logger.info("Avvio pipeline video")

        # Inizializzazione audio
        if self.enable_audio:
            try:
                import cv2
                from moviepy import VideoFileClip

                # Ottieni durata video
                video_clip = VideoFileClip(self.video_path)
                self.video_duration = video_clip.duration
                video_clip.close()

                # Ottieni numero frame
                cap = cv2.VideoCapture(self.video_path)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FPS) * self.video_duration)
                cap.release()

                self.logger.info(f"Informazioni video: durata={self.video_duration}s, frames={self.total_frames}")

                # Avvia audio
                if self.audio_player.initialize():
                    self.audio_player.start()
                    self.logger.info("Riproduzione audio avviata")
                else:
                    self.logger.warning("Inizializzazione audio fallita, continuo senza audio")
                    self.enable_audio = False
            except Exception as e:
                self.logger.error(f"Errore nell'avvio dell'audio: {e}")
                self.enable_audio = False

        # Avvia processo reader
        self.reader_process = multiprocessing.Process(
            target=frame_reader_process,
            args=(
                self.video_path, self.frame_queue, self.should_stop, self.target_fps,
                self.batch_size, self.loop_video
            ),
            daemon=True
        )
        self.reader_process.start()

        # Avvia processo converter
        self.converter_process = multiprocessing.Process(
            target=frame_converter_process,
            args=(
                self.width, self.frame_queue, self.ascii_queue, self.should_stop,
                self.ascii_palette
            ),
            daemon=True
        )
        self.converter_process.start()

        # Avvia thread renderer
        self.renderer_thread = threading.Thread(
            target=self._frame_renderer_thread,
            daemon=True
        )
        self.renderer_thread.start()

    def stop(self):
        """
        Ferma la pipeline video con strategia ottimizzata.

        Implementa una terminazione efficiente e non bloccante di tutti i componenti,
        gestendo in modo sicuro le risorse e le code.
        """
        self.logger.info("Arresto pipeline video")

        # Imposta flag terminazione
        self.should_stop.set()

        # Timeout breve per evitare blocchi
        timeout = 0.5

        # Prima ferma l'audio
        if self.enable_audio and hasattr(self, 'audio_player') and self.audio_player:
            try:
                self.audio_player.stop()
                self.logger.info("Riproduzione audio terminata")
            except Exception as e:
                self.logger.error(f"Errore durante l'arresto dell'audio: {e}")

        # Svuota le code
        try:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get(block=False)
                except:
                    break
        except:
            pass

        try:
            while not self.ascii_queue.empty():
                try:
                    self.ascii_queue.get(block=False)
                except:
                    break
        except:
            pass

        # Gestione parallela dei processi
        processes_to_terminate = []

        if hasattr(self, 'reader_process') and self.reader_process and self.reader_process.is_alive():
            self.reader_process.join(timeout=timeout)
            if self.reader_process.is_alive():
                processes_to_terminate.append(self.reader_process)

        if hasattr(self, 'converter_process') and self.converter_process and self.converter_process.is_alive():
            self.converter_process.join(timeout=timeout)
            if self.converter_process.is_alive():
                processes_to_terminate.append(self.converter_process)

        # Termina processi bloccati
        for process in processes_to_terminate:
            try:
                process.terminate()
            except Exception as e:
                self.logger.error(f"Errore durante la terminazione del processo: {e}")

        # Gestisci thread renderer
        if hasattr(self, 'renderer_thread') and self.renderer_thread and self.renderer_thread.is_alive():
            self.renderer_thread.join(timeout=timeout)

        # Stampa statistiche FPS se richiesto
        if self.log_fps and hasattr(self, 'frame_times') and self.frame_times:
            # Usa solo i frame recenti per maggiore precisione
            recent_frames = self.frame_times[-min(len(self.frame_times), 100):]

            if recent_frames:
                avg_time = sum(recent_frames) / len(recent_frames)
                min_time = min(recent_frames)
                max_time = max(recent_frames)

                avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                min_fps = 1.0 / max_time if max_time > 0 else 0
                max_fps = 1.0 / min_time if min_time > 0 else 0

                self.logger.info(f"FPS medio: {avg_fps:.2f}, Min: {min_fps:.2f}, Max: {max_fps:.2f}")

                # Info target FPS
                if self.target_fps:
                    self.logger.info(f"Target FPS: {self.target_fps}, FPS effettivo: {avg_fps:.2f}")
                    if avg_fps > self.target_fps * 1.1:
                        self.logger.warning(
                            f"L'FPS effettivo ({avg_fps:.2f}) è molto più alto del target ({self.target_fps}). "
                            f"Potrebbe esserci un problema con il controllo FPS.")

        # Ripristina cursore
        from terminal_output_buffer import TerminalOutputBuffer
        import sys

        output_buffer = TerminalOutputBuffer(sys.stdout)
        output_buffer.write('\033[?25h')  # Mostra cursore
        output_buffer.flush()

        self.logger.info("Pipeline video terminata")
