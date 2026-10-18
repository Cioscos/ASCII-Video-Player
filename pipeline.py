"""Pipeline multi-processo della modalità standard: lettura -> conversione -> FrameSource."""
import logging
import math
import multiprocessing
import queue
import time
from collections import deque

# Marker per segnalare la fine del video
END_OF_VIDEO_MARKER = "END_OF_VIDEO"

# Attesa massima di ogni get/put sulle code, per ricontrollare lo stop
_POLL_TIMEOUT = 0.1


def put_until_stopped(q, item, should_stop, timeout=0.1):
    """
    Inserisce un elemento nella coda senza mai scartarlo.

    Ritenta il put finché riesce o finché should_stop è impostato.

    Args:
        q: Coda (multiprocessing.Queue o queue.Queue)
        item: Elemento da inserire
        should_stop: Evento di terminazione (con metodo is_set)
        timeout (float): Attesa massima per ogni tentativo, in secondi

    Returns:
        bool: True se l'elemento è stato inserito, False se è stato richiesto lo stop
    """
    import queue
    while not should_stop.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def frame_reader_process(video_path, width, height, source_fps, target_fps, frame_queue, should_stop,
                         batch_size, loop_video, rotation=0, start_index=0):
    """
    Processo di lettura: decima i frame sorgente, li ridimensiona e li invia in batch.

    Ogni elemento della coda è una lista di (indice_decimato, ndarray ridimensionato
    width x height) oppure END_OF_VIDEO_MARKER. Il loop del video è gestito qui;
    l'indice decimato riparte da 0 a ogni giro (tranne il primo, che parte da start_index).
    La rotazione è applicata dopo il ridimensionamento: il risultato è sempre height x width.

    Args:
        video_path (str): Percorso del file video
        width (int): Larghezza della griglia ASCII
        height (int): Altezza della griglia ASCII
        source_fps (float): FPS del video sorgente (0 o NaN se sconosciuti)
        target_fps (float | None): FPS richiesti
        frame_queue (multiprocessing.Queue): Coda dei frame ridimensionati
        should_stop (multiprocessing.Event): Evento di terminazione
        batch_size (int): Numero di frame per batch (il primo batch di ogni avvio ne ha uno solo)
        loop_video (bool): Se True, riavvia il video alla fine
        rotation (int): Rotazione oraria in gradi (0, 90, 180, 270)
        start_index (int): Indice decimato da cui iniziare (posizionamento del video)
    """
    import signal

    # Ctrl+C arriva a tutti i processi della console: lo gestisce solo il processo
    # principale, che chiude la pipeline impostando should_stop. Va impostato prima
    # degli import pesanti (cv2) per ridurre la finestra in cui il figlio lo riceve.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import cv2
    from playback import FrameDecimator
    from utils import configure_process_logging

    logger = configure_process_logging("Reader", console_level=logging.WARNING)
    decimator = FrameDecimator(source_fps, target_fps)
    video = None
    try:
        logger.info(f"Avvio lettura: {video_path}, {width}x{height}, fps effettivi {decimator.effective_fps}")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Impossibile aprire il video: {video_path}")
            should_stop.set()
            return

        rotate_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotation)
        # Con 90/270 si ridimensiona già in orientamento ruotato, poi si ruota
        resize_size = (height, width) if rotation in (90, 270) else (width, height)

        batch = []
        source_index = decimator.source_index(start_index)
        decimated_index = start_index
        if start_index > 0:
            logger.info(f"Posizionamento al frame sorgente {source_index} (indice decimato {start_index}), "
                        f"rotazione {rotation}°")
            if not video.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                logger.warning(f"Posizionamento al frame sorgente {source_index} non riuscito")
        elif rotation:
            logger.info(f"Rotazione {rotation}°")
        grabbed_in_lap = 0
        offset_lap = start_index > 0  # il giro corrente non parte da 0
        # Il primo batch di ogni avvio parte con un solo frame, così il primo
        # frame compare subito; i successivi (anche dopo il loop) sono interi
        limit = 1
        while not should_stop.is_set():
            if not video.grab():
                if batch:
                    if not put_until_stopped(frame_queue, batch, should_stop):
                        break
                    batch = []
                if not loop_video or (grabbed_in_lap == 0 and not offset_lap):
                    # Fine video senza loop, oppure nessun frame letto in un giro intero
                    if grabbed_in_lap == 0:
                        logger.warning("Nessun frame letto dal video, fine riproduzione")
                    put_until_stopped(frame_queue, END_OF_VIDEO_MARKER, should_stop)
                    logger.info("Fine del video, marker inviato")
                    break
                logger.info("Riavvio del video")
                video.release()
                video = cv2.VideoCapture(video_path)
                source_index = 0
                decimated_index = 0
                grabbed_in_lap = 0
                offset_lap = False
                continue

            grabbed_in_lap += 1
            keep = decimator.should_keep(source_index)
            source_index += 1
            if not keep:
                continue
            ok, frame = video.retrieve()
            if not ok:
                continue
            resized = cv2.resize(frame, resize_size, interpolation=cv2.INTER_NEAREST)
            if rotate_code is not None:
                resized = cv2.rotate(resized, rotate_code)
            batch.append((decimated_index, resized))
            decimated_index += 1
            if len(batch) >= limit:
                if not put_until_stopped(frame_queue, batch, should_stop):
                    break
                batch = []
                limit = batch_size
    except KeyboardInterrupt:
        should_stop.set()
    except Exception as e:
        logger.exception(f"Errore nel processo di lettura: {e}")
        should_stop.set()
    finally:
        if video is not None:
            video.release()
        if should_stop.is_set():
            # Allo stop non si attende lo svuotamento della coda (evita blocchi all'uscita)
            frame_queue.cancel_join_thread()
        logger.info("Processo di lettura terminato")


def frame_converter_process(width, height, palette, frame_queue, ascii_queue, should_stop, log_performance=False,
                            truecolor=False):
    """
    Processo di conversione: trasforma i batch di frame ridimensionati in bytes ANSI.

    Args:
        width (int): Larghezza della griglia ASCII
        height (int): Altezza della griglia ASCII
        palette (str): Caratteri della palette, 'box' oppure 'half'
        frame_queue (multiprocessing.Queue): Coda dei frame ridimensionati
        ascii_queue (multiprocessing.Queue): Coda dei batch di (indice, bytes)
        should_stop (multiprocessing.Event): Evento di terminazione
        log_performance (bool): Se True, logga (DEBUG) la media dei ms/frame ogni 100 frame
        truecolor (bool): Se True, colori a 24 bit invece della palette a 256 colori
    """
    import signal

    # Ctrl+C viene gestito solo dal processo principale (vedi frame_reader_process)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import queue
    import time
    from ascii_converter import make_converter
    from utils import configure_process_logging

    logger = configure_process_logging("Converter", console_level=logging.WARNING)
    if log_performance:
        logger.setLevel(logging.DEBUG)
    try:
        converter = make_converter(width, height, palette, truecolor)
        logger.info(f"Avvio conversione {width}x{height}")
        total_time = 0.0
        count = 0
        while not should_stop.is_set():
            try:
                batch = frame_queue.get(timeout=_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if isinstance(batch, str) and batch == END_OF_VIDEO_MARKER:
                logger.info("Ricevuto marker di fine video")
                put_until_stopped(ascii_queue, END_OF_VIDEO_MARKER, should_stop)
                break

            converted = []
            for index, resized in batch:
                start = time.perf_counter()
                converted.append((index, converter.convert(resized)))
                total_time += time.perf_counter() - start
                count += 1
                if log_performance and count == 100:
                    logger.debug(f"Conversione media: {total_time / count * 1000:.2f} ms/frame")
                    total_time = 0.0
                    count = 0

            if not put_until_stopped(ascii_queue, converted, should_stop):
                break
    except KeyboardInterrupt:
        should_stop.set()
    except Exception as e:
        logger.exception(f"Errore nel processo di conversione: {e}")
        should_stop.set()
    finally:
        if should_stop.is_set():
            ascii_queue.cancel_join_thread()
        logger.info("Processo di conversione terminato")


class _PipelineFrameSource:
    """FrameSource che spacchetta i batch convertiti dalla coda ASCII della pipeline."""

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._pending = deque()
        self._ended = False

    def next_frame(self, timeout=None):
        """
        Restituisce (indice, bytes) oppure None a fine video o con lo stop impostato.

        Se timeout (secondi) è indicato e non arriva alcun frame entro quel tempo,
        solleva TimeoutError; con None attende finché c'è un frame, la fine o lo stop.

        Se il processo di conversione muore senza inviare il marker, imposta lo
        stop e restituisce None invece di attendere per sempre.
        """
        p = self._pipeline
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._pending:
            if self._ended or p.should_stop.is_set():
                return None
            try:
                item = p.ascii_queue.get(timeout=_POLL_TIMEOUT)
            except queue.Empty:
                if p.workers_alive():
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f"Nessun frame entro {timeout} s")
                    continue
                # Ultimo tentativo: i dati potrebbero essere ancora in transito nella pipe
                try:
                    item = p.ascii_queue.get(timeout=0.5)
                except queue.Empty:
                    p.logger.error("Pipeline interrotta: un processo è terminato inaspettatamente")
                    p.should_stop.set()
                    return None
            if isinstance(item, str) and item == END_OF_VIDEO_MARKER:
                self._ended = True
                return None
            self._pending.extend(item)
        if p.should_stop.is_set():
            return None
        return self._pending.popleft()

    def restart(self):
        """No-op: il loop del video è gestito dal processo di lettura."""


class VideoPipeline:
    """
    Pipeline della modalità standard: un processo di lettura e uno di conversione.

    Il rendering avviene nel thread principale tramite run_playback, che legge
    i frame da frame_source().

    Attributes:
        effective_fps (float): FPS reali di riproduzione (mai sopra gli FPS sorgente)
        should_stop (multiprocessing.Event): Evento di terminazione condiviso
    """

    def __init__(self, video_path, width, height, source_fps, target_fps, batch_size, palette, loop_video,
                 log_performance=False, truecolor=False, rotation=0, start_index=0):
        """
        Args:
            video_path (str): Percorso del file video
            width (int): Larghezza della griglia ASCII
            height (int): Altezza della griglia ASCII
            source_fps (float): FPS del video sorgente
            target_fps (float | None): FPS richiesti
            batch_size (int): Numero di frame per batch (il primo batch di ogni avvio ne ha uno solo)
            palette (str): Caratteri della palette, 'box' oppure 'half'
            loop_video (bool): Se True, il video riparte alla fine
            log_performance (bool): Se True, il converter logga i tempi di conversione
            truecolor (bool): Se True, colori a 24 bit
            rotation (int): Rotazione oraria in gradi (0, 90, 180, 270)
            start_index (int): Indice decimato da cui parte il reader
        """
        from ascii_converter import pixel_rows
        from playback import FrameDecimator
        self.video_path = video_path
        self.width = width
        self.height = height
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.batch_size = max(1, int(batch_size))
        self.palette = palette
        self.loop_video = loop_video
        self.log_performance = log_performance
        self.truecolor = truecolor
        self.rotation = rotation
        self.start_index = start_index
        # Righe di pixel che il reader deve produrre (il doppio delle righe di testo con 'half')
        self.resize_height = pixel_rows(height, palette)
        self.effective_fps = FrameDecimator(source_fps, target_fps).effective_fps

        queue_size = max(2, math.ceil(2 * self.effective_fps / self.batch_size))
        self.frame_queue = multiprocessing.Queue(maxsize=queue_size)
        self.ascii_queue = multiprocessing.Queue(maxsize=queue_size)
        self.should_stop = multiprocessing.Event()

        self.reader_process = None
        self.converter_process = None
        self._stopped = False

        self.logger = logging.getLogger('VideoPipeline')
        self.logger.info(
            f"Pipeline: video={video_path}, {width}x{height}, fps effettivi={self.effective_fps}, "
            f"batch={self.batch_size}, code={queue_size}, loop={loop_video}, "
            f"rotazione={rotation}, indice iniziale={start_index}")

    def start(self):
        """Avvia i processi di lettura e conversione."""
        self.reader_process = multiprocessing.Process(
            target=frame_reader_process,
            args=(self.video_path, self.width, self.resize_height, self.source_fps, self.target_fps,
                  self.frame_queue, self.should_stop, self.batch_size, self.loop_video,
                  self.rotation, self.start_index),
            daemon=True,
        )
        self.converter_process = multiprocessing.Process(
            target=frame_converter_process,
            args=(self.width, self.height, self.palette, self.frame_queue, self.ascii_queue,
                  self.should_stop, self.log_performance, self.truecolor),
            daemon=True,
        )
        self.reader_process.start()
        self.converter_process.start()

    def frame_source(self):
        """Restituisce la FrameSource che legge i frame convertiti."""
        return _PipelineFrameSource(self)

    def workers_alive(self):
        """
        True se la pipeline può ancora produrre frame.

        Falso se il converter non è in esecuzione oppure se il reader è terminato
        in modo anomalo (exitcode diverso da 0; 0 è l'uscita normale senza loop).
        """
        if self.converter_process is None or not self.converter_process.is_alive():
            return False
        reader_exit = self.reader_process.exitcode if self.reader_process is not None else None
        if reader_exit not in (None, 0):
            self.logger.error(f"Processo di lettura terminato in modo anomalo (exitcode {reader_exit})")
            return False
        return True

    def stop(self, timeout=1.0):
        """Ferma i processi e svuota le code (idempotente)."""
        if self._stopped:
            return
        self._stopped = True
        self.should_stop.set()
        self.logger.info("Arresto pipeline")

        for q in (self.frame_queue, self.ascii_queue):
            try:
                while True:
                    q.get_nowait()
            except (queue.Empty, OSError, ValueError):
                pass

        for process in (self.reader_process, self.converter_process):
            if process is None:
                continue
            process.join(timeout=timeout)
            if process.is_alive():
                self.logger.warning(f"Processo {process.name} non terminato, forzo la chiusura")
                process.terminate()
                process.join(timeout=timeout)
        self.logger.info("Pipeline terminata")
