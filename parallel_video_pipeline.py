import queue
import time
from multiprocessing import Pool, cpu_count

from converter import frame_to_ascii
from video_pipeline import VideoPipeline


def convert_frame_worker(args):
    """
    Funzione worker per la conversione dei frame in un processo separato.

    Parametri:
        args (tuple): (frame, width, use_cache)

    Ritorna:
        str: Frame ASCII convertito
    """
    frame, width, use_cache = args
    if frame is None:
        return None

    # Usa una cache globale nel processo worker
    global _color_cache
    if not '_color_cache' in globals() and use_cache:
        _color_cache = {}

    # Converte il frame con o senza cache
    if use_cache:
        ascii_frame, _ = frame_to_ascii(frame, width, _color_cache)
    else:
        ascii_frame, _ = frame_to_ascii(frame, width)

    return ascii_frame


class ParallelVideoPipeline(VideoPipeline):
    """
    Pipeline video con conversione parallela dei frame.
    """

    def __init__(self, video_path, width, fps=10, log_fps=False, log_performance=False,
                 batch_size=20, num_processes=None, use_cache=True, logger=None):
        """
        Inizializza la pipeline video con supporto per multiprocessing.

        Parametri:
            video_path (str): Percorso al file video.
            width (int): Larghezza dell'output ASCII.
            fps (int): Frame per secondo per l'estrazione.
            log_fps (bool): Se True, registra i FPS di visualizzazione.
            log_performance (bool): Se True, registra le prestazioni.
            batch_size (int): Dimensione del batch per l'elaborazione dei frame.
            num_processes (int): Numero di processi da utilizzare (None = auto).
            use_cache (bool): Se True, usa la cache dei colori.
            logger (logging.Logger): Logger da utilizzare.
        """
        super().__init__(video_path, width, fps, log_fps, log_performance, batch_size, logger)

        # Imposta il numero di processi
        self.num_processes = num_processes if num_processes else max(1, cpu_count() - 1)
        self.use_cache = use_cache

        # Crea il pool di processi
        self.process_pool = None

    def _init_process_pool(self):
        """Inizializza il pool di processi."""
        if self.process_pool is None:
            self.logger.info(f"Inizializzazione pool con {self.num_processes} processi")
            self.process_pool = Pool(processes=self.num_processes)

    def _frame_converter(self):
        """
        Thread che converte i frame in ASCII utilizzando multiprocessing.
        """
        batch = []
        conversion_count = 0
        total_conversion_time = 0

        # Inizializza il pool di processi
        self._init_process_pool()

        try:
            while self.running and not self.stop_requested:
                try:
                    # Preleva un frame dalla coda
                    frame = self.raw_frame_queue.get(block=True, timeout=0.1)

                    # Controlla se è la fine del video
                    if frame is None:
                        # Elabora il batch rimanente
                        if batch:
                            start_time = time.time()

                            # Prepara gli argomenti per i worker
                            args = [(f, self.width, self.use_cache) for f in batch]

                            # Elabora in parallelo
                            ascii_frames = self.process_pool.map(convert_frame_worker, args)

                            conversion_time = time.time() - start_time

                            if self.log_performance:
                                self.conversion_times.append(conversion_time)
                                conversion_count += 1
                                total_conversion_time += conversion_time
                                avg_time = total_conversion_time / conversion_count
                                self.logger.debug(
                                    f"Conversione batch #{conversion_count}: {conversion_time * 1000:.2f}ms (media: {avg_time * 1000:.2f}ms)")

                            # Metti i frame ASCII in coda
                            for ascii_frame in ascii_frames:
                                if ascii_frame is not None:
                                    self.ascii_frame_queue.put(ascii_frame)

                        # Segnala la fine
                        self.ascii_frame_queue.put(None)
                        break

                    # Aggiungi il frame al batch
                    batch.append(frame)

                    # Se il batch è completo, elaboralo
                    if len(batch) >= self.batch_size:
                        start_time = time.time()

                        # Prepara gli argomenti per i worker
                        args = [(f, self.width, self.use_cache) for f in batch]

                        # Elabora in parallelo
                        ascii_frames = self.process_pool.map(convert_frame_worker, args)

                        conversion_time = time.time() - start_time

                        if self.log_performance:
                            self.conversion_times.append(conversion_time)
                            conversion_count += 1
                            total_conversion_time += conversion_time
                            avg_time = total_conversion_time / conversion_count
                            self.logger.debug(
                                f"Conversione batch #{conversion_count}: {conversion_time * 1000:.2f}ms (media: {avg_time * 1000:.2f}ms)")

                        # Metti i frame ASCII in coda
                        for ascii_frame in ascii_frames:
                            if ascii_frame is not None:
                                try:
                                    self.ascii_frame_queue.put(ascii_frame, block=True, timeout=0.1)
                                except queue.Full:
                                    if not self.stop_requested:
                                        self.logger.warning("Coda dei frame ASCII piena, frame saltato")

                        # Svuota il batch
                        batch = []

                except queue.Empty:
                    time.sleep(0.01)
                except Exception as e:
                    self.logger.error(f"Errore nel thread converter: {e}", exc_info=True)
                    self.running = False
                    self.stop_requested = True
                    break

        finally:
            # Chiudi il pool di processi
            if self.process_pool:
                self.process_pool.close()
                self.process_pool.join()

    def start(self):
        """Override del metodo start per gestire le risorse del multiprocessing."""
        try:
            super().start()
        finally:
            # Assicurati che il pool sia chiuso
            if hasattr(self, 'process_pool') and self.process_pool:
                self.process_pool.close()
                self.process_pool.join()
