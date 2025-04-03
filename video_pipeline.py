"""
Implementazione della pipeline parallela per l'elaborazione e visualizzazione di video ASCII.
"""
import cv2
import threading
import queue
import time
from converter import batch_process_frames
from terminal import print_frame, hide_cursor, show_cursor, clear_terminal
from utils import setup_logging

class VideoPipeline:
    """
    Pipeline parallela per l'elaborazione e la visualizzazione di video ASCII.
    """

    def __init__(self, video_path, width, fps=10, log_fps=False, log_performance=False, batch_size=1, logger=None):
        """
        Inizializza la pipeline video.

        Parametri:
            video_path (str): Percorso al file video.
            width (int): Larghezza dell'output ASCII.
            fps (int): Frame per secondo per l'estrazione.
            log_fps (bool): Se True, registra i FPS di visualizzazione.
            log_performance (bool): Se True, registra le prestazioni di conversione e rendering.
            batch_size (int): Dimensione del batch per l'elaborazione dei frame.
        """
        self.video_path = video_path
        self.width = width
        self.target_fps = fps
        self.log_fps = log_fps
        self.log_performance = log_performance
        self.batch_size = batch_size

        # Code per la comunicazione tra thread
        self.raw_frame_queue = queue.Queue(maxsize=30)  # Buffer limitato
        self.ascii_frame_queue = queue.Queue(maxsize=60)

        # Flag di controllo
        self.running = False
        self.stop_requested = False
        self.height = None

        # Metriche di performance
        self.conversion_times = []
        self.render_times = []
        self.display_fps = []

        # Inizializza il logger
        self.logger = setup_logging() if not logger else logger

    def start(self):
        """
        Avvia la pipeline video.

        Crea e avvia i thread per l'estrazione, la conversione e il rendering dei frame.
        Gestisce le eccezioni e mostra le statistiche di performance se richiesto.
        """
        try:
            self.running = True
            self.stop_requested = False

            # Creare i thread e impostarli come daemon
            extractor_thread = threading.Thread(target=self._frame_extractor, daemon=True)
            converter_thread = threading.Thread(target=self._frame_converter, daemon=True)
            renderer_thread = threading.Thread(target=self._frame_renderer, daemon=True)

            # Thread di monitoraggio per intercettare Ctrl+C
            monitor_thread = threading.Thread(target=self._monitor_interrupt, daemon=True)

            # Nascondi il cursore e pulisci il terminale
            hide_cursor()
            clear_terminal()

            self.logger.info("Avvio della pipeline video...")

            # Avvia tutti i thread
            extractor_thread.start()
            converter_thread.start()
            renderer_thread.start()
            monitor_thread.start()

            # Aspetta che tutti i thread terminino o che venga richiesto lo stop
            while (extractor_thread.is_alive() or
                   converter_thread.is_alive() or
                   renderer_thread.is_alive()) and not self.stop_requested:
                time.sleep(0.1)

            # Se è stato richiesto lo stop, assicurati che tutti i thread terminino
            if self.stop_requested:
                self.running = False
                self.logger.info("Interruzione richiesta, terminazione in corso...")

                # Aspetta al massimo 2 secondi per thread per terminare
                extractor_thread.join(2.0)
                converter_thread.join(2.0)
                renderer_thread.join(2.0)

            # Mostra le statistiche di performance se richiesto
            if self.log_performance:
                self._print_performance_stats()

            # Ripristina il cursore
            show_cursor()

            self.logger.info("Pipeline terminata con successo.")

        except KeyboardInterrupt:
            self.logger.info("Interruzione manuale da tastiera (KeyboardInterrupt).")
            self.running = False
            self.stop_requested = True
            # Ripristina il cursore
            show_cursor()
        except Exception as e:
            self.logger.error(f"Errore durante l'esecuzione: {e}", exc_info=True)
            self.running = False
            # Ripristina il cursore
            show_cursor()

    def _monitor_interrupt(self):
        """
        Thread di monitoraggio per intercettare Ctrl+C.

        Controlla periodicamente se è stato richiesto uno stop
        e gestisce l'arresto pulito della pipeline.
        """
        try:
            while self.running and not self.stop_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Interruzione da tastiera rilevata.")
            self.stop_requested = True
            self.running = False

    def _frame_extractor(self):
        """
        Thread che estrae i frame dal video e li mette in coda.

        Legge i frame dal file video alla frequenza specificata
        e li inserisce nella coda raw_frame_queue.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                error_msg = f"Impossibile aprire il file video: {self.video_path}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Ottieni alcune informazioni sul video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Video caricato: {self.video_path}")
            self.logger.info(f"Frame totali: {total_frames}, FPS originali: {original_fps:.2f}, FPS target: {self.target_fps}")

            # Calcola il tempo tra i frame
            frame_delay = 1.0 / self.target_fps

            # Leggi i frame
            last_frame_time = time.time()
            frames_processed = 0

            while self.running and not self.stop_requested:
                # Controlla se è il momento di estrarre un nuovo frame
                current_time = time.time()
                if current_time - last_frame_time >= frame_delay:
                    ret, frame = cap.read()
                    if not ret:
                        # Fine del video
                        self.logger.info("Fine del video raggiunta.")
                        self.running = False
                        break

                    frames_processed += 1

                    # Metti il frame in coda se c'è spazio
                    try:
                        self.raw_frame_queue.put(frame, block=True, timeout=0.5)
                        last_frame_time = current_time

                        # Log ogni 100 frame
                        if frames_processed % 100 == 0 and self.log_performance:
                            self.logger.debug(f"Estratti {frames_processed}/{total_frames} frames")

                    except queue.Full:
                        if not self.stop_requested:
                            self.logger.warning("Coda dei frame raw piena, salto un frame")
                else:
                    # Aspetta un po' per risparmiare CPU
                    time.sleep(0.001)

            # Segnala la fine del video
            try:
                self.raw_frame_queue.put(None, block=True, timeout=1.0)
                self.logger.info("Estrazione frame completata.")
            except queue.Full:
                self.logger.warning("Impossibile segnalare la fine del video, la coda è piena.")

        except Exception as e:
            self.logger.error(f"Errore nel thread di estrazione: {e}", exc_info=True)
            self.running = False
            self.stop_requested = True

            # Segnala l'errore agli altri thread
            try:
                self.raw_frame_queue.put(None, block=False)
            except queue.Full:
                pass

        finally:
            # Assicurati che la videocamera sia rilasciata
            if 'cap' in locals() and cap.isOpened():
                cap.release()
                self.logger.debug("Risorse video rilasciate.")

    def _frame_converter(self):
        """
        Thread che converte i frame in ASCII e li mette in coda.

        Prende i frame dalla coda raw_frame_queue, li elabora in batch
        e inserisce i frame ASCII nella coda ascii_frame_queue.
        """
        batch = []

        # Buffer per tracciare le metriche di conversione
        conversion_count = 0
        total_conversion_time = 0

        while self.running and not self.stop_requested:
            try:
                # Preleva un frame dalla coda
                frame = self.raw_frame_queue.get(block=True, timeout=0.1)

                # Controlla se è la fine del video
                if frame is None:
                    # Elabora il batch rimanente
                    if batch:
                        start_time = time.time()
                        ascii_frames = batch_process_frames(batch, self.width)
                        conversion_time = time.time() - start_time

                        if self.log_performance:
                            self.conversion_times.append(conversion_time)

                            # Aggiorna le metriche di conversione
                            conversion_count += 1
                            total_conversion_time += conversion_time
                            avg_time = total_conversion_time / conversion_count
                            self.logger.debug(
                                f"Conversione batch #{conversion_count}: {conversion_time * 1000:.2f}ms (media: {avg_time * 1000:.2f}ms)")

                        # Metti i frame ASCII in coda
                        for ascii_frame in ascii_frames:
                            self.ascii_frame_queue.put(ascii_frame)

                    # Segnala la fine
                    self.ascii_frame_queue.put(None)
                    break

                # Aggiungi il frame al batch
                batch.append(frame)

                # Se il batch è completo, elaboralo
                if len(batch) >= self.batch_size:
                    start_time = time.time()
                    ascii_frames = batch_process_frames(batch, self.width)
                    conversion_time = time.time() - start_time

                    if self.log_performance:
                        self.conversion_times.append(conversion_time)

                        # Aggiorna le metriche di conversione
                        conversion_count += 1
                        total_conversion_time += conversion_time
                        avg_time = total_conversion_time / conversion_count
                        self.logger.debug(
                            f"Conversione batch #{conversion_count}: {conversion_time * 1000:.2f}ms (media: {avg_time * 1000:.2f}ms)")

                    # Metti i frame ASCII in coda
                    for ascii_frame in ascii_frames:
                        try:
                            self.ascii_frame_queue.put(ascii_frame, block=True, timeout=0.1)
                        except queue.Full:
                            # Se la coda è piena, rileva e registra nei log
                            if not self.stop_requested:
                                self.logger.warning("Coda dei frame ASCII piena, frame saltato")

                    # Svuota il batch
                    batch = []

            except queue.Empty:
                # Se la coda è vuota, aspetta un po'
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Errore nel thread converter: {e}", exc_info=True)
                self.running = False
                self.stop_requested = True
                break

    def _frame_renderer(self):
        """
        Thread che mostra i frame ASCII nel terminale.
        """
        last_fps_check = time.time()
        frames_count = 0
        frames_rendered = 0

        while self.running and not self.stop_requested:
            try:
                # Preleva un frame ASCII dalla coda
                ascii_frame = self.ascii_frame_queue.get(block=True, timeout=0.1)

                # Controlla se è la fine del video
                if ascii_frame is None:
                    break

                # Stampa il frame (senza bisogno del frame precedente)
                start_time = time.time()
                print_frame(ascii_frame)
                render_time = time.time() - start_time

                if self.log_performance:
                    self.render_times.append(render_time)

                # Aggiorna il conteggio dei frame per il calcolo degli FPS
                frames_count += 1
                frames_rendered += 1
                current_time = time.time()
                elapsed = current_time - last_fps_check

                # Calcola e registra gli FPS ogni secondo
                if elapsed >= 1.0:
                    fps = frames_count / elapsed
                    if self.log_fps:
                        # Usa \033[K per cancellare il resto della riga
                        print(f"\033[0m\nFPS: {fps:.2f} | Frames: {frames_rendered}\033[K", end="")
                    if self.log_performance:
                        self.display_fps.append(fps)
                    frames_count = 0
                    last_fps_check = current_time

                    # Log dettagliato delle performance
                    if self.log_performance and fps < self.target_fps * 0.9:
                        self.logger.debug(f"Performance sotto target: {fps:.2f} FPS vs {self.target_fps} target")

            except queue.Empty:
                # Se la coda è vuota, aspetta un po'
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Errore nel thread renderer: {e}", exc_info=True)
                self.running = False
                self.stop_requested = True
                break

    def _print_performance_stats(self):
        """
        Registra le statistiche di performance utilizzando il logger.

        Calcola e registra i tempi medi di conversione e rendering,
        e gli FPS medi di visualizzazione, oltre a metriche aggiuntive.
        """
        if not self.conversion_times or not self.render_times:
            self.logger.warning("Nessuna statistica di performance disponibile.")
            return

        # Calcola statistiche di base
        avg_conversion_time = sum(self.conversion_times) / len(self.conversion_times)
        avg_render_time = sum(self.render_times) / len(self.render_times)
        avg_fps = sum(self.display_fps) / len(self.display_fps) if self.display_fps else 0

        # Calcola statistiche aggiuntive
        max_conversion_time = max(self.conversion_times)
        min_conversion_time = min(self.conversion_times)
        max_render_time = max(self.render_times)
        min_render_time = min(self.render_times)
        max_fps = max(self.display_fps) if self.display_fps else 0
        min_fps = min(self.display_fps) if self.display_fps else 0

        # Registra tutte le statistiche
        self.logger.info("--- STATISTICHE DI PERFORMANCE ---")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Frame target FPS: {self.target_fps}")
        self.logger.info(f"Tempo medio di conversione per batch: {avg_conversion_time*1000:.2f} ms (min: {min_conversion_time*1000:.2f} ms, max: {max_conversion_time*1000:.2f} ms)")
        self.logger.info(f"Tempo medio di rendering per frame: {avg_render_time*1000:.2f} ms (min: {min_render_time*1000:.2f} ms, max: {max_render_time*1000:.2f} ms)")
        self.logger.info(f"FPS medi di visualizzazione: {avg_fps:.2f} (min: {min_fps:.2f}, max: {max_fps:.2f})")
        self.logger.info(f"Numero totale di batch elaborati: {len(self.conversion_times)}")
        self.logger.info(f"Numero totale di frame renderizzati: {len(self.render_times)}")

        # Mostra anche a schermo per l'utente
        # print("\n\nStatistiche di performance (salvate nei log):")
        # print(f"Tempo medio di conversione per batch: {avg_conversion_time*1000:.2f} ms")
        # print(f"Tempo medio di rendering per frame: {avg_render_time*1000:.2f} ms")
        # print(f"FPS medi di visualizzazione: {avg_fps:.2f}")
