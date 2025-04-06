import sounddevice as sd
from moviepy import *
import time
import threading
import queue
import logging


class AudioPlayer:
    """
    Classe per la riproduzione sincronizzata dell'audio di un video.
    Gestisce la sincronizzazione A/V anche quando gli FPS sono più lenti del previsto.

    Attributes:
        video_path (str): Percorso del file video
        target_fps (float): FPS target per la riproduzione
        should_stop (threading.Event): Flag per la terminazione
        audio_thread (threading.Thread): Thread per la riproduzione dell'audio
        sync_queue (queue.Queue): Coda per la sincronizzazione A/V
        audio_time (float): Tempo corrente di riproduzione audio
        video_time (float): Tempo corrente di riproduzione video
        sync_tolerance (float): Tolleranza di sincronizzazione in secondi
    """

    def __init__(self, video_path, target_fps=None):
        """
        Inizializza il player audio.

        Args:
            video_path (str): Percorso del file video
            target_fps (float, optional): FPS target per la riproduzione
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.should_stop = threading.Event()
        self.audio_thread = None
        self.sync_queue = queue.Queue()
        self.audio_time = 0.0
        self.video_time = 0.0
        self.sync_tolerance = 0.1  # 100ms di tolleranza per la sincronizzazione
        self.logger = logging.getLogger('AudioPlayer')
        self.playback_started = False
        self.resync_event = threading.Event()
        self.audio_clip = None
        self.stream = None
        self.audio_array = None
        self.samplerate = 0
        self.initialized = False

    def initialize(self):
        """
        Inizializza l'audio estraendolo dal video.
        """
        try:
            # Estrai audio dal video usando moviepy
            self.logger.info(f"Estrazione audio da {self.video_path}")
            video_clip = VideoFileClip(self.video_path)

            # Ottieni l'audio clip dal video
            self.audio_clip = video_clip.audio
            if self.audio_clip is None:
                self.logger.warning("Il video non contiene audio")
                return False

            # Ottieni i dati audio come array numpy
            self.audio_array = self.audio_clip.to_soundarray()
            self.samplerate = self.audio_clip.fps

            # Converti in mono se stereo (opzionale ma semplifica)
            if len(self.audio_array.shape) > 1 and self.audio_array.shape[1] > 1:
                self.audio_array = self.audio_array  # Mantieni stereo

            self.logger.info(f"Audio estratto: {self.samplerate} Hz, durata: {self.audio_clip.duration} sec")
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione dell'audio: {e}")
            return False

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Callback per il sounddevice stream.
        Fornisce i dati audio da riprodurre e gestisce la sincronizzazione.

        Args:
            outdata: Buffer di output per i dati audio
            frames: Numero di frame da fornire
            time_info: Informazioni sul tempo
            status: Stato della riproduzione
        """
        if status:
            self.logger.warning(f"Status audio: {status}")

        # Calcola l'indice di inizio per i dati audio
        start_idx = int(self.audio_time * self.samplerate)
        end_idx = start_idx + frames

        # Verifica se abbiamo raggiunto la fine dell'audio
        if start_idx >= len(self.audio_array):
            # Fine dell'audio raggiunta
            outdata.fill(0)
            if not self.should_stop.is_set():
                self.should_stop.set()
            return sd.CallbackStop

        # Aggiorna il tempo audio corrente
        self.audio_time += frames / self.samplerate

        # Verifica di sincronizzazione con il video
        try:
            # Controlla se ci sono comandi di sincronizzazione nella coda
            while not self.sync_queue.empty():
                cmd = self.sync_queue.get_nowait()
                if cmd["type"] == "set_time":
                    # Aggiorna il tempo audio
                    new_time = cmd["time"]
                    self.logger.debug(f"Sincronizzazione: audio={self.audio_time:.3f}, video={new_time:.3f}")

                    # Se la differenza è significativa, risincronizza
                    if abs(self.audio_time - new_time) > self.sync_tolerance:
                        self.audio_time = new_time
                        start_idx = int(new_time * self.samplerate)
                        end_idx = start_idx + frames
                        self.logger.info(f"Risincronizzazione audio a {new_time:.3f}s")
        except queue.Empty:
            pass

        # Copia i dati audio nel buffer di output
        if end_idx <= len(self.audio_array):
            if len(self.audio_array.shape) > 1:  # Stereo
                outdata[:] = self.audio_array[start_idx:end_idx]
            else:  # Mono
                outdata[:, 0] = self.audio_array[start_idx:end_idx]
        else:
            # Gestisci il caso in cui raggiungiamo la fine dell'audio
            remaining = len(self.audio_array) - start_idx
            if remaining > 0:
                if len(self.audio_array.shape) > 1:  # Stereo
                    outdata[:remaining] = self.audio_array[start_idx:]
                else:  # Mono
                    outdata[:remaining, 0] = self.audio_array[start_idx:]
                outdata[remaining:] = 0
            else:
                outdata.fill(0)

            # Segnala la fine dell'audio
            if not self.should_stop.is_set():
                self.should_stop.set()
            return sd.CallbackStop

    def _audio_thread_func(self):
        """
        Funzione per il thread di riproduzione audio.
        """
        self.logger.info("Thread audio avviato")

        try:
            # Crea uno stream audio
            with sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=2 if len(self.audio_array.shape) > 1 else 1,
                    callback=self._audio_callback
            ) as self.stream:
                self.logger.info("Riproduzione audio avviata")
                self.playback_started = True

                # Attendi che il thread debba terminare
                while not self.should_stop.is_set():
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Errore nella riproduzione audio: {e}")
        finally:
            self.logger.info("Thread audio terminato")

    def start(self):
        """
        Avvia la riproduzione dell'audio.
        """
        if not self.initialized and not self.initialize():
            self.logger.warning("Audio non disponibile o non inizializzato")
            return False

        # Avvia il thread audio
        self.audio_thread = threading.Thread(
            target=self._audio_thread_func,
            daemon=True
        )
        self.audio_thread.start()
        return True

    def update_video_time(self, current_time):
        """
        Aggiorna il tempo di riproduzione video e gestisce la sincronizzazione.

        Args:
            current_time (float): Tempo corrente di riproduzione video in secondi
        """
        self.video_time = current_time

        # Invia un comando di sincronizzazione se la differenza è significativa
        if abs(self.audio_time - self.video_time) > self.sync_tolerance:
            self.sync_queue.put({
                "type": "set_time",
                "time": current_time
            })

    def stop(self):
        """
        Ferma la riproduzione dell'audio con gestione ottimizzata delle risorse.
        """
        self.logger.info("Arresto riproduzione audio")
        self.should_stop.set()

        # Timeout ridotto per il join del thread audio
        # Questo evita che l'applicazione rimanga bloccata troppo a lungo
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.5)
            # Non aspettiamo all'infinito che il thread termini
            # Se il thread è ancora attivo dopo il timeout, lo consideriamo comunque terminato
            if self.audio_thread.is_alive():
                self.logger.warning("Thread audio non ha terminato entro il timeout")
                # Non chiamiamo più thread.join() perché questo può bloccare il processo

        # Chiudi le risorse audio in modo sicuro
        if self.stream:
            try:
                # Interrompi il callback loop di sounddevice
                self.stream.abort()
                self.stream.close()
            except Exception as e:
                self.logger.error(f"Errore durante la chiusura dello stream audio: {e}")
            finally:
                self.stream = None

        # Chiudi l'audio clip
        if self.audio_clip:
            try:
                self.audio_clip.close()
            except Exception as e:
                self.logger.error(f"Errore durante la chiusura dell'audio clip: {e}")
            finally:
                self.audio_clip = None

        self.initialized = False
        self.playback_started = False
