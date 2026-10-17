import sounddevice as sd
import numpy as np
import time
import threading
import logging


class AudioPlayer:
    """
    Classe per la riproduzione sincronizzata dell'audio di un video.

    Gestisce la sincronizzazione audio/video anche quando gli FPS sono più lenti del previsto,
    adattando la riproduzione audio in tempo reale in base all'avanzamento dei frame video.

    Attributes:
        video_path (str): Percorso del file video
        target_fps (float): FPS target per la riproduzione
        should_stop (threading.Event): Flag per la terminazione
        audio_thread (threading.Thread): Thread per la riproduzione dell'audio
        _pending_seek (float | None): Risincronizzazione pendente, applicata dal callback
        _paused (bool): In pausa il callback emette silenzio senza avanzare audio_time
        underflows (int): Numero di callback con status non vuoto (underflow/overflow)
        audio_time (float): Tempo corrente di riproduzione audio in secondi
        video_time (float): Tempo corrente di riproduzione video in secondi
        sync_tolerance (float): Tolleranza di sincronizzazione in secondi
        playback_started (bool): Indica se la riproduzione è iniziata
        resync_event (threading.Event): Flag per forzare la risincronizzazione
        initialized (bool): Indica se l'audio è stato inizializzato
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
        self._pending_seek = None
        self._paused = False
        self.underflows = 0
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

    def load_array(self, samples, samplerate):
        """
        Carica i campioni audio in memoria.

        Converte in float32 C-contiguo; mono viene duplicato su 2 canali e
        più di 2 canali vengono mediati su 2 canali.

        Args:
            samples (numpy.ndarray): Campioni di forma (n,) o (n, canali)
            samplerate (int): Frequenza di campionamento in Hz
        """
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
        elif arr.shape[1] > 2:
            arr = np.repeat(arr.mean(axis=1, keepdims=True), 2, axis=1)
        self.audio_array = np.ascontiguousarray(arr, dtype=np.float32)
        self.samplerate = int(samplerate)
        self.initialized = True

    def initialize(self):
        """
        Inizializza l'audio estraendolo dal video.

        Estrae la traccia audio dal video usando moviepy e la prepara per la riproduzione
        con sounddevice.

        Returns:
            bool: True se l'inizializzazione è avvenuta con successo, False altrimenti
        """
        video_clip = None
        try:
            from moviepy import VideoFileClip

            self.logger.info(f"Estrazione audio da {self.video_path}")
            video_clip = VideoFileClip(self.video_path)

            audio_clip = video_clip.audio
            if audio_clip is None:
                self.logger.warning("Il video non contiene audio")
                return False

            samples = audio_clip.to_soundarray()
            samplerate = audio_clip.fps
            duration = audio_clip.duration
            self.load_array(samples, samplerate)
            self.logger.info(f"Audio estratto: {self.samplerate} Hz, durata: {duration} sec")
            return True
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione dell'audio: {e}")
            return False
        finally:
            if video_clip is not None:
                try:
                    video_clip.close()
                except Exception as e:
                    self.logger.warning(f"Errore nella chiusura del video: {e}")

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Callback per il sounddevice stream (eseguito sul thread audio).

        Applica l'eventuale risincronizzazione pendente, copia i campioni con
        wrap-around (l'audio va in loop insieme al video) e avanza audio_time.
        Se il tempo richiesto è oltre la fine dell'audio emette silenzio.
        Non registra log né restituisce mai CallbackStop.

        Args:
            outdata (numpy.ndarray): Buffer di output per i dati audio
            frames (int): Numero di frame da fornire
            time_info (dict): Informazioni sul tempo
            status (CallbackFlags): Stato della riproduzione
        """
        if status:
            self.underflows += 1

        audio = self.audio_array
        if audio is None:
            outdata.fill(0)
            return
        total = len(audio)
        if total == 0:
            outdata.fill(0)
            return

        seek = self._pending_seek
        if seek is not None:
            self._pending_seek = None
            self.audio_time = seek

        if self._paused:
            # In pausa: silenzio e audio_time fermo (il seek pendente resta applicato).
            outdata.fill(0)
            return

        if self.audio_time * self.samplerate >= total:
            # Video oltre la fine dell'audio (audio più corto): silenzio, senza
            # ripartire dall'inizio, finché un seek non torna sotto la durata.
            # audio_time segue il video, così non si generano seek a ogni frame.
            outdata.fill(0)
            self.audio_time += frames / self.samplerate
            return

        pos = int(self.audio_time * self.samplerate) % total
        done = 0
        while done < frames:
            n = min(frames - done, total - pos)
            outdata[done:done + n] = audio[pos:pos + n]
            done += n
            pos = (pos + n) % total

        self.audio_time = (self.audio_time + frames / self.samplerate) % (total / self.samplerate)

    def _audio_thread_func(self):
        """
        Funzione per il thread di riproduzione audio.

        Avvia lo stream audio e gestisce il ciclo di vita della riproduzione.
        """
        self.logger.info("Thread audio avviato")

        try:
            # Crea uno stream audio
            with sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=2,
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

        Inizializza l'audio se non è già stato fatto e avvia il thread di riproduzione.

        Returns:
            bool: True se l'avvio è avvenuto con successo, False altrimenti
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

        Confronta il tempo video con il tempo audio e registra una risincronizzazione
        pendente quando necessario.

        Args:
            current_time (float): Tempo corrente di riproduzione video in secondi
        """
        self.video_time = current_time

        # Richiede una risincronizzazione (applicata dal callback) se la differenza è significativa
        if abs(self.audio_time - self.video_time) > self.sync_tolerance:
            self._pending_seek = current_time

    def set_paused(self, paused):
        """
        Mette in pausa o riprende l'audio.

        Il flag è letto dal callback sul thread audio (assegnazione atomica,
        come per _pending_seek): in pausa il callback emette silenzio.

        Args:
            paused (bool): True per mettere in pausa, False per riprendere
        """
        self._paused = bool(paused)

    def seek(self, t):
        """
        Sposta subito l'audio al tempo indicato, anche sotto sync_tolerance.

        Args:
            t (float): Tempo di destinazione in secondi
        """
        self.video_time = t
        self._pending_seek = t

    def stop(self):
        """
        Ferma la riproduzione dell'audio con gestione ottimizzata delle risorse.

        Implementa una chiusura non bloccante con un timeout molto breve per evitare
        di bloccare l'applicazione durante la chiusura.
        """
        if not hasattr(self, 'should_stop') or self.should_stop.is_set():
            # Già in fase di chiusura o non inizializzato
            return

        self.logger.info("Arresto riproduzione audio")
        if self.underflows:
            self.logger.warning(f"Underflow/overflow audio: {self.underflows}")
        self.should_stop.set()

        # Timeout ridotto per il join del thread audio
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
            # Timeout di soli 100ms - se non termina velocemente, continuiamo
            self.audio_thread.join(timeout=0.1)

        # Chiudi le risorse audio in modo non bloccante
        if hasattr(self, 'stream') and self.stream:
            try:
                # Interrompi il callback loop di sounddevice
                self.stream.abort()
            except Exception as e:
                self.logger.error(f"Errore durante l'abort dello stream audio: {e}")
            finally:
                self.stream = None

        # Rilascia l'audio clip
        if hasattr(self, 'audio_clip') and self.audio_clip:
            try:
                self.audio_clip = None  # Rilascia immediatamente il riferimento
            except Exception as e:
                self.logger.error(f"Errore durante la chiusura dell'audio clip: {e}")

        # Rilascia immediatamente le risorse di memoria
        if hasattr(self, 'audio_array'):
            self.audio_array = None

        self.initialized = False
        self.playback_started = False
        self.logger.info("Risorse audio rilasciate")
