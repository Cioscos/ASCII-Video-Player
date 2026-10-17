"""Logica di riproduzione condivisa tra le modalità (senza dipendenze pesanti)."""
import math
import time


class FrameDecimator:
    """
    Decide quali frame sorgente tenere per ottenere gli FPS richiesti.

    La decisione dipende solo dall'indice del frame sorgente (nessuno stato),
    quindi resta corretta anche dopo il riavvio del loop del video.

    Attributes:
        effective_fps (float): FPS reali di riproduzione (mai sopra gli FPS sorgente)
    """

    def __init__(self, source_fps, target_fps):
        """
        Args:
            source_fps (float): FPS del video sorgente (può essere 0 o NaN se sconosciuto)
            target_fps (float | None): FPS richiesti dall'utente
        """
        valid = isinstance(source_fps, (int, float)) and math.isfinite(source_fps) and source_fps > 0
        if not valid:
            # FPS sorgente sconosciuti: si tengono tutti i frame
            self.effective_fps = float(target_fps or 30.0)
            self.ratio = 1.0
        else:
            self.effective_fps = float(min(target_fps, source_fps)) if target_fps else float(source_fps)
            self.ratio = self.effective_fps / source_fps

    def should_keep(self, source_index):
        """Restituisce True se il frame sorgente con questo indice va tenuto."""
        if self.ratio >= 1.0 or source_index <= 0:
            return True
        return math.floor(source_index * self.ratio + 1e-9) > math.floor((source_index - 1) * self.ratio + 1e-9)


class PlaybackClock:
    """
    Orologio di riproduzione a scadenze assolute (nessuna deriva cumulativa).

    La scadenza del frame n è t0 + n / fps, indipendente dal tempo impiegato
    per i frame precedenti. `n` è un contatore di riproduzione monotono, non
    l'indice del frame video (che si azzera al loop).
    """

    def __init__(self, fps, now=time.perf_counter, sleep=time.sleep):
        """
        Args:
            fps (float): FPS di riproduzione
            now (Callable[[], float]): sorgente del tempo (iniettabile nei test)
            sleep (Callable[[float], None]): funzione di attesa (iniettabile nei test)
        """
        self.fps = fps
        self._now = now
        self._sleep = sleep
        self.t0 = None

    def start(self):
        """Ancora l'istante di partenza."""
        self.t0 = self._now()

    def deadline(self, n):
        """Istante in cui va mostrato il frame n."""
        return self.t0 + n / self.fps

    def wait_for(self, n):
        """Attende fino alla scadenza del frame n, se è nel futuro."""
        if self.t0 is None:
            self.start()
        remaining = self.deadline(n) - self._now()
        if remaining > 0:
            self._sleep(remaining)

    def lateness(self, n):
        """Ritardo (secondi) rispetto alla scadenza del frame n; negativo se in anticipo."""
        return self._now() - self.deadline(n)

    def rebase(self, n):
        """Riancora l'orologio in modo che il frame n sia in scadenza adesso."""
        self.t0 = self._now() - n / self.fps
