"""Logica di riproduzione condivisa tra le modalità (senza dipendenze pesanti)."""
import logging
import math
import threading
import time
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


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

    def now(self):
        """Istante corrente secondo la sorgente del tempo dell'orologio."""
        return self._now()

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


class FrameSource(Protocol):
    """Sorgente di frame già convertiti, letta dal thread principale."""

    def next_frame(self) -> Optional[tuple]:
        """Restituisce (indice, frame in bytes) oppure None a fine video."""
        ...

    def restart(self) -> None:
        """Riporta la sorgente al primo frame (per il loop)."""
        ...


def run_playback(source, fps, display, should_stop: threading.Event, loop,
                 stats=None, audio=None, clock: Optional[PlaybackClock] = None) -> int:
    """
    Loop di riproduzione comune a entrambe le modalità.

    Il contatore `n` del clock avanza per ogni frame estratto (disegnato o
    saltato). I frame in ritardo di almeno un frame (ma meno di 1 s) vengono
    saltati; con ritardo >= 1 s (stallo) l'orologio viene riancorato.

    Returns:
        int: numero di frame disegnati
    """
    if clock is None:
        clock = PlaybackClock(fps)
    if clock.t0 is None:
        clock.start()
    frame_time = 1.0 / fps
    n = 0
    drawn = 0
    last_draw = None
    got_frame = False  # almeno un frame ottenuto dall'ultimo avvio/restart

    while not should_stop.is_set():
        item = source.next_frame()
        if item is None:
            if not loop or not got_frame:
                break  # fine video, oppure sorgente vuota dopo il restart
            source.restart()
            got_frame = False
            continue
        got_frame = True
        index, frame = item

        late = clock.lateness(n)
        if late >= 1.0:
            clock.rebase(n)
            logger.debug("Stallo di %.2fs: orologio riancorato al frame %d", late, n)
        elif late > frame_time:
            logger.debug("Frame %d saltato (ritardo %.3fs)", index, late)
            n += 1
            continue
        else:
            clock.wait_for(n)
        n += 1

        if audio is not None:
            audio.update_video_time(index / fps)

        overlay = None
        if stats is not None and drawn % 5 == 0:
            overlay = stats.overlay_lines(f"Frame {index}")
        display.show(frame, overlay)
        drawn += 1

        now = clock.now()
        if stats is not None and last_draw is not None:
            stats.record(now - last_draw)
        last_draw = now

    return drawn
