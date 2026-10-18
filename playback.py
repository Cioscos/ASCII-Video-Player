"""Logica di riproduzione condivisa tra le modalità (senza dipendenze pesanti)."""
import logging
import math
import threading
import time
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

SEEK_SHORT_S = 5
SEEK_LONG_S = 30
PAUSE_TEXT = "⏸ In pausa"
SEEK_UNAVAILABLE_TEXT = "Seek disponibile solo con --high-performance"
STATUS_SECONDS = 2.0
ROTATING_TEXT = "Rotazione {}°…"
ROTATE_FAILED_TEXT = "Rotazione non riuscita"
SEEK_WAIT_TEXT = "Seek…"

_QUIT_KEYS = ("q", "esc")
_ROTATE_STEPS = {"r": 1, "R": -1}
_SEEK_SECONDS = {"left": -SEEK_SHORT_S, "right": SEEK_SHORT_S, "down": -SEEK_LONG_S, "up": SEEK_LONG_S}


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

    def source_index(self, decimated_index):
        """Indice sorgente del frame decimato `decimated_index` (inverso di `should_keep`)."""
        if self.ratio >= 1.0 or decimated_index <= 0:
            return max(0, int(decimated_index))
        # Formula chiusa: il frame decimato i è il primo sorgente s con floor(s * ratio) >= i
        s = max(0, math.ceil((decimated_index - 1e-9) / self.ratio))
        # Correzione degli errori di arrotondamento in virgola mobile
        while s > 0 and not self.should_keep(s):
            s += 1
        while s > 1 and self.should_keep(s - 1) and math.floor((s - 1) * self.ratio + 1e-9) >= decimated_index:
            s -= 1
        return s


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


def seek_target(current: int, delta: int, total: int, loop: bool) -> int:
    """
    Indice di destinazione di un seek di `delta` frame a partire da `current`.

    Indietro oltre l'inizio → 0; avanti oltre la fine → 0 con il loop,
    altrimenti l'ultimo frame. Con `total <= 0` (numero di frame sconosciuto)
    il seek in avanti non ha limite superiore.
    """
    target = current + delta
    if target < 0:
        return 0
    if total <= 0:
        return target
    if target >= total:
        return 0 if loop else total - 1
    return target


class _Overlay:
    """Righe di overlay correnti: statistiche più un'eventuale riga di stato a tempo."""

    def __init__(self):
        self.stats_lines = []
        self.status = None
        self.status_until = 0.0
        self.dirty = False  # l'overlay va ridisegnato al prossimo frame

    def lines(self, extra=None):
        """Lista esplicita delle righe (mai None, così l'overlay precedente viene sostituito)."""
        lines = list(self.stats_lines)
        if self.status:
            lines.append(self.status)
        if extra:
            lines.append(extra)
        return lines

    def set_status(self, text, now):
        self.status = text
        self.status_until = now + STATUS_SECONDS
        self.dirty = True

    def expire(self, now):
        """Rimuove la riga di stato scaduta."""
        if self.status and now >= self.status_until:
            self.status = None
            self.dirty = True


class _KeyHandler:
    """Gestione dei tasti di controllo (pausa, seek, uscita) per run_playback."""

    def __init__(self, controls, source, fps, loop, display, clock, audio, should_stop, overlay):
        self.controls = controls
        self.source = source
        self.fps = fps
        self.loop = loop
        self.display = display
        self.clock = clock
        self.audio = audio
        self.should_stop = should_stop
        self.overlay = overlay
        self.interrupted = False  # l'ultima azione ha interrotto il ritmo (pausa, rotazione, seek lento)
        self.pending = None  # tasto letto durante lo svuotamento, da gestire per primo

    def next_key(self):
        """Tasto in sospeso, se c'è, altrimenti il prossimo tasto disponibile."""
        if self.pending is not None:
            key, self.pending = self.pending, None
            return key
        return self.controls.poll()

    def handle(self, key, n, last):
        """
        Esegue l'azione del tasto.

        Args:
            key (str): nome del tasto
            n (int): contatore del clock del prossimo frame da estrarre
            last (tuple | None): (indice, frame) dell'ultimo frame disegnato

        Returns:
            bool: True se la riproduzione deve terminare
        """
        self.interrupted = False
        if key in _QUIT_KEYS:
            return True
        if key == "space":
            self.interrupted = True
            return self._pause(n, last)
        if self._is_slow(key):
            return self._slow_action(key, n, last)
        if key in _SEEK_SECONDS:
            self._seek(_SEEK_SECONDS[key], n, last)
        return False

    def _can_rotate(self):
        return hasattr(self.source, "rotate")

    def _can_seek(self):
        return hasattr(self.source, "seek")

    def _is_slow(self, key):
        """True per i tasti che avviano un'operazione lenta (rotazione o seek con pipeline)."""
        if key in _ROTATE_STEPS:
            return self._can_rotate()
        return key in _SEEK_SECONDS and self._can_seek() and getattr(self.source, "slow_seek", False)

    def _slow_action(self, key, n, last):
        """
        Rotazione e/o seek lenti, accorpando i tasti già in attesa.

        Prima dell'operazione si svuotano i tasti disponibili: i passi di r/R
        si sommano (modulo 4) e i secondi delle frecce si sommano in un unico
        seek; un tasto di uscita termina subito, qualsiasi altro tasto ferma lo
        svuotamento e resta in sospeso. Se ci sono sia rotazione sia seek, si
        applica prima la rotazione e poi il seek. Dopo l'operazione si scartano
        r/R e frecce arrivati durante l'attesa (uscita sempre rispettata).

        Returns:
            bool: True se la riproduzione deve terminare
        """
        steps = 0
        seconds = 0
        while key is not None:
            if key in _QUIT_KEYS:
                return True
            if key in _ROTATE_STEPS:
                if self._can_rotate():
                    steps += _ROTATE_STEPS[key]
            elif key in _SEEK_SECONDS:
                if self._can_seek():
                    seconds += _SEEK_SECONDS[key]
            else:
                self.pending = key
                break
            key = self.controls.poll()

        net = steps % 4
        if net == 3:
            net = -1  # tre quarti di giro orari = un quarto antiorario
        if net:
            self._rotate(net, n, last)
        if seconds:
            self._seek(seconds, n, last)
        if net or seconds:
            return self._discard_repeats()
        return False

    def _discard_repeats(self):
        """
        Scarta r/R e frecce arrivati durante un'operazione lenta.

        Un tasto di uscita termina la riproduzione; il primo altro tasto resta
        in sospeso. Se c'è già un tasto in sospeso non si legge nulla, per non
        alterare l'ordine dei tasti premuti prima dell'operazione.

        Returns:
            bool: True se la riproduzione deve terminare
        """
        if self.pending is not None:
            return False
        while True:
            key = self.controls.poll()
            if key is None:
                return False
            if key in _QUIT_KEYS:
                return True
            if key in _ROTATE_STEPS or key in _SEEK_SECONDS:
                continue
            self.pending = key
            return False

    def _pause(self, n, last):
        """Pausa fino a `space` (ripresa) o `q`/`esc`/should_stop (uscita)."""
        if last is not None:
            self.display.show(last[1], self.overlay.lines(PAUSE_TEXT))
        if self.audio is not None:
            self.audio.set_paused(True)
        logger.info("Riproduzione in pausa")
        while not self.should_stop.is_set():
            key = self.controls.read(0.05)
            if key in _QUIT_KEYS:
                return True
            if key == "space":
                break
        else:
            return True  # should_stop: l'audio viene fermato dal chiamante
        if self.audio is not None:
            self.audio.set_paused(False)
        self.clock.rebase(n)  # nessun frame saltato per il tempo passato in pausa
        if last is not None:
            self.display.show(last[1], self.overlay.lines())
        logger.info("Riproduzione ripresa")
        return False

    def _wait_screen(self, text, last):
        """Mostra l'ultimo frame con la riga di attesa e mette in pausa l'audio."""
        if last is not None:
            self.display.show(last[1], self.overlay.lines(text))
        if self.audio is not None:
            self.audio.set_paused(True)

    def _rotate(self, step, n, last):
        """Ruota il video di step quarti di giro (positivo = orario) se la sorgente lo supporta."""
        if not hasattr(self.source, "rotate"):
            return
        self.interrupted = True
        target = (getattr(self.source, "rotation", 0) + 90 * step) % 360
        self._wait_screen(ROTATING_TEXT.format(target), last)
        ok = self.source.rotate(step)
        self.display.invalidate()  # le dimensioni del frame cambiano: ridisegno completo
        self.overlay.dirty = True
        if not ok:
            self.overlay.set_status(ROTATE_FAILED_TEXT, self.clock.now())
        self.clock.rebase(n)
        if self.audio is not None:
            self.audio.seek(((last[0] + 1) if last is not None else 0) / self.fps)
            self.audio.set_paused(False)
        logger.info("Rotazione di %+d° (angolo %d, esito %s)", 90 * step, target, ok)

    def _seek(self, seconds, n, last):
        """Seek di `seconds` secondi rispetto all'ultimo frame disegnato."""
        if not hasattr(self.source, "seek"):
            self.overlay.set_status(SEEK_UNAVAILABLE_TEXT, self.clock.now())
            return
        current = last[0] if last is not None else 0
        target = seek_target(current, round(seconds * self.fps), len(self.source), self.loop)
        slow = getattr(self.source, "slow_seek", False)
        if slow:
            self.interrupted = True
            self._wait_screen(SEEK_WAIT_TEXT, last)
        self.source.seek(target)
        self.clock.rebase(n)  # il frame di destinazione viene disegnato subito
        if self.audio is not None:
            self.audio.seek(target / self.fps)
            if slow:
                self.audio.set_paused(False)
        if slow:
            self.overlay.dirty = True  # toglie la riga di attesa
        logger.info("Seek di %+ds: frame %d → %d", seconds, current, target)


def run_playback(source, fps, display, should_stop: threading.Event, loop,
                 stats=None, audio=None, clock: Optional[PlaybackClock] = None,
                 controls=None, seek_loop=None) -> int:
    """
    Loop di riproduzione comune a entrambe le modalità.

    Il contatore `n` del clock avanza per ogni frame estratto (disegnato o
    saltato). I frame in ritardo di almeno un frame (ma meno di 1 s) vengono
    saltati; con ritardo >= 1 s (stallo) l'orologio viene riancorato.

    Con `controls` (oggetto con `poll()` e `read(timeout)`) i tasti vengono
    letti una volta per frame: `space` pausa, `q`/`esc` uscita, frecce seek
    (solo se la sorgente ha `seek`), `r`/`R` rotazione (solo se ha `rotate`).
    Prima di un'operazione lenta (rotazione, seek con `slow_seek`) i tasti
    ripetuti già in attesa vengono accorpati in una sola operazione, e quelli
    arrivati durante l'attesa vengono scartati (vedi `_KeyHandler._slow_action`).

    `seek_loop` decide se un seek in avanti oltre la fine torna a 0 (True) o si
    ferma all'ultimo frame (False); None usa `loop`. Serve quando il loop è
    gestito dalla sorgente (modalità standard) e run_playback riceve loop=False.

    Returns:
        int: numero di frame estratti dalla sorgente e disegnati
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
    last = None  # (indice, frame) dell'ultimo frame disegnato
    overlay_state = _Overlay()
    keys = None
    if controls is not None:
        keys = _KeyHandler(controls, source, fps, loop if seek_loop is None else seek_loop,
                           display, clock, audio, should_stop, overlay_state)

    while not should_stop.is_set():
        if keys is not None:
            key = keys.next_key()
            if key is not None:
                if keys.handle(key, n, last):
                    break
                if keys.interrupted:
                    last_draw = None  # l'intervallo a cavallo di pausa, rotazione o seek lento non entra nelle statistiche

        fetch_start = clock.now()
        item = source.next_frame()
        fetch_time = clock.now() - fetch_start
        if item is None:
            if not loop or not got_frame:
                break  # fine video, oppure sorgente vuota dopo il restart
            source.restart()
            got_frame = False
            continue
        got_frame = True
        index, frame = item

        late = clock.lateness(n)
        if fetch_time > frame_time:
            # Ritardo dovuto alla sorgente lenta: si riancora, senza saltare
            clock.rebase(n)
            logger.debug("Sorgente lenta (%.3fs): orologio riancorato al frame %d", fetch_time, n)
        elif late >= 1.0:
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

        if stats is not None and drawn % 5 == 0:
            overlay_state.stats_lines = stats.overlay_lines(f"Frame {index}")
            overlay_state.dirty = True
        overlay_state.expire(clock.now())
        overlay = None  # None mantiene l'overlay già mostrato
        if overlay_state.dirty:
            overlay = overlay_state.lines()
            overlay_state.dirty = False
        display.show(frame, overlay)
        drawn += 1
        last = item

        now = clock.now()
        if stats is not None and last_draw is not None:
            stats.record(now - last_draw)
        last_draw = now

    return drawn
