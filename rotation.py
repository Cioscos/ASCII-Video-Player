"""Funzioni pure per la rotazione del video in riproduzione."""

from ascii_converter import auto_width, compute_ascii_height

# Angoli (gradi, senso orario) in ordine di ciclo
ROTATIONS = (0, 90, 180, 270)


def rotated_dimensions(src_w, src_h, rotation, width, term_cols, term_rows, reserved_rows):
    """
    Dimensioni ASCII (larghezza, altezza in caratteri) del video ruotato.

    A 90/270 gradi larghezza e altezza sorgente si scambiano. Con `width=None`
    la larghezza è calcolata da `auto_width`; una larghezza esplicita a 90/270
    viene ridotta se non entra nel terminale.
    """
    if rotation in (90, 270):
        src_w, src_h = src_h, src_w
    fit = auto_width(src_w, src_h, term_cols, term_rows, reserved_rows)
    if width is None:
        w = fit
    elif rotation in (90, 270):
        w = min(width, fit)
    else:
        w = width
    return w, compute_ascii_height(src_w, src_h, w)


class RotatingSource:
    """
    FrameSource che alterna cache (a 0 gradi) e pipeline dal vivo (altri angoli).

    Senza cache usa sempre una pipeline. Ricorda l'indice dell'ultimo frame
    restituito: la nuova sorgente riparte dal frame successivo.
    """

    def __init__(self, make_pipeline, total, cache=None, start_timeout=15.0):
        self._make_pipeline = make_pipeline
        self._total = total
        self._cache = cache
        self._start_timeout = start_timeout
        self.rotation = 0
        self._pipeline = None
        self._psource = None
        self._pending = None  # primo frame già letto dalla pipeline
        self._last = -1
        self._failed = False

    @property
    def slow_seek(self):
        """True quando la sorgente corrente è una pipeline."""
        return self._pipeline is not None

    def __len__(self):
        return self._total

    def start(self):
        """Senza cache avvia la pipeline a 0 gradi (senza attendere il primo frame)."""
        if self._cache is None and self._pipeline is None:
            self._launch(0, 0)

    def next_frame(self):
        """Restituisce (indice, frame) oppure None a fine video o errore."""
        if self._pending is not None:
            item, self._pending = self._pending, None
        elif self._failed:
            return None
        elif self._pipeline is None:
            item = self._cache.next_frame() if self._cache is not None else None
        else:
            item = self._psource.next_frame()
        if item is not None:
            self._last = item[0]
        return item

    def restart(self):
        """Riparte dal primo frame (le pipeline si riavviano da sole)."""
        if self._pipeline is None and self._cache is not None:
            self._cache.restart()
            self._last = -1

    def seek(self, index):
        """Il prossimo frame sarà quello con questo indice."""
        self._pending = None
        self._failed = False
        if self._pipeline is None and self._cache is not None:
            self._cache.seek(index)
        else:
            self._stop_pipeline()
            try:
                self._launch(self.rotation, index, wait=True)
            except Exception:
                self._failed = True
        self._last = index - 1

    def rotate(self, step):
        """Ruota di step * 90 gradi. Restituisce False (invariato) se fallisce."""
        previous = self.rotation
        angle = ROTATIONS[(ROTATIONS.index(previous) + step) % len(ROTATIONS)]
        start = self._last + 1
        self._stop_pipeline()
        self._pending = None
        if angle == 0 and self._cache is not None:
            self._cache.seek(start)
            self.rotation = 0
            self._failed = False
            return True
        try:
            self._launch(angle, start, wait=True)
        except Exception:
            self._stop_pipeline()
            self._restore(previous, start)
            return False
        self.rotation = angle
        self._failed = False
        return True

    def close(self):
        """Ferma la pipeline corrente (idempotente)."""
        self._stop_pipeline()

    def _launch(self, rotation, start, wait=False):
        """Crea e avvia una pipeline; con wait tiene il primo frame come prossimo."""
        pipeline = self._make_pipeline(rotation, start)
        self._pipeline = pipeline
        self._psource = None
        pipeline.start()
        self._psource = pipeline.frame_source()
        if wait:
            first = self._psource.next_frame(timeout=self._start_timeout)
            if first is None:
                raise RuntimeError("Nessun frame dalla pipeline")
            self._pending = first

    def _stop_pipeline(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._psource = None

    def _restore(self, rotation, start):
        """Ripristina la sorgente precedente dopo una rotazione fallita."""
        self._pending = None
        self.rotation = rotation
        if rotation == 0 and self._cache is not None:
            self._cache.seek(start)
            return
        try:
            self._launch(rotation, start)
        except Exception:
            self._stop_pipeline()
            self._failed = True
