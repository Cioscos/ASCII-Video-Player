"""Logica di riproduzione condivisa tra le modalità (senza dipendenze pesanti)."""
import math


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
