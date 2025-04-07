import time
import sys
import logging


class AsciiRenderer:
    """
    Classe che gestisce il rendering dei frame ASCII sul terminale.

    Si occupa della visualizzazione sincronizzata dei frame ASCII
    rispettando il target FPS e fornendo statistiche sulla performance.

    Attributes:
        target_fps (int): FPS target per la visualizzazione
        log_fps (bool): Se True, registra le informazioni sugli FPS effettivi
        frame_times (list): Lista di tempi di rendering per il calcolo degli FPS
        last_render_time (float): Timestamp dell'ultimo rendering
    """

    def __init__(self, target_fps=None, log_fps=False):
        """
        Inizializza il renderer ASCII.

        Args:
            target_fps (int, optional): FPS target per la visualizzazione.
                                       Se None, visualizza alla massima velocità possibile.
            log_fps (bool): Se True, registra le informazioni sugli FPS effettivi.
        """
        self.target_fps = target_fps
        self.log_fps = log_fps
        self.frame_times = []
        self.last_render_time = None

        # Sequenze ANSI per il controllo del terminale
        self.CURSOR_HOME = '\033[H'  # Sposta il cursore all'inizio
        self.CLEAR_SCREEN = '\033[2J'  # Pulisce lo schermo

        self.logger = logging.getLogger('AsciiRenderer')
        self.logger.info(f"Inizializzato renderer con target_fps={target_fps}, log_fps={log_fps}")

    def render_frame(self, ascii_frame):
        """
        Renderizza un frame ASCII sul terminale.

        Gestisce il timing in base al target FPS e calcola le statistiche di performance.

        Args:
            ascii_frame (str): Il frame ASCII da renderizzare

        Returns:
            float: Tempo impiegato per renderizzare il frame
        """
        now = time.time()

        # Calcola il tempo trascorso dall'ultimo rendering
        if self.last_render_time is not None:
            elapsed = now - self.last_render_time
            # Se abbiamo un target FPS, aspetta il tempo necessario
            if self.target_fps and elapsed < 1.0 / self.target_fps:
                time.sleep(1.0 / self.target_fps - elapsed)
                now = time.time()

        # Pulisci schermo al primo frame
        if self.last_render_time is None:
            sys.stdout.write(self.CLEAR_SCREEN)

        # Rendering del frame
        sys.stdout.write(self.CURSOR_HOME)
        sys.stdout.write(ascii_frame)
        sys.stdout.flush()

        render_time = time.time() - now
        self.last_render_time = now

        # Registra le statistiche FPS
        if self.log_fps:
            self.frame_times.append(render_time)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)  # Mantieni solo gli ultimi 100 frame

        return render_time

    def get_fps_stats(self):
        """
        Calcola le statistiche sugli FPS.

        Elabora i tempi di rendering per fornire dati sulla performance.

        Returns:
            dict: Dizionario con statistiche sugli FPS {avg_fps, min_fps, max_fps}
        """
        if not self.frame_times:
            return {"avg_fps": 0, "min_fps": 0, "max_fps": 0}

        avg_time = sum(self.frame_times) / len(self.frame_times)
        min_time = min(self.frame_times)
        max_time = max(self.frame_times)

        return {
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "min_fps": 1.0 / max_time if max_time > 0 else 0,
            "max_fps": 1.0 / min_time if min_time > 0 else 0
        }
