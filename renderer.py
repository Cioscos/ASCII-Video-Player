import os
import time
import sys
import logging


class AsciiRenderer:
    """
    Classe che gestisce il rendering dei frame ASCII sul terminale.

    Attributes:
        target_fps (int): FPS target per la visualizzazione
        log_fps (bool): Se True, registra le informazioni sugli FPS effettivi
        frame_times (list): Lista di tempi di rendering per il calcolo degli FPS
        last_render_time (float): Timestamp dell'ultimo rendering
        clear_command (str): Comando per pulire lo schermo del terminale
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

        # Determina il metodo di pulizia dello schermo in base al sistema operativo
        if os.name == 'nt':  # Windows
            self.clear_command = 'cls'
        else:  # Unix/Linux/MacOS
            self.clear_command = 'clear'

        self.logger = logging.getLogger('AsciiRenderer')
        self.logger.info(f"Inizializzato renderer con target_fps={target_fps}, log_fps={log_fps}")

    def render_frame(self, ascii_frame):
        """
        Renderizza un frame ASCII sul terminale.

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

        # Pulisci lo schermo
        os.system(self.clear_command)

        # Stampa il frame usando sys.stdout.write
        sys.stdout.write(ascii_frame)
        sys.stdout.flush()

        render_time = time.time() - now
        self.last_render_time = now

        # Registra le informazioni sugli FPS
        if self.log_fps:
            self.frame_times.append(render_time)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)

        return render_time

    def get_fps_stats(self):
        """
        Calcola le statistiche sugli FPS.

        Returns:
            dict: Dizionario con le statistiche sugli FPS
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
