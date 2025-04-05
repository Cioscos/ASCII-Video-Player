import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging


class FrameConverter:
    """
    Classe che gestisce la conversione dei frame video in caratteri ASCII.
    Utilizza il parallelismo e ottimizzazioni numpy per massimizzare le prestazioni.

    Attributes:
        width (int): Larghezza desiderata dell'output ASCII
        max_workers (int): Numero di worker per l'elaborazione parallela
        ascii_chars (numpy.ndarray): Array di caratteri ASCII ordinati per intensità
        height_scale (float): Fattore di scala per l'altezza del frame
        last_shape (tuple): Forma dell'ultimo frame processato
    """

    def __init__(self, width, max_workers=None):
        """
        Inizializza il convertitore di frame.

        Args:
            width (int): Larghezza desiderata dell'output ASCII
            max_workers (int, optional): Numero massimo di worker per l'elaborazione parallela.
                                        Se None, viene determinato automaticamente.
        """
        self.width = width
        # Usa tutti i core disponibili se non specificato diversamente
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        # Caratteri ASCII ordinati per intensità (dal più scuro al più chiaro)
        self.ascii_chars = np.array(list('@%#*+=-:. '))

        # Prealloca le matrici per le operazioni di resize
        self.height_scale = None
        self.last_shape = None

        self.logger = logging.getLogger('FrameConverter')
        self.logger.info(f"Inizializzato convertitore con width={width}, workers={self.max_workers}")

    def _convert_frame_to_ascii(self, frame):
        """
        Converte un singolo frame in ASCII art usando ottimizzazioni numpy.

        Args:
            frame (numpy.ndarray): Il frame da convertire

        Returns:
            str: Rappresentazione ASCII del frame
        """
        # Ridimensiona il frame alla larghezza desiderata
        height, width, _ = frame.shape

        # Calcola l'altezza proporzionale alla larghezza desiderata
        if self.height_scale is None or self.last_shape != (height, width):
            self.height_scale = self.width / width
            self.last_shape = (height, width)

        new_height = int(height * self.height_scale)
        resized = cv2.resize(frame, (self.width, new_height))

        # Converti a scala di grigi
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Normalizza i valori di grigio su tutta la gamma di caratteri ASCII
        # e mappali al rispettivo indice nell'array ascii_chars
        indices = (gray / 255 * (len(self.ascii_chars) - 1)).astype(np.int)

        # Usa numpy per creare velocemente l'output
        chars = self.ascii_chars[indices]

        # Aggiungi i caratteri di newline
        rows = [''.join(row) for row in chars]
        ascii_frame = '\n'.join(rows)

        return ascii_frame

    def convert_batch(self, frames):
        """
        Converte un batch di frame in ASCII art in parallelo.

        Args:
            frames (list): Lista di frame da convertire

        Returns:
            list: Lista di frame convertiti in ASCII
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._convert_frame_to_ascii, frames))
