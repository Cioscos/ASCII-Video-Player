import sys
from typing import Optional, TextIO


class TerminalOutputBuffer:
    """
    Buffer ottimizzato per l'output del terminale con minimizzazione degli overhead.

    Accumula testo in un buffer interno e lo scrive al terminale solo quando necessario,
    riducendo significativamente il numero di operazioni I/O e migliorando le prestazioni.
    """

    def __init__(self, stdout: Optional[TextIO] = None, max_buffer_size: int = 1024 * 1024):
        """
        Inizializza il buffer di output con configurazione ottimizzata.

        Args:
            stdout (TextIO, optional): Stream di output, default: sys.stdout
            max_buffer_size (int): Dimensione massima del buffer in bytes prima del flush automatico
        """
        self.buffer = []  # Lista di stringhe da combinare
        self.stdout = stdout or sys.stdout
        self.max_buffer_size = max_buffer_size
        self.buffered_bytes = 0

    def write(self, text: str) -> None:
        """
        Aggiunge testo al buffer interno.

        Il testo non viene scritto immediatamente sullo stream di output ma viene
        accumulato fino a quando non viene chiamato flush() o la dimensione supera
        max_buffer_size.

        Args:
            text (str): Testo da aggiungere al buffer
        """
        self.buffer.append(text)
        self.buffered_bytes += len(text)

        # Flush automatico solo se superiamo significativamente la dimensione massima
        if self.buffered_bytes >= self.max_buffer_size:
            self.flush()

    def flush(self) -> None:
        """
        Scrive tutto il contenuto del buffer sullo stream di output e svuota il buffer.

        Combina tutte le stringhe in un'unica operazione di scrittura per minimizzare
        le operazioni di I/O e migliorare le prestazioni.
        """
        if not self.buffer:
            return

        try:
            # Una singola operazione di scrittura per tutto il buffer
            combined_text = ''.join(self.buffer)
            self.stdout.write(combined_text)
            self.stdout.flush()
        except Exception:
            # Gestione silenziosa degli errori di I/O
            pass

        # Reset del buffer
        self.buffer.clear()
        self.buffered_bytes = 0

    def clear(self) -> None:
        """
        Svuota il buffer senza scrivere sullo stream di output.

        Utile quando si desidera annullare l'output accumulato senza visualizzarlo.
        """
        self.buffer.clear()
        self.buffered_bytes = 0
