import sys
from typing import Optional, TextIO


class TerminalOutputBuffer:
    """
    Buffer ottimizzato per l'output del terminale con minimizzazione degli overhead.
    Versione semplificata e ottimizzata per prestazioni.
    """

    def __init__(self, stdout: Optional[TextIO] = None, max_buffer_size: int = 1024 * 1024):
        """
        Inizializza il buffer di output con configurazione ottimizzata.

        Args:
            stdout: Stream di output (default: sys.stdout)
            max_buffer_size: Dimensione massima del buffer in bytes prima del flush automatico
        """
        self.buffer = []  # Lista di stringhe da combinare
        self.stdout = stdout or sys.stdout
        self.max_buffer_size = max_buffer_size
        self.buffered_bytes = 0

    def write(self, text: str) -> None:
        """
        Aggiunge testo al buffer.
        Versione semplificata per massimizzare le prestazioni.

        Args:
            text (str): Testo da aggiungere al buffer
        """
        self.buffer.append(text)
        self.buffered_bytes += len(text)  # Stima approssimativa

        # Flush automatico solo se superiamo di molto la dimensione massima
        # per ridurre significativamente il numero di operazioni di flush
        if self.buffered_bytes >= self.max_buffer_size:
            self.flush()

    def flush(self) -> None:
        """
        Scrive tutto il contenuto del buffer sullo stream di output e svuota il buffer.
        Versione ottimizzata per minimizzare le operazioni di I/O.
        """
        if not self.buffer:
            return

        try:
            # Combina tutte le stringhe in un'unica operazione di scrittura
            combined_text = ''.join(self.buffer)
            self.stdout.write(combined_text)
            self.stdout.flush()
        except Exception:
            # Gestisce errori di I/O silenziosamente
            pass

        # Resetta il buffer
        self.buffer.clear()
        self.buffered_bytes = 0

    def clear(self) -> None:
        """
        Svuota il buffer senza scrivere sullo stream di output.
        """
        self.buffer.clear()
        self.buffered_bytes = 0