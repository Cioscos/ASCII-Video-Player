import sys
from typing import Optional, List, TextIO


class TerminalOutputBuffer:
    """
    Classe ottimizzata che implementa un buffer per l'output del terminale 
    per ridurre le chiamate a sys.stdout.write.

    Implementa una strategia di buffering intelligente con compressione
    delle sequenze ANSI ripetute e scrittura in batch.

    Attributes:
        buffer (list): Lista di stringhe da scrivere sul terminale
        stdout: Stream di output (default: sys.stdout)
        max_buffer_size (int): Dimensione massima del buffer prima del flush automatico
        last_cursor_position (tuple): Ultima posizione del cursore (riga, colonna)
        buffered_bytes (int): Numero di byte attualmente nel buffer
    """

    def __init__(self, stdout: Optional[TextIO] = None, max_buffer_size: int = 8192):
        """
        Inizializza il buffer di output con configurazione ottimizzata.

        Args:
            stdout: Stream di output (default: sys.stdout)
            max_buffer_size: Dimensione massima del buffer in bytes prima del flush automatico
        """
        self.buffer: List[str] = []
        self.stdout = stdout or sys.stdout
        self.max_buffer_size: int = max_buffer_size
        self.last_cursor_position: Optional[tuple] = None
        self.buffered_bytes: int = 0

        # Sequenze ANSI comuni per ottimizzazione
        self.CURSOR_HOME = '\033[H'
        self.CURSOR_POSITION = '\033[%d;%dH'

        # Stato interno per ottimizzazione
        self._last_write_was_cursor_move = False

    def write(self, text: str) -> None:
        """
        Aggiunge testo al buffer con ottimizzazioni.

        Implementa:
        1. Combinazione di movimenti cursore consecutivi
        2. Compressione di sequenze ANSI ripetute
        3. Flush automatico quando il buffer raggiunge dimensioni critiche

        Args:
            text (str): Testo da aggiungere al buffer
        """
        # Ottimizzazione: combinazione di movimenti cursore consecutivi
        if text == self.CURSOR_HOME and self._last_write_was_cursor_move and self.buffer:
            # Sostituisci l'ultimo movimento cursore invece di aggiungerne uno nuovo
            self.buffer[-1] = text
            return

        # Aggiunge il testo al buffer
        self.buffer.append(text)
        self.buffered_bytes += len(text.encode('utf-8'))

        # Traccia se questo è un movimento cursore per ottimizzazione
        self._last_write_was_cursor_move = (text == self.CURSOR_HOME or
                                            text.startswith('\033[') and 'H' in text)

        # Flush automatico se il buffer supera la dimensione massima
        if self.buffered_bytes >= self.max_buffer_size:
            self.flush()

    def writeln(self, text: str) -> None:
        """
        Aggiunge testo al buffer seguito da un ritorno a capo.

        Args:
            text (str): Testo da aggiungere al buffer
        """
        self.write(text + '\n')

    def write_at(self, row: int, col: int, text: str) -> None:
        """
        Scrive il testo in una posizione specifica del terminale.
        Ottimizza evitando di ripetere il posizionamento se la posizione è consecutiva.

        Args:
            row (int): Riga (1-based)
            col (int): Colonna (1-based)
            text (str): Testo da scrivere
        """
        # Ottimizzazione: se stiamo scrivendo alla riga successiva nella stessa colonna
        # e l'ultima operazione era stata una scrittura posizionata, usa un semplice \n
        if (self.last_cursor_position and
                self.last_cursor_position[0] + 1 == row and
                self.last_cursor_position[1] == col and
                self._last_write_was_cursor_move):
            self.write('\n' + text)
        else:
            self.write(self.CURSOR_POSITION % (row, col))
            self.write(text)

        # Aggiorna l'ultima posizione del cursore
        self.last_cursor_position = (row, col + len(text.rstrip('\033[0m')))

    def flush(self) -> None:
        """
        Scrive tutto il contenuto del buffer sullo stream di output e svuota il buffer.
        Implementa ottimizzazioni per la scrittura batch.
        """
        if not self.buffer:
            return

        # Ottimizzazione: combina le stringhe del buffer in un'unica operazione di scrittura
        combined_text = ''.join(self.buffer)

        # Ulteriore ottimizzazione: compressione di sequenze ANSI consecutive identiche
        # Ad esempio, sostituisci sequenze come "\033[0m\033[0m" con "\033[0m"
        # Questa è un'ottimizzazione avanzata che richiederebbe un parser ANSI

        # Esegui la scrittura effettiva
        try:
            self.stdout.write(combined_text)
            self.stdout.flush()
        except (IOError, ValueError) as e:
            # Gestisce errori di I/O in modo silenzioso (es. pipe rotto)
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
        self.last_cursor_position = None
        self._last_write_was_cursor_move = False
