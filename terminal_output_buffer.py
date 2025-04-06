class TerminalOutputBuffer:
    """
    Classe che implementa un buffer per l'output del terminale per ridurre le chiamate a sys.stdout.write.

    Attributes:
        buffer (list): Lista di stringhe da scrivere sul terminale
        stdout: Stream di output (default: sys.stdout)
    """

    def __init__(self, stdout=None):
        """
        Inizializza il buffer di output.

        Args:
            stdout: Stream di output (default: sys.stdout)
        """
        self.buffer = []
        self.stdout = stdout or sys.stdout

    def write(self, text):
        """
        Aggiunge testo al buffer.

        Args:
            text (str): Testo da aggiungere al buffer
        """
        self.buffer.append(text)

    def flush(self):
        """
        Scrive tutto il contenuto del buffer sullo stream di output e svuota il buffer.
        """
        if self.buffer:
            # Unisci tutte le stringhe nel buffer in un'unica operazione di scrittura
            combined_text = ''.join(self.buffer)
            self.stdout.write(combined_text)
            self.stdout.flush()
            self.buffer.clear()

    def clear(self):
        """
        Svuota il buffer senza scrivere sullo stream di output.
        """
        self.buffer.clear()
