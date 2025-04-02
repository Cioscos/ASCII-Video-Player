"""
Funzioni di utilità per l'applicazione ASCII Video.
"""
import os
import shutil
import logging
import time

# Configurazione del logger
def setup_logging():
    """
    Configura il sistema di logging.

    Configura un logger che scrive su file e su console con timestamp e livello.

    Ritorna:
        logging.Logger: Logger configurato.
    """
    # Crea una directory per i log se non esiste
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Nome file log con timestamp
    log_filename = os.path.join(log_dir, f"ascii_video_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Configurazione logger
    logger = logging.getLogger('ascii_video')
    logger.setLevel(logging.INFO)

    # Handler per il file
    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Handler per la console
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Aggiungi gli handler al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log inizializzato in: {log_filename}")

    return logger

def get_terminal_size():
    """
    Ottiene le dimensioni attuali del terminale.

    Ritorna:
        tuple: (colonne, righe) del terminale.
    """
    return shutil.get_terminal_size()


def estimate_height(width, video_aspect_ratio=None):
    """
    Stima l'altezza basata sulla larghezza e sull'aspect ratio del video.

    Parametri:
        width (int): Larghezza in caratteri.
        video_aspect_ratio (float, optional): Aspect ratio del video (altezza/larghezza).
                                            Se None, usa un valore predefinito di 9/16.

    Ritorna:
        int: Altezza stimata in caratteri.
    """
    # Se non è fornito un aspect ratio, assumiamo 16:9 (comune per video)
    if video_aspect_ratio is None:
        video_aspect_ratio = 9 / 16

    # Consideriamo che i caratteri del terminale sono circa il doppio in altezza rispetto alla larghezza
    return int(width * video_aspect_ratio * 2)
