import logging
import os
import sys
from datetime import datetime


def setup_logging(log_fps=False, log_performance=False):
    """
    Configura il sistema di logging.

    Args:
        log_fps (bool): Se True, abilita il logging degli FPS
        log_performance (bool): Se True, abilita il logging delle prestazioni

    Returns:
        logging.Logger: L'oggetto logger configurato
    """
    # Crea la directory dei log se non esiste
    os.makedirs('logs', exist_ok=True)

    # Genera un nome file univoco basato sulla data e ora corrente
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/ascii_video_{timestamp}.log'

    # Configura il logger root
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Handler per il file di log
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Handler per la console
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Configura il livello di logging in base ai parametri
    if log_fps or log_performance:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Logging configurato. File di log: {log_filename}")

    return logger


def get_terminal_size():
    """
    Ottiene le dimensioni del terminale.

    Returns:
        tuple: (width, height) del terminale
    """
    try:
        columns, lines = os.get_terminal_size()
        return columns, lines
    except:
        # Valori di default in caso di errore
        return 80, 24
