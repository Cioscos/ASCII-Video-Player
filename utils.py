import logging
import os
import sys
from datetime import datetime


def setup_logging(log_fps=False, log_performance=False, console_level=logging.WARNING):
    """
    Configura il sistema di logging.

    Args:
        log_fps (bool): Se True, abilita il logging degli FPS
        log_performance (bool): Se True, abilita il logging delle prestazioni
        console_level (int): Livello di logging per la console (default: WARNING)

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

    # Imposta il livello di logging globale in base ai parametri
    if log_fps or log_performance:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Handler per il file di log - tutti i messaggi secondo il livello globale
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Handler per la console - solo messaggi di livello WARNING o superiore per default
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)  # Importante: imposta il livello del handler
    logger.addHandler(console_handler)

    logger.info(f"Logging configurato. File di log: {log_filename}")

    return logger


def configure_process_logging(process_name, console_level=logging.WARNING):
    """
    Configura il logging per un processo o thread specifico.
    Da usare nei processi figlio per configurare il logging separatamente.

    Args:
        process_name (str): Nome del processo/thread per identificare la fonte dei log
        console_level (int): Livello di logging per la console

    Returns:
        logging.Logger: Logger configurato per il processo
    """
    # Configura il logging di base
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - %(levelname)s - {process_name} - %(message)s',
        handlers=[]  # Non aggiungere handler di default
    )

    logger = logging.getLogger(process_name)

    # Solo i log al file, non al terminale durante il rendering
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # File handler - tutti i messaggi vengono registrati
    file_handler = logging.FileHandler(f"{log_dir}/{process_name}.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - solo messaggi importanti (WARNING+)
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

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
