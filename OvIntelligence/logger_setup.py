# logger_setup.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_logger(
    name: Optional[str] = None,
    log_file: str = "streamlit_app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with a RotatingFileHandler and StreamHandler.

    Parameters:
        name (Optional[str]): The name of the logger. If None, the root logger is used.
        log_file (str): The name of the log file to be stored in the 'logs' directory.
        level (int): Logging level.
        max_bytes (int): Maximum bytes before the log is rotated.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevents duplicate logging in the root logger

    # Ensure the logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_path = log_dir / log_file

    # Only add handlers if they are not already present
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Rotating File Handler
        file_handler = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream Handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# Create a default logger that can be imported from other modules
logger = setup_logger(name="default_logger")