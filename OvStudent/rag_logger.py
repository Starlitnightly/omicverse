# rag_logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler
import os

class RAGLogger:
    def __init__(self, name: str, log_file: str = 'rag_system.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Rotating file handler
        handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10 MB
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Stream handler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)
