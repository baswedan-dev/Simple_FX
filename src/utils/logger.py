import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import sys
from .config_loader import load_config

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with file and console handlers

    Args:
        name: Logger name
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    config = load_config()
    log_config = config.get('logging', {})

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_config.get('level', 'INFO'))

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Create formatter
    formatter = logging.Formatter(log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file or 'file' in log_config:
        log_file_path = log_file or log_config['file']
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger