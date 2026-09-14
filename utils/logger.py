"""Logging utility module for Driver Drowsiness Detection System.

Configures application-wide logging handlers and formatters to ensure
consistent, structured logging and prevent raw print statements across
the entire codebase.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "drowsiness_detection",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Set up and configure a structured logger.

    Args:
        name: Name of the logger instance.
        log_level: Standard logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional Path to log file for persistent disk logs.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter with timestamp, level, module name, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console stream handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application logger instance
logger = setup_logger()
