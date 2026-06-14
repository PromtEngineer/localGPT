"""
Centralized logging configuration for localGPT backend.
Provides structured logging to both console and file with rotation.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class LogConfig:
    """Centralized logging configuration manager."""

    @staticmethod
    def setup_logging(
        name: str = "localgpt",
        level: str = "INFO",
        log_dir: str = "logs",
        log_file: str = "localgpt.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_enabled: bool = True,
    ) -> logging.Logger:
        """
        Configure logger with file and optional console output.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            log_file: Log filename
            max_bytes: Max size before rotation (bytes)
            backup_count: Number of backup files to keep
            console_enabled: Whether to output to console

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Set level
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Prevent duplicate handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create logs directory if needed
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Formatter for detailed logging
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (simple format for readability)
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler with rotation (detailed format)
        log_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Prevent propagation to root logger (avoids duplicate logs)
        logger.propagate = False

        return logger


def get_logger(name: str = "localgpt") -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
