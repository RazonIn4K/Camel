"""Logging utilities for Gray Swan Arena."""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """Set up logging for the Gray Swan Arena project.

    Args:
        name: Logger name. Defaults to None for root logger.
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to a file (default: True)
        log_filename: Name of the log file (default: None, will generate a timestamped filename)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Get or create the logger
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured already
    if not logger.handlers:
        logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

        # Set up file handler if requested
        if log_to_file:
            if log_filename is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_filename = f"grayswan_{current_date}.log"

            file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logging("LoggingTest")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
