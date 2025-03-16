"""
Logging utilities for Gray Swan Arena.

This module provides utilities for setting up and configuring logging
throughout the Gray Swan system.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    logger_name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (if None, logs to console only)
        log_format: Custom log format (if None, uses default format)
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Skip if logger is already configured
    if logger.handlers:
        return logger
    
    # Set log level
    logger.setLevel(log_level)
    
    # Default log format
    if log_format is None:
        log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add timestamp to log file name if not already present
        if not any(x in log_file for x in ['%Y', '%m', '%d', '%H', '%M', '%S']):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base, ext = os.path.splitext(log_file)
            log_file = f"{base}_{timestamp}{ext}"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
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
