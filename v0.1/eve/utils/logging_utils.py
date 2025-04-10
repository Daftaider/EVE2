"""
Logging utilities for the EVE2 system.

This module provides functions for setting up and configuring logging.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Remove direct import of old config
# from eve import config 

def setup_logging(
    level: str,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    log_to_console: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """Set up logging configuration.
    
    Args:
        level: The logging level (e.g., 'DEBUG', 'INFO', etc.)
        log_file: Optional path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        log_to_console: Whether to log to console
        log_format: Format string for log messages
        date_format: Format string for timestamps
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove all existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(log_format, date_format)
    
    if log_to_console:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if log_file:
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for different modules
    
    # Reduce noise from audio system
    logging.getLogger('eve.speech.speech_recognizer').setLevel(logging.ERROR)
    logging.getLogger('eve.speech.audio_capture').setLevel(logging.ERROR)
    
    # Ensure display/input events are visible
    lcd_logger = logging.getLogger('eve.display.lcd_controller')
    lcd_logger.setLevel(logging.INFO)
    
    # Create a special handler for keyboard/mouse events
    class InputEventFilter(logging.Filter):
        def filter(self, record):
            return (
                'Key pressed' in record.msg or
                'Mouse click' in record.msg or
                'Double-click' in record.msg
            )
    
    # Add special handler for input events
    input_handler = logging.StreamHandler()
    input_handler.setFormatter(formatter)
    input_handler.addFilter(InputEventFilter())
    lcd_logger.addHandler(input_handler)
    
    # Log the configuration
    logging.info(f"Logging configured with level: {level}")
    logging.debug("Debug logging enabled")

def get_module_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name
        
    Returns:
        A logger configured for the specified module
    """
    return logging.getLogger(name)

def log_exception(logger: logging.Logger, e: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with detailed information.
    
    Args:
        logger: The logger to use
        e: The exception to log
        message: Optional message to include (default: "An error occurred")
    """
    logger.error(f"{message}: {str(e)}", exc_info=True) 