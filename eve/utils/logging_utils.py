"""
Logging utilities for the EVE2 system.

This module provides functions for setting up and configuring logging.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from eve import config

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    log_date_format: Optional[str] = None,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None
) -> None:
    """
    Configure the logging system for EVE2.
    
    Args:
        log_level: Logging level (default: from config)
        log_file: Path to log file (default: from config)
        log_format: Log message format (default: from config)
        log_date_format: Log date format (default: from config)
        log_to_console: Whether to log to console (default: from config)
        log_to_file: Whether to log to file (default: from config)
    """
    # Use configuration values if parameters not provided
    log_level = log_level or config.logging.LOG_LEVEL
    log_file = log_file or config.logging.LOG_FILE
    log_format = log_format or config.logging.LOG_FORMAT
    log_date_format = log_date_format or config.logging.LOG_DATE_FORMAT
    log_to_console = log_to_console if log_to_console is not None else config.logging.LOG_TO_CONSOLE
    log_to_file = log_to_file if log_to_file is not None else config.logging.LOG_TO_FILE
    
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, log_date_format)
    
    # Set up console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Set up file handler if enabled and log file is specified
    if log_to_file and log_file:
        # Create the log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log the configuration
    logging.info("Logging initialized")
    logging.debug(f"Log level: {log_level}")
    if log_to_file and log_file:
        logging.debug(f"Log file: {log_file}")

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