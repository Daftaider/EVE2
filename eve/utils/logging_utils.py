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

def setup_logging():
    """Set up logging configuration for EVE"""
    # Use environment variable or default to INFO
    log_level_name = os.environ.get("EVE_LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log file path is specified
    log_file = os.environ.get("EVE_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific loggers to different levels if needed
    if hasattr(config, 'LOGGER_LEVELS'):
        for logger_name, logger_level in config.LOGGER_LEVELS.items():
            logging.getLogger(logger_name).setLevel(logger_level)
    
    logging.info("Logging system initialized")

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