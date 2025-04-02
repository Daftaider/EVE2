"""
Logging utilities for the EVE2 system.

This module provides functions for setting up and configuring logging.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Remove direct import of old config
# from eve import config 

def setup_logging(level: str = 'INFO', 
                  log_file: Optional[str] = None, 
                  max_bytes: int = 10 * 1024 * 1024, 
                  backup_count: int = 3, 
                  log_to_console: bool = True,
                  log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                  date_format: str = '%Y-%m-%d %H:%M:%S'):
    """Set up logging configuration based on provided parameters."""
    try:
        # Get the numeric log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level) # Set level on root logger

        # Remove existing handlers to avoid duplicates during re-configuration
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handlers properly

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Configure Console Handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Configure File Handler
        if log_file:
            try:
                # Ensure log directory exists
                log_dir = Path(log_file).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Use RotatingFileHandler
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as file_log_err:
                 # Log error to console if possible, but don't crash
                 print(f"ERROR: Failed to configure file logging to '{log_file}': {file_log_err}", file=sys.stderr)

        # Log confirmation message using the new configuration
        root_logger.info("Logging system initialized.")
        
        # --- REMOVE OLD LOGIC ---
        # # Use environment variable or default to INFO
        # log_level_name = os.environ.get("EVE_LOG_LEVEL", "INFO")
        # log_level = getattr(logging, log_level_name, logging.INFO)
        # 
        # # Configure the root logger
        # logging.basicConfig(
        #     level=log_level,
        #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S'
        # )
        # 
        # # Add file handler if log file path is specified
        # log_file = os.environ.get("EVE_LOG_FILE")
        # ... (rest of old file handler logic)
        # 
        # # Set specific loggers (this should be handled by root logger level now)
        # if hasattr(config, 'LOGGER_LEVELS'):
        #     for logger_name, logger_level in config.LOGGER_LEVELS.items():
        #         logging.getLogger(logger_name).setLevel(logger_level)
        # logging.info("Logging system initialized")

    except Exception as setup_err:
         # Fallback basic config if setup fails catastrophically
         logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.critical(f"CRITICAL ERROR during logging setup: {setup_err}. Using fallback config.", exc_info=True)

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