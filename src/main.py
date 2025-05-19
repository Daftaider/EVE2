"""
Main entry point for EVE2.
"""
import logging
import time
import os
import yaml # Added for reading config
from pathlib import Path
from services.interaction_manager import InteractionManager

# logger will be configured in setup_logging
logger = logging.getLogger(__name__)

def setup_logging(default_config_path='config/settings.yaml') -> None:
    """Set up logging configuration from YAML file."""
    config = {}
    # Determine project root to correctly locate the config file
    try:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / default_config_path
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                log_config = config.get('logging', {})
        else:
            print(f"Warning: Logging configuration file not found at {config_path}. Using basic console logging.")
            log_config = {}

    except Exception as e:
        print(f"Error reading logging config: {e}. Using basic console logging.")
        log_config = {}

    log_level_str = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_path_str = log_config.get('file', 'src/logs/eve2.log') # Default if not in config

    log_level = getattr(logging, log_level_str, logging.INFO)

    # Basic configuration for console
    logging.basicConfig(level=log_level, format=log_format)

    if log_file_path_str:
        try:
            # Ensure log_file_path is relative to project_root if it's a relative path like 'src/logs/eve2.log'
            # If it's absolute, Path will handle it correctly.
            log_file_path = Path(log_file_path_str)
            if not log_file_path.is_absolute():
                log_file_path = project_root / log_file_path_str
            
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Get the root logger and add the file handler
            # If also wanting console output, basicConfig might already handle it.
            # If only file, then need to be careful.
            # For now, let's assume basicConfig handles console and we add file handler.
            logging.getLogger().addHandler(file_handler)
            # If we want to set the level for ALL handlers including console from config:
            logging.getLogger().setLevel(log_level)

            print(f"Logging to file: {log_file_path} at level {log_level_str}")
            
        except Exception as e:
            print(f"Error setting up file logger: {e}")


def main():
    """Main entry point."""
    # Setup logging first using the config file for paths and levels
    # Resolve project_root and config_path once for both logging and InteractionManager
    project_root_path = Path(__file__).resolve().parent.parent
    default_config_file = 'config/settings.yaml'
    config_path_for_logging = project_root_path / default_config_file
    
    # Pass the string path to setup_logging, it will resolve relative to project root
    setup_logging(default_config_path=str(default_config_file))


    logger.info(f"Using configuration file: {config_path_for_logging}")

    try:
        # Create and start EVE2 with the correct config path
        # InteractionManager expects a string path
        with InteractionManager(config_path=str(config_path_for_logging)) as eve:
            # Main loop
            try:
                while True:
                    time.sleep(0.1)  # Small sleep to prevent CPU hogging
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                
    except Exception as e:
        logger.error(f"Error running EVE2: {e}", exc_info=True) # Added exc_info for more detail
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 