"""
Main entry point for EVE2.
"""
import logging
import time
import os
from pathlib import Path
from services.interaction_manager import InteractionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    try:
        # Get the absolute path to the config file
        # Go up one level from src/main.py location to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config', 'settings.yaml')
        logger.info(f"Using configuration file: {config_path}") # Log the resolved path

        # Create and start EVE2 with the correct config path
        with InteractionManager(config_path=config_path) as eve:
            # Main loop
            try:
                while True:
                    time.sleep(0.1)  # Small sleep to prevent CPU hogging
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                
    except Exception as e:
        logger.error(f"Error running EVE2: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 