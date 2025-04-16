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
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.yaml')
        
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