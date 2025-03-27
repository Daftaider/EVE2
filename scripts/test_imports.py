import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python path
logger.info("Python path:")
for path in sys.path:
    logger.info(f"  {path}")

# Print current directory
logger.info(f"Current directory: {os.getcwd()}")

# Try imports
try:
    import eve
    logger.info("Successfully imported eve")
    
    from eve.vision import camera_utils
    logger.info("Successfully imported camera_utils")
    
except ImportError as e:
    logger.error(f"Import failed: {e}")
    
except Exception as e:
    logger.error(f"Unexpected error: {e}") 