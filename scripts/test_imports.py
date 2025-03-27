import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    
    # Print module file locations
    logger.info(f"eve module location: {eve.__file__}")
    logger.info(f"camera_utils location: {camera_utils.__file__}")
    
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("sys.path: %s", sys.path)
    
except Exception as e:
    logger.error(f"Unexpected error: {e}") 