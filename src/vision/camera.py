"""
Camera module for handling video input.
"""
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Camera:
    """Camera class for handling video input."""
    
    def __init__(self, resolution: Tuple[int, int] = (800, 480)):
        """Initialize camera with specified resolution."""
        self.resolution = resolution
        self.cap = None
        self.running = False
        
    def start(self) -> bool:
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            self.running = True
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
            
    def stop(self) -> None:
        """Stop camera capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False
        logger.info("Camera stopped")
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from camera."""
        if not self.running or self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
                
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 