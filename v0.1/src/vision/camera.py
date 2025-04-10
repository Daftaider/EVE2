"""
Camera module for handling video input.
"""
import logging
from typing import Optional, Tuple, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Camera:
    """Camera class for handling video input."""
    
    def __init__(self, resolution: Tuple[int, int] = (800, 480), device_index: int = 0):
        """Initialize camera with specified resolution."""
        self.resolution = resolution
        self.device_index = device_index
        self.cap = None
        self.running = False
        
    def _get_available_cameras(self) -> List[int]:
        """Get list of available camera indices."""
        available = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
        
    def start(self) -> bool:
        """Start camera capture."""
        try:
            # Check available cameras
            available = self._get_available_cameras()
            if not available:
                logger.error("No cameras found")
                return False
                
            # Try to use specified device index, or fall back to first available
            if self.device_index not in available:
                logger.warning(f"Camera {self.device_index} not available, using {available[0]} instead")
                self.device_index = available[0]
                
            # Initialize camera
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
                
            # Get current camera capabilities
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Camera {self.device_index} initialized with resolution {width}x{height}")
            
            # Try to set resolution
            if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0]):
                logger.warning(f"Failed to set width to {self.resolution[0]}")
            if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1]):
                logger.warning(f"Failed to set height to {self.resolution[1]}")
                
            # Verify resolution
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Camera resolution set to {actual_width}x{actual_height}")
            
            self.running = True
            logger.info(f"Camera {self.device_index} started successfully")
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
        logger.info(f"Camera {self.device_index} stopped")
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from camera."""
        if not self.running or self.cap is None:
            logger.warning("Camera not running or not initialized")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from camera {self.device_index}")
                return None
                
            if frame is None or frame.size == 0:
                logger.warning(f"Empty frame received from camera {self.device_index}")
                return None
                
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame from camera {self.device_index}: {e}")
            return None
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 