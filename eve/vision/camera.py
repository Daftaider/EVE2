"""
Simple camera interface for EVE2.
"""
import logging
import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logger.warning("Picamera2 not available, falling back to OpenCV camera")

class Camera:
    """Simple camera interface that works with both Picamera2 and OpenCV."""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
        """Initialize camera with specified resolution and FPS."""
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self._capture_thread = None
        
    def start(self) -> bool:
        """Start the camera capture."""
        try:
            if PICAMERA_AVAILABLE:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                
            self.running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
            
    def stop(self) -> None:
        """Stop the camera capture."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            
        if PICAMERA_AVAILABLE and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()
            
        self.camera = None
            
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the camera."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def _capture_loop(self) -> None:
        """Continuous capture loop."""
        while self.running:
            try:
                if PICAMERA_AVAILABLE:
                    frame = self.camera.capture_array()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                        
                    # Convert BGR to RGB for consistency
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                with self.frame_lock:
                    self.latest_frame = frame
                    
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 