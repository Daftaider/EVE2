import numpy as np
import cv2
import time
import logging

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_mock = False
        self.frame_count = 0
        self.logger = logging.getLogger(__name__)
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.logger.warning(f"Failed to open camera {camera_index}, using mock camera")
                self.is_mock = True
            else:
                self.logger.info(f"Camera {camera_index} initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            self.is_mock = True
            
    def get_frame(self):
        """Get a frame from the camera or generate a mock frame if camera not available"""
        if self.is_mock:
            return self._get_mock_frame()
            
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.logger.warning("Failed to capture frame from camera")
                return self._get_mock_frame()
        else:
            return self._get_mock_frame()
            
    def _get_mock_frame(self):
        """Generate a mock frame for testing"""
        self.frame_count += 1
        
        # Create a black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a moving face-like circle
        center_x = 320 + int(50 * np.sin(self.frame_count / 30))
        cv2.circle(frame, (center_x, 240), 100, (200, 200, 200), -1)
        
        # Draw eyes
        left_eye_x = center_x - 40
        right_eye_x = center_x + 40
        cv2.circle(frame, (left_eye_x, 220), 20, (255, 255, 255), -1)
        cv2.circle(frame, (right_eye_x, 220), 20, (255, 255, 255), -1)
        
        # Draw mouth - smile or frown based on position
        smile = np.sin(self.frame_count / 50) > 0
        if smile:
            cv2.ellipse(frame, (center_x, 280), (50, 20), 0, 0, 180, (255, 255, 255), -1)
        else:
            cv2.ellipse(frame, (center_x, 300), (50, 20), 0, 180, 360, (255, 255, 255), -1)
            
        # Add some text
        cv2.putText(frame, "EVE Mock Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Simulate processing delay
        time.sleep(0.03)
        
        return frame
        
    def release(self):
        """Release the camera resources"""
        if self.cap and not self.is_mock:
            self.cap.release() 