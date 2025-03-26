import numpy as np
import cv2
import time
import logging
import os
import queue

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0, resolution=(640, 480), mock_if_failed=True):
        """Initialize camera with fallback to mock"""
        self.logger = logging.getLogger(__name__)
        self.resolution = resolution
        self.mock_mode = False
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_index}")
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Test capture
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture test frame")
                
        except Exception as e:
            self.logger.warning(f"Failed to open camera {camera_index}, using mock camera")
            if mock_if_failed:
                self.mock_mode = True
                self._init_mock_camera()
            else:
                raise

    def _init_mock_camera(self):
        """Initialize mock camera with generated patterns"""
        self.mock_patterns = []
        
        # Generate some test patterns
        for i in range(3):
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            
            # Add some shapes
            cv2.circle(frame, 
                      (self.resolution[0]//2, self.resolution[1]//2),
                      min(self.resolution)//4,
                      (0, 255, 0),
                      2)
            
            cv2.rectangle(frame,
                         (self.resolution[0]//4, self.resolution[1]//4),
                         (3*self.resolution[0]//4, 3*self.resolution[1]//4),
                         (0, 0, 255),
                         2)
            
            # Add some text
            cv2.putText(frame,
                       f"Mock Camera Frame {i+1}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (255, 255, 255),
                       2)
            
            self.mock_patterns.append(frame)
        
        self.current_pattern = 0
        self.pattern_change_time = time.time()
        self.logger.info(f"Mock camera initialized with {len(self.mock_patterns)} test patterns")

    def _get_mock_frame(self):
        """Generate mock frame with movement simulation"""
        current_time = time.time()
        
        # Change pattern every 3 seconds
        if current_time - self.pattern_change_time > 3.0:
            self.current_pattern = (self.current_pattern + 1) % len(self.mock_patterns)
            self.pattern_change_time = current_time
        
        # Get base pattern
        frame = self.mock_patterns[self.current_pattern].copy()
        
        # Add some random noise to simulate movement
        noise = np.random.normal(0, 5, frame.shape).astype(np.int8)
        frame = cv2.add(frame, noise)
        
        return True, frame

    def read(self):
        """Read a frame from the camera"""
        if self.mock_mode:
            return self._get_mock_frame()
        elif self.cap is not None:
            return self.cap.read()
        return False, None

    def release(self):
        """Release camera resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def get_frame(self):
        """Get a frame from the camera or generate a mock frame if camera not available"""
        # Limit frame rate
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        target_interval = 1.0 / self.fps
        
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
            
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # Return real or mock frame
        if self.mock_mode:
            return self._get_mock_frame()
            
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
            else:
                logger.warning("Failed to capture frame from camera")
                return self._get_mock_frame()
        else:
            return self._get_mock_frame()
        
    def _get_mock_frame(self):
        """Generate a mock frame for testing"""
        # If we have test images, cycle through them
        if self.test_images:
            index = self.frame_count % len(self.test_images)
            return self.test_images[index].copy()
        
        # Otherwise generate a synthetic frame
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving face
        t = self.frame_count / 30.0  # Time parameter
        center_x = width // 2 + int(100 * np.sin(t))
        center_y = height // 2 + int(50 * np.cos(t * 0.7))
        
        # Draw face
        cv2.circle(frame, (center_x, center_y), 100, (200, 200, 200), -1)
        
        # Draw eyes that blink occasionally
        blink = (self.frame_count % 50) < 3  # Blink every ~2 seconds
        eye_h = 5 if blink else 20
        
        cv2.ellipse(frame, (center_x - 30, center_y - 20), (20, eye_h), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (center_x + 30, center_y - 20), (20, eye_h), 0, 0, 360, (255, 255, 255), -1)
        
        # Draw mouth that changes expression
        smile = np.sin(t * 0.5) > 0
        if smile:
            cv2.ellipse(frame, (center_x, center_y + 30), (50, 20), 0, 0, 180, (255, 255, 255), -1)
        else:
            cv2.ellipse(frame, (center_x, center_y + 40), (50, 20), 0, 180, 360, (255, 255, 255), -1)
        
        # Add frame info
        cv2.putText(frame, f"Mock Camera: Frame {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def release(self):
        """Release the camera resources"""
        if self.cap and not self.mock_mode:
            self.cap.release() 