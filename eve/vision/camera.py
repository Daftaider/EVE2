import numpy as np
import cv2
import time
import logging
import os
import queue

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, config=None, camera_index=0, resolution=(640, 480), fps=30, **kwargs):
        self.logger = logging.getLogger(__name__)
        
        # Initialize with defaults or config values
        if config:
            self.camera_index = getattr(config, 'CAMERA_INDEX', camera_index)
            self.resolution = getattr(config, 'RESOLUTION', resolution)
            self.fps = getattr(config, 'FPS', fps)
        else:
            self.camera_index = camera_index
            self.resolution = resolution
            self.fps = fps
        
        self.cap = None
        self.mock_mode = False
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test capture
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture test frame")
                
        except Exception as e:
            self.logger.warning(f"Failed to open camera {self.camera_index}, using mock camera")
            self.mock_mode = True
            self._init_mock_camera()

    def _init_mock_camera(self):
        """Initialize mock camera with test patterns"""
        self.mock_patterns = []
        
        # Generate test patterns
        for i in range(3):
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            
            # Add face-like features
            center_x = self.resolution[0] // 2
            center_y = self.resolution[1] // 2
            
            # Draw face outline
            radius = min(self.resolution) // 3
            cv2.circle(frame, (center_x, center_y), radius, (200, 200, 200), 2)
            
            # Draw eyes
            eye_radius = radius // 4
            left_eye = (center_x - radius//2, center_y - radius//4)
            right_eye = (center_x + radius//2, center_y - radius//4)
            cv2.circle(frame, left_eye, eye_radius, (255, 255, 255), -1)
            cv2.circle(frame, right_eye, eye_radius, (255, 255, 255), -1)
            
            # Draw mouth
            mouth_y = center_y + radius//3
            cv2.line(frame, 
                    (center_x - radius//2, mouth_y),
                    (center_x + radius//2, mouth_y),
                    (255, 255, 255), 2)
            
            self.mock_patterns.append(frame)
        
        self.current_pattern = 0
        self.pattern_change_time = time.time()
        self.logger.info(f"Mock camera initialized with {len(self.mock_patterns)} test patterns")

    def read(self):
        """Read a frame from the camera or mock camera"""
        if self.mock_mode:
            return self._get_mock_frame()
        
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                self.last_frame_time = time.time()
            return ret, frame
        
        return False, None

    def _get_mock_frame(self):
        """Generate a mock frame"""
        current_time = time.time()
        
        # Change pattern every 3 seconds
        if current_time - self.pattern_change_time > 3.0:
            self.current_pattern = (self.current_pattern + 1) % len(self.mock_patterns)
            self.pattern_change_time = current_time
        
        frame = self.mock_patterns[self.current_pattern].copy()
        
        # Add some random noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.int8)
        frame = cv2.add(frame, noise)
        
        self.frame_count += 1
        self.last_frame_time = time.time()
        
        return True, frame

    def get_fps(self):
        """Calculate actual FPS"""
        if self.frame_count == 0:
            return 0.0
        elapsed_time = time.time() - self.last_frame_time
        if elapsed_time == 0:
            return self.fps
        return min(self.frame_count / elapsed_time, self.fps)

    def is_open(self):
        """Check if camera is open and operational"""
        if self.mock_mode:
            return True
        return self.cap is not None and self.cap.isOpened()

    def get_resolution(self):
        """Get current resolution"""
        return self.resolution

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.cap = None
        self.frame_count = 0
        self.last_frame_time = 0 