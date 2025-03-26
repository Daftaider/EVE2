import numpy as np
import cv2
import time
import logging
import os
import queue

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0, resolution=(640, 480), mock_if_failed=True, fps=30):
        """Initialize camera with fallback to mock"""
        self.logger = logging.getLogger(__name__)
        self.resolution = resolution
        self.fps = fps
        self.mock_mode = False
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest frames
        
        # Frame timing
        self.last_frame_time = time.time()
        self.frame_interval = 1.0 / fps
        self.frame_count = 0
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_index}")
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
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
        
        # Generate test patterns
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
            
            # Add face-like features
            eye_color = (255, 255, 255)
            eye_size = min(self.resolution) // 16
            left_eye_pos = (self.resolution[0]//3, self.resolution[1]//2)
            right_eye_pos = (2*self.resolution[0]//3, self.resolution[1]//2)
            
            cv2.circle(frame, left_eye_pos, eye_size, eye_color, -1)
            cv2.circle(frame, right_eye_pos, eye_size, eye_color, -1)
            
            # Add mouth
            mouth_start = (self.resolution[0]//3, 2*self.resolution[1]//3)
            mouth_end = (2*self.resolution[0]//3, 2*self.resolution[1]//3)
            cv2.line(frame, mouth_start, mouth_end, (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(frame,
                       f"Mock Frame {i+1}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (255, 255, 255),
                       2)
            
            self.mock_patterns.append(frame)
        
        self.current_pattern = 0
        self.pattern_change_time = time.time()
        self.last_frame_time = time.time()
        self.logger.info(f"Mock camera initialized with {len(self.mock_patterns)} test patterns")

    def read(self):
        """Read a frame from the camera"""
        current_time = time.time()
        
        # Enforce frame rate
        if current_time - self.last_frame_time < self.frame_interval:
            time.sleep(max(0, self.frame_interval - (current_time - self.last_frame_time)))
        
        if self.mock_mode:
            ret, frame = self._get_mock_frame()
        elif self.cap is not None:
            ret, frame = self.cap.read()
        else:
            return False, None
            
        if ret:
            self.last_frame_time = time.time()
            self.frame_count += 1
            
        return ret, frame

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
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.frame_count = 0
        self.last_frame_time = 0 