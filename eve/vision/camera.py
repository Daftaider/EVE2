import numpy as np
import cv2
import time
import logging
import os
import queue

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        """Initialize camera with specified parameters
        
        Args:
            camera_index (int): Index of the camera to use
            resolution (tuple): Desired resolution as (width, height)
            fps (int): Desired frames per second
        """
        self.logger = logging.getLogger(__name__)
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        
        self.cap = None
        self.mock_mode = False
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.current_frame = None
        
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
        """Initialize mock camera with test patterns including faces"""
        self.mock_patterns = []
        
        # Generate test patterns with faces
        for i in range(3):
            # Create base frame
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame.fill(64)  # Dark gray background
            
            # Add face-like features
            center_x = self.resolution[0] // 2
            center_y = self.resolution[1] // 2
            
            # Draw face outline (skin colored)
            radius = min(self.resolution) // 3
            face_color = (205, 200, 255)  # BGR format
            cv2.ellipse(frame, 
                       (center_x, center_y),
                       (radius, int(radius * 1.2)),
                       0, 0, 360, face_color, -1)
            
            # Draw eyes
            eye_color = (255, 255, 255)
            eye_radius = radius // 6
            eye_offset_x = radius // 2
            eye_offset_y = radius // 4
            
            # Left eye
            cv2.circle(frame, 
                      (center_x - eye_offset_x, center_y - eye_offset_y),
                      eye_radius, eye_color, -1)
            cv2.circle(frame,
                      (center_x - eye_offset_x, center_y - eye_offset_y),
                      eye_radius // 2, (50, 50, 50), -1)
            
            # Right eye
            cv2.circle(frame,
                      (center_x + eye_offset_x, center_y - eye_offset_y),
                      eye_radius, eye_color, -1)
            cv2.circle(frame,
                      (center_x + eye_offset_x, center_y - eye_offset_y),
                      eye_radius // 2, (50, 50, 50), -1)
            
            # Draw mouth (slight smile)
            mouth_color = (150, 150, 150)
            mouth_start = (center_x - radius//2, center_y + radius//3)
            mouth_end = (center_x + radius//2, center_y + radius//3)
            mouth_control = (center_x, center_y + radius//2)
            
            # Draw curved mouth using Bezier curve
            pts = np.array([mouth_start, mouth_control, mouth_end], np.int32)
            cv2.polylines(frame, [pts], False, mouth_color, 2)
            
            # Add frame counter
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
        self.logger.info(f"Mock camera initialized with {len(self.mock_patterns)} test patterns")

    def _get_mock_frame(self):
        """Generate a mock frame with proper type handling"""
        current_time = time.time()
        
        # Change pattern every 3 seconds
        if current_time - self.pattern_change_time > 3.0:
            self.current_pattern = (self.current_pattern + 1) % len(self.mock_patterns)
            self.pattern_change_time = current_time
        
        # Get base pattern
        frame = self.mock_patterns[self.current_pattern].copy()
        
        # Add subtle movement to simulate video
        shift_x = int(np.sin(current_time * 2) * 5)
        shift_y = int(np.cos(current_time * 2) * 5)
        
        # Create translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply the translation
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Ensure frame is valid
        if frame is None or frame.size == 0:
            self.logger.error("Failed to generate mock frame")
            return False, None
        
        self.frame_count += 1
        self.last_frame_time = time.time()
        self.current_frame = frame
        
        return True, frame

    def get_frame(self):
        """Get the latest frame"""
        ret, frame = self.read()
        return ret, frame

    def read(self):
        """Read a frame from the camera or mock camera"""
        if self.mock_mode:
            return self._get_mock_frame()
        
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                self.last_frame_time = time.time()
                self.current_frame = frame
            return ret, frame
        
        return False, None

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
        return self.mock_mode or (self.cap is not None and self.cap.isOpened())

    def get_resolution(self):
        """Get current resolution"""
        return self.resolution

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.cap = None
        self.current_frame = None
        self.frame_count = 0 