import cv2
import time
import logging
import numpy as np
import os
import queue

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        """Initialize camera with specified parameters or fall back to mock mode
        
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
        
        # Try to initialize real camera
        if self._try_initialize_camera():
            self.logger.info(f"Camera initialized successfully: index={self.camera_index}, "
                            f"resolution={self.resolution}, fps={self.fps}")
        else:
            self.logger.info("Using mock camera mode")
            self.mock_mode = True
            self._init_mock_camera()

    def _try_initialize_camera(self):
        """Try to initialize a real camera, return True if successful"""
        try:
            # Redirect stderr temporarily to suppress OpenCV warnings
            original_stderr = os.dup(2)
            os.close(2)
            os.open(os.devnull, os.O_WRONLY)
            
            try:
                # Try to open camera
                self.cap = cv2.VideoCapture(self.camera_index)
                success = self.cap.isOpened()
                
                if success:
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Test capture
                    ret, frame = self.cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self.logger.warning(f"Camera {self.camera_index} opened but couldn't capture frame")
                        success = False
            finally:
                # Restore stderr
                os.close(2)
                os.dup(original_stderr)
                os.close(original_stderr)
            
            # Clean up if not successful
            if not success and self.cap is not None:
                self.cap.release()
                self.cap = None
                
            return success
            
        except Exception as e:
            self.logger.warning(f"Error initializing camera: {e}")
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            return False

    def _init_mock_camera(self):
        """Initialize mock camera with realistic face patterns"""
        self.mock_patterns = []
        
        # Create 3 different mock face patterns
        for i in range(3):
            frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame.fill(64)  # Dark gray background
            
            # Vary face position slightly in each pattern
            offset_x = (i - 1) * 50  # -50, 0, 50 pixels
            center_x = self.resolution[0] // 2 + offset_x
            center_y = self.resolution[1] // 2
            
            # Draw face outline
            radius = min(self.resolution) // 3
            face_color = (205, 200, 255)  # BGR skin tone
            cv2.ellipse(frame, 
                       (center_x, center_y),
                       (radius, int(radius * 1.2)),
                       0, 0, 360, face_color, -1)
            
            # Add eyes, eyebrows, nose, mouth
            eye_radius = radius // 6
            eye_offset_x = radius // 2
            eye_offset_y = radius // 4
            
            # Eyes
            for x_mult in [-1, 1]:  # Left and right eyes
                x = center_x + x_mult * eye_offset_x
                y = center_y - eye_offset_y
                # White of eye
                cv2.circle(frame, (x, y), eye_radius, (255, 255, 255), -1)
                # Pupil
                cv2.circle(frame, (x, y), eye_radius // 2, (50, 50, 50), -1)
                # Eyebrow
                eyebrow_y = y - eye_radius - 5
                cv2.line(frame,
                        (x - eye_radius - 5, eyebrow_y),
                        (x + eye_radius + 5, eyebrow_y),
                        (100, 100, 100), 3)
            
            # Nose
            nose_top = (center_x, center_y - radius//8)
            nose_bottom = (center_x, center_y + radius//8)
            nose_width = radius // 4
            cv2.line(frame, nose_top, nose_bottom, (180, 180, 180), 2)
            cv2.ellipse(frame, 
                       nose_bottom,
                       (nose_width, nose_width//2),
                       0, 0, 180, (180, 180, 180), 2)
            
            # Mouth (slight smile, neutral, or slight frown based on index)
            mouth_y = center_y + radius//2
            mouth_width = radius - 10
            if i == 0:  # Slight smile
                cv2.ellipse(frame,
                          (center_x, mouth_y - 10),
                          (mouth_width//2, mouth_width//3),
                          0, 0, 180, (150, 150, 150), 2)
            elif i == 1:  # Neutral
                cv2.line(frame,
                        (center_x - mouth_width//2, mouth_y),
                        (center_x + mouth_width//2, mouth_y),
                        (150, 150, 150), 2)
            else:  # Slight frown
                cv2.ellipse(frame,
                          (center_x, mouth_y + 10),
                          (mouth_width//2, mouth_width//3),
                          0, 180, 360, (150, 150, 150), 2)
            
            # Add text label
            mood = ["Happy", "Neutral", "Sad"][i]
            cv2.putText(frame,
                      f"Mock Face: {mood}",
                      (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7,
                      (255, 255, 255),
                      2)
            
            # Add mock camera indicator
            cv2.putText(frame,
                      "MOCK CAMERA - NO REAL CAMERA AVAILABLE",
                      (10, self.resolution[1] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.5,
                      (0, 0, 255),
                      1)
            
            self.mock_patterns.append(frame)
        
        self.current_pattern = 0
        self.pattern_change_time = time.time()
        self.logger.info(f"Mock camera initialized with {len(self.mock_patterns)} face patterns")

    def _get_mock_frame(self):
        """Generate a mock frame with simulated motion"""
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
            else:
                # Camera may have been disconnected
                self.logger.warning("Failed to read from camera, switching to mock mode")
                self.mock_mode = True
                self._init_mock_camera()
                return self._get_mock_frame()
            return ret, frame
        
        # Fallback to mock if cap is not valid
        if not self.mock_mode:
            self.logger.warning("Camera no longer available, switching to mock mode")
            self.mock_mode = True
            self._init_mock_camera()
        return self._get_mock_frame()

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