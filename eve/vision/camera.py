import numpy as np
import cv2
import time
import logging
import os

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for accessing video frames"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_mock = False
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 30  # Target FPS
        
        # Try to initialize the actual camera
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                logger.warning(f"Failed to open camera {camera_index}, using mock camera")
                self.is_mock = True
                self._init_mock_camera()
            else:
                logger.info(f"Camera {camera_index} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.is_mock = True
            self._init_mock_camera()
    
    def _init_mock_camera(self):
        """Initialize mock camera resources"""
        # Try to load test images if available
        self.test_images = []
        test_dir = os.path.join("assets", "test_images")
        
        if os.path.exists(test_dir):
            # Load all images from test directory
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(test_dir, file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.test_images.append(img)
                    except Exception as e:
                        logger.error(f"Error loading test image {file}: {e}")
        
        logger.info(f"Mock camera initialized with {len(self.test_images)} test images")
            
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
        if self.is_mock:
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
        if self.cap and not self.is_mock:
            self.cap.release() 