import logging
import subprocess
import os
import cv2
import numpy as np
import time
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import picamera2, but don't fail if it's not available
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA = True
except ImportError:
    HAVE_PICAMERA = False
    logger.warning("picamera2 not available, will try other camera options")

def check_camera_permissions():
    """Check if we have permission to access the camera"""
    try:
        for i in range(10):
            device = f"/dev/video{i}"
            if os.path.exists(device):
                stats = os.stat(device)
                readable = os.access(device, os.R_OK)
                writable = os.access(device, os.W_OK)
                logger.info(f"Camera device {device}:")
                logger.info(f"  Permissions: {oct(stats.st_mode)[-3:]}")
                logger.info(f"  Readable: {readable}")
                logger.info(f"  Writable: {writable}")
    except Exception as e:
        logger.error(f"Error checking camera permissions: {e}")

def check_camera_hardware():
    """Check what camera hardware is available"""
    try:
        # Check for Raspberry Pi camera module
        if HAVE_PICAMERA:
            try:
                result = subprocess.run(['vcgencmd', 'get_camera'], 
                                    capture_output=True, text=True)
                if 'detected=1' in result.stdout:
                    logger.info("Raspberry Pi camera module detected")
                    return 'picamera'
            except Exception as e:
                logger.warning(f"Error checking Pi camera: {e}")
        
        # Check for USB cameras
        for i in range(10):  # Check first 10 possible camera indices
            if os.path.exists(f'/dev/video{i}'):
                logger.info(f"USB camera detected at /dev/video{i}")
                return 'usb'
        
        logger.warning("No cameras detected")
        return 'none'
    except Exception as e:
        logger.error(f"Error checking camera hardware: {e}")
        return 'unknown'

class Camera:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None
        self.mock_mode = False
        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize camera with tested working configuration."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Set tested working configuration
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Verify settings
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise RuntimeError("Failed to capture test frame")

            logger.info("Camera initialized successfully")
            logger.info(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                       f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            logger.info(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
            self.mock_mode = False

        except Exception as e:
            logger.warning(f"Failed to initialize camera: {e}, using mock camera")
            self.mock_mode = True
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.mock_mode:
            return self._get_mock_frame()
        
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera disconnected, attempting to reinitialize...")
            self._init_camera()
            if self.mock_mode:
                return self._get_mock_frame()

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame, switching to mock mode")
            self.mock_mode = True
            return self._get_mock_frame()

        return True, frame

    def _get_mock_frame(self) -> Tuple[bool, np.ndarray]:
        """Generate a mock frame for testing."""
        # Create a 640x480 frame with a moving pattern
        t = time.time()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a moving circle
        cx = int(320 + 200 * np.cos(t))
        cy = int(240 + 200 * np.sin(t))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        # Add text
        cv2.putText(frame, "Mock Camera", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return True, frame

    def release(self) -> None:
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.camera_type = None
        
        # Add debug info
        self.logger.info(f"OpenCV version: {cv2.__version__}")
        self.logger.info(f"Available backends: {[x.name for x in cv2.videoio_registry.getBackends()]}")
        
        # Check camera permissions
        check_camera_permissions()

    def _try_usb_camera(self):
        """Try to initialize USB camera with different settings"""
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for backend, name in backends:
            self.logger.info(f"Trying {name} backend...")
            try:
                # Try direct device access first
                for device_index in range(10):
                    if not os.path.exists(f"/dev/video{device_index}"):
                        continue
                        
                    self.logger.info(f"Trying device: /dev/video{device_index}")
                    cap = cv2.VideoCapture(device_index)
                    
                    if not cap.isOpened():
                        self.logger.warning(f"Could not open device {device_index}")
                        cap.release()
                        continue
                    
                    # Try to set properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Try to read a frame
                    for _ in range(5):  # Multiple attempts
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.logger.info(f"Successfully initialized camera {device_index}")
                            self.logger.info(f"Frame size: {frame.shape}")
                            return cap
                        time.sleep(0.1)
                    
                    cap.release()
                    
            except Exception as e:
                self.logger.error(f"Error with {name} backend: {e}")
                
        return None

    def initialize(self):
        """Initialize camera with available hardware"""
        try:
            # Try USB camera first since we detected one
            self.camera = self._try_usb_camera()
            if self.camera is not None:
                self.camera_type = 'usb'
                self.logger.info("USB camera initialized successfully")
                return True

            # Try Raspberry Pi camera if available
            if HAVE_PICAMERA:
                try:
                    self.camera = Picamera2()
                    config = self.camera.create_preview_configuration(
                        main={"size": (self.width, self.height),
                              "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    self.camera.start()
                    self.camera_type = 'picamera'
                    self.logger.info("Initialized Raspberry Pi camera")
                    return True
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Pi camera: {e}")
                    if self.camera:
                        self.camera.stop()
                        self.camera = None

            # If no camera available, use mock camera
            self.camera_type = 'mock'
            self.logger.warning("No camera available, using mock camera")
            return False

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def read_frame(self):
        """Read a frame from the camera"""
        try:
            if self.camera_type == 'picamera':
                frame = self.camera.capture_array()
                return True, frame
            elif self.camera_type == 'usb':
                for _ in range(3):  # Multiple attempts
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        return True, frame
                    time.sleep(0.1)
                return False, self._create_mock_frame()
            else:
                # Return mock frame
                frame = self._create_mock_frame()
                return True, frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return False, self._create_mock_frame()

    def _create_mock_frame(self):
        """Create a mock frame for testing"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        text = "No Camera Available"
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (self.width - textsize[0]) // 2
        text_y = (self.height + textsize[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
        return frame

    def release(self):
        """Release the camera"""
        try:
            if self.camera_type == 'picamera':
                self.camera.stop()
            elif self.camera_type == 'usb':
                self.camera.release()
            self.camera = None
            self.camera_type = None
        except Exception as e:
            self.logger.error(f"Error releasing camera: {e}") 