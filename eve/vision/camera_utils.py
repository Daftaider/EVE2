import logging
import subprocess
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import picamera2, but don't fail if it's not available
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA = True
except ImportError:
    HAVE_PICAMERA = False
    logger.warning("picamera2 not available, will try other camera options")

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

class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.camera_type = None

    def initialize(self):
        """Initialize camera with available hardware"""
        try:
            # First try Raspberry Pi camera if available
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

            # Try USB camera with different backends
            backends = [
                cv2.CAP_V4L2,
                cv2.CAP_GSTREAMER,
                cv2.CAP_ANY
            ]
            
            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(0 + backend)
                    if self.camera.isOpened():
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            self.camera_type = 'usb'
                            self.logger.info(f"Initialized USB camera with backend {backend}")
                            return True
                        self.camera.release()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize USB camera with backend {backend}: {e}")
                    if self.camera:
                        self.camera.release()
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
                return self.camera.read()
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