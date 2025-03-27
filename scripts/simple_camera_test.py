import cv2
import time
import logging
import subprocess
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_v4l_devices() -> None:
    """Check available video devices using v4l2-ctl."""
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True)
        logger.info("Available video devices:\n%s", result.stdout)
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found. Install with: sudo apt-get install v4l-utils")
    except Exception as e:
        logger.error("Failed to list video devices: %s", e)

def get_camera_capabilities(device_id: int = 0) -> None:
    """Get camera capabilities using v4l2-ctl."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', f'--device=/dev/video{device_id}', '--all'],
            capture_output=True, text=True
        )
        logger.info("Camera capabilities:\n%s", result.stdout)
    except Exception as e:
        logger.error("Failed to get camera capabilities: %s", e)

def try_camera_formats(cap: cv2.VideoCapture) -> Optional[Tuple[int, int, int]]:
    """Try different camera formats and resolutions."""
    formats = [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')),
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V')),
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B','G','R','3')),
    ]
    
    resolutions = [(640, 480), (320, 240), (1280, 720)]
    fps_options = [30, 15, 10]

    for fmt, codec in formats:
        cap.set(fmt, codec)
        logger.info("Trying format: %s", codec)
        
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            for fps in fps_options:
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    logger.info(f"Success with: {actual_width}x{actual_height} @ {actual_fps}fps")
                    return actual_width, actual_height, actual_fps
    
    return None

def main():
    logger.info("OpenCV version: %s", cv2.__version__)
    logger.info("OpenCV backend: %s", cv2.getBuildInformation())
    
    # Check available devices
    check_v4l_devices()
    
    # Try to open camera
    logger.info("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully")
    
    # Get initial camera properties
    logger.info("Initial camera properties:")
    logger.info("Backend: %s", cap.getBackendName())
    logger.info("Width: %s", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    logger.info("Height: %s", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("FPS: %s", cap.get(cv2.CAP_PROP_FPS))
    
    # Get camera capabilities
    get_camera_capabilities()
    
    # Try different formats
    logger.info("Testing different camera formats...")
    result = try_camera_formats(cap)
    
    if result:
        width, height, fps = result
        logger.info(f"Found working configuration: {width}x{height} @ {fps}fps")
        
        # Test continuous frame capture
        logger.info("Testing continuous frame capture...")
        start_time = time.time()
        frames_captured = 0
        
        while frames_captured < 30:  # Capture 30 frames
            ret, frame = cap.read()
            if ret:
                frames_captured += 1
                logger.info(f"Successfully captured frame {frames_captured}")
            else:
                logger.error(f"Failed to capture frame {frames_captured + 1}")
            time.sleep(1/fps)  # Respect FPS timing
        
        duration = time.time() - start_time
        logger.info(f"Captured {frames_captured} frames in {duration:.2f} seconds")
        logger.info(f"Effective FPS: {frames_captured/duration:.2f}")
    else:
        logger.error("Could not find working camera configuration")
    
    # Release the camera
    cap.release()
    logger.info("Camera test completed")

if __name__ == "__main__":
    main() 