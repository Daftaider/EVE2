import os
import sys
import logging
import time
import cv2

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now we can import eve modules
from eve.vision.camera_utils import check_camera_hardware, CameraManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting camera test...")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {os.getcwd()}")

    try:
        # Check what type of camera is available
        camera_type = check_camera_hardware()
        logger.info(f"Detected camera type: {camera_type}")

        # Initialize the camera
        camera = CameraManager(width=640, height=480, fps=30)
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return

        # Create display window
        window_name = "Camera Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("Starting capture loop...")
        try:
            while True:
                ret, frame = camera.read_frame()
                if ret and frame is not None:
                    cv2.imshow(window_name, frame)
                    
                    # Break on ESC key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            camera.release()
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        logger.info("Test completed")

if __name__ == "__main__":
    main() 