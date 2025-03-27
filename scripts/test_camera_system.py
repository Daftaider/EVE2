import logging
import time
from eve.vision.camera_utils import check_camera_hardware, CameraManager
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera():
    # Check available camera hardware
    camera_type = check_camera_hardware()
    logger.info(f"Detected camera type: {camera_type}")
    
    # Initialize camera manager
    camera = CameraManager(width=640, height=480, fps=30)
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        return
    
    # Create window
    cv2.namedWindow("Test Camera", cv2.WINDOW_NORMAL)
    
    try:
        # Capture frames for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = camera.read_frame()
            if ret and frame is not None:
                cv2.imshow("Test Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
            
            time.sleep(1/30)  # ~30 FPS
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera() 