import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_camera():
    logger.info("Testing simple camera access...")
    
    # Try to open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return
    
    logger.info("Camera opened successfully")
    
    # Try to read 5 frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            logger.info(f"Frame {i+1}: Shape = {frame.shape}")
        else:
            logger.error(f"Failed to read frame {i+1}")
        time.sleep(0.5)
    
    # Release the camera
    cap.release()
    logger.info("Camera test completed")

if __name__ == "__main__":
    test_simple_camera() 