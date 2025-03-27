import cv2
import logging
import sys
import signal
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera():
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        # Create window
        cv2.namedWindow('Test Camera', cv2.WINDOW_NORMAL)
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Display frame
            cv2.imshow('Test Camera', frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera() 