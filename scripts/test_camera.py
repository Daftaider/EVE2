import cv2
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_camera():
    # Force backend preferences
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'
    
    # Try different backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend, name in backends:
        logger.info(f"\nTrying {name} backend...")
        try:
            cap = cv2.VideoCapture(0 + backend)
            if not cap.isOpened():
                logger.warning(f"{name} backend failed to open camera")
                continue
                
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"{name} backend failed to read frame")
                cap.release()
                continue
                
            logger.info(f"Successfully captured frame with {name} backend")
            logger.info(f"Frame shape: {frame.shape}")
            
            # Save test frame
            cv2.imwrite(f"test_frame_{name}.jpg", frame)
            logger.info(f"Saved test frame as test_frame_{name}.jpg")
            
            cap.release()
            return True
            
        except Exception as e:
            logger.error(f"Error with {name} backend: {e}")
            continue
            
    logger.error("All backends failed")
    return False

if __name__ == "__main__":
    test_camera() 