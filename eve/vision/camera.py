# Add a simple camera mock
def get_mock_frame():
    """Generate a mock frame for testing"""
    import numpy as np
    import cv2
    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a face-like circle
    cv2.circle(frame, (320, 240), 100, (200, 200, 200), -1)
    # Draw eyes
    cv2.circle(frame, (280, 220), 20, (255, 255, 255), -1)
    cv2.circle(frame, (360, 220), 20, (255, 255, 255), -1)
    # Draw mouth
    cv2.ellipse(frame, (320, 280), (50, 20), 0, 0, 180, (255, 255, 255), -1)
    return frame 