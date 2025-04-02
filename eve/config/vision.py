"""
Vision system configuration for EVE2
"""

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_ROTATION = 0
CAMERA_FLIP_HORIZONTAL = False
CAMERA_FLIP_VERTICAL = False

# Processing settings
VISION_PROCESSING_INTERVAL = 0.1  # seconds between processing frames
VISION_ENABLED = True
VISION_DEBUG = True
VISION_OVERLAY = True

# Face detection settings
FACE_DETECTION_ENABLED = True
FACE_DETECTION_INTERVAL_SEC = 0.5
FACE_DETECTION_INTERVAL_MS = 500  # Add this line - millisecond version of the interval
FACE_DETECTION_MODEL = "haarcascade"  # Options: "haarcascade", "dnn", "ssd", "mtcnn"
FACE_DETECTION_CONFIDENCE = 0.5
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (30, 30)

# Face recognition settings
FACE_RECOGNITION_ENABLED = False
FACE_RECOGNITION_MODEL = "facenet"
FACE_RECOGNITION_THRESHOLD = 0.6
FACE_RECOGNITION_TOLERANCE = 0.6  # Tolerance for face matching
FACE_DATABASE_PATH = "data/faces"
KNOWN_FACES_DIR = "data/known_faces"
FACE_ENCODING_MODEL = "hog"  # Options: "hog", "cnn"
FACE_RECOGNITION_INTERVAL = 1.0  # Seconds

# Emotion detection settings
EMOTION_DETECTION_ENABLED = True
EMOTION_DETECTION_MODEL = "fer"  # Facial Emotion Recognition model
EMOTION_DETECTION_INTERVAL_SEC = 1.0
EMOTION_THRESHOLD = 0.5  # Minimum confidence for emotion detection
EMOTION_CONFIDENCE_THRESHOLD = 0.5  # Add this line - alias for EMOTION_THRESHOLD

# Object detection
OBJECT_DETECTION_ENABLED = True  # Enable object detection
OBJECT_DETECTION_MODEL = "yolov8n.pt"  # Use YOLOv8 nano model
OBJECT_DETECTION_CONFIDENCE = 0.5  # Increase confidence threshold back to 0.5
OBJECT_DETECTION_INTERVAL = 0.2  # Process more frequently for testing
OBJECT_DETECTION_CLASSES = ["person", "cat", "dog", "bird", "car", "bicycle", "motorcycle", "bus", "truck"]  # Common classes to detect

# Tracking settings
TRACKING_ENABLED = False
TRACKING_ALGORITHM = "CSRT"  # Options: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"

from types import SimpleNamespace

# Vision configuration
VISION = SimpleNamespace(**{
    'KNOWN_FACES_DIR': 'data/known_faces',
    'CAMERA_INDEX': 0,
    'FRAME_WIDTH': 640,
    'FRAME_HEIGHT': 480,
    'FPS': 30
})

# Camera settings
CAMERA = SimpleNamespace(**{
    'WIDTH': 640,
    'HEIGHT': 480,
    'FPS': 30
})

# Face detection settings
FACE_DETECTION = SimpleNamespace(**{
    'MIN_FACE_SIZE': 30,
    'SCALE_FACTOR': 1.1,
    'MIN_NEIGHBORS': 5
})

class VisionConfig:
    # Camera settings
    CAMERA_INDEX = 0
    RESOLUTION = (640, 480)
    FPS = 30
    
    # Face detection settings
    MIN_FACE_SIZE = (30, 30)
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5
    
    # Object detection settings
    OBJECT_DETECTION_ENABLED = True
    OBJECT_DETECTION_MODEL = "yolov8n.pt"
    OBJECT_DETECTION_CONFIDENCE = 0.5
    OBJECT_DETECTION_INTERVAL = 0.5
    OBJECT_DETECTION_CLASSES = ["person", "cat", "dog", "bird", "car", "bicycle", "motorcycle", "bus", "truck"]
    
    # Debug settings
    DEBUG = False
    SHOW_DETECTION = True  # Show detection boxes by default 