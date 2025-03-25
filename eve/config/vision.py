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
VISION_DEBUG = False
VISION_OVERLAY = True

# Face detection settings
FACE_DETECTION_ENABLED = True
FACE_DETECTION_INTERVAL_SEC = 0.5
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
KNOWN_FACES_DIR = "data/faces/known"
FACE_ENCODING_MODEL = "hog"  # Options: "hog", "cnn"
FACE_RECOGNITION_INTERVAL = 1.0  # Seconds

# Emotion detection settings
EMOTION_DETECTION_ENABLED = True
EMOTION_DETECTION_MODEL = "fer"  # Facial Emotion Recognition model
EMOTION_DETECTION_INTERVAL_SEC = 1.0
EMOTION_THRESHOLD = 0.5  # Minimum confidence for emotion detection

# Object detection
OBJECT_DETECTION_ENABLED = False
OBJECT_DETECTION_MODEL = "yolo"
OBJECT_DETECTION_CONFIDENCE = 0.5
OBJECT_DETECTION_INTERVAL = 1.0  # Seconds
OBJECT_CLASSES_FILE = "models/object_detection/coco.names"

# Tracking settings
TRACKING_ENABLED = False
TRACKING_ALGORITHM = "CSRT"  # Options: "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" 