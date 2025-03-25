"""
Vision system configuration
"""

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection settings
FACE_DETECTION_INTERVAL_SEC = 0.5
FACE_DETECTION_MODEL = "haarcascade"  # Options: "haarcascade", "dnn", "ssd", "mtcnn"
FACE_DETECTION_CONFIDENCE = 0.5

# Face recognition settings
FACE_RECOGNITION_ENABLED = False
FACE_RECOGNITION_MODEL = "facenet"
FACE_RECOGNITION_THRESHOLD = 0.6
FACE_DATABASE_PATH = "data/faces"
KNOWN_FACES_DIR = "data/faces/known"  # Directory for storing known face encodings

# Emotion detection settings
EMOTION_DETECTION_ENABLED = True
EMOTION_DETECTION_MODEL = "fer"  # Facial Emotion Recognition model
EMOTION_DETECTION_INTERVAL_SEC = 1.0

# Object detection
OBJECT_DETECTION_ENABLED = False
OBJECT_DETECTION_MODEL = "yolo"
OBJECT_DETECTION_CONFIDENCE = 0.5 