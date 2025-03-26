"""
Communication system configuration for EVE2
"""

# Message queue settings
MESSAGE_QUEUE_SIZE = 100
MESSAGE_TIMEOUT_SEC = 0.5
MESSAGE_PRIORITY_LEVELS = 3
MESSAGE_RETRY_COUNT = 3
MESSAGE_LOGGING = True

# API settings
API_ENABLED = True
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = False
API_THREADED = True
API_ALLOWED_ORIGINS = ["*"]
API_RATE_LIMIT = 100  # requests per minute

# WebSocket settings
WEBSOCKET_ENABLED = True
WEBSOCKET_PORT = 5001
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PATH = "/ws"
WEBSOCKET_PING_INTERVAL = 30  # seconds

# Network security
API_TOKEN_REQUIRED = False
API_TOKEN = "eve2_default_token"
ENCRYPTION_ENABLED = False
ENCRYPTION_KEY = "default_key_change_me"

# Protocol settings
PROTOCOLS = ["HTTP", "WEBSOCKET"]
PROTOCOL_TIMEOUT = 5.0  # seconds
HEARTBEAT_INTERVAL = 10.0  # seconds
CONNECTION_RETRY_INTERVAL = 5.0  # seconds

# Message topics for internal communication
TOPICS = {
    'FACE_DETECTED': 'face_detected',
    'FACE_LOST': 'face_lost',
    'EMOTION_DETECTED': 'emotion_detected',
    'SPEECH_DETECTED': 'speech_detected',
    'SPEECH_RECOGNIZED': 'speech_recognized',
    'AUDIO_LEVEL': 'audio_level',
    'SYSTEM_STATUS': 'system_status',
    'ERROR': 'error',
    'COMMAND': 'command',
    'RESPONSE': 'response',
    'DISPLAY_UPDATE': 'display_update',
    'VISION_STATUS': 'vision_status',
    'AUDIO_STATUS': 'audio_status',
    'DISPLAY_STATUS': 'display_status'
}

# Message priorities
PRIORITIES = {
    'HIGH': 0,
    'MEDIUM': 1,
    'LOW': 2
}

# Communication timeouts (in seconds)
TIMEOUTS = {
    'DEFAULT': 1.0,
    'FACE_DETECTION': 0.1,
    'EMOTION_DETECTION': 0.2,
    'SPEECH_RECOGNITION': 2.0,
    'COMMAND_EXECUTION': 5.0
} 