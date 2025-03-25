"""
Communication system configuration
"""

# Message queue settings
MESSAGE_QUEUE_SIZE = 100
MESSAGE_TIMEOUT_SEC = 0.5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 5000
API_ENABLED = True

# WebSocket settings
WEBSOCKET_ENABLED = True
WEBSOCKET_PORT = 5001

# Communication protocols
PROTOCOLS = ["HTTP", "WEBSOCKET"]

# Security
API_TOKEN_REQUIRED = False
API_TOKEN = "eve2_default_token" 