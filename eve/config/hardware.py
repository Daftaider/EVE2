"""
Hardware configuration settings for EVE2
"""

# System role (e.g., 'master', 'display', 'audio', etc.)
ROLE = "master"

# Distributed mode - determines if EVE runs as a single system or distributed across multiple devices
DISTRIBUTED_MODE = False

# Device capabilities (enable/disable hardware components)
CAMERA_ENABLED = True
CAMERA_INDEX = 0  # For selecting specific camera device when multiple are available

# Audio settings
AUDIO_INPUT_ENABLED = True  # Microphone
AUDIO_OUTPUT_ENABLED = True  # Speaker
MICROPHONE_ENABLED = True    # Alternative name for backward compatibility
SPEAKER_ENABLED = True       # Alternative name for backward compatibility

# Display settings
DISPLAY_ENABLED = True
DISPLAY_RESOLUTION = (800, 480)
GPIO_ENABLED = True

# Network settings for distributed mode
NETWORK = {
    "master_ip": "192.168.1.100",
    "master_port": 5000,
    "discovery_port": 5001,
    "heartbeat_interval": 5.0  # seconds
}

# Hardware capabilities (more detailed than simple enabled flags)
CAPABILITIES = {
    "display": True,
    "camera": True,
    "microphone": True,
    "speaker": True,
    "gpio": True
}

# GPIO pin mappings
GPIO_PINS = {
    "button_1": 17,
    "button_2": 27,
    "led_1": 22,
    "led_2": 23
}

# Display settings
DISPLAY_TYPE = "lcd"
DISPLAY_ROTATION = 0
DISPLAY_BRIGHTNESS = 80  # percentage

# Camera settings
CAMERA_TYPE = "picamera"
CAMERA_ROTATION = 0
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30
CAMERA_FPS = 30  # Add this line (same as FRAMERATE for compatibility)

# Audio settings
AUDIO_INPUT_DEVICE = "default"
AUDIO_OUTPUT_DEVICE = "default"
AUDIO_VOLUME = 80  # percentage
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1 