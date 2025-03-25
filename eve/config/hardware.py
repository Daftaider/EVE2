"""
Hardware configuration settings for EVE2
"""

# System role (e.g., 'master', 'display', 'audio', etc.)
ROLE = "master"

# Hardware capabilities
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

# Audio settings
AUDIO_INPUT_DEVICE = "default"
AUDIO_OUTPUT_DEVICE = "default"
AUDIO_VOLUME = 80  # percentage 