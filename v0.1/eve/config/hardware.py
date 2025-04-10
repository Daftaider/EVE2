"""
Hardware configuration settings for EVE2
"""

# System role and mode
ROLE = "master"  # Options: "master", "display", "audio", etc.
DISTRIBUTED_MODE = False
MODE = "normal"  # Options: "normal", "debug", "demo", "test"

# Device capabilities
CAMERA_ENABLED = True
CAMERA_INDEX = 0
AUDIO_INPUT_ENABLED = True
AUDIO_OUTPUT_ENABLED = True
MICROPHONE_ENABLED = True  # Legacy alias
SPEAKER_ENABLED = True  # Legacy alias
DISPLAY_ENABLED = True
GPIO_ENABLED = True
BLUETOOTH_ENABLED = False
WIFI_ENABLED = True

# Display hardware settings
DISPLAY_RESOLUTION = (800, 480)
FULLSCREEN = False
DISPLAY_TYPE = "lcd"  # Options: "lcd", "oled", "hdmi", "none"
DISPLAY_ROTATION = 0  # 0, 90, 180, 270 degrees
DISPLAY_BRIGHTNESS = 80  # percentage
DISPLAY_POWER_SAVE = True
DISPLAY_TIMEOUT = 300  # seconds before dimming display

# Camera hardware settings
CAMERA_TYPE = "picamera"  # Options: "picamera", "opencv", "gstreamer", "none"
CAMERA_ROTATION = 0  # 0, 90, 180, 270 degrees
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30
CAMERA_FPS = 30  # Same as FRAMERATE for compatibility
CAMERA_AWB = "auto"  # Auto white balance
CAMERA_EXPOSURE_MODE = "auto"

# Audio hardware settings
AUDIO_INPUT_DEVICE = "default"
audio_input_device = "default"  # Add lowercase version for compatibility
AUDIO_OUTPUT_DEVICE = "default"
audio_output_device = "default"  # Add lowercase version for compatibility
AUDIO_VOLUME = 80  # percentage
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BOOST = False

# Network settings
NETWORK = {
    "master_ip": "192.168.1.100",
    "master_port": 5000,
    "discovery_port": 5001,
    "heartbeat_interval": 5.0  # seconds
}

# GPIO settings
GPIO_PINS = {
    "button_1": 17,
    "button_2": 27,
    "led_1": 22,
    "led_2": 23,
    "servo": 18,
    "sensor": 4
}

# Power management
ENABLE_POWER_MANAGEMENT = True
LOW_POWER_MODE = False
BATTERY_MONITOR_ENABLED = True
SHUTDOWN_ON_LOW_BATTERY = True
BATTERY_LOW_THRESHOLD = 15  # percentage

# Hardware capabilities (more detailed than simple enabled flags)
CAPABILITIES = {
    "display": True,
    "camera": True,
    "microphone": True,
    "speaker": True,
    "gpio": True
} 