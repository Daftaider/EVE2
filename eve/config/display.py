"""
Display configuration settings
"""

# Default emotion to display when no specific emotion is active
DEFAULT_EMOTION = "neutral"

# Available emotions that can be displayed
AVAILABLE_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "confused",
    "thinking"
]

# Display dimensions
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

# Colors
BACKGROUND_COLOR = (0, 0, 0)  # Black background
TEXT_COLOR = (255, 255, 255)  # White text
ACCENT_COLOR = (0, 120, 215)  # Blue accent

# Animation settings
ANIMATION_FPS = 30
BLINK_INTERVAL_SEC = 3.0
ANIMATION_PATH = "assets/animations"

# UI elements
SHOW_STATUS_BAR = True
SHOW_TIME = True
FONT_NAME = "Arial"
FONT_SIZE = 24 