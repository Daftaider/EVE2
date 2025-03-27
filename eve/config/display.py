"""
Display configuration settings for EVE2
"""

from enum import Enum, auto

class Emotion(Enum):
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    CONFUSED = auto()

class DisplayConfig:
    # Display settings
    WINDOW_SIZE = (800, 480)
    FPS = 30
    FULLSCREEN = False
    
    # Emotion settings
    DEFAULT_EMOTION = Emotion.NEUTRAL
    EMOTIONS = {
        Emotion.NEUTRAL: "neutral",
        Emotion.HAPPY: "happy",
        Emotion.SAD: "sad",
        Emotion.ANGRY: "angry",
        Emotion.SURPRISED: "surprised",
        Emotion.CONFUSED: "confused"
    }
    
    # Asset paths
    ASSET_DIR = "assets/emotions"
    
    # Animation settings
    TRANSITION_SPEED = 0.5  # seconds
    
    @classmethod
    def get_emotion_path(cls, emotion: Emotion) -> str:
        """Get the file path for an emotion's image."""
        return f"{cls.ASSET_DIR}/{cls.EMOTIONS[emotion]}.png"

# General display settings
DISPLAY_ENABLED = True
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
DISPLAY_FPS = 30
DISPLAY_DRIVER = "pygame"  # Options: "pygame", "kivy", "tkinter", "custom"

# Emotions and animations
DEFAULT_EMOTION = "neutral"
EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "confused",
    "thinking"
]
AVAILABLE_EMOTIONS = EMOTIONS  # Keep for backward compatibility

# Colors
BACKGROUND_COLOR = (0, 0, 0)  # Black background
TEXT_COLOR = (255, 255, 255)  # White text
ACCENT_COLOR = (0, 120, 215)  # Blue accent
EYE_COLOR = (0, 191, 255)  # Deep Sky Blue
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow
ERROR_COLOR = (255, 0, 0)  # Red
SUCCESS_COLOR = (0, 255, 0)  # Green

# Animation settings
ANIMATION_FPS = 30
BLINK_INTERVAL_SEC = 3.0
ANIMATION_PATH = "assets/animations"
ENABLE_BLINKING = True
ENABLE_EMOTIONS = True
ANIMATION_SMOOTHING = True
EMOTION_TRANSITION_TIME_MS = 500  # Time to transition between emotions

# Face settings
EYE_SIZE = 50
EYE_SPACING = 100
MOUTH_WIDTH = 120
MOUTH_HEIGHT = 20

# UI elements
SHOW_STATUS_BAR = True
SHOW_TIME = True
SHOW_WEATHER = False
SHOW_BATTERY = False
FONT_NAME = "Arial"
FONT_SIZE = 24
UI_PADDING = 10
STATUS_BAR_HEIGHT = 30

# Transitions
ENABLE_TRANSITIONS = True
TRANSITION_DURATION = 0.3  # seconds

# Add to your display configuration
HEADLESS_MODE = True  # Set to True to skip display initialization 