"""
Display configuration settings for EVE2
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Union

class Emotion(Enum):
    """Enumeration of possible emotions."""
    NEUTRAL = 0
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    SURPRISED = 4
    CONFUSED = 5

    @classmethod
    def from_value(cls, value: Union[int, str, 'Emotion']) -> 'Emotion':
        """Convert various input types to Emotion enum."""
        if isinstance(value, cls):
            return value
        elif isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return cls.NEUTRAL
        elif isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                return cls.NEUTRAL
        return cls.NEUTRAL

    @property
    def filename(self) -> str:
        """Get the filename for this emotion."""
        return f"{self.name.lower()}.png"

class DisplayConfig:
    """Configuration for the display subsystem."""
    
    # Display settings
    WINDOW_SIZE: Tuple[int, int] = (800, 480)
    FPS: int = 30
    FULLSCREEN: bool = False
    
    # Emotion settings
    DEFAULT_EMOTION: Emotion = Emotion.NEUTRAL
    EMOTION_FILENAMES: Dict[Emotion, str] = {
        emotion: emotion.filename for emotion in Emotion
    }
    
    # Color settings
    DEFAULT_BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    DEFAULT_EYE_COLOR: Tuple[int, int, int] = (255, 255, 255)
    
    # Asset paths
    ASSET_DIR: str = "assets/emotions"
    
    # Animation settings
    TRANSITION_SPEED: float = 0.5
    
    @classmethod
    def get_emotion_path(cls, emotion: Union[int, str, Emotion]) -> str:
        """Get the file path for an emotion's image."""
        emotion_enum = Emotion.from_value(emotion)
        return f"{cls.ASSET_DIR}/{emotion_enum.filename}"

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