"""
Display configuration settings for EVE2
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Union, Any
import logging
import pygame
from dataclasses import dataclass, field, fields, MISSING
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class Emotion(Enum):
    """Enumeration of possible emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"

    @classmethod
    def from_value(cls, value: Union[str, int, 'Emotion', None]) -> 'Emotion':
        """Convert various input types to Emotion enum, defaulting to NEUTRAL."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NEUTRAL
        
        normalized_value = str(value).strip().upper()
        
        # Try direct name lookup
        if normalized_value in cls.__members__:
            return cls[normalized_value]
        
        # Try value lookup
        for member in cls:
            if member.value.upper() == normalized_value:
                return member
                
        # Try index lookup if integer
        if isinstance(value, int):
            try:
                return list(cls)[value]
            except IndexError:
                pass  # Fall through to default

        logger.warning(f"Invalid emotion value: '{value}', defaulting to NEUTRAL.")
        return cls.NEUTRAL

    def __str__(self) -> str:
        return self.value

    @property
    def filename(self) -> str:
        """Get the filename for this emotion."""
        return f"{self.value}.png"

@dataclass
class DisplayConfig:
    """Configuration for the display subsystem."""
    
    # Display settings
    WINDOW_SIZE: Tuple[int, int] = (800, 480)
    FPS: int = 30
    FULLSCREEN: bool = False
    
    # Emotion settings
    DEFAULT_EMOTION: Emotion = Emotion.NEUTRAL
    EMOTION_FILENAMES: Dict[Emotion, str] = field(
        default_factory=lambda: {e: e.filename for e in Emotion}
    )
    
    # Color settings
    DEFAULT_BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    DEFAULT_EYE_COLOR: Tuple[int, int, int] = (255, 255, 255)
    
    # Asset paths
    ASSET_DIR: str = "assets/emotions"
    
    # Animation settings
    TRANSITION_SPEED: float = 0.5
    
    @classmethod
    def get_emotion_path(cls, emotion: Union[str, Emotion, None]) -> str:
        """Get the file path for an emotion's image."""
        emotion_enum = Emotion.from_value(emotion)
        return os.path.join(cls.ASSET_DIR, emotion_enum.filename)

    @staticmethod
    def parse_color(color_input: Union[Tuple[int, int, int], str, None], default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Parse color input (string or tuple) to RGB tuple."""
        if color_input is None:
            return default
        if isinstance(color_input, tuple) and len(color_input) == 3:
            return color_input
        if isinstance(color_input, str):
            try:
                # Use pygame's color parsing which is quite flexible
                color = pygame.Color(color_input)
                return (color.r, color.g, color.b)
            except ValueError:
                logger.warning(f"Invalid color string '{color_input}', using default {default}.")
                return default
        logger.warning(f"Invalid color format '{color_input}', using default {default}.")
        return default

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DisplayConfig':
        """Create DisplayConfig from dict, applying defaults and type conversions."""
        instance_kwargs = {}
        defined_fields = {f.name: f for f in fields(cls)}

        for field_name, field_def in defined_fields.items():
            if field_def.default_factory is not MISSING:
                 default_value = field_def.default_factory()
            else:
                 default_value = field_def.default

            raw_value = config_dict.get(field_name, default_value)
            
            if raw_value is MISSING:
                if field_def.init:
                    logger.warning(f"Required config field '{field_name}' missing and no default provided. Skipping.")
                continue

            try:
                if field_name == 'WINDOW_SIZE':
                    instance_kwargs[field_name] = tuple(map(int, raw_value))
                elif field_name == 'FPS':
                     instance_kwargs[field_name] = int(raw_value)
                elif field_name == 'FULLSCREEN':
                     instance_kwargs[field_name] = bool(raw_value)
                elif field_name == 'DEFAULT_EMOTION':
                     instance_kwargs[field_name] = Emotion.from_value(raw_value)
                elif field_name == 'DEFAULT_BACKGROUND_COLOR':
                     instance_kwargs[field_name] = cls.parse_color(raw_value, cls.DEFAULT_BACKGROUND_COLOR)
                elif field_name == 'DEFAULT_EYE_COLOR':
                      instance_kwargs[field_name] = cls.parse_color(raw_value, cls.DEFAULT_EYE_COLOR)
                elif field_name == 'ASSET_DIR':
                     instance_kwargs[field_name] = str(raw_value)
                elif field_name == 'TRANSITION_SPEED':
                     instance_kwargs[field_name] = float(raw_value)
                elif field_name == 'EMOTION_FILENAMES':
                     if isinstance(raw_value, dict):
                         instance_kwargs[field_name] = raw_value
                     else:
                         instance_kwargs[field_name] = default_value
                else:
                    instance_kwargs[field_name] = raw_value
            except (ValueError, TypeError) as e:
                 logger.warning(f"Error processing config field '{field_name}' with value '{raw_value}': {e}. Using default.")
                 instance_kwargs[field_name] = default_value

        return cls(**instance_kwargs)

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