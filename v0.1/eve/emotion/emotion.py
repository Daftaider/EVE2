"""
Emotion module for EVE2.

This module defines the Emotion enum class that represents all possible emotions
that can be displayed on the LCD screen.
"""
from enum import Enum, auto

class Emotion(Enum):
    """Enum representing all possible emotions for EVE's display."""
    
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    FEARFUL = auto()
    DISGUSTED = auto()
    BLINK = auto()
    
    @property
    def filename(self) -> str:
        """Get the filename for this emotion's image."""
        return f"{self.name.lower()}.png" 