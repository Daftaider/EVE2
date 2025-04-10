"""
Display state management for the LCD display.
"""
import pygame
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class DisplayMode(Enum):
    """Enum for different display modes."""
    NORMAL = auto()
    DEBUG = auto()
    MENU = auto()

@dataclass
class DisplayState:
    """Data class for display state."""
    mode: DisplayMode = DisplayMode.NORMAL
    rotation: int = 0
    hardware_display: bool = True
    eye_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (0, 0, 0)
    fps: float = 0.0
    listening: bool = False
    debug_info: Dict[str, Any] = None

class DisplayStateManager:
    """Manages the display state and transitions."""
    
    def __init__(self, screen: pygame.Surface):
        """Initialize the display state manager."""
        self.screen = screen
        self.state = DisplayState()
        self._previous_state = None
        
    def set_mode(self, mode: DisplayMode) -> None:
        """Set the display mode."""
        self._previous_state = self.state.mode
        self.state.mode = mode
        logger.debug(f"Display mode changed from {self._previous_state} to {mode}")
        
    def set_rotation(self, rotation: int) -> None:
        """Set the display rotation."""
        self.state.rotation = rotation
        logger.debug(f"Display rotation set to {rotation}")
        
    def toggle_hardware_display(self) -> None:
        """Toggle hardware display mode."""
        self.state.hardware_display = not self.state.hardware_display
        logger.debug(f"Hardware display mode toggled to {self.state.hardware_display}")
        
    def set_eye_color(self, color: tuple[int, int, int]) -> None:
        """Set the eye color."""
        self.state.eye_color = color
        logger.debug(f"Eye color set to {color}")
        
    def set_background_color(self, color: tuple[int, int, int]) -> None:
        """Set the background color."""
        self.state.background_color = color
        logger.debug(f"Background color set to {color}")
        
    def update_fps(self, fps: float) -> None:
        """Update the FPS counter."""
        self.state.fps = fps
        
    def set_listening(self, listening: bool) -> None:
        """Set the listening state."""
        self.state.listening = listening
        logger.debug(f"Listening state set to {listening}")
        
    def update_debug_info(self, info: Dict[str, Any]) -> None:
        """Update debug information."""
        self.state.debug_info = info
        
    def get_state(self) -> DisplayState:
        """Get the current display state."""
        return self.state
        
    def restore_previous_mode(self) -> None:
        """Restore the previous display mode."""
        if self._previous_state is not None:
            self.state.mode = self._previous_state
            self._previous_state = None
            logger.debug(f"Restored previous display mode: {self.state.mode}") 