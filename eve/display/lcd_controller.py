"""
LCD display controller module.

This module manages the LCD display and renders eye animations.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import pygame
import numpy as np
from PIL import Image, ImageDraw
import cv2

from eve import config
from eve.config.display import Emotion, DisplayConfig

logger = logging.getLogger(__name__)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two boxes.
    Boxes are expected in (startX, startY, endX, endY) format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

IOU_THRESHOLD = 0.7 # Overlap threshold to consider a match for correction

class LCDController:
    """
    LCD display controller for rendering eye animations.
    
    This class handles the initialization of the display and
    manages the rendering of emotive eye animations.
    """
    
    def __init__(self, 
                 config: Optional[DisplayConfig] = None, 
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 fps: Optional[int] = None,
                 default_emotion: Optional[Union[str, int, Emotion]] = None,
                 background_color: Optional[Union[Tuple[int, int, int], str]] = None,
                 eye_color: Optional[Union[Tuple[int, int, int], str]] = None,
                 rotation: int = 0,
                 use_hardware_display: bool = True):
        """
        Initialize the LCD Controller.
        
        Args:
            config: Display configuration object
            width: Optional window width (overrides config)
            height: Optional window height (overrides config)
            fps: Optional frames per second (overrides config)
            default_emotion: Optional starting emotion (overrides config)
            background_color: Optional background color as RGB tuple or string
            eye_color: Optional eye color as RGB tuple or string
            rotation: Display rotation in degrees (0, 90, 180, 270)
            use_hardware_display: Whether to use hardware LCD display
        """
        # Set environment variables for display
        if use_hardware_display:
            os.environ['SDL_VIDEODRIVER'] = 'fbcon'  # Use framebuffer console
            os.environ['SDL_FBDEV'] = '/dev/fb0'     # Primary framebuffer device
            # Disable cursor for hardware display
            os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1'
        else:
            os.environ['SDL_VIDEODRIVER'] = 'x11'    # Use X11 for testing
        
        self.config = config or DisplayConfig
        self.use_hardware_display = use_hardware_display
        self.rotation = rotation % 360  # Normalize rotation
        
        # Override config values if parameters are provided
        if width is not None and height is not None:
            self.window_size = (width, height)
        else:
            self.window_size = self.config.WINDOW_SIZE
            
        self.fps = fps if fps is not None else self.config.FPS
        
        # Convert and validate emotion using the from_value method
        self._current_emotion = Emotion.from_value(default_emotion)
        
        # Handle colors
        self.background_color = self._parse_color(background_color) if background_color else self.config.BACKGROUND_COLOR
        self.eye_color = self._parse_color(eye_color) if eye_color else self.config.EYE_COLOR
        
        # Animation settings
        self.blink_interval = self.config.BLINK_INTERVAL
        self.blink_duration = self.config.BLINK_DURATION
        self.is_blinking = False
        self.last_blink_time = time.time()
        
        # Initialize display system
        self._init_display()
        
        # Font for debug text
        self.debug_font = pygame.font.SysFont(None, self.config.DEBUG_FONT_SIZE)
        self.debug_ui_elements: Dict[str, pygame.Rect] = {}
        
        # Log initialization parameters
        logger.info(f"LCD Controller initialized with: size={self.window_size}, "
                    f"fps={self.fps}, default_emotion={self._current_emotion.name}, "
                    f"background_color={self.background_color}, "
                    f"eye_color={self.eye_color}, rotation={self.rotation}, "
                    f"hardware_display={self.use_hardware_display}")

    def _parse_color(self, color: Union[Tuple[int, int, int], str, None]) -> Tuple[int, int, int]:
        """Convert color to RGB tuple."""
        if color is None:
            return (255, 255, 255)
        if isinstance(color, tuple) and len(color) == 3:
            return color
        if isinstance(color, str):
            try:
                rgb = pygame.Color(color)
                return (rgb.r, rgb.g, rgb.b)
            except ValueError:
                logger.warning(f"Invalid color string: {color}, using default")
                return (255, 255, 255)
        return (255, 255, 255)

    def _init_display(self):
        """Initialize the display with current settings."""
        try:
            pygame.init()
            
            # Set window position (centered)
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            
            # Create the window with a title
            pygame.display.set_caption("EVE2 Display")
            
            # Initialize the display
            self.screen = pygame.display.set_mode(
                self.window_size,
                pygame.SHOWN | (pygame.FULLSCREEN if self.use_hardware_display else 0)
            )
            
            # Apply rotation if needed
            if self.rotation != 0:
                self.screen = pygame.transform.rotate(self.screen, -self.rotation)
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            # Load images
            self._load_emotion_images()
            
            # Initial display update
            self.screen.fill(self.background_color)
            pygame.display.flip()
            
            logger.info("Display initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            self._init_fallback_mode()

    def _init_fallback_mode(self):
        """Initialize a fallback mode for headless operation."""
        logger.info("Initializing display in fallback mode")
        pygame.init()
        self.screen = pygame.Surface(self.window_size)
        self.clock = pygame.time.Clock()
        self._load_emotion_images()

    def _rotate_surface(self, surface: pygame.Surface) -> pygame.Surface:
        """Rotate a surface according to the display rotation setting."""
        if self.rotation == 0:
            return surface
        return pygame.transform.rotate(surface, -self.rotation)

    def _apply_eye_color(self, surface: pygame.Surface, eye_color: Tuple[int, int, int]) -> pygame.Surface:
        """Apply eye color to white pixels in the surface."""
        # Create a copy of the surface
        new_surface = surface.copy()
        
        # Get pixel array of the surface
        pixels = pygame.PixelArray(new_surface)
        
        # Find white pixels (255, 255, 255) and replace with eye color
        white = surface.map_rgb((255, 255, 255))
        pixels.replace(white, surface.map_rgb(eye_color))
        
        # Free the pixel array
        pixels.close()
        
        return new_surface

    def _load_emotion_images(self):
        """Load all emotion images into memory."""
        self.emotion_images = {}
        for emotion in Emotion:
            try:
                image_path = self.config.get_emotion_path(emotion)
                original = pygame.image.load(image_path)
                
                # Scale image if needed
                if original.get_size() != self.window_size:
                    original = pygame.transform.scale(original, self.window_size)
                
                # Apply eye color
                colored = self._apply_eye_color(original, self.eye_color)
                self.emotion_images[emotion] = colored
                
            except Exception as e:
                logger.warning(f"Failed to load emotion image for {emotion}: {e}")
                # Create a fallback surface
                surface = pygame.Surface(self.window_size)
                surface.fill(self._get_fallback_color(emotion))
                self.emotion_images[emotion] = surface

    def _get_fallback_color(self, emotion: Emotion) -> Tuple[int, int, int]:
        """Get a fallback color for an emotion."""
        colors = {
            Emotion.NEUTRAL: (128, 128, 128),    # Gray
            Emotion.HAPPY: (255, 255, 0),        # Yellow
            Emotion.SAD: (0, 0, 255),            # Blue
            Emotion.ANGRY: (255, 0, 0),          # Red
            Emotion.SURPRISED: (255, 165, 0),    # Orange
            Emotion.FEARFUL: (128, 0, 128),      # Purple
            Emotion.DISGUSTED: (0, 128, 0),      # Green
            Emotion.BLINK: (0, 0, 0),            # Black for blinking
        }
        return colors.get(emotion, (0, 0, 0))

    def update(self, 
               emotion: Optional[Emotion] = None, 
               debug_menu_active: bool = False,
               current_debug_view: Optional[str] = None,
               debug_frame: Optional[np.ndarray] = None,
               detections: Optional[List[Dict[str, Any]]] = None,
               camera_info: Optional[Dict[str, Any]] = None,
               camera_rotation: int = 0,
               is_correcting: bool = False,
               input_buffer: str = "",
               corrections: Optional[List[Dict[str, Any]]] = None,
               audio_debug_listen_always: bool = False,
               last_recognized_text: str = "",
               available_audio_devices: Optional[List[Dict]] = None,
               selected_audio_device_index: Optional[int] = None,
               last_audio_rms: float = 0.0,
               current_porcupine_sensitivity: float = 0.5
               ) -> None:
        """Update the display with current emotion or debug view."""
        # Check for blink
        current_time = time.time()
        if current_time - self.last_blink_time > self.blink_interval:
            self.blink()
            self.last_blink_time = current_time

        # Normal Emotion Mode (if debug menu isn't active)
        if not debug_menu_active:
            if emotion is not None:
                self._current_emotion = emotion
                
                try:
                    self.screen.fill(self.background_color)
                    if self._current_emotion in self.emotion_images:
                        emotion_surface = self.emotion_images[self._current_emotion]
                        if self.rotation != 0:
                            emotion_surface = self._rotate_surface(emotion_surface)
                        self.screen.blit(emotion_surface, (0, 0))
                    pygame.display.flip()
                except Exception as e:
                    logger.error(f"Error updating display (normal mode): {e}")
                return

        # ... rest of the update method (debug mode) remains unchanged ...

    def cleanup(self) -> None:
        """Clean up pygame resources and restore display settings."""
        try:
            # Save final frame
            if hasattr(self, 'screen'):
                pygame.image.save(self.screen, self.config.CURRENT_FRAME_PATH)
            
            # Quit pygame
            pygame.quit()
            
            # If using hardware display, try to restore default settings
            if self.use_hardware_display:
                try:
                    # Reset framebuffer settings if needed
                    os.system('echo 0 > /sys/class/graphics/fb0/rotate')
                    # Restore cursor
                    os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '0'
                except Exception as e:
                    logger.warning(f"Could not reset display settings: {e}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def set_rotation(self, rotation: int) -> None:
        """Set display rotation (0, 90, 180, 270 degrees)."""
        self.rotation = rotation % 360
        if hasattr(self, 'screen'):
            self.screen = pygame.transform.rotate(self.screen, -self.rotation)
        logger.info(f"Display rotation set to {self.rotation} degrees")

    def toggle_hardware_display(self, use_hardware: bool) -> None:
        """Toggle between hardware display and window mode."""
        if self.use_hardware_display != use_hardware:
            self.use_hardware_display = use_hardware
            os.environ['SDL_VIDEODRIVER'] = 'fbcon' if use_hardware else 'x11'
            os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1' if use_hardware else '0'
            self._init_display()  # Reinitialize display with new settings
            logger.info(f"Display mode changed to {'hardware' if use_hardware else 'window'}")

    def get_debug_ui_element_at(self, pos: Tuple[int, int]) -> Optional[str]:
        """Check if a position overlaps with a known debug UI element.

        Args:
            pos: The (x, y) coordinate to check.

        Returns:
            The string ID of the clicked element (e.g., 'select_cam_1') or None.
        """
        for element_id, rect in self.debug_ui_elements.items():
            if rect.collidepoint(pos):
                return element_id
        return None

    def set_eye_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """Set a new eye color and reload images."""
        self.eye_color = self._parse_color(color)
        self._load_emotion_images()  # Reload images with new eye color
        self.update()

    def set_background_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """Set a new background color."""
        self.background_color = self._parse_color(color)
        self._load_emotion_images()  # Reload images with new background
        self.update()

    def get_current_emotion(self) -> Emotion:
        """Get the current emotion being displayed."""
        return self._current_emotion

    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        if not isinstance(emotion, Emotion):
            raise ValueError(f"Expected Emotion enum, got {type(emotion)}")
        self._current_emotion = emotion
        self.update()

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            
            # Save current emotion
            current_emotion = self._current_emotion
            
            # Set to blink emotion
            self._current_emotion = Emotion.BLINK
            self.update()
            
            # Wait for blink duration
            time.sleep(self.blink_duration)
            
            # Restore previous emotion
            self._current_emotion = current_emotion
            self.update()
            
            self.is_blinking = False
            logger.debug("Completed blink animation")
        except Exception as e:
            logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False

    print("libcamera imported successfully!") 