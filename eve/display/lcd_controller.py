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
    """Controls the LCD display and renders eye animations."""
    
    def __init__(self, config: DisplayConfig):
        """Initialize the LCD controller.
        
        Args:
            config: Display configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize display settings
        self.width, self.height = config.WINDOW_SIZE
        self.fps = config.FPS
        self.fullscreen = config.FULLSCREEN
        self.use_hardware_display = config.USE_HARDWARE_DISPLAY
        self.rotation = config.DISPLAY_ROTATION
        
        # Initialize colors
        self.background_color = config.BACKGROUND_COLOR
        self.eye_color = config.EYE_COLOR
        self.text_color = config.TEXT_COLOR
        
        # Initialize animation settings
        self.blink_interval = config.BLINK_INTERVAL_SEC
        self.blink_duration = config.BLINK_DURATION
        self.transition_speed = config.TRANSITION_SPEED
        
        # Initialize debug settings
        self.debug_menu_enabled = config.DEBUG_MENU_ENABLED
        self.debug_font_size = config.DEBUG_FONT_SIZE
        
        # Initialize file paths
        self.asset_dir = config.ASSET_DIR
        self.current_frame_path = config.CURRENT_FRAME_PATH
        
        # Initialize emotion settings
        self.current_emotion = config.DEFAULT_EMOTION
        self.emotion_images = {}
        
        # Initialize Pygame
        if not pygame.get_init():
            pygame.init()
        
        # Set up display
        if self.use_hardware_display:
            try:
                os.environ['SDL_VIDEODRIVER'] = 'fbcon'
                os.environ['SDL_FBDEV'] = '/dev/fb0'
                os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1'
                os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
                # Test if fbcon is available
                pygame.display.init()
                pygame.display.quit()
            except pygame.error:
                self.logger.warning("Framebuffer console (fbcon) not available, falling back to X11 display mode")
                self.use_hardware_display = False
                os.environ['SDL_VIDEODRIVER'] = 'x11'
        
        # Create display surface
        try:
            if self.fullscreen:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
            else:
                self.screen = pygame.display.set_mode((self.width, self.height))
        except pygame.error as e:
            self.logger.error(f"Failed to create display surface: {e}")
            raise
        
        # Load emotion images
        self._load_emotion_images()
        
        # Initialize animation state
        self.last_blink_time = time.time()
        self.is_blinking = False
        self.blink_start_time = 0
        
        self.logger.info("LCD Controller initialized with config: %s", config)
    
    def update(self, emotion: Emotion = None, is_listening: bool = False):
        """Update the display with the current emotion and state.
        
        Args:
            emotion: Current emotion to display
            is_listening: Whether EVE is currently listening
        """
        if emotion is not None:
            self.current_emotion = emotion
        
        # Handle blinking
        current_time = time.time()
        if current_time - self.last_blink_time >= self.blink_interval:
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_start_time = current_time
            elif current_time - self.blink_start_time >= self.blink_duration:
                self.is_blinking = False
                self.last_blink_time = current_time
        
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw current emotion
        if self.current_emotion in self.emotion_images:
            image = self.emotion_images[self.current_emotion]
            
            # Apply blinking effect
            if self.is_blinking:
                # Scale down the image during blink
                scale = 0.2
                scaled_size = (int(image.get_width() * scale), int(image.get_height() * scale))
                image = pygame.transform.scale(image, scaled_size)
            
            # Center the image
            x = (self.width - image.get_width()) // 2
            y = (self.height - image.get_height()) // 2
            
            # Apply rotation if needed
            if self.rotation != 0:
                image = pygame.transform.rotate(image, self.rotation)
            
            self.screen.blit(image, (x, y))
        
        # Draw debug menu if enabled
        if self.debug_menu_enabled:
            self._draw_debug_menu(is_listening)
        
        # Update display
        pygame.display.flip()
        
        # Save current frame if path is set
        if self.current_frame_path:
            pygame.image.save(self.screen, self.current_frame_path)
    
    def _draw_debug_menu(self, is_listening: bool):
        """Draw the debug menu on the screen.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        font = pygame.font.Font(None, self.debug_font_size)
        
        # Draw emotion text
        emotion_text = f"Emotion: {self.current_emotion.name}"
        text_surface = font.render(emotion_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 10))
        
        # Draw listening status
        listening_text = "Listening: Yes" if is_listening else "Listening: No"
        text_surface = font.render(listening_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 40))
        
        # Draw FPS
        fps_text = f"FPS: {int(self.fps)}"
        text_surface = font.render(fps_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 70))
    
    def cleanup(self):
        """Clean up resources."""
        pygame.quit()
        self.logger.info("LCD Controller cleaned up")

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
        
        # Create assets directory if it doesn't exist
        os.makedirs(self.asset_dir, exist_ok=True)
        
        for emotion in Emotion:
            try:
                # Construct the image path using the asset directory and emotion filename
                image_path = os.path.join(self.asset_dir, emotion.filename)
                
                # Check if file exists
                if not os.path.exists(image_path):
                    self.logger.warning(f"Emotion image not found: {image_path}")
                    # Create a fallback surface with emotion color
                    surface = pygame.Surface((self.width, self.height))
                    surface.fill(self._get_fallback_color(emotion))
                    self.emotion_images[emotion] = surface
                    continue
                
                original = pygame.image.load(image_path)
                
                # Scale image if needed
                if original.get_size() != (self.width, self.height):
                    original = pygame.transform.scale(original, (self.width, self.height))
                
                # Apply eye color
                colored = self._apply_eye_color(original, self.eye_color)
                self.emotion_images[emotion] = colored
                
            except Exception as e:
                self.logger.warning(f"Failed to load emotion image for {emotion.name}: {e}")
                # Create a fallback surface
                surface = pygame.Surface((self.width, self.height))
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

    def start(self) -> bool:
        """Start the display controller."""
        try:
            logger.info("Starting LCD Controller...")
            # Initial display update
            self.update()
            logger.info("LCD Controller started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start LCD Controller: {e}")
            return False

    def stop(self):
        """Stop the display controller."""
        try:
            logger.info("Stopping LCD Controller...")
            # Clean up resources
            pygame.quit()
            logger.info("LCD Controller stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping LCD Controller: {e}")

    def cleanup(self):
        """Clean up resources."""
        self.stop()

    def set_rotation(self, rotation: int):
        """Set the display rotation."""
        self.rotation = rotation % 360
        logger.info(f"Display rotation set to {self.rotation} degrees")

    def toggle_hardware_display(self, use_hardware: bool):
        """Toggle hardware display mode."""
        if self.use_hardware_display != use_hardware:
            self.use_hardware_display = use_hardware
            # Reinitialize display with new settings
            self._init_display()
            logger.info(f"Hardware display {'enabled' if use_hardware else 'disabled'}")

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
        return self.current_emotion

    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        if not isinstance(emotion, Emotion):
            raise ValueError(f"Expected Emotion enum, got {type(emotion)}")
        self.current_emotion = emotion
        self.update()

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            
            # Save current emotion
            current_emotion = self.current_emotion
            
            # Set to blink emotion
            self.current_emotion = Emotion.BLINK
            self.update()
            
            # Wait for blink duration
            time.sleep(self.blink_duration)
            
            # Restore previous emotion
            self.current_emotion = current_emotion
            self.update()
            
            self.is_blinking = False
            logger.debug("Completed blink animation")
        except Exception as e:
            logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False

    print("libcamera imported successfully!") 