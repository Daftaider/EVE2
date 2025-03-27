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

from eve import config
from eve.config.display import Emotion, DisplayConfig

logger = logging.getLogger(__name__)

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
                 default_emotion: Optional[Emotion] = None,
                 background_color: Optional[Union[Tuple[int, int, int], str]] = None,
                 eye_color: Optional[Union[Tuple[int, int, int], str]] = None):
        """
        Initialize the LCD Controller.
        
        Args:
            config: Display configuration object
            width: Optional window width (overrides config)
            height: Optional window height (overrides config)
            fps: Optional frames per second (overrides config)
            default_emotion: Optional starting emotion (overrides config)
            background_color: Optional background color as RGB tuple or string (default: black)
            eye_color: Optional eye color as RGB tuple or string (default: white)
        """
        self.config = config or DisplayConfig
        
        # Override config values if parameters are provided
        if width is not None and height is not None:
            self.window_size = (width, height)
        else:
            self.window_size = self.config.WINDOW_SIZE
            
        self.fps = fps if fps is not None else self.config.FPS
        
        # Ensure default_emotion is a valid Emotion enum
        if default_emotion is not None and not isinstance(default_emotion, Emotion):
            logging.warning(f"Invalid default_emotion type: {type(default_emotion)}, using DEFAULT_EMOTION")
            default_emotion = None
        self._current_emotion = default_emotion if default_emotion is not None else self.config.DEFAULT_EMOTION
        
        # Handle colors
        self.background_color = self._parse_color(background_color) if background_color else self.config.DEFAULT_BACKGROUND_COLOR
        self.eye_color = self._parse_color(eye_color) if eye_color else self.config.DEFAULT_EYE_COLOR
        
        # Initialize display system
        self._init_display()
        
        # Log initialization parameters
        logging.info(f"LCD Controller initialized with: size={self.window_size}, "
                    f"fps={self.fps}, default_emotion={self._current_emotion.name}, "
                    f"background_color={self.background_color}, "
                    f"eye_color={self.eye_color}")

    def _parse_color(self, color: Union[Tuple[int, int, int], str]) -> Tuple[int, int, int]:
        """Convert color string or tuple to RGB tuple."""
        if isinstance(color, tuple) and len(color) == 3:
            return color
        elif isinstance(color, str):
            try:
                return pygame.Color(color)[:3]
            except ValueError:
                logging.warning(f"Invalid color string: {color}, using default")
                return (255, 255, 255)  # Default to white
        else:
            logging.warning(f"Invalid color format: {color}, using default")
            return (255, 255, 255)  # Default to white

    def _init_display(self):
        """Initialize the display with current settings."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode(
                self.window_size,
                pygame.FULLSCREEN if self.config.FULLSCREEN else 0
            )
            self.clock = pygame.time.Clock()
            self._load_emotion_images()
            self.screen.fill(self.background_color)
            pygame.display.flip()
            logging.info(f"Display initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize display: {e}")
            self._init_fallback_mode()

    def _init_fallback_mode(self):
        """Initialize a fallback mode for headless operation."""
        logging.info("Initializing display in fallback mode")
        pygame.init()
        self.screen = pygame.Surface(self.window_size)
        self.screen.fill(self.background_color)
        self.clock = pygame.time.Clock()
        self._load_emotion_images()

    def _load_emotion_images(self):
        """Load all emotion images into memory."""
        self.emotion_images = {}
        for emotion in Emotion:
            try:
                image_path = self.config.get_emotion_path(emotion)
                original = pygame.image.load(image_path)
                
                # Create a new surface for the emotion
                image = pygame.Surface(self.window_size)
                image.fill(self.background_color)
                
                # Scale and center the emotion image
                scaled = pygame.transform.scale(original, self.window_size)
                
                # Apply eye color if the image is not None
                if scaled is not None:
                    # Convert eye color pixels (assuming white pixels are eyes)
                    pixels = pygame.PixelArray(scaled)
                    white_pixels = pixels.compare(pygame.Color('white'))
                    pixels[white_pixels] = self.eye_color
                    pixels.close()
                
                image.blit(scaled, (0, 0))
                self.emotion_images[emotion] = image
                logging.debug(f"Loaded emotion image for {emotion}")
                
            except Exception as e:
                logging.warning(f"Failed to load emotion image for {emotion}: {e}")
                # Create a fallback colored rectangle
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
            Emotion.CONFUSED: (128, 0, 128),     # Purple
        }
        return colors.get(emotion, (0, 0, 0))

    def update(self, emotion: Optional[Emotion] = None) -> None:
        """Update the display with the given emotion."""
        if emotion is not None and isinstance(emotion, Emotion):
            self._current_emotion = emotion

        try:
            self.screen.fill(self.background_color)
            self.screen.blit(self.emotion_images[self._current_emotion], (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
        except Exception as e:
            logging.error(f"Error updating display: {e}")

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

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        try:
            pygame.quit()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _render_loop(self):
        """Main rendering loop"""
        last_error_time = 0
        error_cooldown = 5  # seconds between error messages
        
        while self.running:
            try:
                self._render_current_emotion()
                self.clock.tick(self.fps)
            except Exception as e:
                current_time = time.time()
                if current_time - last_error_time > error_cooldown:
                    self.logger.error(f"Error in render loop: {e}")
                    last_error_time = current_time
                time.sleep(0.1)  # Prevent tight error loop

    def _render_current_emotion(self):
        """Render the current emotion to the surface"""
        try:
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Get current emotion surface
            emotion_surface = self.emotion_images.get(
                'blink' if self.is_blinking else self._current_emotion,
                self.emotion_images[Emotion.NEUTRAL]
            )
            
            # Blit emotion onto screen
            self.screen.blit(emotion_surface, (0, 0))
            
            # Save the current frame to a file for external display
            pygame.image.save(self.screen, "current_display.png")
            
        except Exception as e:
            self.logger.error(f"Error rendering emotion: {e}")

    def start(self):
        """Start the display controller"""
        if not self.running:
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True
            self.render_thread.start()
            self.logger.info("Display controller started")

    def stop(self):
        """Stop the display controller"""
        self.running = False
        if self.render_thread:
            try:
                self.render_thread.join(timeout=1.0)
            except Exception as e:
                self.logger.error(f"Error stopping render thread: {e}")
        
        try:
            pygame.quit()
        except Exception as e:
            self.logger.error(f"Error during pygame cleanup: {e}")
        
        self.logger.info("Display controller stopped")

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            time.sleep(self.blink_duration)
            self.is_blinking = False
            self.logger.debug("Completed blink animation")
        except Exception as e:
            self.logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False 