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
    
    def __init__(self, config=None):
        self.config = config or DisplayConfig
        self._current_emotion = self.config.DEFAULT_EMOTION
        self._init_display()

    def _init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            self.config.WINDOW_SIZE,
            pygame.FULLSCREEN if self.config.FULLSCREEN else 0
        )
        self.clock = pygame.time.Clock()
        self._load_emotion_images()

    def _load_emotion_images(self):
        """Load all emotion images into memory."""
        self.emotion_images = {}
        for emotion in Emotion:
            try:
                image_path = self.config.get_emotion_path(emotion)
                self.emotion_images[emotion] = pygame.image.load(image_path)
            except Exception as e:
                logging.error(f"Failed to load emotion image for {emotion}: {e}")
                # Create a colored rectangle as fallback
                surface = pygame.Surface(self.config.WINDOW_SIZE)
                surface.fill(self._get_fallback_color(emotion))
                self.emotion_images[emotion] = surface

    def _get_fallback_color(self, emotion: Emotion):
        """Get a fallback color for an emotion."""
        colors = {
            Emotion.NEUTRAL: (128, 128, 128),    # Gray
            Emotion.HAPPY: (255, 255, 0),        # Yellow
            Emotion.SAD: (0, 0, 255),            # Blue
            Emotion.ANGRY: (255, 0, 0),          # Red
            Emotion.SURPRISED: (255, 165, 0),    # Orange
            Emotion.CONFUSED: (128, 0, 128),     # Purple
        }
        return colors.get(emotion, (0, 0, 0))    # Default to black

    def update(self, emotion: Emotion = None):
        """Update the display with the given emotion."""
        if emotion is not None and isinstance(emotion, Emotion):
            self._current_emotion = emotion

        try:
            self.screen.blit(self.emotion_images[self._current_emotion], (0, 0))
            pygame.display.flip()
            self.clock.tick(self.config.FPS)
        except Exception as e:
            logging.error(f"Error updating display: {e}")

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()

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

    def set_emotion(self, emotion):
        """Set the current emotion"""
        if emotion not in self.emotion_images:
            self.logger.warning(f"Unknown emotion: {emotion}, falling back to neutral")
            emotion = Emotion.NEUTRAL
        
        self._current_emotion = emotion
        return True

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