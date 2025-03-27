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

logger = logging.getLogger(__name__)

class LCDController:
    """
    LCD display controller for rendering eye animations.
    
    This class handles the initialization of the display and
    manages the rendering of emotive eye animations.
    """
    
    def __init__(self, width=800, height=480, fps=30, default_emotion="neutral",
                 background_color=(0, 0, 0), eye_color=(0, 191, 255), **kwargs):
        """Initialize the LCD Controller"""
        self.logger = logging.getLogger(__name__)
        
        # Force pure software mode before pygame init
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Start with dummy driver
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        
        self.width = width
        self.height = height
        self.fps = fps
        self.background_color = background_color
        self.eye_color = eye_color
        self.default_emotion = default_emotion
        self.current_emotion = default_emotion
        
        self.running = False
        self.render_thread = None
        self.is_blinking = False
        self.blink_duration = 0.15
        self.emotion_images = {}
        
        # Initialize pygame in software-only mode
        self._init_display()

    def _init_display(self):
        """Initialize display in software mode"""
        try:
            # Initialize pygame
            if not pygame.get_init():
                pygame.init()
            
            # Create a software surface only
            self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            
            # Load or generate assets
            self._load_assets()
            
            self.logger.info("Display controller initialized in software mode")
            
        except Exception as e:
            self.logger.error(f"Error initializing display: {e}")
            raise

    def _load_assets(self):
        """Load or generate emotion assets"""
        try:
            # Generate basic emotions using shapes
            self._generate_basic_emotions()
            self.logger.info("Generated basic emotion assets")
        except Exception as e:
            self.logger.error(f"Error loading assets: {e}")
            raise

    def _generate_basic_emotions(self):
        """Generate basic emotion patterns using pygame shapes"""
        emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'blink']
        
        for emotion in emotions:
            surface = pygame.Surface((self.width, self.height))
            surface.fill(self.background_color)
            
            # Eye positions
            left_eye_x = self.width // 3
            right_eye_x = (self.width * 2) // 3
            eye_y = self.height // 2
            eye_size = 30
            
            if emotion == 'neutral':
                pygame.draw.circle(surface, self.eye_color, (left_eye_x, eye_y), eye_size)
                pygame.draw.circle(surface, self.eye_color, (right_eye_x, eye_y), eye_size)
            
            elif emotion == 'happy':
                pygame.draw.circle(surface, self.eye_color, (left_eye_x, eye_y - 10), eye_size)
                pygame.draw.circle(surface, self.eye_color, (right_eye_x, eye_y - 10), eye_size)
            
            elif emotion == 'sad':
                pygame.draw.circle(surface, self.eye_color, (left_eye_x, eye_y + 10), eye_size)
                pygame.draw.circle(surface, self.eye_color, (right_eye_x, eye_y + 10), eye_size)
            
            elif emotion == 'angry':
                pygame.draw.circle(surface, self.eye_color, (left_eye_x, eye_y), eye_size)
                pygame.draw.circle(surface, self.eye_color, (right_eye_x, eye_y), eye_size)
                # Add angry eyebrows
                pygame.draw.line(surface, self.eye_color, 
                               (left_eye_x - 30, eye_y - 30),
                               (left_eye_x + 30, eye_y - 10), 5)
                pygame.draw.line(surface, self.eye_color,
                               (right_eye_x - 30, eye_y - 10),
                               (right_eye_x + 30, eye_y - 30), 5)
            
            elif emotion == 'surprised':
                pygame.draw.circle(surface, self.eye_color, (left_eye_x, eye_y), eye_size + 10)
                pygame.draw.circle(surface, self.eye_color, (right_eye_x, eye_y), eye_size + 10)
            
            elif emotion == 'blink':
                # Draw simple lines for closed eyes
                pygame.draw.line(surface, self.eye_color,
                               (left_eye_x - 30, eye_y),
                               (left_eye_x + 30, eye_y), 5)
                pygame.draw.line(surface, self.eye_color,
                               (right_eye_x - 30, eye_y),
                               (right_eye_x + 30, eye_y), 5)
            
            self.emotion_images[emotion] = surface

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
                'blink' if self.is_blinking else self.current_emotion,
                self.emotion_images['neutral']
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
            emotion = 'neutral'
        
        self.current_emotion = emotion
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