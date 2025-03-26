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
    
    def __init__(self, config=None, width=800, height=480, fps=30, 
                 default_emotion="neutral", background_color=(0, 0, 0), 
                 eye_color=(0, 191, 255), **kwargs):  # Added **kwargs to handle extra arguments
        self.logger = logging.getLogger(__name__)
        
        # Initialize with defaults or config values
        if config:
            self.width = getattr(config, 'WIDTH', width)
            self.height = getattr(config, 'HEIGHT', height)
            self.fps = getattr(config, 'FPS', fps)
            self.default_emotion = getattr(config, 'DEFAULT_EMOTION', default_emotion)
            self.background_color = getattr(config, 'BACKGROUND_COLOR', background_color)
            self.eye_color = getattr(config, 'EYE_COLOR', eye_color)
        else:
            self.width = width
            self.height = height
            self.fps = fps
            self.default_emotion = default_emotion
            self.background_color = background_color
            self.eye_color = eye_color

        # Handle resolution from kwargs if provided
        if 'resolution' in kwargs and kwargs['resolution']:
            self.width, self.height = kwargs['resolution']
        
        self.current_emotion = self.default_emotion
        self.running = False
        self.render_thread = None
        self.use_fallback = 'DISPLAY' not in os.environ
        self.emotion_images = {}
        
        self._init_display()
        self._load_assets()

    def _init_display(self):
        try:
            os.environ['SDL_VIDEODRIVER'] = 'dummy' if self.use_fallback else 'x11'
            pygame.init()
            self.screen = (pygame.Surface((self.width, self.height)) if self.use_fallback 
                         else pygame.display.set_mode((self.width, self.height)))
            self.clock = pygame.time.Clock()
            self.logger.info("Display initialized in {} mode".format(
                'headless' if self.use_fallback else 'display'))
        except Exception as e:
            self.logger.error("Display initialization error: {}".format(str(e)))
            self.use_fallback = True
            self.screen = pygame.Surface((self.width, self.height))
            pygame.init()

    def _generate_default_emotion(self):
        try:
            image = Image.new('RGB', (self.width, self.height), self.background_color)
            draw = ImageDraw.Draw(image)
            
            eye_size = min(self.width, self.height) // 10
            center_y = self.height // 2
            left_x = self.width // 3
            right_x = (self.width * 2) // 3
            
            for x in (left_x, right_x):
                draw.ellipse([x - eye_size, center_y - eye_size, 
                            x + eye_size, center_y + eye_size], 
                           fill=self.eye_color)
            
            return pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        except Exception as e:
            self.logger.error("Error generating default emotion: {}".format(str(e)))
            surface = pygame.Surface((self.width, self.height))
            surface.fill(self.background_color)
            return surface

    def _load_assets(self):
        self.logger.info("Loading display assets")
        basic_emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'confused']
        assets_path = os.path.join(os.path.dirname(__file__), 'assets', 'emotions')
        
        try:
            os.makedirs(assets_path, exist_ok=True)
            for emotion in basic_emotions:
                image_path = os.path.join(assets_path, "{}.png".format(emotion))
                self.emotion_images[emotion] = (
                    pygame.image.load(image_path) if os.path.exists(image_path)
                    else self._generate_default_emotion()
                )
        except Exception as e:
            self.logger.error("Error loading assets: {}".format(str(e)))
            self.emotion_images['neutral'] = self._generate_default_emotion()

    def _render_loop(self):
        last_state_log = 0
        state_log_interval = 5.0
        
        while self.running:
            try:
                if not self.use_fallback:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                
                self.screen.fill(self.background_color)
                
                if self.current_emotion in self.emotion_images:
                    image = self.emotion_images[self.current_emotion]
                    x = (self.width - image.get_width()) // 2
                    y = (self.height - image.get_height()) // 2
                    self.screen.blit(image, (x, y))
                
                current_time = time.time()
                if not self.use_fallback:
                    try:
                        pygame.display.flip()
                    except pygame.error:
                        self.use_fallback = True
                elif current_time - last_state_log >= state_log_interval:
                    self.logger.debug("Headless mode - Current emotion: {}".format(
                        self.current_emotion))
                    last_state_log = current_time
                
                self.clock.tick(self.fps)
                
            except Exception as e:
                self.logger.error("Render loop error: {}".format(str(e)))
                time.sleep(1)

    def start(self):
        if not self.running:
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True
            self.render_thread.start()
            self.logger.info("Display controller started")

    def stop(self):
        self.running = False
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
        if not self.use_fallback:
            try:
                pygame.quit()
            except pygame.error:
                pass
        self.logger.info("Display controller stopped")

    def set_emotion(self, emotion):
        if emotion not in self.emotion_images:
            self.logger.warning("Unknown emotion: {}, using neutral".format(emotion))
            emotion = 'neutral'
        self.current_emotion = emotion
        return True

    def is_active(self):
        return self.running and self.render_thread and self.render_thread.is_alive()

    def _create_default_assets(self):
        """Create default assets for testing when real assets are not available"""
        logger.info("Creating default assets")
        # Create minimal assets in memory
        import pygame
        
        # Create a simple background
        self.background = pygame.Surface((800, 480))
        self.background.fill((0, 0, 0))  # Black background
        
        # Create simple eye assets
        self.eye = pygame.Surface((50, 50))
        pygame.draw.circle(self.eye, (0, 191, 255), (25, 25), 25)
        
        # Add text
        font = pygame.font.SysFont(None, 36)
        text = font.render("EVE2 Default Display", True, (255, 255, 255))
        self.background.blit(text, (400 - text.get_width() // 2, 50)) 
        self.background.blit(text, (400 - text.get_width() // 2, 50)) 