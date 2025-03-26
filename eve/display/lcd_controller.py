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
    
    def __init__(self, width=800, height=480, fullscreen=False, resolution=None, fps=30, 
                 default_emotion="neutral", background_color=(0, 0, 0), eye_color=(0, 191, 255),
                 transition_time_ms=500, headless=False):
        """Initialize the LCD Controller with display parameters and emotion assets"""
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.fps = fps
        self.background_color = background_color
        self.eye_color = eye_color
        self.transition_time_ms = transition_time_ms
        self.default_emotion = default_emotion
        self.current_emotion = default_emotion
        self.target_emotion = default_emotion
        
        # Initialize state
        self.running = False
        self.render_thread = None
        self.use_fallback = headless  # Start in fallback mode if headless
        self.emotion_images = {}
        self.clock = None
        self.screen = None
        
        # Initialize pygame in headless mode if specified
        self._init_display()
        self._load_assets()

    def _generate_default_emotion(self):
        """Generate a default emotion face using PIL"""
        try:
            # Create a blank image with background color
            image = Image.new('RGB', (self.width, self.height), self.background_color)
            draw = ImageDraw.Draw(image)
            
            # Draw simple eyes
            eye_size = min(self.width, self.height) // 10
            center_y = self.height // 2
            left_x = self.width // 3
            right_x = (self.width * 2) // 3
            
            # Draw eyes as circles
            draw.ellipse([left_x - eye_size, center_y - eye_size, 
                         left_x + eye_size, center_y + eye_size], 
                        fill=self.eye_color)
            draw.ellipse([right_x - eye_size, center_y - eye_size, 
                         right_x + eye_size, center_y + eye_size], 
                        fill=self.eye_color)
            
            # Convert PIL image to pygame surface
            mode = image.mode
            size = image.size
            data = image.tobytes()
            return pygame.image.fromstring(data, size, mode)
            
        except Exception as e:
            self.logger.error("Error generating default emotion: {}".format(str(e)))
            # Return a simple colored surface as last resort
            surface = pygame.Surface((self.width, self.height))
            surface.fill(self.background_color)
            return surface

    def _load_assets(self):
        """Load emotion assets with fallback to generated images"""
        self.logger.info("Loading display assets")
        
        # Define basic emotions
        basic_emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'confused']
        assets_path = os.path.join(os.path.dirname(__file__), 'assets', 'emotions')
        
        try:
            # Create assets directory if it doesn't exist
            os.makedirs(assets_path, exist_ok=True)
            
            # Load or generate each emotion
            for emotion in basic_emotions:
                image_path = os.path.join(assets_path, "{}.png".format(emotion))
                
                if os.path.exists(image_path):
                    try:
                        self.emotion_images[emotion] = pygame.image.load(image_path)
                    except pygame.error:
                        self.logger.warning("Failed to load image for emotion: {}".format(emotion))
                        self.emotion_images[emotion] = self._generate_default_emotion()
                else:
                    self.logger.warning("No image found for emotion: {}".format(emotion))
                    self.emotion_images[emotion] = self._generate_default_emotion()
                    
                    # Save generated emotion for future use
                    try:
                        pygame.image.save(self.emotion_images[emotion], image_path)
                    except pygame.error as e:
                        self.logger.warning("Could not save generated emotion: {}".format(str(e)))
                        
        except Exception as e:
            self.logger.error("Error loading assets: {}".format(str(e)))
            # Ensure at least neutral emotion exists
            self.emotion_images['neutral'] = self._generate_default_emotion()

    def _init_display(self):
        """Initialize display with automatic headless detection"""
        try:
            # Check for display availability
            if 'DISPLAY' not in os.environ:
                self.logger.info("No display detected, running in headless mode")
                self.use_fallback = True
            
            if self.use_fallback:
                # Headless mode initialization
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
                self.screen = pygame.Surface((self.width, self.height))
                self.logger.info("Initialized in headless mode")
            else:
                # Try hardware acceleration first
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                pygame.init()
                try:
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((self.width, self.height), 
                                                            pygame.FULLSCREEN | pygame.HWSURFACE)
                    else:
                        self.screen = pygame.display.set_mode((self.width, self.height))
                    pygame.display.set_caption("EVE2 Display")
                except pygame.error:
                    self.logger.warning("Failed to initialize hardware-accelerated display")
                    self.use_fallback = True
                    self.screen = pygame.Surface((self.width, self.height))
            
            self.clock = pygame.time.Clock()
            
        except Exception as e:
            self.logger.error("Display initialization error: {}".format(str(e)))
            self.use_fallback = True
            self.screen = pygame.Surface((self.width, self.height))
            pygame.init()

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
            self.render_thread.join(timeout=1.0)
        if not self.use_fallback:
            try:
                pygame.quit()
            except pygame.error:
                pass
        self.logger.info("Display controller stopped")

    def set_emotion(self, emotion):
        """Set the current emotion to display"""
        if emotion not in self.emotion_images:
            self.logger.warning("Unknown emotion: {}, using neutral".format(emotion))
            emotion = 'neutral'
        self.target_emotion = emotion
        self.current_emotion = emotion
        
        if self.use_fallback:
            self.logger.debug("Emotion set to {} (headless mode)".format(emotion))
        return True

    def _render_loop(self):
        """Main rendering loop with headless mode support"""
        self.logger.info("Starting render loop in {} mode".format(
            'headless' if self.use_fallback else 'display'))
        last_state_log = 0
        state_log_interval = 5.0  # Log state every 5 seconds in headless mode
        
        while self.running:
            try:
                # Process events only in display mode
                if not self.use_fallback:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                
                # Update the screen content
                self.screen.fill(self.background_color)
                self._render_current_state()
                
                # Update display or log state in headless mode
                current_time = time.time()
                if not self.use_fallback:
                    try:
                        pygame.display.flip()
                    except pygame.error as e:
                        self.logger.debug("Display update failed: {}".format(str(e)))
                        self.use_fallback = True
                elif current_time - last_state_log >= state_log_interval:
                    self.logger.debug("Headless mode - Current emotion: {}".format(
                        self.current_emotion))
                    last_state_log = current_time
                
                # Maintain frame rate
                self.clock.tick(self.fps)
                
            except Exception as e:
                self.logger.error("Render loop error: {}".format(str(e)))
                time.sleep(1)  # Prevent tight error loop

    def _render_current_state(self):
        """Render the current display state"""
        try:
            if self.current_emotion in self.emotion_images:
                image = self.emotion_images[self.current_emotion]
                x = (self.width - image.get_width()) // 2
                y = (self.height - image.get_height()) // 2
                self.screen.blit(image, (x, y))
        except Exception as e:
            self.logger.debug("Error rendering emotion: {}".format(str(e)))

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