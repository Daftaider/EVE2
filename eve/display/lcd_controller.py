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
                 transition_time_ms=500):
        """
        Initialize the display controller
        
        Args:
            width (int): Display width in pixels
            height (int): Display height in pixels
            fullscreen (bool): Whether to use fullscreen mode
            resolution (tuple): Optional (width, height) tuple
            fps (int): Target frames per second for the display
            default_emotion (str): Default emotion to display on startup
            background_color (tuple): RGB color tuple for background
            eye_color (tuple): RGB color tuple for eyes
            transition_time_ms (int): Time in milliseconds for emotion transitions
        """
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.fps = fps
        self.background_color = background_color
        self.eye_color = eye_color
        self.transition_time_ms = transition_time_ms
        self.current_emotion = default_emotion
        self.target_emotion = default_emotion
        
        # Initialize state
        self.running = False
        self.render_thread = None
        self.use_fallback = False
        self.gl_context_error_count = 0
        self.max_gl_errors = 3  # Switch to fallback after this many errors
        
        # Try to initialize display
        self._init_display()
        
    def _init_display(self):
        """Initialize display with fallback options"""
        try:
            # Try SDL video driver first
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            pygame.init()
            pygame.display.init()
        except pygame.error:
            self.logger.warning("Failed to initialize X11 display, trying dummy driver")
            try:
                # Try dummy driver as fallback
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
                pygame.display.init()
                self.use_fallback = True
            except pygame.error as e:
                self.logger.error(f"Failed to initialize display: {e}")
                self.use_fallback = True
                return

        try:
            if not self.use_fallback:
                # Try to set up the display
                if self.fullscreen:
                    self.screen = pygame.display.set_mode((self.width, self.height), 
                                                        pygame.FULLSCREEN | pygame.HWSURFACE)
                else:
                    self.screen = pygame.display.set_mode((self.width, self.height))
                
                pygame.display.set_caption("EVE2 Display")
            else:
                # Create a surface for fallback mode
                self.screen = pygame.Surface((self.width, self.height))
            
            self.clock = pygame.time.Clock()
            self._load_assets()
        except Exception as e:
            self.logger.error(f"Display initialization error: {e}")
            self.use_fallback = True
        
    def _load_assets(self) -> None:
        """Load emotion assets or generate default ones."""
        logger.info("Loading display assets")
        
        # Load emotion images
        self._load_emotion_images()
        
        logger.info(f"Loaded {len(self.emotion_images)} emotion assets")
    
    def _load_emotion_images(self) -> None:
        """Load emotion image assets"""
        # Get list of emotions
        emotions = getattr(config.display, "EMOTIONS", 
                  getattr(config.display, "AVAILABLE_EMOTIONS", 
                  ["neutral", "happy", "sad", "angry", "surprised", "confused", "thinking"]))
        
        assets_dir = getattr(config, "ASSETS_DIR", "assets")
        emotions_dir = os.path.join(assets_dir, "emotions")
        
        # Create directory if it doesn't exist
        if not os.path.exists(emotions_dir):
            try:
                os.makedirs(emotions_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating emotions directory: {e}")
        
        # Load each emotion image
        for emotion in emotions:
            # Try multiple image formats
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_path = os.path.join(emotions_dir, f"{emotion}{ext}")
                if os.path.exists(image_path):
                    try:
                        self.emotion_images[emotion] = pygame.image.load(image_path)
                        found = True
                        break
                    except Exception as e:
                        logger.error(f"Error loading image for {emotion}: {e}")
            
            if not found:
                logger.warning(f"No image found for emotion: {emotion}, generating default")
                self.emotion_images[emotion] = self._generate_emotion_image(emotion)
    
    def _generate_emotion_image(self, emotion: str) -> pygame.Surface:
        """Generate a default image for an emotion"""
        # Create a surface
        img = pygame.Surface((300, 200))
        img.fill(self.background_color)
        
        # Draw eyes using the specified eye color
        if emotion == "neutral":
            pygame.draw.circle(img, self.eye_color, (100, 80), 30)
            pygame.draw.circle(img, self.eye_color, (200, 80), 30)
            pygame.draw.rect(img, self.eye_color, (100, 140, 100, 20))
        elif emotion == "happy":
            pygame.draw.circle(img, self.eye_color, (100, 80), 30)
            pygame.draw.circle(img, self.eye_color, (200, 80), 30)
            pygame.draw.arc(img, self.eye_color, (75, 120, 150, 60), 0, 3.14, 5)
        elif emotion == "sad":
            pygame.draw.arc(img, self.eye_color, (75, 150, 150, 60), 3.14, 6.28, 5)
        elif emotion == "angry":
            pygame.draw.polygon(img, self.eye_color, [(70, 80), (130, 60), (130, 90)])
            pygame.draw.polygon(img, self.eye_color, [(170, 60), (230, 80), (170, 90)])
            pygame.draw.rect(img, self.eye_color, (100, 150, 100, 20))
        elif emotion == "surprised":
            pygame.draw.circle(img, self.eye_color, (100, 80), 40)
            pygame.draw.circle(img, self.eye_color, (200, 80), 40)
            pygame.draw.circle(img, self.eye_color, (150, 150), 30)
        elif emotion == "confused":
            pygame.draw.ellipse(img, self.eye_color, (170, 60, 60, 40))
            pygame.draw.arc(img, self.eye_color, (75, 120, 150, 60), 0.5, 3.64, 5)
        else:  # default or "thinking"
            pygame.draw.arc(img, self.eye_color, (75, 140, 150, 30), 0, 3.14, 5)
            pygame.draw.circle(img, self.eye_color, (250, 50), 15)
        
        # Add text for the emotion
        font = pygame.font.SysFont(None, 24)
        text = font.render(emotion, True, (255, 255, 255))
        img.blit(text, (150 - text.get_width() // 2, 10))
        
        return img
    
    def start(self) -> bool:
        """Start the display controller."""
        if self.running:
            logger.warning("Display controller is already running")
            return False
        
        if self.screen is None or self.clock is None:
            logger.error("Display controller not initialized properly")
            return False
        
        logger.info("Starting display controller")
        
        # Set the initial emotion
        self.current_emotion = self.default_emotion
        self.target_emotion = self.default_emotion
        
        # Start the rendering thread
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        
        logger.info("Display controller started successfully")
        return True
    
    def stop(self) -> bool:
        """Stop the display controller."""
        if not self.running:
            logger.warning("Display controller is not running")
            return False
        
        logger.info("Stopping display controller")
        
        # Signal the rendering thread to stop
        self.running = False
        
        # Wait for the rendering thread to finish
        if hasattr(self, 'render_thread') and self.render_thread.is_alive():
            self.render_thread.join(timeout=2.0)
        
        # Quit pygame
        pygame.quit()
        
        logger.info("Display controller stopped successfully")
        return True
    
    def set_emotion(self, emotion: str) -> bool:
        """
        Set the emotion to display.
        
        Args:
            emotion: The emotion to display
        """
        # Check if the emotion is valid
        if emotion not in self.emotion_images:
            logger.warning(f"Unknown emotion: {emotion}, using default")
            emotion = self.default_emotion
        
        # Skip if the target emotion is already set
        if emotion == self.target_emotion:
            return True
        
        logger.info(f"Setting emotion to: {emotion}")
        
        # Start a transition to the new emotion
        self.target_emotion = emotion
        self.transition_start_time = time.time()
        self.transition_progress = 0.0
        return True
    
    def _render_loop(self) -> None:
        """Main rendering loop with error handling and fallback mode"""
        self.logger.info("Starting render loop")
        last_error_time = 0
        error_cooldown = 5  # Seconds between logging repeated errors
        
        while self.running:
            try:
                # Process events only in non-fallback mode
                if not self.use_fallback:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                
                # Clear screen
                self.screen.fill(self.background_color)
                
                # Render current emotion
                self._render_current_state()
                
                # Update display with error handling
                if not self.use_fallback:
                    try:
                        pygame.display.flip()
                    except pygame.error as e:
                        current_time = time.time()
                        if current_time - last_error_time > error_cooldown:
                            self.logger.error(f"Error updating display: {e}")
                            last_error_time = current_time
                            self.gl_context_error_count += 1
                            
                        if self.gl_context_error_count >= self.max_gl_errors:
                            self.logger.warning("Too many GL errors, switching to fallback mode")
                            self.use_fallback = True
                else:
                    # In fallback mode, just log the current state periodically
                    if time.time() - last_error_time > error_cooldown:
                        self.logger.info(f"Fallback mode active - Current emotion: {self.current_emotion}")
                        last_error_time = time.time()
                
                # Maintain frame rate
                self.clock.tick(self.fps)
                
            except Exception as e:
                current_time = time.time()
                if current_time - last_error_time > error_cooldown:
                    self.logger.error(f"Render loop error: {e}")
                    last_error_time = current_time
    
    def _render_current_state(self):
        """Render the current display state"""
        try:
            if self.current_emotion in self.emotion_images:
                image = self.emotion_images[self.current_emotion]
                if not self.use_fallback:
                    x = (self.width - image.get_width()) // 2
                    y = (self.height - image.get_height()) // 2
                    self.screen.blit(image, (x, y))
        except Exception as e:
            self.logger.error(f"Error rendering emotion: {e}")

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