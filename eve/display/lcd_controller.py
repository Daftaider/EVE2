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
                 default_emotion="neutral", background_color=(0, 0, 0)):
        """
        Initialize the display controller
        
        Args:
            width (int): Display width in pixels
            height (int): Display height in pixels
            fullscreen (bool): Whether to use fullscreen mode
            resolution (tuple): Optional (width, height) tuple
            fps (int): Target frames per second for the display
            default_emotion (str): Default emotion to display on startup
            background_color (tuple): RGB color tuple for background (0-255 for each component)
        """
        # If resolution is provided, use it instead of width/height parameters
        if resolution and isinstance(resolution, tuple) and len(resolution) == 2:
            self.width, self.height = resolution
        else:
            self.width = width
            self.height = height
        
        self.fullscreen = fullscreen
        self.fps = fps
        self.default_emotion = default_emotion
        self.background_color = background_color
        self.running = False
        self.render_thread = None
        self.current_emotion = default_emotion
        self.emotion_images = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize pygame with software rendering fallback
        self._init_pygame()
        
        # Load display assets
        self._load_assets()
        
    def _init_pygame(self) -> None:
        """Initialize pygame with fallback options"""
        try:
            logger.info("Initializing pygame")
            pygame.init()
            
            # Try to set hardware acceleration first
            try:
                # Set the SDL_VIDEODRIVER environment variable to 'x11' for Raspberry Pi
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                
                # Try to create the display
                flags = pygame.HWSURFACE | pygame.DOUBLEBUF
                if self.fullscreen:
                    flags |= pygame.FULLSCREEN
                    
                self.screen = pygame.display.set_mode((self.width, self.height), flags)
                logger.info("Using hardware accelerated rendering")
                
            except pygame.error as e:
                logger.warning(f"Hardware acceleration failed: {e}")
                
                # Fall back to software rendering
                os.environ['SDL_VIDEODRIVER'] = 'fbcon'  # Try framebuffer console
                try:
                    flags = pygame.SWSURFACE
                    if self.fullscreen:
                        flags |= pygame.FULLSCREEN
                    self.screen = pygame.display.set_mode((self.width, self.height), flags)
                    logger.info("Using framebuffer console rendering")
                except pygame.error:
                    # Final fallback - dummy display
                    os.environ['SDL_VIDEODRIVER'] = 'dummy'
                    self.screen = pygame.display.set_mode((self.width, self.height))
                    logger.warning("Using dummy display driver - no visible output")
            
            pygame.display.set_caption("EVE2")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            logger.info("Pygame initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            raise
    
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
        img.fill(self.background_color)  # Black background
        
        # Draw eyes
        pygame.draw.circle(img, self.eye_color, (100, 80), 30)
        pygame.draw.circle(img, self.eye_color, (200, 80), 30)
        
        # Draw different eye shapes based on the emotion
        if emotion == "neutral":
            pygame.draw.rect(img, self.eye_color, (100, 140, 100, 20))
        elif emotion == "happy":
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
        """Main rendering loop running in a separate thread."""
        self.logger.info("Rendering loop started")
        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                
                # Clear screen with the background color
                self.screen.fill(self.background_color)
                
                # Update emotion transition
                self._update_transition()
                
                # Render the current frame
                self._render_frame()
                
                # Cap the frame rate
                self.clock.tick(self.fps)
            
        except Exception as e:
            logger.error(f"Error in rendering loop: {e}")
            time.sleep(0.1)
        
        logger.info("Rendering loop stopped")
    
    def _update_transition(self) -> None:
        """Update the emotion transition progress."""
        # Skip if no transition is in progress
        if self.transition_progress >= 1.0:
            return
        
        # Calculate time elapsed since transition started
        elapsed = time.time() - self.transition_start_time
        
        # Calculate transition progress (0.0 to 1.0)
        self.transition_progress = min(1.0, elapsed / (self.transition_time_ms / 1000.0))
        
        # If transition is complete, update current emotion
        if self.transition_progress >= 1.0:
            self.current_emotion = self.target_emotion
    
    def _render_frame(self) -> None:
        """Render the current frame to the display."""
        if self.transition_progress < 1.0:
            # During transition, blend between current and target emotions
            if self.current_emotion in self.emotion_images and self.target_emotion in self.emotion_images:
                current_img = self.emotion_images[self.current_emotion]
                target_img = self.emotion_images[self.target_emotion]
                
                # Create a temporary surface for the blended result
                temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                
                # Draw the current emotion image with fading alpha
                current_alpha = int(255 * (1.0 - self.transition_progress))
                current_img.set_alpha(current_alpha)
                temp_surface.blit(current_img, (0, 0))
                
                # Draw the target emotion image with increasing alpha
                target_alpha = int(255 * self.transition_progress)
                target_img.set_alpha(target_alpha)
                temp_surface.blit(target_img, (0, 0))
                
                # Draw the blended result to the screen
                self.screen.blit(temp_surface, (0, 0))
            else:
                # Fallback if images are missing
                if self.target_emotion in self.emotion_images:
                    self.screen.blit(self.emotion_images[self.target_emotion], (0, 0))
                elif self.current_emotion in self.emotion_images:
                    self.screen.blit(self.emotion_images[self.current_emotion], (0, 0))
                else:
                    # Last resort: draw default eyes
                    self._generate_emotion_image(self.default_emotion)
        else:
            # No transition, just draw the current emotion
            if self.current_emotion in self.emotion_images:
                self.screen.blit(self.emotion_images[self.current_emotion], (0, 0))
            else:
                # Fallback if image is missing
                self._generate_emotion_image(self.default_emotion)
        
        # Update the display
        pygame.display.flip()

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