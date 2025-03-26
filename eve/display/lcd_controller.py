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
                 background_color=(0, 0, 0), eye_color=(0, 191, 255)):
        """Initialize the LCD Controller with display parameters"""
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.fps = fps
        self.background_color = background_color
        self.eye_color = eye_color
        self.default_emotion = default_emotion
        self.current_emotion = default_emotion
        
        self.running = False
        self.render_thread = None
        self.use_fallback = 'DISPLAY' not in os.environ
        self.emotion_images = {}
        self.screen = None
        self.clock = None
        
        # Initialize pygame and display
        self._init_display()
        
        # Load emotions (or generate them if missing)
        self._load_assets()
        
        # Ensure we have at least a neutral emotion
        if not self.emotion_images:
            self.logger.warning("No emotion images loaded, generating default")
            self.emotion_images[self.default_emotion] = self._generate_default_emotion()
            
        # Force initial render to prevent blank screen
        self._render_current_emotion()
        if not self.use_fallback:
            try:
                pygame.display.flip()
                self.logger.info("Initial screen render complete")
            except pygame.error as e:
                self.logger.error(f"Error in initial screen render: {e}")

    def _init_display(self):
        """Initialize the display with proper error handling"""
        try:
            # Initialize pygame
            pygame.init()
            
            # Check for display
            if 'DISPLAY' not in os.environ:
                self.logger.info("No display detected, using dummy driver")
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                self.use_fallback = True
            
            # Try to set up the display
            if not self.use_fallback:
                try:
                    self.screen = pygame.display.set_mode((self.width, self.height))
                    pygame.display.set_caption("EVE2 Display")
                    self.logger.info(f"Display initialized: {self.width}x{self.height}")
                except pygame.error as e:
                    self.logger.error(f"Error setting display mode: {e}")
                    self.use_fallback = True
            
            # Use fallback surface if needed
            if self.use_fallback:
                self.screen = pygame.Surface((self.width, self.height))
                self.logger.info("Using fallback surface for headless mode")
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
        except Exception as e:
            self.logger.error(f"Error initializing display: {e}")
            # Ensure we at least have a surface
            self.use_fallback = True
            self.screen = pygame.Surface((self.width, self.height))

    def _generate_default_emotion(self):
        """Generate a default emotion face"""
        try:
            # Create a blank image
            surface = pygame.Surface((self.width, self.height))
            surface.fill(self.background_color)
            
            # Draw eyes
            eye_size = min(self.width, self.height) // 10
            center_y = self.height // 2
            left_x = self.width // 3
            right_x = (self.width * 2) // 3
            
            # Draw left eye
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size
            )
            
            # Draw right eye
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (right_x, center_y), 
                eye_size
            )
            
            # Draw mouth
            mouth_y = center_y + self.height // 6
            pygame.draw.line(
                surface,
                self.eye_color,
                (left_x, mouth_y),
                (right_x, mouth_y),
                3
            )
            
            # Add debug text
            font = pygame.font.SysFont(None, 36)
            text = font.render("DEFAULT EMOTION", True, (255, 255, 255))
            surface.blit(text, (self.width//2 - text.get_width()//2, 30))
            
            return surface
            
        except Exception as e:
            self.logger.error(f"Error generating default emotion: {e}")
            # Create a very simple fallback
            fallback = pygame.Surface((self.width, self.height))
            fallback.fill((255, 0, 0))  # Red background to show there's an issue
            return fallback

    def _load_assets(self):
        """Load emotion assets or generate defaults"""
        self.logger.info("Loading display assets")
        
        # List of basic emotions to ensure we have
        basic_emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'confused']
        assets_path = os.path.join(os.path.dirname(__file__), 'assets', 'emotions')
        
        try:
            # Create assets directory if it doesn't exist
            os.makedirs(assets_path, exist_ok=True)
            
            # Try to load each emotion, generate if missing
            for emotion in basic_emotions:
                image_path = os.path.join(assets_path, f"{emotion}.png")
                
                if os.path.exists(image_path):
                    try:
                        self.emotion_images[emotion] = pygame.image.load(image_path)
                        self.logger.info(f"Loaded emotion image: {emotion}")
                    except pygame.error:
                        self.logger.warning(f"Failed to load image for {emotion}, generating default")
                        self.emotion_images[emotion] = self._generate_emotion(emotion)
                else:
                    self.logger.warning(f"No image found for {emotion}, generating default")
                    self.emotion_images[emotion] = self._generate_emotion(emotion)
                    
                    # Try to save the generated emotion
                    try:
                        pygame.image.save(self.emotion_images[emotion], image_path)
                        self.logger.info(f"Saved generated {emotion} image to {image_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not save generated emotion: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error loading assets: {e}")
            # Ensure we at least have a neutral emotion
            self.emotion_images['neutral'] = self._generate_default_emotion()

    def _generate_emotion(self, emotion):
        """Generate a specific emotion face"""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(self.background_color)
        
        # Base parameters
        eye_size = min(self.width, self.height) // 10
        center_y = self.height // 2
        left_x = self.width // 3
        right_x = (self.width * 2) // 3
        
        # Customize based on emotion
        if emotion == 'happy':
            # Happy eyes (slightly closed)
            pygame.draw.ellipse(
                surface, 
                self.eye_color, 
                (left_x - eye_size, center_y - eye_size//2, eye_size*2, eye_size),
                0
            )
            pygame.draw.ellipse(
                surface, 
                self.eye_color, 
                (right_x - eye_size, center_y - eye_size//2, eye_size*2, eye_size),
                0
            )
            
            # Smile
            pygame.draw.arc(
                surface,
                self.eye_color,
                (left_x, center_y, right_x - left_x, eye_size*2),
                0, 3.14,
                3
            )
            
        elif emotion == 'sad':
            # Sad eyes (droopy)
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size
            )
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (right_x, center_y), 
                eye_size
            )
            
            # Frown
            pygame.draw.arc(
                surface,
                self.eye_color,
                (left_x, center_y + eye_size, right_x - left_x, eye_size*2),
                3.14, 2*3.14,
                3
            )
            
        elif emotion == 'angry':
            # Angry eyes (narrowed)
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size
            )
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (right_x, center_y), 
                eye_size
            )
            
            # Angry eyebrows
            pygame.draw.line(
                surface,
                self.eye_color,
                (left_x - eye_size, center_y - eye_size),
                (left_x + eye_size//2, center_y - eye_size*1.5),
                3
            )
            pygame.draw.line(
                surface,
                self.eye_color,
                (right_x + eye_size, center_y - eye_size),
                (right_x - eye_size//2, center_y - eye_size*1.5),
                3
            )
            
            # Angry mouth
            pygame.draw.line(
                surface,
                self.eye_color,
                (left_x + eye_size//2, center_y + eye_size*1.5),
                (right_x - eye_size//2, center_y + eye_size*1.5),
                3
            )
            
        elif emotion == 'surprised':
            # Surprised eyes (wide)
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size * 1.5
            )
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (right_x, center_y), 
                eye_size * 1.5
            )
            
            # Surprised mouth (O shape)
            pygame.draw.circle(
                surface,
                self.eye_color,
                (self.width//2, center_y + eye_size*2),
                eye_size,
                3
            )
            
        elif emotion == 'confused':
            # Confused eyes (one squinting)
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size
            )
            pygame.draw.ellipse(
                surface, 
                self.eye_color, 
                (right_x - eye_size, center_y - eye_size//2, eye_size*2, eye_size),
                0
            )
            
            # Confused eyebrow
            pygame.draw.line(
                surface,
                self.eye_color,
                (right_x - eye_size, center_y - eye_size),
                (right_x + eye_size, center_y - eye_size*1.5),
                3
            )
            
            # Confused mouth (squiggly)
            points = [
                (left_x, center_y + eye_size*1.5),
                (left_x + (right_x - left_x)//3, center_y + eye_size),
                (right_x - (right_x - left_x)//3, center_y + eye_size*2),
                (right_x, center_y + eye_size*1.5)
            ]
            pygame.draw.lines(
                surface,
                self.eye_color,
                False,
                points,
                3
            )
            
        else:  # neutral or default
            # Neutral eyes
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (left_x, center_y), 
                eye_size
            )
            pygame.draw.circle(
                surface, 
                self.eye_color, 
                (right_x, center_y), 
                eye_size
            )
            
            # Neutral mouth
            pygame.draw.line(
                surface,
                self.eye_color,
                (left_x, center_y + eye_size*1.5),
                (right_x, center_y + eye_size*1.5),
                3
            )
        
        # Add label showing which emotion this is
        font = pygame.font.SysFont(None, 36)
        text = font.render(emotion.upper(), True, (255, 255, 255))
        surface.blit(text, (self.width//2 - text.get_width()//2, 30))
            
        return surface

    def _render_current_emotion(self):
        """Render the current emotion to the screen"""
        try:
            # Clear the screen
            self.screen.fill(self.background_color)
            
            # Get the current emotion image
            emotion = self.current_emotion
            if emotion not in self.emotion_images:
                self.logger.warning(f"Unknown emotion: {emotion}, using neutral")
                emotion = 'neutral'
                if 'neutral' not in self.emotion_images:
                    self.emotion_images['neutral'] = self._generate_default_emotion()
            
            # Draw the emotion
            image = self.emotion_images[emotion]
            if image.get_width() == self.width and image.get_height() == self.height:
                # Full-screen image
                self.screen.blit(image, (0, 0))
            else:
                # Centered image
                x = (self.width - image.get_width()) // 2
                y = (self.height - image.get_height()) // 2
                self.screen.blit(image, (x, y))
                
            # Add status text at bottom
            try:
                font = pygame.font.SysFont(None, 24)
                status = f"EVE2 Display - Current Emotion: {emotion}"
                text = font.render(status, True, (200, 200, 200))
                self.screen.blit(text, (10, self.height - 30))
            except:
                pass  # Ignore text rendering errors
                
        except Exception as e:
            self.logger.error(f"Error rendering emotion: {e}")
            # Try to show an error message
            try:
                self.screen.fill((64, 0, 0))  # Dark red background
                font = pygame.font.SysFont(None, 36)
                text = font.render("ERROR RENDERING EMOTION", True, (255, 255, 255))
                self.screen.blit(text, (self.width//2 - text.get_width()//2, self.height//2))
            except:
                pass  # Last resort, just ignore if even this fails

    def _render_loop(self):
        """Main rendering loop"""
        self.logger.info(f"Starting render loop in {'headless' if self.use_fallback else 'display'} mode")
        
        while self.running:
            try:
                # Handle pygame events
                if not self.use_fallback:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                
                # Render the current emotion
                self._render_current_emotion()
                
                # Update the display
                if not self.use_fallback:
                    try:
                        pygame.display.flip()
                    except pygame.error as e:
                        self.logger.error(f"Error updating display: {e}")
                        self.use_fallback = True
                
                # Maintain frame rate
                self.clock.tick(self.fps)
                
            except Exception as e:
                self.logger.error(f"Error in render loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop

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
            except:
                pass
        
        # Clean up pygame
        try:
            pygame.quit()
        except:
            pass
            
        self.logger.info("Display controller stopped")

    def set_emotion(self, emotion):
        """Set the current emotion to display"""
        if emotion not in self.emotion_images:
            self.logger.warning(f"Unknown emotion: {emotion}, using neutral")
            emotion = 'neutral'
            # Generate neutral if we don't have it
            if emotion not in self.emotion_images:
                self.emotion_images[emotion] = self._generate_default_emotion()
                
        self.current_emotion = emotion
        
        # Force immediate render if we're not running the loop
        if not self.running and not self.use_fallback:
            self._render_current_emotion()
            try:
                pygame.display.flip()
            except:
                pass
                
        return True

    def is_active(self):
        """Check if display controller is active"""
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