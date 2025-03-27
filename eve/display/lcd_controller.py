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
        
        # Force software rendering
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        os.environ['SDL_RENDERER_DRIVER'] = 'software'
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
        self.use_fallback = False
        self.emotion_images = {}
        
        # Initialize pygame in software mode
        try:
            pygame.init()
            pygame.display.init()
            
            # Try to create the display with software rendering
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.SWSURFACE | pygame.HWSURFACE
            )
            self.clock = pygame.time.Clock()
            
            # Load emotions
            self._load_assets()
            
            self.logger.info("Display controller initialized in software mode")
            self.is_blinking = False
            self.blink_duration = 0.15  # seconds for each blink phase
        except Exception as e:
            self.logger.error(f"Error initializing display: {e}")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback mode without display"""
        try:
            self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self._load_assets()
            self.logger.info("Display controller initialized in fallback mode")
        except Exception as e:
            self.logger.error(f"Error initializing fallback mode: {e}")

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
        """Load emotion assets"""
        try:
            # Load standard emotions
            emotions_path = os.path.join('assets', 'emotions')
            for emotion in ['neutral', 'happy', 'sad', 'angry', 'surprised']:
                image_path = os.path.join(emotions_path, f'{emotion}.png')
                if os.path.exists(image_path):
                    self.emotion_images[emotion] = pygame.image.load(image_path)
                else:
                    self.emotion_images[emotion] = self._generate_default_emotion()

            # Create blink emotion (eyes closed)
            blink_surface = pygame.Surface((self.width, self.height))
            blink_surface.fill(self.background_color)
            # Draw closed eyes (simple lines)
            eye_width = 100
            eye_height = 10
            left_eye_pos = ((self.width // 2) - eye_width - 50, self.height // 2)
            right_eye_pos = ((self.width // 2) + 50, self.height // 2)
            
            for pos in [left_eye_pos, right_eye_pos]:
                pygame.draw.line(blink_surface, self.eye_color,
                               (pos[0], pos[1]),
                               (pos[0] + eye_width, pos[1]),
                               eye_height)
            
            self.emotion_images['blink'] = blink_surface
            
            self.logger.info("Loaded all emotion assets")
        except Exception as e:
            self.logger.error(f"Error loading assets: {e}")
            # Create default emotion as fallback
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
        """Render the current emotion"""
        try:
            if not pygame.display.get_init():
                return

            # Clear screen
            self.screen.fill(self.background_color)
            
            # If blinking, render closed eyes, otherwise render current emotion
            emotion_surface = self.emotion_images.get(
                'blink' if self.is_blinking else self.current_emotion,
                self.emotion_images.get('neutral')
            )
            
            if emotion_surface:
                # Center the emotion on screen
                pos_x = (self.width - emotion_surface.get_width()) // 2
                pos_y = (self.height - emotion_surface.get_height()) // 2
                self.screen.blit(emotion_surface, (pos_x, pos_y))
            
            pygame.display.flip()
            
        except Exception as e:
            self.logger.error(f"Error rendering emotion: {e}")

    def _render_loop(self):
        """Main rendering loop"""
        while self.running:
            try:
                if not pygame.display.get_init():
                    self.logger.error("Display surface not initialized")
                    break
                
                self._render_current_emotion()
                self.clock.tick(self.fps)
                
            except pygame.error as e:
                self.logger.error(f"Pygame error: {e}")
                break
            except Exception as e:
                self.logger.error(f"Error rendering emotion: {e}")
                time.sleep(0.1)  # Prevent tight error loop
        
        self.logger.info("Render loop ended")

    def start(self):
        """Start the display controller in headless mode"""
        if not self.running:
            self.running = True
            self.render_thread = threading.Thread(target=self._render_loop)
            self.render_thread.daemon = True
            self.render_thread.start()
            self.logger.info("Display controller started in headless mode")

    def stop(self):
        """Stop the display controller gracefully"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop render thread
        if self.render_thread:
            try:
                self.render_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error stopping render thread: {e}")
        
        # Clean up pygame
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception as e:
            self.logger.error(f"Error during pygame cleanup: {e}")
        
        self.logger.info("Display controller stopped")

    def set_emotion(self, emotion):
        """Set the emotion with fallback to neutral if not found"""
        if emotion not in self.emotion_images:
            self.logger.warning(f"Unknown emotion: {emotion}, falling back to neutral")
            emotion = 'neutral'
            if emotion not in self.emotion_images:
                self.emotion_images[emotion] = self._generate_default_emotion()
        
        # Update current emotion
        previous = self.current_emotion
        self.current_emotion = emotion
        
        if previous != emotion:
            self.logger.info(f"Emotion changed: {previous} → {emotion}")
        
        return True

    def is_active(self):
        """Check if display controller is running"""
        return self.running and self.render_thread and self.render_thread.is_alive()

    def get_status(self):
        """Get display controller status"""
        return {
            'running': self.is_active(),
            'fallback_mode': self.use_fallback,
            'current_emotion': self.current_emotion,
            'emotions_available': list(self.emotion_images.keys())
        }

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

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            # Store current emotion
            current = self.current_emotion
            
            # Close eyes
            self.set_emotion('blink')
            time.sleep(self.blink_duration)
            
            # Open eyes
            self.set_emotion(current)
            time.sleep(self.blink_duration)
            
            self.is_blinking = False
            self.logger.debug("Completed blink animation")
        except Exception as e:
            self.logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False 