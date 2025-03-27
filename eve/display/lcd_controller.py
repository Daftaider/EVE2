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
        self.use_fallback = False
        self.emotion_images = {}
        self.screen = None
        self.clock = None
        self.gl_error_count = 0
        
        # Initial setup
        self._init_display()
        self._load_assets()
        
        # Log the mode we're in
        self.logger.info(f"Display controller initialized in {'fallback' if self.use_fallback else 'standard'} mode")

    def _init_display(self):
        """Try various display modes until one works"""
        # Try methods in order of preference
        methods = [
            self._try_standard_mode,
            self._try_software_mode,
            self._try_dummy_mode
        ]
        
        for method in methods:
            if method():
                return
                
        # If all methods fail, use minimal fallback
        self.logger.error("All display initialization methods failed")
        self.use_fallback = True
        self.screen = pygame.Surface((self.width, self.height))

    def _try_standard_mode(self):
        """Try normal display mode"""
        try:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("EVE2 Display")
            self.clock = pygame.time.Clock()
            
            # Test if it works
            self.screen.fill(self.background_color)
            pygame.display.flip()
            
            self.logger.info("Standard display mode initialized successfully")
            return True
        except Exception as e:
            self.logger.debug(f"Standard mode failed: {e}")
            return False

    def _try_software_mode(self):
        """Try software rendering mode"""
        try:
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            os.environ['SDL_RENDERER_DRIVER'] = 'software'
            
            pygame.display.quit()
            pygame.display.init()
            
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.SWSURFACE
            )
            self.clock = pygame.time.Clock()
            
            # Test if it works
            self.screen.fill(self.background_color)
            pygame.display.flip()
            
            self.logger.info("Software rendering mode initialized successfully")
            return True
        except Exception as e:
            self.logger.debug(f"Software mode failed: {e}")
            return False

    def _try_dummy_mode(self):
        """Use dummy driver for headless operation"""
        try:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
            pygame.display.quit()
            pygame.display.init()
            
            self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.use_fallback = True
            
            self.logger.info("Dummy display mode initialized successfully")
            return True
        except Exception as e:
            self.logger.debug(f"Dummy mode failed: {e}")
            return False

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
        """Main rendering loop with minimal logging in fallback mode"""
        self.logger.info(f"Render loop started in {'fallback' if self.use_fallback else 'display'} mode")
        
        last_log_time = 0
        fallback_log_interval = 60.0  # Log fallback status once per minute
        
        while self.running:
            try:
                # Handle events if not in fallback mode
                if not self.use_fallback:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                
                # Always render current emotion to keep state consistent
                self._render_current_emotion()
                
                # Update display if not in fallback mode
                if not self.use_fallback:
                    try:
                        pygame.display.flip()
                        self.gl_error_count = 0  # Reset on success
                    except pygame.error as e:
                        self.gl_error_count += 1
                        
                        # Log first error and every tenth error
                        if self.gl_error_count == 1 or self.gl_error_count % 10 == 0:
                            self.logger.error(f"Display update error ({self.gl_error_count}): {e}")
                        
                        # Switch to fallback after repeated errors
                        if self.gl_error_count >= 3:
                            self.logger.warning("Too many display errors, switching to fallback mode")
                            self._switch_to_fallback()
                else:
                    # Periodically log fallback status
                    current_time = time.time()
                    if current_time - last_log_time > fallback_log_interval:
                        self.logger.info(f"Display running in fallback mode - current emotion: {self.current_emotion}")
                        last_log_time = current_time
                
                # Control frame rate
                self.clock.tick(self.fps)
                
            except Exception as e:
                self.logger.error(f"Error in render loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop

    def _switch_to_fallback(self):
        """Switch to fallback mode during runtime"""
        try:
            self.use_fallback = True
            
            # Try to reinitialize with dummy driver
            try:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.display.quit()
                pygame.display.init()
                self.screen = pygame.Surface((self.width, self.height))
            except:
                # If that fails, just use a Surface
                self.screen = pygame.Surface((self.width, self.height))
                
            self.logger.info("Switched to fallback display mode")
        except Exception as e:
            self.logger.error(f"Error switching to fallback mode: {e}")

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
        try:
            pygame.quit()
        except:
            pass
        self.logger.info("Display controller stopped")

    def set_emotion(self, emotion):
        """Set the emotion with fallback to neutral if not found"""
        if emotion not in self.emotion_images:
            self.logger.warning(f"Unknown emotion: {emotion}, falling back to neutral")
            emotion = 'neutral'
            if emotion not in self.emotion_images:
                self.emotion_images[emotion] = self._generate_default_emotion()
        
        # Always update current emotion even in fallback mode
        previous = self.current_emotion
        self.current_emotion = emotion
        
        if not self.use_fallback and previous != emotion:
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