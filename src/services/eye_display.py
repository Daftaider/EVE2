"""
Eye display service for rendering EVE-inspired eyes.
"""
import logging
import pygame
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)

class Emotion(Enum):
    """Enumeration of possible emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"

class EyeDisplay:
    """Eye display service for rendering EVE-inspired eyes."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize eye display service."""
        self.config = self._load_config(config_path)
        self.screen: Optional[pygame.Surface] = None # Screen will be set by InteractionManager
        self.running = False
        self.current_emotion = Emotion.NEUTRAL
        self.eye_sprites: Dict[Emotion, pygame.Surface] = {}
        self.clock = pygame.time.Clock() # Clock can still be owned here, ticked by InteractionManager
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def prepare_resources(self) -> bool:
        """Load eye sprites and prepare for rendering."""
        try:
            # Pygame.init() and screen creation are handled by InteractionManager
            self._load_eye_sprites()
            self.running = True # Service is ready to have its update() called
            logger.info("Eye display resources prepared successfully")
            return True
        except Exception as e:
            logger.error(f"Error preparing eye display resources: {e}")
            return False
            
    def _load_eye_sprites(self) -> None:
        """Load eye sprites for each emotion."""
        try:
            assets_dir = Path("src/assets/eyes")
            for emotion in Emotion:
                sprite_path = assets_dir / f"{emotion.value}.png"
                if sprite_path.exists():
                    try:
                        # Ensure Pygame is initialized before loading images
                        if pygame.get_init():
                            self.eye_sprites[emotion] = pygame.image.load(str(sprite_path))
                        else:
                            logger.error("Pygame not initialized when trying to load eye sprites.")
                            # Optionally, raise an error or handle this state
                            return # Can't load sprites if Pygame isn't up
                    except Exception as e:
                        logger.error(f"Failed to load sprite {sprite_path}: {e}")
                else:
                    logger.warning(f"Missing sprite for emotion: {emotion.value}")
        except Exception as e:
            logger.error(f"Error loading eye sprites: {e}")
            
    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        self.current_emotion = emotion
        logger.info(f"Emotion set to: {emotion.value}")
        
    def update(self) -> None:
        """Update the display by drawing the current emotion. Called by InteractionManager."""
        if not self.running or self.screen is None:
            # If screen is None here, it means InteractionManager hasn't set it yet or there was an issue.
            if self.running and self.screen is None:
                 logger.warning("EyeDisplay.update() called but screen is not set.")
            return
            
        try:
            self.screen.fill((0, 0, 0)) 
            
            if self.current_emotion in self.eye_sprites:
                sprite = self.eye_sprites[self.current_emotion]
                x = (self.screen.get_width() - sprite.get_width()) // 2
                y = (self.screen.get_height() - sprite.get_height()) // 2
                self.screen.blit(sprite, (x, y))
            # No pygame.display.flip() or clock.tick() - handled by InteractionManager
        except Exception as e:
            logger.error(f"Error updating eye display visuals: {e}")
            
    def stop(self) -> None:
        """Stop the eye display service."""
        # pygame.quit() is handled by InteractionManager
        self.running = False
        # self.screen = None # InteractionManager manages the screen lifecycle
        logger.info("Eye display service stopped")
        
    # __enter__ and __exit__ might need adjustment if start() is no longer the main init point
    # For now, let's assume InteractionManager calls prepare_resources()
    # If these are still used directly, they should call prepare_resources()
    def __enter__(self):
        self.prepare_resources()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 