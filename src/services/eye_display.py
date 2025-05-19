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
        self.screen: Optional[pygame.Surface] = None # Type hint for clarity
        self.running = False
        self.current_emotion = Emotion.NEUTRAL
        self.eye_sprites: Dict[Emotion, pygame.Surface] = {}
        self.clock = pygame.time.Clock() # Keep clock for InteractionManager to use
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def start(self) -> bool:
        """Start the eye display. Initializes Pygame and the screen."""
        try:
            pygame.init() # Initialize all Pygame modules
            width = self.config.get('display', {}).get('width', 800)
            height = self.config.get('display', {}).get('height', 480)
            
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("EVE2 Eyes")
            
            self._load_eye_sprites()
            
            self.running = True
            logger.info("Eye display started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting eye display: {e}")
            return False
            
    def _load_eye_sprites(self) -> None:
        """Load eye sprites for each emotion."""
        try:
            # Use src/assets/eyes directory
            assets_dir = Path("src/assets/eyes")
            for emotion in Emotion:
                sprite_path = assets_dir / f"{emotion.value}.png"
                if sprite_path.exists():
                    try:
                        self.eye_sprites[emotion] = pygame.image.load(str(sprite_path))
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
        """Update the display by drawing the current emotion. Does not flip the display."""
        if not self.running or self.screen is None:
            return

        try:
            self.screen.fill((0, 0, 0)) # Clear screen
            
            if self.current_emotion in self.eye_sprites:
                sprite = self.eye_sprites[self.current_emotion]
                x = (self.screen.get_width() - sprite.get_width()) // 2
                y = (self.screen.get_height() - sprite.get_height()) // 2
                self.screen.blit(sprite, (x, y))
                
        except Exception as e:
            logger.error(f"Error updating eye display visuals: {e}")
            
    def stop(self) -> None:
        """Stop the eye display."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        self.running = False
        logger.info("Eye display stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 