"""
LCD controller module for handling display output.
"""
import logging
from enum import Enum
from typing import List, Optional
import pygame
import numpy as np
from vision.object_detector import Detection

logger = logging.getLogger(__name__)

class Emotion(Enum):
    """Enumeration of possible emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"

class LCDController:
    """LCD controller class for handling display output."""
    
    def __init__(self, width: int = 800, height: int = 480):
        """Initialize LCD controller with specified dimensions."""
        self.width = width
        self.height = height
        self.screen = None
        self.running = False
        self.current_emotion = Emotion.NEUTRAL
        self.current_frame = None
        self.current_detections: List[Detection] = []
        
    def start(self) -> bool:
        """Start the display."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("EVE2 Display")
            self.running = True
            logger.info("Display started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting display: {e}")
            return False
            
    def stop(self) -> None:
        """Stop the display."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        self.running = False
        logger.info("Display stopped")
        
    def update_frame(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Update the current frame and detections."""
        self.current_frame = frame
        self.current_detections = detections
        
    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        self.current_emotion = emotion
        
    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def update(self) -> None:
        """Update the display."""
        if not self.running or self.screen is None:
            return
            
        try:
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Draw current frame if available
            if self.current_frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                # Convert to pygame surface
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                # Draw frame
                self.screen.blit(frame_surface, (0, 0))
                
            # Draw emotion text
            font = pygame.font.Font(None, 36)
            text = font.render(f"Emotion: {self.current_emotion.value}", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            
            # Update display
            pygame.display.flip()
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 