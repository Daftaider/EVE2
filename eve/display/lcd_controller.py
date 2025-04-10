"""
Simple LCD controller for EVE2.
"""
import logging
import pygame
import numpy as np
from typing import Optional, Dict, List, Tuple
import threading
import time
from pathlib import Path
from enum import Enum
import cv2

logger = logging.getLogger(__name__)

class Emotion(Enum):
    """Basic emotions for EVE2."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    TALKING = "talking"

class LCDController:
    """Simple LCD controller for displaying video feed and emotions."""
    
    def __init__(self, width: int = 800, height: int = 480):
        """Initialize the LCD controller."""
        self.width = width
        self.height = height
        self.screen = None
        self.running = False
        self.current_emotion = Emotion.NEUTRAL
        self.emotion_images = {}
        self.font = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("EVE2")
        self.font = pygame.font.Font(None, 36)
        
        # Load emotion images
        self._load_emotion_images()
        
    def _load_emotion_images(self) -> None:
        """Load emotion images from assets directory."""
        assets_dir = Path(__file__).parent.parent.parent / "assets" / "emotions"
        if not assets_dir.exists():
            logger.warning(f"Emotions directory not found: {assets_dir}")
            return
            
        for emotion in Emotion:
            image_path = assets_dir / f"{emotion.value}.png"
            if image_path.exists():
                try:
                    image = pygame.image.load(str(image_path))
                    image = pygame.transform.scale(image, (self.width, self.height))
                    self.emotion_images[emotion] = image
                except Exception as e:
                    logger.error(f"Failed to load emotion image {image_path}: {e}")
                    
    def start(self) -> None:
        """Start the LCD controller."""
        self.running = True
        
    def stop(self) -> None:
        """Stop the LCD controller."""
        self.running = False
        pygame.quit()
        
    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        self.current_emotion = emotion
        
    def update_frame(self, frame: np.ndarray, detections: Optional[List[Dict]] = None) -> None:
        """Update the video frame and detections."""
        if frame is None:
            return
            
        # Convert frame to pygame surface
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        
        with self.frame_lock:
            self.latest_frame = pygame.surfarray.make_surface(frame)
            self.latest_detections = detections if detections else []
            
    def _draw_detections(self) -> None:
        """Draw object detection boxes and labels."""
        if not self.latest_detections:
            return
            
        for det in self.latest_detections:
            # Get detection info
            box = det.get('box', None)
            if not box:
                continue
                
            x1, y1, x2, y2 = box
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            
            # Draw box
            pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            text_surface = self.font.render(text, True, (0, 255, 0))
            self.screen.blit(text_surface, (x1, max(0, y1 - 20)))
            
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
    def _draw_fps(self) -> None:
        """Draw FPS counter."""
        fps_text = f"FPS: {self.fps:.1f}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (10, 10))
        
    def update(self) -> None:
        """Update the display."""
        if not self.running:
            return
            
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw video frame if available
        with self.frame_lock:
            if self.latest_frame is not None:
                self.screen.blit(self.latest_frame, (0, 0))
                self._draw_detections()
            else:
                # Draw emotion
                emotion_image = self.emotion_images.get(self.current_emotion)
                if emotion_image:
                    self.screen.blit(emotion_image, (0, 0))
                    
        # Draw FPS
        self._update_fps()
        self._draw_fps()
        
        # Update display
        pygame.display.flip()
        
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 