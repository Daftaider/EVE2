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
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (800, 480),
        fullscreen: bool = False,
        fps: int = 30,
        default_emotion: str = "neutral",
        background_color: Tuple[int, int, int] = (0, 0, 0),
        eye_color: Tuple[int, int, int] = (0, 191, 255),
        transition_time_ms: int = 500
    ) -> None:
        """
        Initialize the LCD controller.
        
        Args:
            resolution: Display resolution as (width, height) (default: (800, 480))
            fullscreen: Whether to start in fullscreen mode (default: False)
            fps: Target frames per second (default: 30)
            default_emotion: Default emotion to display (default: "neutral")
            background_color: Background color in RGB (default: (0, 0, 0))
            eye_color: Eye color in RGB (default: (0, 191, 255))
            transition_time_ms: Transition time between emotions in milliseconds (default: 500)
        """
        self.resolution = resolution
        self.fullscreen = fullscreen
        self.fps = fps
        self.default_emotion = default_emotion
        self.background_color = background_color
        self.eye_color = eye_color
        self.transition_time_ms = transition_time_ms
        
        # State variables
        self.running = False
        self.current_emotion = default_emotion
        self.target_emotion = default_emotion
        self.transition_start_time = 0
        self.transition_progress = 1.0  # 1.0 means transition is complete
        
        # Pygame objects
        self.screen = None
        self.clock = None
        
        # Assets
        self.emotion_surfaces: Dict[str, pygame.Surface] = {}
        
        # Initialize pygame and assets
        self._init_pygame()
        self._load_assets()
        
    def _init_pygame(self) -> None:
        """Initialize pygame and the display."""
        logger.info("Initializing pygame")
        try:
            pygame.init()
            pygame.display.set_caption("EVE2 Display")
            
            # Set up the display
            if self.fullscreen:
                self.screen = pygame.display.set_mode(
                    self.resolution,
                    pygame.FULLSCREEN
                )
            else:
                self.screen = pygame.display.set_mode(
                    self.resolution
                )
            
            # Create a clock for FPS control
            self.clock = pygame.time.Clock()
            
            logger.info("Pygame initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            raise
    
    def _load_assets(self) -> None:
        """Load emotion assets or generate default ones."""
        logger.info("Loading display assets")
        
        # Check if assets directory exists
        assets_dir = config.ASSETS_DIR
        if os.path.exists(assets_dir):
            # Load emotion images from files
            self._load_emotion_images(assets_dir)
        else:
            # Generate default emotion graphics
            self._generate_default_emotions()
        
        logger.info(f"Loaded {len(self.emotion_surfaces)} emotion assets")
    
    def _load_emotion_images(self, assets_dir: Path) -> None:
        """
        Load emotion images from the assets directory.
        
        Args:
            assets_dir: Directory containing emotion image files
        """
        try:
            # Check for image files for each emotion
            for emotion in config.display.EMOTIONS:
                # Look for common image formats
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    image_path = os.path.join(assets_dir, "emotions", f"{emotion}{ext}")
                    if os.path.exists(image_path):
                        # Load the image
                        image = pygame.image.load(str(image_path))
                        
                        # Resize to fit the screen if needed
                        if image.get_width() > self.resolution[0] or image.get_height() > self.resolution[1]:
                            image = pygame.transform.scale(
                                image,
                                self.resolution
                            )
                        
                        # Store the surface
                        self.emotion_surfaces[emotion] = image
                        break
                
                # If no image was found for this emotion, generate a default one
                if emotion not in self.emotion_surfaces:
                    logger.warning(f"No image found for emotion: {emotion}, generating default")
                    self._generate_emotion_surface(emotion)
        
        except Exception as e:
            logger.error(f"Error loading emotion images: {e}")
            # Fall back to generated emotions
            self._generate_default_emotions()
    
    def _generate_default_emotions(self) -> None:
        """Generate default emotion graphics for all configured emotions."""
        for emotion in config.display.EMOTIONS:
            self._generate_emotion_surface(emotion)
    
    def _generate_emotion_surface(self, emotion: str) -> None:
        """
        Generate a default surface for the specified emotion.
        
        Args:
            emotion: The emotion to generate a surface for
        """
        # Create a new surface
        surface = pygame.Surface(self.resolution)
        surface.fill(self.background_color)
        
        # Get the center of the screen
        center_x = self.resolution[0] // 2
        center_y = self.resolution[1] // 2
        
        # Draw different eye shapes based on the emotion
        if emotion == "neutral":
            # Neutral: Simple circular eyes
            self._draw_circular_eyes(surface, center_x, center_y, 80)
            
        elif emotion == "happy":
            # Happy: Curved eyes (upward curve)
            self._draw_curved_eyes(surface, center_x, center_y, 80, -0.5)
            
        elif emotion == "sad":
            # Sad: Curved eyes (downward curve)
            self._draw_curved_eyes(surface, center_x, center_y, 80, 0.5)
            
        elif emotion == "angry":
            # Angry: Angled eyes
            self._draw_angled_eyes(surface, center_x, center_y, 80, -0.3)
            
        elif emotion == "surprised":
            # Surprised: Large circular eyes
            self._draw_circular_eyes(surface, center_x, center_y, 100)
            
        elif emotion == "fearful":
            # Fearful: Smaller, wider eyes
            self._draw_elliptical_eyes(surface, center_x, center_y, 60, 40)
            
        elif emotion == "disgusted":
            # Disgusted: Squinted eyes
            self._draw_elliptical_eyes(surface, center_x, center_y, 80, 20)
            
        else:
            # Default: Simple circular eyes
            self._draw_circular_eyes(surface, center_x, center_y, 80)
        
        # Store the generated surface
        self.emotion_surfaces[emotion] = surface
    
    def _draw_circular_eyes(self, surface: pygame.Surface, center_x: int, center_y: int, radius: int) -> None:
        """
        Draw circular eyes on the surface.
        
        Args:
            surface: The surface to draw on
            center_x: X-coordinate of the center of the screen
            center_y: Y-coordinate of the center of the screen
            radius: Radius of the eyes
        """
        # Distance between eyes
        eye_distance = radius * 3
        
        # Draw left eye
        pygame.draw.circle(
            surface,
            self.eye_color,
            (center_x - eye_distance // 2, center_y),
            radius
        )
        
        # Draw right eye
        pygame.draw.circle(
            surface,
            self.eye_color,
            (center_x + eye_distance // 2, center_y),
            radius
        )
    
    def _draw_elliptical_eyes(self, surface: pygame.Surface, center_x: int, center_y: int, 
                            width: int, height: int) -> None:
        """
        Draw elliptical eyes on the surface.
        
        Args:
            surface: The surface to draw on
            center_x: X-coordinate of the center of the screen
            center_y: Y-coordinate of the center of the screen
            width: Width of the ellipse
            height: Height of the ellipse
        """
        # Distance between eyes
        eye_distance = width * 3
        
        # Draw left eye
        pygame.draw.ellipse(
            surface,
            self.eye_color,
            (center_x - eye_distance // 2 - width // 2, center_y - height // 2, width, height)
        )
        
        # Draw right eye
        pygame.draw.ellipse(
            surface,
            self.eye_color,
            (center_x + eye_distance // 2 - width // 2, center_y - height // 2, width, height)
        )
    
    def _draw_curved_eyes(self, surface: pygame.Surface, center_x: int, center_y: int, 
                         width: int, curve_factor: float) -> None:
        """
        Draw curved eyes on the surface.
        
        Args:
            surface: The surface to draw on
            center_x: X-coordinate of the center of the screen
            center_y: Y-coordinate of the center of the screen
            width: Width of the eye
            curve_factor: Factor controlling the curvature (-1.0 to 1.0)
        """
        # Distance between eyes
        eye_distance = width * 3
        
        # Height of the eye
        height = width // 2
        
        # Calculate curve points for left eye
        left_eye_points = []
        for x in range(width + 1):
            # Calculate relative x position (-1 to 1)
            rel_x = (x / width) * 2 - 1
            # Calculate y position based on curve
            y = curve_factor * (rel_x ** 2 - 1) * height
            # Add point to list
            left_eye_points.append(
                (center_x - eye_distance // 2 - width // 2 + x, center_y + y)
            )
        
        # Calculate curve points for right eye
        right_eye_points = []
        for x in range(width + 1):
            # Calculate relative x position (-1 to 1)
            rel_x = (x / width) * 2 - 1
            # Calculate y position based on curve
            y = curve_factor * (rel_x ** 2 - 1) * height
            # Add point to list
            right_eye_points.append(
                (center_x + eye_distance // 2 - width // 2 + x, center_y + y)
            )
        
        # Draw left eye
        if len(left_eye_points) >= 2:
            pygame.draw.lines(surface, self.eye_color, False, left_eye_points, width=height // 2)
        
        # Draw right eye
        if len(right_eye_points) >= 2:
            pygame.draw.lines(surface, self.eye_color, False, right_eye_points, width=height // 2)
    
    def _draw_angled_eyes(self, surface: pygame.Surface, center_x: int, center_y: int, 
                         width: int, angle_factor: float) -> None:
        """
        Draw angled eyes on the surface.
        
        Args:
            surface: The surface to draw on
            center_x: X-coordinate of the center of the screen
            center_y: Y-coordinate of the center of the screen
            width: Width of the eye
            angle_factor: Factor controlling the angle (-1.0 to 1.0)
        """
        # Distance between eyes
        eye_distance = width * 3
        
        # Height of the eye
        height = width // 3
        
        # Calculate angle offset
        angle_offset = int(angle_factor * height)
        
        # Draw left eye
        pygame.draw.polygon(
            surface,
            self.eye_color,
            [
                (center_x - eye_distance // 2 - width // 2, center_y),
                (center_x - eye_distance // 2 + width // 2, center_y + angle_offset),
                (center_x - eye_distance // 2 + width // 2, center_y + angle_offset + height),
                (center_x - eye_distance // 2 - width // 2, center_y + height)
            ]
        )
        
        # Draw right eye
        pygame.draw.polygon(
            surface,
            self.eye_color,
            [
                (center_x + eye_distance // 2 - width // 2, center_y + angle_offset),
                (center_x + eye_distance // 2 + width // 2, center_y),
                (center_x + eye_distance // 2 + width // 2, center_y + height),
                (center_x + eye_distance // 2 - width // 2, center_y + angle_offset + height)
            ]
        )
    
    def start(self) -> None:
        """Start the display controller."""
        if self.running:
            logger.warning("Display controller is already running")
            return
        
        if self.screen is None or self.clock is None:
            logger.error("Display controller not initialized properly")
            return
        
        logger.info("Starting display controller")
        
        # Set the initial emotion
        self.current_emotion = self.default_emotion
        self.target_emotion = self.default_emotion
        
        # Start the rendering thread
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        
        logger.info("Display controller started successfully")
    
    def stop(self) -> None:
        """Stop the display controller."""
        if not self.running:
            logger.warning("Display controller is not running")
            return
        
        logger.info("Stopping display controller")
        
        # Signal the rendering thread to stop
        self.running = False
        
        # Wait for the rendering thread to finish
        if hasattr(self, 'render_thread') and self.render_thread.is_alive():
            self.render_thread.join(timeout=2.0)
        
        # Quit pygame
        pygame.quit()
        
        logger.info("Display controller stopped successfully")
    
    def set_emotion(self, emotion: str) -> None:
        """
        Set the emotion to display.
        
        Args:
            emotion: The emotion to display
        """
        # Check if the emotion is valid
        if emotion not in config.display.EMOTIONS:
            logger.warning(f"Invalid emotion: {emotion}, using default")
            emotion = self.default_emotion
        
        # Skip if the target emotion is already set
        if emotion == self.target_emotion:
            return
        
        logger.info(f"Setting emotion to: {emotion}")
        
        # Start a transition to the new emotion
        self.target_emotion = emotion
        self.transition_start_time = time.time()
        self.transition_progress = 0.0
    
    def _render_loop(self) -> None:
        """Main rendering loop running in a separate thread."""
        logger.info("Rendering loop started")
        
        while self.running:
            try:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE and self.fullscreen:
                            self.running = False
                            break
                
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
        # Clear the screen
        self.screen.fill(self.background_color)
        
        if self.transition_progress < 1.0:
            # During transition, blend between current and target emotions
            if self.current_emotion in self.emotion_surfaces and self.target_emotion in self.emotion_surfaces:
                current_surface = self.emotion_surfaces[self.current_emotion]
                target_surface = self.emotion_surfaces[self.target_emotion]
                
                # Create a temporary surface for the blended result
                temp_surface = pygame.Surface(self.resolution, pygame.SRCALPHA)
                
                # Draw the current emotion surface with fading alpha
                current_alpha = int(255 * (1.0 - self.transition_progress))
                current_surface.set_alpha(current_alpha)
                temp_surface.blit(current_surface, (0, 0))
                
                # Draw the target emotion surface with increasing alpha
                target_alpha = int(255 * self.transition_progress)
                target_surface.set_alpha(target_alpha)
                temp_surface.blit(target_surface, (0, 0))
                
                # Draw the blended result to the screen
                self.screen.blit(temp_surface, (0, 0))
            else:
                # Fallback if surfaces are missing
                if self.target_emotion in self.emotion_surfaces:
                    self.screen.blit(self.emotion_surfaces[self.target_emotion], (0, 0))
                elif self.current_emotion in self.emotion_surfaces:
                    self.screen.blit(self.emotion_surfaces[self.current_emotion], (0, 0))
                else:
                    # Last resort: draw default eyes
                    self._draw_circular_eyes(
                        self.screen,
                        self.resolution[0] // 2,
                        self.resolution[1] // 2,
                        80
                    )
        else:
            # No transition, just draw the current emotion
            if self.current_emotion in self.emotion_surfaces:
                self.screen.blit(self.emotion_surfaces[self.current_emotion], (0, 0))
            else:
                # Fallback if surface is missing
                self._draw_circular_eyes(
                    self.screen,
                    self.resolution[0] // 2,
                    self.resolution[1] // 2,
                    80
                )
        
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