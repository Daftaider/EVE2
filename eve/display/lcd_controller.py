"""
LCD display controller module.

This module manages the LCD display and renders eye animations.
"""
import logging
import os
import threading
import time
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import pygame
import numpy as np
from PIL import Image, ImageDraw
import cv2

from eve import config
from eve.config.display import Emotion, DisplayConfig
from eve.emotion.emotion import Emotion as EveEmotion

logger = logging.getLogger(__name__)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two boxes.
    Boxes are expected in (startX, startY, endX, endY) format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

IOU_THRESHOLD = 0.7 # Overlap threshold to consider a match for correction

class LCDController:
    """Controls the LCD display and renders eye animations."""
    
    def __init__(self, config: Dict):
        """Initialize the LCD controller.
        
        Args:
            config: Configuration dictionary containing display settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Display settings
        self.width = config.get('display', {}).get('resolution', [800, 480])[0]
        self.height = config.get('display', {}).get('resolution', [800, 480])[1]
        self.fps = config.get('display', {}).get('fps', 30)
        self.fullscreen = config.get('display', {}).get('fullscreen', True)
        self.rotation = config.get('display', {}).get('rotation', 0)
        self.use_hardware_display = config.get('display', {}).get('use_hardware_display', True)
        
        # Color settings
        self.background_color = self._parse_color(config.get('display', {}).get('background_color', (0, 0, 0)))
        self.eye_color = self._parse_color(config.get('display', {}).get('eye_color', (255, 255, 255)))
        self.text_color = self._parse_color(config.get('display', {}).get('text_color', (255, 255, 255)))
        
        # Animation settings
        self.blink_interval = config.get('display', {}).get('blink_interval_sec', 4.0)
        self.blink_duration = config.get('display', {}).get('blink_duration_sec', 0.15)
        self.transition_speed = config.get('display', {}).get('transition_speed', 0.1)
        
        # Debug settings
        self.debug_menu_enabled = config.get('display', {}).get('debug_menu_enabled', True)
        self.debug_font_size = config.get('display', {}).get('debug_font_size', 24)
        self.debug_ui_elements = {}
        
        # File paths
        self.asset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'display')
        self.current_frame_path = None
        
        # State
        self.running = True
        self.is_blinking = False
        self.current_emotion = Emotion.NEUTRAL
        self.debug_mode = None
        self.last_blink_time = time.time()
        
        # Initialize pygame and display
        self._init_display()
        
        # Set up signal handler for CTRL+C
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start blink thread
        self.blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
        self.blink_thread.start()
    
    def _signal_handler(self, signum, frame):
        """Handle CTRL+C signal."""
        self.logger.info("Received CTRL+C signal")
        self.running = False
        pygame.quit()
        sys.exit(0)
    
    def update(self, is_listening: bool = False):
        """Update the display with current state.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        if not self.running:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                self._handle_key_event(event)
        
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Check if we should blink
        current_time = time.time()
        if current_time - self.last_blink_time >= self.blink_interval:
            self.blink()
            self.last_blink_time = current_time
        
        # Get current emotion image
        image = self.emotion_images.get(self.current_emotion)
        if image is None:
            self.logger.error(f"No image found for emotion: {self.current_emotion}")
            return
        
        # Center the image
        x = (self.width - image.get_width()) // 2
        y = (self.height - image.get_height()) // 2
        
        # Apply rotation if needed
        if self.rotation != 0:
            image = pygame.transform.rotate(image, self.rotation)
        
        self.screen.blit(image, (x, y))
        
        # Draw debug menu if enabled
        if self.debug_menu_enabled:
            self._draw_debug_menu(is_listening)
        
        # Update display
        pygame.display.flip()
        
        # Save current frame if path is set
        if self.current_frame_path:
            pygame.image.save(self.screen, self.current_frame_path)
        
        # Cap the frame rate
        self.clock.tick(self.fps)
    
    def _handle_key_event(self, event):
        """Handle keyboard events."""
        if event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
            # CTRL+C - Exit
            self.logger.info("CTRL+C pressed, exiting...")
            self.running = False
        elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
            # CTRL+S - Toggle debug mode
            if self.debug_mode is None:
                # Show debug mode selection menu
                self._show_debug_mode_menu()
            else:
                # Exit debug mode
                self.debug_mode = None
                self.logger.info("Exiting debug mode")
        elif self.debug_mode == 'video':
            # Video debug mode controls
            if event.key == pygame.K_r:
                # Rotate display
                self.rotation = (self.rotation + 90) % 360
                self.logger.info(f"Display rotation set to {self.rotation} degrees")
            elif event.key == pygame.K_ESCAPE:
                # Exit video debug mode
                self.debug_mode = None
        elif self.debug_mode == 'audio':
            # Audio debug mode controls
            if event.key == pygame.K_ESCAPE:
                # Exit audio debug mode
                self.debug_mode = None
    
    def _show_debug_mode_menu(self):
        """Show the debug mode selection menu."""
        self.screen.fill(self.background_color)
        
        # Draw menu title
        font = pygame.font.Font(None, 36)
        title = font.render("Debug Mode Selection", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)
        
        # Draw options
        font = pygame.font.Font(None, 24)
        video_option = font.render("1. Video Debug (Object Detection)", True, self.text_color)
        audio_option = font.render("2. Audio Debug (Voice Detection)", True, self.text_color)
        exit_option = font.render("3. Exit Menu", True, self.text_color)
        
        video_rect = video_option.get_rect(center=(self.width // 2, self.height // 2))
        audio_rect = audio_option.get_rect(center=(self.width // 2, self.height // 2 + 40))
        exit_rect = exit_option.get_rect(center=(self.width // 2, self.height // 2 + 80))
        
        self.screen.blit(video_option, video_rect)
        self.screen.blit(audio_option, audio_rect)
        self.screen.blit(exit_option, exit_rect)
        
        pygame.display.flip()
        
        # Wait for selection
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.debug_mode = 'video'
                        waiting = False
                    elif event.key == pygame.K_2:
                        self.debug_mode = 'audio'
                        waiting = False
                    elif event.key == pygame.K_3 or event.key == pygame.K_ESCAPE:
                        waiting = False
    
    def _update_video_debug(self):
        """Update the video debug display."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Video Debug Mode", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "R: Rotate display",
            "ESC: Exit debug mode",
            "CTRL+C: Exit application"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (20, 80 + i * 30))
        
        # Draw current rotation
        rotation_text = f"Current rotation: {self.rotation}°"
        rotation_surface = font.render(rotation_text, True, self.text_color)
        self.screen.blit(rotation_surface, (20, 200))
        
        # Draw placeholder for object detection
        # In a real implementation, this would show the camera feed with detected objects
        placeholder_text = "Object Detection Feed"
        placeholder_surface = font.render(placeholder_text, True, self.text_color)
        placeholder_rect = placeholder_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(placeholder_surface, placeholder_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _update_audio_debug(self):
        """Update the audio debug display."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Audio Debug Mode", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "ESC: Exit debug mode",
            "CTRL+C: Exit application"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (20, 80 + i * 30))
        
        # Draw placeholder for audio visualization
        # In a real implementation, this would show audio levels and detected speech
        placeholder_text = "Audio Visualization"
        placeholder_surface = font.render(placeholder_text, True, self.text_color)
        placeholder_rect = placeholder_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(placeholder_surface, placeholder_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_debug_menu(self, is_listening: bool):
        """Draw the debug menu on the screen.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        font = pygame.font.Font(None, self.debug_font_size)
        
        # Draw emotion text
        emotion_text = f"Emotion: {self.current_emotion.name}"
        text_surface = font.render(emotion_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 10))
        
        # Draw listening status
        listening_text = "Listening: Yes" if is_listening else "Listening: No"
        text_surface = font.render(listening_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 40))
        
        # Draw FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        text_surface = font.render(fps_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 70))
        
        # Draw keyboard shortcuts
        shortcuts = [
            "CTRL+C: Exit",
            "CTRL+S: Debug Menu"
        ]
        
        for i, text in enumerate(shortcuts):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (10, 100 + i * 30))
    
    def cleanup(self):
        """Clean up resources."""
        pygame.quit()
        self.logger.info("LCD Controller cleaned up")

    def _parse_color(self, color: Union[Tuple[int, int, int], str, None]) -> Tuple[int, int, int]:
        """Convert color to RGB tuple."""
        if color is None:
            return (255, 255, 255)
        if isinstance(color, tuple) and len(color) == 3:
            return color
        if isinstance(color, str):
            try:
                rgb = pygame.Color(color)
                return (rgb.r, rgb.g, rgb.b)
            except ValueError:
                logger.warning(f"Invalid color string: {color}, using default")
                return (255, 255, 255)
        return (255, 255, 255)

    def _init_display(self):
        """Initialize the display with current settings."""
        try:
            pygame.init()
            
            # Set window position (centered)
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            
            # Create the window with a title
            pygame.display.set_caption("EVE2 Display")
            
            # Initialize the display
            flags = pygame.FULLSCREEN if self.fullscreen else 0
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            # Load images
            self._load_emotion_images()
            
            # Initial display update
            self.screen.fill(self.background_color)
            pygame.display.flip()
            
            self.logger.info("Display initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize display: {e}")
            self._init_fallback_mode()

    def _init_fallback_mode(self):
        """Initialize a fallback mode for headless operation."""
        logger.info("Initializing display in fallback mode")
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self._load_emotion_images()

    def _rotate_surface(self, surface: pygame.Surface) -> pygame.Surface:
        """Rotate a surface according to the display rotation setting."""
        if self.rotation == 0:
            return surface
        return pygame.transform.rotate(surface, -self.rotation)

    def _apply_eye_color(self, surface: pygame.Surface, eye_color: Tuple[int, int, int]) -> pygame.Surface:
        """Apply eye color to white pixels in the surface."""
        # Create a copy of the surface
        new_surface = surface.copy()
        
        # Get pixel array of the surface
        pixels = pygame.PixelArray(new_surface)
        
        # Find white pixels (255, 255, 255) and replace with eye color
        white = surface.map_rgb((255, 255, 255))
        pixels.replace(white, surface.map_rgb(eye_color))
        
        # Free the pixel array
        pixels.close()
        
        return new_surface

    def _load_emotion_images(self):
        """Load all emotion images into memory."""
        self.emotion_images = {}
        
        # Create assets directory if it doesn't exist
        os.makedirs(self.asset_dir, exist_ok=True)
        
        for emotion in Emotion:
            try:
                # Construct the image path using the asset directory and emotion filename
                image_path = os.path.join(self.asset_dir, emotion.filename)
                
                # Check if file exists
                if not os.path.exists(image_path):
                    self.logger.warning(f"Emotion image not found: {image_path}")
                    # Create a fallback surface with emotion color
                    surface = pygame.Surface((self.width, self.height))
                    surface.fill(self._get_fallback_color(emotion))
                    self.emotion_images[emotion] = surface
                    continue
                
                original = pygame.image.load(image_path)
                
                # Scale image if needed
                if original.get_size() != (self.width, self.height):
                    original = pygame.transform.scale(original, (self.width, self.height))
                
                # Apply eye color
                colored = self._apply_eye_color(original, self.eye_color)
                self.emotion_images[emotion] = colored
                
            except Exception as e:
                self.logger.warning(f"Failed to load emotion image for {emotion.name}: {e}")
                # Create a fallback surface
                surface = pygame.Surface((self.width, self.height))
                surface.fill(self._get_fallback_color(emotion))
                self.emotion_images[emotion] = surface

    def _get_fallback_color(self, emotion: Emotion) -> Tuple[int, int, int]:
        """Get a fallback color for an emotion."""
        colors = {
            Emotion.NEUTRAL: (128, 128, 128),    # Gray
            Emotion.HAPPY: (255, 255, 0),        # Yellow
            Emotion.SAD: (0, 0, 255),            # Blue
            Emotion.ANGRY: (255, 0, 0),          # Red
            Emotion.SURPRISED: (255, 165, 0),    # Orange
            Emotion.FEARFUL: (128, 0, 128),      # Purple
            Emotion.DISGUSTED: (0, 128, 0),      # Green
            Emotion.BLINK: (0, 0, 0),            # Black for blinking
        }
        return colors.get(emotion, (0, 0, 0))

    def start(self) -> bool:
        """Start the display controller."""
        try:
            logger.info("Starting LCD Controller...")
            # Initial display update
            self.update()
            logger.info("LCD Controller started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start LCD Controller: {e}")
            return False

    def stop(self):
        """Stop the display controller."""
        try:
            logger.info("Stopping LCD Controller...")
            # Clean up resources
            pygame.quit()
            logger.info("LCD Controller stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping LCD Controller: {e}")

    def set_rotation(self, rotation: int):
        """Set the display rotation."""
        self.rotation = rotation % 360
        logger.info(f"Display rotation set to {self.rotation} degrees")

    def toggle_hardware_display(self, use_hardware: bool):
        """Toggle hardware display mode."""
        if self.use_hardware_display != use_hardware:
            self.use_hardware_display = use_hardware
            # Reinitialize display with new settings
            self._init_display()
            logger.info(f"Hardware display {'enabled' if use_hardware else 'disabled'}")

    def get_debug_ui_element_at(self, pos: Tuple[int, int]) -> Optional[str]:
        """Check if a position overlaps with a known debug UI element.

        Args:
            pos: The (x, y) coordinate to check.

        Returns:
            The string ID of the clicked element (e.g., 'select_cam_1') or None.
        """
        for element_id, rect in self.debug_ui_elements.items():
            if rect.collidepoint(pos):
                return element_id
        return None

    def set_eye_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """Set a new eye color and reload images."""
        self.eye_color = self._parse_color(color)
        self._load_emotion_images()  # Reload images with new eye color
        self.update()

    def set_background_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """Set a new background color."""
        self.background_color = self._parse_color(color)
        self._load_emotion_images()  # Reload images with new background
        self.update()

    def get_current_emotion(self) -> Emotion:
        """Get the current emotion being displayed."""
        return self.current_emotion

    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        if not isinstance(emotion, Emotion):
            raise ValueError(f"Expected Emotion enum, got {type(emotion)}")
        self.current_emotion = emotion
        self.update()

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            
            # Save current emotion
            current_emotion = self.current_emotion
            
            # Set to blink emotion
            self.current_emotion = Emotion.BLINK
            self.update()
            
            # Wait for blink duration
            time.sleep(self.blink_duration)
            
            # Restore previous emotion
            self.current_emotion = current_emotion
            self.update()
            
            self.is_blinking = False
            logger.debug("Completed blink animation")
        except Exception as e:
            logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False

    def _blink_loop(self):
        """Loop for blinking animation."""
        while self.running:
            if self.is_blinking:
                self.update()
            time.sleep(0.01)  # Wait between checks

    def _show_debug_mode_menu(self):
        """Show the debug mode selection menu."""
        self.screen.fill(self.background_color)
        
        # Draw menu title
        font = pygame.font.Font(None, 36)
        title = font.render("Debug Mode Selection", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)
        
        # Draw options
        font = pygame.font.Font(None, 24)
        video_option = font.render("1. Video Debug (Object Detection)", True, self.text_color)
        audio_option = font.render("2. Audio Debug (Voice Detection)", True, self.text_color)
        exit_option = font.render("3. Exit Menu", True, self.text_color)
        
        video_rect = video_option.get_rect(center=(self.width // 2, self.height // 2))
        audio_rect = audio_option.get_rect(center=(self.width // 2, self.height // 2 + 40))
        exit_rect = exit_option.get_rect(center=(self.width // 2, self.height // 2 + 80))
        
        self.screen.blit(video_option, video_rect)
        self.screen.blit(audio_option, audio_rect)
        self.screen.blit(exit_option, exit_rect)
        
        pygame.display.flip()
        
        # Wait for selection
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.debug_mode = 'video'
                        waiting = False
                    elif event.key == pygame.K_2:
                        self.debug_mode = 'audio'
                        waiting = False
                    elif event.key == pygame.K_3 or event.key == pygame.K_ESCAPE:
                        waiting = False
    
    def _update_video_debug(self):
        """Update the video debug display."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Video Debug Mode", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "R: Rotate display",
            "ESC: Exit debug mode",
            "CTRL+C: Exit application"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (20, 80 + i * 30))
        
        # Draw current rotation
        rotation_text = f"Current rotation: {self.rotation}°"
        rotation_surface = font.render(rotation_text, True, self.text_color)
        self.screen.blit(rotation_surface, (20, 200))
        
        # Draw placeholder for object detection
        # In a real implementation, this would show the camera feed with detected objects
        placeholder_text = "Object Detection Feed"
        placeholder_surface = font.render(placeholder_text, True, self.text_color)
        placeholder_rect = placeholder_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(placeholder_surface, placeholder_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _update_audio_debug(self):
        """Update the audio debug display."""
        # Clear screen
        self.screen.fill(self.background_color)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Audio Debug Mode", True, self.text_color)
        title_rect = title.get_rect(center=(self.width // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "ESC: Exit debug mode",
            "CTRL+C: Exit application"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (20, 80 + i * 30))
        
        # Draw placeholder for audio visualization
        # In a real implementation, this would show audio levels and detected speech
        placeholder_text = "Audio Visualization"
        placeholder_surface = font.render(placeholder_text, True, self.text_color)
        placeholder_rect = placeholder_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(placeholder_surface, placeholder_rect)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_debug_menu(self, is_listening: bool):
        """Draw the debug menu on the screen.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        font = pygame.font.Font(None, self.debug_font_size)
        
        # Draw emotion text
        emotion_text = f"Emotion: {self.current_emotion.name}"
        text_surface = font.render(emotion_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 10))
        
        # Draw listening status
        listening_text = "Listening: Yes" if is_listening else "Listening: No"
        text_surface = font.render(listening_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 40))
        
        # Draw FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        text_surface = font.render(fps_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 70))
        
        # Draw keyboard shortcuts
        shortcuts = [
            "CTRL+C: Exit",
            "CTRL+S: Debug Menu"
        ]
        
        for i, text in enumerate(shortcuts):
            text_surface = font.render(text, True, self.text_color)
            self.screen.blit(text_surface, (10, 100 + i * 30)) 