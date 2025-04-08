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
from eve.emotion.emotion import Emotion
from eve.config.config import DisplayConfig

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
    
    def __init__(self, config: DisplayConfig):
        """Initialize the LCD controller.
        
        Args:
            config: DisplayConfig object containing display settings
        """
        try:
            # Set up logger with proper name
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)  # Ensure input events are visible
            
            # Display settings
            self.width = config.WINDOW_SIZE[0]
            self.height = config.WINDOW_SIZE[1]
            self.fps = config.FPS
            self.fullscreen = config.FULLSCREEN
            self.rotation = config.DISPLAY_ROTATION
            self.use_hardware_display = config.USE_HARDWARE_DISPLAY
            
            # Color settings
            self.background_color = self._parse_color(config.BACKGROUND_COLOR)
            self.eye_color = self._parse_color(config.EYE_COLOR)
            self.text_color = self._parse_color(config.TEXT_COLOR)
            
            # Animation settings
            self.blink_interval = config.BLINK_INTERVAL_SEC
            self.blink_duration = config.BLINK_DURATION
            self.transition_speed = config.TRANSITION_SPEED
            
            # Debug settings
            self.debug_menu_enabled = config.DEBUG_MENU_ENABLED
            self.debug_font_size = config.DEBUG_FONT_SIZE
            self.debug_ui_elements = {}
            
            # File paths
            self.asset_dir = os.path.join(os.path.dirname(__file__), '..', '..', config.ASSET_DIR)
            self.current_frame_path = config.CURRENT_FRAME_PATH
            
            # State
            self.running = True
            self.is_blinking = False
            self.current_emotion = Emotion.NEUTRAL
            self.debug_mode = None
            self.last_blink_time = time.time()
            
            # Double-click detection
            self.last_click_time = 0
            self.double_click_threshold = 0.5  # seconds
            self.last_click_pos = None
            
            # Check if we're in a headless environment
            self.headless_mode = self._is_headless_environment()
            if self.headless_mode:
                self.logger.info("Running in headless mode - display will be simulated")
            
            # Initialize pygame and display
            self._init_display()
            
            # Set up signal handler for CTRL+C
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Start blink thread
            self.blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
            self.blink_thread.start()
            
            self.logger.info("LCD Controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LCD Controller: {e}", exc_info=True)
            raise
    
    def _is_headless_environment(self) -> bool:
        """Check if we're running in a headless environment."""
        # Check if DISPLAY environment variable is set
        if 'DISPLAY' not in os.environ:
            return True
        
        # Check if we're running in a container or virtual environment
        if os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
            return True
        
        # Check if we're running on a Raspberry Pi without a display
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry pi' in model and not os.path.exists('/dev/fb0'):
                    return True
        
        return False
    
    def _init_display(self):
        """Initialize the display with current settings."""
        try:
            pygame.init()
            
            # Set window position (centered)
            os.environ['SDL_VIDEO_CENTERED'] = '1'
            
            # Create the window with a title
            pygame.display.set_caption("EVE2 Display")
            
            # Initialize the display
            if self.headless_mode:
                # Use a dummy display in headless mode
                self.screen = pygame.Surface((self.width, self.height))
                self.logger.info("Using dummy display in headless mode")
            else:
                # Try to use hardware display if available
                if self.use_hardware_display:
                    try:
                        # Check if we're on a Raspberry Pi
                        is_raspberry_pi = os.path.exists('/proc/device-tree/model')
                        if is_raspberry_pi:
                            with open('/proc/device-tree/model', 'r') as f:
                                model = f.read().lower()
                                is_raspberry_pi = 'raspberry pi' in model
                        
                        if is_raspberry_pi:
                            # Set environment variables for framebuffer
                            os.environ['SDL_VIDEODRIVER'] = 'fbcon'
                            os.environ['SDL_FBDEV'] = '/dev/fb0'
                            os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1'
                            
                            # Create fullscreen display
                            flags = pygame.FULLSCREEN
                            self.screen = pygame.display.set_mode((self.width, self.height), flags)
                            self.logger.info("Using hardware display with framebuffer on Raspberry Pi")
                        else:
                            raise RuntimeError("Not running on a Raspberry Pi")
                            
                    except Exception as fb_err:
                        self.logger.warning(f"Failed to use framebuffer: {fb_err}. Falling back to windowed mode.")
                        # Reset SDL video driver to default
                        if 'SDL_VIDEODRIVER' in os.environ:
                            del os.environ['SDL_VIDEODRIVER']
                        if 'SDL_FBDEV' in os.environ:
                            del os.environ['SDL_FBDEV']
                        if 'SDL_VIDEO_CURSOR_HIDDEN' in os.environ:
                            del os.environ['SDL_VIDEO_CURSOR_HIDDEN']
                        
                        # Fall back to windowed mode
                        flags = pygame.FULLSCREEN if self.fullscreen else 0
                        self.screen = pygame.display.set_mode((self.width, self.height), flags)
                        self.logger.info("Using windowed display mode")
                else:
                    # Use windowed mode
                    flags = pygame.FULLSCREEN if self.fullscreen else 0
                    self.screen = pygame.display.set_mode((self.width, self.height), flags)
                    self.logger.info("Using windowed display mode (hardware display disabled)")
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            # Enable key repeat for better keyboard handling
            pygame.key.set_repeat(500, 50)  # 500ms delay, 50ms interval
            
            # Load images
            self._load_emotion_images()
            
            # Initial display update
            self.screen.fill(self.background_color)
            if not self.headless_mode:
                pygame.display.flip()
            
            self.logger.info("Display initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize display: {e}")
            self._init_fallback_mode()
    
    def _init_fallback_mode(self):
        """Initialize a fallback mode for headless operation."""
        self.logger.info("Initializing display in fallback mode")
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self._load_emotion_images()
        self.headless_mode = True
    
    def _load_emotion_images(self):
        """Load all emotion images into memory."""
        self.emotion_images = {}
        
        # Create assets directory if it doesn't exist
        os.makedirs(self.asset_dir, exist_ok=True)
        
        # Create default emotion images if they don't exist
        self._create_default_emotion_images()
        
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
    
    def _create_default_emotion_images(self):
        """Create default emotion images if they don't exist."""
        # Define basic shapes for each emotion
        emotion_shapes = {
            Emotion.NEUTRAL: self._create_neutral_face,
            Emotion.HAPPY: self._create_happy_face,
            Emotion.SAD: self._create_sad_face,
            Emotion.ANGRY: self._create_angry_face,
            Emotion.SURPRISED: self._create_surprised_face,
            Emotion.FEARFUL: self._create_fearful_face,
            Emotion.DISGUSTED: self._create_disgusted_face,
            Emotion.BLINK: self._create_blink_face
        }
        
        # Create each emotion image if it doesn't exist
        for emotion, shape_func in emotion_shapes.items():
            image_path = os.path.join(self.asset_dir, emotion.filename)
            if not os.path.exists(image_path):
                self.logger.info(f"Creating default image for {emotion.name}")
                # Create a new surface
                surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                surface.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw the emotion shape
                shape_func(surface)
                
                # Save the image
                pygame.image.save(surface, image_path)
    
    def _create_neutral_face(self, surface):
        """Create a neutral face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw mouth (straight line)
        mouth_y = 2 * self.height // 3
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3, mouth_y), 
                        (2 * self.width // 3, mouth_y), 3)
    
    def _create_happy_face(self, surface):
        """Create a happy face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw mouth (smile)
        mouth_y = 2 * self.height // 3
        pygame.draw.arc(surface, eye_color, 
                       (self.width // 3, mouth_y - 20, 
                        self.width // 3, 40), 0, 3.14, 3)
    
    def _create_sad_face(self, surface):
        """Create a sad face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw mouth (frown)
        mouth_y = 2 * self.height // 3
        pygame.draw.arc(surface, eye_color, 
                       (self.width // 3, mouth_y - 20, 
                        self.width // 3, 40), 3.14, 6.28, 3)
    
    def _create_angry_face(self, surface):
        """Create an angry face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw eyebrows
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3 - 20, self.height // 2 - 20), 
                        (self.width // 3 + 20, self.height // 2 - 10), 3)
        
        pygame.draw.line(surface, eye_color, 
                        (2 * self.width // 3 - 20, self.height // 2 - 10), 
                        (2 * self.width // 3 + 20, self.height // 2 - 20), 3)
        
        # Draw mouth (straight line)
        mouth_y = 2 * self.height // 3
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3, mouth_y), 
                        (2 * self.width // 3, mouth_y), 3)
    
    def _create_surprised_face(self, surface):
        """Create a surprised face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw mouth (O shape)
        mouth_y = 2 * self.height // 3
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 2, mouth_y), 20, 3)
    
    def _create_fearful_face(self, surface):
        """Create a fearful face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw eyebrows
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3 - 20, self.height // 2 - 10), 
                        (self.width // 3 + 20, self.height // 2 - 20), 3)
        
        pygame.draw.line(surface, eye_color, 
                        (2 * self.width // 3 - 20, self.height // 2 - 20), 
                        (2 * self.width // 3 + 20, self.height // 2 - 10), 3)
        
        # Draw mouth (open)
        mouth_y = 2 * self.height // 3
        pygame.draw.ellipse(surface, eye_color, 
                          (self.width // 2 - 20, mouth_y - 10, 40, 20), 3)
    
    def _create_disgusted_face(self, surface):
        """Create a disgusted face."""
        # Draw eyes
        eye_color = self.eye_color
        eye_size = min(self.width, self.height) // 16
        
        # Left eye
        pygame.draw.circle(surface, eye_color, 
                          (self.width // 3, self.height // 2), eye_size)
        
        # Right eye
        pygame.draw.circle(surface, eye_color, 
                          (2 * self.width // 3, self.height // 2), eye_size)
        
        # Draw mouth (disgusted)
        mouth_y = 2 * self.height // 3
        pygame.draw.arc(surface, eye_color, 
                       (self.width // 3, mouth_y - 20, 
                        self.width // 3, 40), 0, 3.14, 3)
        
        # Draw tongue
        pygame.draw.line(surface, eye_color, 
                        (self.width // 2, mouth_y + 10), 
                        (self.width // 2, mouth_y + 30), 3)
    
    def _create_blink_face(self, surface):
        """Create a blinking face."""
        # Draw eyes (closed)
        eye_color = self.eye_color
        
        # Left eye (closed)
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3 - 20, self.height // 2), 
                        (self.width // 3 + 20, self.height // 2), 3)
        
        # Right eye (closed)
        pygame.draw.line(surface, eye_color, 
                        (2 * self.width // 3 - 20, self.height // 2), 
                        (2 * self.width // 3 + 20, self.height // 2), 3)
        
        # Draw mouth (straight line)
        mouth_y = 2 * self.height // 3
        pygame.draw.line(surface, eye_color, 
                        (self.width // 3, mouth_y), 
                        (2 * self.width // 3, mouth_y), 3)

    def _signal_handler(self, signum, frame):
        """Handle CTRL+C signal."""
        self.logger.info("Received CTRL+C signal")
        self.running = False
        pygame.quit()
        sys.exit(0)
    
    def _process_events(self):
        """Process all pending events and return True if any were handled."""
        handled = False
        for event in pygame.event.get():
            # Log all events at debug level
            self.logger.debug(f"Processing event: {event}")
            
            if event.type == pygame.QUIT:
                self.logger.info("Received QUIT event")
                self.running = False
                handled = True
                break
            
            elif event.type == pygame.KEYDOWN:
                # Log key press with modifiers
                key_name = pygame.key.name(event.key)
                mod_keys = []
                if event.mod & pygame.KMOD_CTRL: mod_keys.append('CTRL')
                if event.mod & pygame.KMOD_SHIFT: mod_keys.append('SHIFT')
                if event.mod & pygame.KMOD_ALT: mod_keys.append('ALT')
                mod_str = '+'.join(mod_keys) if mod_keys else 'NO_MOD'
                self.logger.info(f"Key pressed: {key_name}, Modifiers: {mod_str}")
                
                # Handle CTRL+C
                if event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
                    self.logger.info("CTRL+C pressed, exiting...")
                    self.running = False
                    pygame.quit()
                    sys.exit(0)
                
                # Handle CTRL+S
                elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                    self.logger.info("CTRL+S pressed, toggling debug mode")
                    if self.debug_mode is None:
                        self._show_debug_mode_menu()
                    else:
                        self.debug_mode = None
                    handled = True
                
                # Handle ESC
                elif event.key == pygame.K_ESCAPE:
                    if self.debug_mode is not None:
                        self.logger.info("ESC pressed, exiting debug mode")
                        self.debug_mode = None
                    else:
                        self.logger.info("ESC pressed, exiting application")
                        self.running = False
                        pygame.quit()
                        sys.exit(0)
                    handled = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    current_time = time.time()
                    current_pos = event.pos
                    self.logger.info(f"Mouse click at position: {current_pos}")
                    
                    # Check for double-click with improved position checking
                    if (self.last_click_time and 
                        current_time - self.last_click_time < self.double_click_threshold and 
                        self.last_click_pos and 
                        abs(current_pos[0] - self.last_click_pos[0]) < 20 and  # Increased threshold
                        abs(current_pos[1] - self.last_click_pos[1]) < 20):    # Increased threshold
                        self.logger.info("Double-click detected, toggling debug mode")
                        if self.debug_mode is None:
                            self._show_debug_mode_menu()
                        else:
                            self.debug_mode = None
                        handled = True
                        # Reset click tracking after double-click
                        self.last_click_time = None
                        self.last_click_pos = None
                    else:
                        # Update last click time and position
                        self.last_click_time = current_time
                        self.last_click_pos = current_pos
        
        return handled

    def update(self, is_listening: bool = False):
        """Update the display with current state.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        if not self.running:
            return
        
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
        
        # Draw the emotion image
        self.screen.blit(image, (x, y))
        
        # Draw debug menu if enabled
        if self.debug_menu_enabled:
            self._draw_debug_menu(is_listening)
        
        # Update display
        if not self.headless_mode:
            pygame.display.flip()
        
        # Save current frame if path is set
        if self.current_frame_path:
            pygame.image.save(self.screen, self.current_frame_path)
        
        # Cap the frame rate and update FPS counter
        self.clock.tick(self.fps)
        
        # Log FPS periodically
        current_fps = self.clock.get_fps()
        if current_fps > 0:
            self.logger.debug(f"Current FPS: {int(current_fps)}")
    
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
    
    def _draw_debug_menu(self, is_listening: bool):
        """Draw the debug menu on the screen.
        
        Args:
            is_listening: Whether EVE is currently listening
        """
        font = pygame.font.Font(None, self.debug_font_size)
        
        # Draw emotion text with background for better visibility
        emotion_text = f"Emotion: {self.current_emotion.name}"
        text_surface = font.render(emotion_text, True, self.text_color)
        text_rect = text_surface.get_rect(topleft=(10, 10))
        pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
        self.screen.blit(text_surface, text_rect)
        
        # Draw listening status with more visible formatting
        listening_text = "Listening: YES" if is_listening else "Listening: NO"
        listening_color = (0, 255, 0) if is_listening else (255, 0, 0)  # Green for Yes, Red for No
        text_surface = font.render(listening_text, True, listening_color)
        text_rect = text_surface.get_rect(topleft=(10, 40))
        pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
        self.screen.blit(text_surface, text_rect)
        
        # Draw FPS with background
        current_fps = self.clock.get_fps()
        fps_text = f"FPS: {int(current_fps) if current_fps > 0 else 0}"
        text_surface = font.render(fps_text, True, self.text_color)
        text_rect = text_surface.get_rect(topleft=(10, 70))
        pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
        self.screen.blit(text_surface, text_rect)
        
        # Draw keyboard shortcuts with background
        shortcuts = [
            "CTRL+C: Exit",
            "CTRL+S: Debug Menu",
            "ESC: Exit/Back",
            "Double-Click: Debug Menu"
        ]
        
        for i, text in enumerate(shortcuts):
            text_surface = font.render(text, True, self.text_color)
            text_rect = text_surface.get_rect(topleft=(10, 100 + i * 30))
            pygame.draw.rect(self.screen, (0, 0, 0, 128), text_rect.inflate(10, 5))
            self.screen.blit(text_surface, text_rect)

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
            self.logger.info("Starting LCD Controller...")
            
            # Initial display update
            self.update()
            
            # In headless mode, we don't need to wait for display events
            if self.headless_mode:
                self.logger.info("Running in headless mode - display updates will be simulated")
            
            self.logger.info("LCD Controller started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start LCD Controller: {e}")
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
    
    def handle_event(self, event):
        """Handle a Pygame event.
        
        Args:
            event: The Pygame event to handle
            
        Returns:
            bool: True if the event was handled, False otherwise
        """
        if event.type == pygame.QUIT:
            self.logger.info("Received QUIT event")
            self.running = False
            return True
        
        elif event.type == pygame.KEYDOWN:
            # Log key press with modifiers
            key_name = pygame.key.name(event.key)
            mod_keys = []
            if event.mod & pygame.KMOD_CTRL: mod_keys.append('CTRL')
            if event.mod & pygame.KMOD_SHIFT: mod_keys.append('SHIFT')
            if event.mod & pygame.KMOD_ALT: mod_keys.append('ALT')
            mod_str = '+'.join(mod_keys) if mod_keys else 'NO_MOD'
            self.logger.info(f"Key pressed: {key_name}, Modifiers: {mod_str}")
            
            # Handle CTRL+C
            if event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
                self.logger.info("CTRL+C pressed, exiting...")
                self.running = False
                pygame.quit()
                sys.exit(0)
            
            # Handle CTRL+S
            elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                self.logger.info("CTRL+S pressed, toggling debug mode")
                if self.debug_mode is None:
                    self._show_debug_mode_menu()
                else:
                    self.debug_mode = None
                return True
            
            # Handle ESC
            elif event.key == pygame.K_ESCAPE:
                if self.debug_mode is not None:
                    self.logger.info("ESC pressed, exiting debug mode")
                    self.debug_mode = None
                else:
                    self.logger.info("ESC pressed, exiting application")
                    self.running = False
                    pygame.quit()
                    sys.exit(0)
                return True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                current_time = time.time()
                current_pos = event.pos
                self.logger.info(f"Mouse click at position: {current_pos}")
                
                # Check for double-click with improved position checking
                if (self.last_click_time and 
                    current_time - self.last_click_time < self.double_click_threshold and 
                    self.last_click_pos and 
                    abs(current_pos[0] - self.last_click_pos[0]) < 20 and  # Increased threshold
                    abs(current_pos[1] - self.last_click_pos[1]) < 20):    # Increased threshold
                    self.logger.info("Double-click detected, toggling debug mode")
                    if self.debug_mode is None:
                        self._show_debug_mode_menu()
                    else:
                        self.debug_mode = None
                    # Reset click tracking after double-click
                    self.last_click_time = None
                    self.last_click_pos = None
                    return True
                else:
                    # Update last click time and position
                    self.last_click_time = current_time
                    self.last_click_pos = current_pos
        
        return False 