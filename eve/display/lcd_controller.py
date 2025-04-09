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
import traceback
import random
import queue

import pygame
import numpy as np
from PIL import Image, ImageDraw
import cv2

from eve import config
from eve.emotion.emotion import Emotion
from eve.config.config import DisplayConfig
from eve.vision.rpi_ai_camera import RPiAICamera

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
            
            # FPS tracking
            self.current_fps = 0
            self.fps_counter = 0
            self.last_fps_time = time.time()
            
            # Double-click detection
            self.last_click_time = 0
            self.double_click_threshold = 0.5  # seconds
            self.last_click_pos = None
            
            # Thread synchronization
            self._blink_stop_event = threading.Event()
            self._blink_queue = queue.Queue()
            self._blink_thread = None
            
            # Check if we're in a headless environment
            self.headless_mode = self._is_headless_environment()
            if self.headless_mode:
                self.logger.info("Running in headless mode - display will be simulated")
            
            # Initialize pygame and display
            self._init_display()
            
            # Initialize font
            self.font = pygame.font.Font(None, self.debug_font_size)
            
            # Set up signal handler for CTRL+C
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Start blink thread
            self._start_blink_thread()
            
            # Initialize camera and object detector references
            self.camera = None
            self.object_detector = None
            
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
            # Initialize pygame if not already initialized
            if not pygame.get_init():
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
                            self.logger.info(f"Detected Raspberry Pi model: {model.strip()}")
                        
                        if is_raspberry_pi:
                            # Clear any existing SDL environment variables
                            for var in ['SDL_VIDEODRIVER', 'SDL_FBDEV', 'SDL_VIDEO_CURSOR_HIDDEN']:
                                if var in os.environ:
                                    del os.environ[var]
                            
                            # Set environment variables for framebuffer
                            os.environ['SDL_VIDEODRIVER'] = 'fbcon'
                            os.environ['SDL_FBDEV'] = '/dev/fb0'
                            os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1'
                            
                            # Create fullscreen display
                            flags = pygame.FULLSCREEN
                            self.logger.info("Attempting to create fullscreen display with framebuffer...")
                            self.screen = pygame.display.set_mode((self.width, self.height), flags)
                            self.logger.info("Successfully initialized hardware display with framebuffer")
                        else:
                            raise RuntimeError("Not running on a Raspberry Pi")
                            
                    except Exception as fb_err:
                        self.logger.warning(f"Failed to use framebuffer: {fb_err}. Falling back to windowed mode.")
                        # Reset SDL video driver to default
                        for var in ['SDL_VIDEODRIVER', 'SDL_FBDEV', 'SDL_VIDEO_CURSOR_HIDDEN']:
                            if var in os.environ:
                                del os.environ[var]
                        
                        # Fall back to windowed mode
                        flags = pygame.FULLSCREEN if self.fullscreen else 0
                        self.logger.info("Attempting to create windowed display...")
                        self.screen = pygame.display.set_mode((self.width, self.height), flags)
                        self.logger.info("Successfully initialized windowed display")
                else:
                    # Use windowed mode
                    flags = pygame.FULLSCREEN if self.fullscreen else 0
                    self.logger.info("Hardware display disabled, using windowed mode...")
                    self.screen = pygame.display.set_mode((self.width, self.height), flags)
                    self.logger.info("Successfully initialized windowed display")
            
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
            
            self.logger.info("Display initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize display: {e}")
            self.logger.error(traceback.format_exc())
            self._init_fallback_mode()
            return False

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
                        # Handle single click for person name assignment
                        if self.debug_mode == 'video' and self.object_detector is not None:  # Add check for object_detector
                            try:
                                detections = self.object_detector.get_latest_detections()
                                if detections:
                                    for detection in detections:
                                        x1, y1, x2, y2 = detection['box']
                                        if (x1 <= current_pos[0] <= x2 and 
                                            y1 <= current_pos[1] <= y2 and 
                                            detection['label'] == 'person'):
                                            # Prompt for name
                                            name = self._prompt_for_name()
                                            if name:
                                                detection['name'] = name
                                                handled = True
                            except Exception as e:
                                self.logger.error(f"Error getting detections: {e}", exc_info=True)
                        # Update last click time and position
                        self.last_click_time = current_time
                        self.last_click_pos = current_pos
        
        return handled

    def handle_event(self, event):
        """Handle a Pygame event."""
        try:
            if event.type == pygame.QUIT:
                self.running = False
                return True
            
            elif event.type == pygame.KEYDOWN:
                # Handle camera rotation
                if event.key == pygame.K_RIGHT:  # Rotate right
                    self.rotation = (self.rotation + 90) % 360
                    logger.info(f"Camera rotated to {self.rotation}°")
                    return True
                elif event.key == pygame.K_LEFT:  # Rotate left
                    self.rotation = (self.rotation - 90) % 360
                    logger.info(f"Camera rotated to {self.rotation}°")
                    return True
                elif event.key == pygame.K_s and not (event.mod & pygame.KMOD_CTRL):  # Save rotation
                    self._save_rotation()
                    return True
                
                # Handle other key events
                if event.key == pygame.K_ESCAPE:
                    self.debug_mode = None
                    return True
                elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                    self.debug_mode = None
                    return True
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                if self.debug_mode == 'video':
                    # Check rotation controls
                    if hasattr(self, '_debug_controls'):
                        pos = event.pos
                        if self._debug_controls['left_button'].collidepoint(pos):
                            self.rotation = (self.rotation - 90) % 360
                            logger.info(f"Camera rotated to {self.rotation}°")
                            return True
                        elif self._debug_controls['right_button'].collidepoint(pos):
                            self.rotation = (self.rotation + 90) % 360
                            logger.info(f"Camera rotated to {self.rotation}°")
                            return True
                        elif self._debug_controls['save_button'].collidepoint(pos):
                            self._save_rotation()
                            return True
                    
                    # Check for person detection and training
                    if self.object_detector is not None:
                        try:
                            detections = self.object_detector.get_latest_detections()
                            if detections:
                                for detection in detections:
                                    # Check if click is on a person detection
                                    if detection.get('label', '').lower() == 'person':
                                        # Check if click is on the detection box
                                        if 'box' in detection:
                                            x1, y1, x2, y2 = detection['box']
                                            if (x1 <= event.pos[0] <= x2 and 
                                                y1 <= event.pos[1] <= y2):
                                                # Check if click is on train button
                                                if 'train_button' in detection and detection['train_button'].collidepoint(event.pos):
                                                    self._train_person(detection)
                                                    return True
                                                # Otherwise prompt for name
                                                else:
                                                    name = self._prompt_for_name()
                                                    if name:
                                                        detection['name'] = name
                                                        return True
                        except Exception as e:
                            logger.error(f"Error handling detection click: {e}", exc_info=True)
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling event: {e}", exc_info=True)
            return False

    def _train_person(self, detection):
        """Train EVE to recognize a person."""
        try:
            if not self.camera:
                logger.warning("No camera available for training")
                return
            
            # Get current frame
            if isinstance(self.camera, RPiAICamera):
                frame = self.camera.get_latest_frame()
            else:
                _, frame = self.camera.get_frame()
            
            if frame is None:
                logger.warning("No frame available for training")
                return
            
            # Extract face region
            x1, y1, x2, y2 = detection['box']
            face_region = frame[y1:y2, x1:x2]
            
            # TODO: Implement face training logic here
            # This would involve:
            # 1. Extracting face features
            # 2. Storing the features with the person's name
            # 3. Updating the face recognition system
            
            logger.info(f"Training initiated for person in box {detection['box']}")
            
        except Exception as e:
            logger.error(f"Error training person: {e}", exc_info=True)

    def update(self, display_state=None):
        """Update the display.
        
        Args:
            display_state: Optional display state information from the orchestrator
        """
        try:
            # Process any pending events
            handled = self._process_events()
            
            # Check for blink requests from blink thread
            try:
                while not self._blink_queue.empty():
                    action, _ = self._blink_queue.get_nowait()
                    if action == 'blink':
                        self._perform_blink()
            except queue.Empty:
                pass
            
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Handle different display modes
            if self.debug_mode == 'video':
                # Draw video debug info
                self._update_video_debug()
            elif self.debug_mode == 'audio':
                # Draw audio debug info
                self._update_audio_debug()
            else:
                # Normal display mode
                if self.current_emotion in self.emotion_images:
                    image = self.emotion_images[self.current_emotion]
                    
                    # Handle image rotation if needed
                    if self.rotation != 0:
                        image = pygame.transform.rotate(image, self.rotation)
                    
                    # Draw the emotion image
                    rect = image.get_rect(center=self.screen.get_rect().center)
                    self.screen.blit(image, rect)
            
            # Draw debug menu if enabled
            if self.debug_mode:
                self._draw_debug_menu(display_state)
            
            # Update display
            pygame.display.flip()
            
            # Save current frame if path is set
            if self.current_frame_path:
                pygame.image.save(self.screen, self.current_frame_path)
                self.current_frame_path = None
            
            # Update FPS counter
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = current_time
                self.logger.debug(f"FPS: {self.current_fps}")
            
            # Cap frame rate
            self.clock.tick(self.fps)
            
            # Force display update if events were handled
            if handled:
                pygame.display.flip()
            
        except Exception as e:
            self.logger.error(f"Error updating display: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _show_debug_mode_menu(self):
        """Show the debug mode selection menu."""
        try:
            self.screen.fill(self.background_color)
            
            # Draw menu title
            title = self.font.render("Debug Mode Selection", True, (255, 255, 255))
            title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
            self.screen.blit(title, title_rect)
            
            # Draw options
            video_option = self.font.render("1. Video Debug (Object Detection)", True, (255, 255, 255))
            audio_option = self.font.render("2. Audio Debug (Voice Detection)", True, (255, 255, 255))
            exit_option = self.font.render("3. Exit Menu", True, (255, 255, 255))
            
            video_rect = video_option.get_rect(center=(self.width // 2, self.height // 2))
            audio_rect = audio_option.get_rect(center=(self.width // 2, self.height // 2 + 40))
            exit_rect = exit_option.get_rect(center=(self.width // 2, self.height // 2 + 80))
            
            self.screen.blit(video_option, video_rect)
            self.screen.blit(audio_option, audio_rect)
            self.screen.blit(exit_option, exit_rect)
            
            # Update display
            if not self.headless_mode:
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
                            self.logger.info("Selected Video Debug Mode")
                            waiting = False
                        elif event.key == pygame.K_2:
                            self.debug_mode = 'audio'
                            self.logger.info("Selected Audio Debug Mode")
                            waiting = False
                        elif event.key == pygame.K_3 or event.key == pygame.K_ESCAPE:
                            self.debug_mode = None
                            self.logger.info("Exiting Debug Menu")
                            waiting = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Check if click is within the video option area
                        if video_rect.collidepoint(event.pos):
                            self.debug_mode = 'video'
                            self.logger.info("Selected Video Debug Mode")
                            waiting = False
                        # Check if click is within the audio option area
                        elif audio_rect.collidepoint(event.pos):
                            self.debug_mode = 'audio'
                            self.logger.info("Selected Audio Debug Mode")
                            waiting = False
                        # Check if click is within the exit option area
                        elif exit_rect.collidepoint(event.pos):
                            self.debug_mode = None
                            self.logger.info("Exiting Debug Menu")
                            waiting = False
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
            
            # Update the display based on the selected debug mode
            if self.debug_mode == 'video':
                self._update_video_debug()
            elif self.debug_mode == 'audio':
                self._update_audio_debug()
            else:
                # Return to normal display
                self.update()
                
        except Exception as e:
            self.logger.error(f"Error showing debug mode menu: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _draw_debug_menu(self, display_state=None):
        """Draw the debug menu on the screen.
        
        Args:
            display_state: Optional display state information from the orchestrator
        """
        font = pygame.font.Font(None, self.debug_font_size)
        
        # Draw FPS counter with background
        fps_text = f"FPS: {self.current_fps}"
        fps_surface = font.render(fps_text, True, (255, 255, 255))
        fps_rect = fps_surface.get_rect(topleft=(10, 10))
        
        # Draw semi-transparent background for FPS
        fps_bg = pygame.Surface((fps_rect.width + 10, fps_rect.height + 5))
        fps_bg.fill((0, 0, 0))
        fps_bg.set_alpha(128)
        self.screen.blit(fps_bg, (fps_rect.left - 5, fps_rect.top - 2))
        self.screen.blit(fps_surface, fps_rect)
        
        # Draw listening status with more visible formatting
        is_listening = False
        if display_state and hasattr(display_state, 'is_listening'):
            is_listening = display_state.is_listening
        elif hasattr(self, 'is_listening'):
            is_listening = self.is_listening
            
        listening_text = "Listening: YES" if is_listening else "Listening: NO"
        listening_color = (0, 255, 0) if is_listening else (255, 0, 0)  # Green for Yes, Red for No
        text_surface = font.render(listening_text, True, listening_color)
        text_rect = text_surface.get_rect(topleft=(10, 40))
        
        # Draw semi-transparent background for listening status
        listening_bg = pygame.Surface((text_rect.width + 10, text_rect.height + 5))
        listening_bg.fill((0, 0, 0))
        listening_bg.set_alpha(128)
        self.screen.blit(listening_bg, (text_rect.left - 5, text_rect.top - 2))
        self.screen.blit(text_surface, text_rect)
        
        # Draw keyboard shortcuts
        shortcuts = [
            "CTRL+C: Exit",
            "CTRL+S: Toggle Debug",
            "ESC: Exit"
        ]
        
        for i, shortcut in enumerate(shortcuts):
            shortcut_surface = font.render(shortcut, True, (200, 200, 200))
            shortcut_rect = shortcut_surface.get_rect(topleft=(10, 70 + i * 25))
            
            # Draw semi-transparent background for shortcuts
            shortcut_bg = pygame.Surface((shortcut_rect.width + 10, shortcut_rect.height + 5))
            shortcut_bg.fill((0, 0, 0))
            shortcut_bg.set_alpha(128)
            self.screen.blit(shortcut_bg, (shortcut_rect.left - 5, shortcut_rect.top - 2))
            self.screen.blit(shortcut_surface, shortcut_rect)

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

    def _start_blink_thread(self):
        """Start the blink animation thread."""
        try:
            self._blink_thread = threading.Thread(target=self._blink_loop, daemon=True)
            self._blink_thread.start()
            self.logger.info("Blink thread started")
        except Exception as e:
            self.logger.error(f"Failed to start blink thread: {e}")
            self.logger.error(traceback.format_exc())

    def _blink_loop(self):
        """Handle blinking animation in a separate thread."""
        try:
            while self.running and not self._blink_stop_event.is_set():
                try:
                    # Skip blinking if in debug mode
                    if self.debug_mode is not None:
                        time.sleep(0.1)  # Small sleep to prevent CPU hogging
                        continue
                        
                    # Signal main thread to perform blink
                    self._blink_queue.put(('blink', None))
                    # Wait for next blink
                    time.sleep(random.uniform(2.0, 5.0))
                except Exception as e:
                    self.logger.error(f"Error in blink loop: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    time.sleep(1.0)  # Prevent rapid error loops
        except Exception as e:
            self.logger.error(f"Fatal error in blink loop: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("Blink loop stopped")

    def _perform_blink(self):
        """Perform a single blink animation."""
        try:
            # Save current emotion
            current_emotion = self.current_emotion
            
            # Set blink emotion
            self.current_emotion = Emotion.BLINK
            
            # Draw blink emotion
            self._draw_current_emotion()
            
            # Wait for blink duration
            time.sleep(self.blink_duration)
            
            # Restore previous emotion
            self.current_emotion = current_emotion
            
            # Draw restored emotion
            self._draw_current_emotion()
            
        except Exception as e:
            self.logger.error(f"Error in blink animation: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _draw_current_emotion(self):
        """Draw the current emotion on the screen."""
        try:
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Get current emotion image
            emotion_image = self.emotion_images.get(self.current_emotion)
            if emotion_image:
                # Draw emotion image
                self.screen.blit(emotion_image, (0, 0))
                
                # Draw debug info if enabled
                if self.debug_mode:
                    self._draw_debug_info()
            
            # Update display
            if not self.headless_mode:
                pygame.display.flip()
                
        except Exception as e:
            self.logger.error(f"Error drawing emotion: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _stop_blink(self):
        """Stop the blink animation thread."""
        try:
            if hasattr(self, '_blink_stop_event'):
                self._blink_stop_event.set()
            if hasattr(self, '_blink_thread') and self._blink_thread.is_alive():
                self._blink_thread.join(timeout=1.0)
                if self._blink_thread.is_alive():
                    self.logger.warning("Blink thread did not stop gracefully")
        except Exception as e:
            self.logger.error(f"Error stopping blink thread: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _update_video_debug(self):
        """Update the display with video debug information."""
        try:
            if not self.camera:
                logger.warning("No camera available for video debug")
                return

            # Get the latest frame based on camera type
            if isinstance(self.camera, RPiAICamera):
                frame = self.camera.get_latest_frame()
            else:
                _, frame = self.camera.get_frame()

            if frame is None:
                logger.warning("No frame available for video debug")
                return

            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype == np.uint8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = (frame * 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply rotation if needed
            if self.rotation != 0:
                if self.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Calculate video display area (centered, 80% of screen width)
            video_width = int(self.width * 0.8)
            video_height = int(video_width * frame.shape[0] / frame.shape[1])
            video_x = (self.width - video_width) // 2
            video_y = (self.height - video_height) // 2

            # Resize frame to fit display area
            frame = cv2.resize(frame, (video_width, video_height))

            # Convert to Pygame surface
            frame_surface = pygame.surfarray.make_surface(frame)

            # Clear screen
            self.screen.fill((0, 0, 0))

            # Draw video frame
            self.screen.blit(frame_surface, (video_x, video_y))

            # Draw object detection boxes if available
            if self.object_detector is not None:
                try:
                    detections = self.object_detector.get_latest_detections()
                    if detections:
                        for detection in detections:
                            # Get box coordinates - handle both 'box' and 'bbox' formats
                            if 'box' in detection:
                                x1, y1, x2, y2 = detection['box']
                            elif 'bbox' in detection:
                                x1, y1, x2, y2 = detection['bbox']
                            else:
                                logger.warning(f"Invalid detection format: {detection}")
                                continue
                            
                            # Scale coordinates to match video display area
                            x1 = int(x1 * video_width / frame.shape[1]) + video_x
                            y1 = int(y1 * video_height / frame.shape[0]) + video_y
                            x2 = int(x2 * video_width / frame.shape[1]) + video_x
                            y2 = int(y2 * video_height / frame.shape[0]) + video_y
                            
                            # Get label and confidence
                            confidence = detection.get('confidence', 0.0)
                            label = detection.get('label', 'unknown')
                            if 'class' in detection:
                                label = detection['class']
                            
                            # Draw box
                            pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)
                            
                            # Draw label and confidence
                            text = f"{label}: {confidence:.2f}"
                            if 'name' in detection:
                                text = f"{detection['name']} ({text})"
                            text_surface = self.font.render(text, True, (0, 255, 0))
                            self.screen.blit(text_surface, (x1, y1 - 20))

                            # Draw training button for person detections
                            if label.lower() == 'person' and not detection.get('name'):
                                train_button = pygame.Rect(x2 + 5, y1, 80, 30)
                                pygame.draw.rect(self.screen, (255, 165, 0), train_button)
                                train_text = self.font.render("Train", True, (0, 0, 0))
                                self.screen.blit(train_text, (train_button.x + 10, train_button.y + 5))
                                # Store button rect for click detection
                                detection['train_button'] = train_button
                except Exception as e:
                    logger.error(f"Error drawing detections: {e}", exc_info=True)

            # Draw rotation controls panel on the right
            panel_width = self.width - video_width - video_x
            panel_x = video_x + video_width + 10
            panel_y = video_y
            panel_height = video_height

            # Draw panel background
            pygame.draw.rect(self.screen, (40, 40, 40), (panel_x, panel_y, panel_width, panel_height))

            # Draw rotation controls
            control_y = panel_y + 20
            control_spacing = 50

            # Title
            title = self.font.render("Camera Controls", True, (255, 255, 255))
            self.screen.blit(title, (panel_x + 10, control_y))
            control_y += control_spacing

            # Current rotation
            rotation_text = self.font.render(f"Rotation: {self.rotation}°", True, (255, 255, 255))
            self.screen.blit(rotation_text, (panel_x + 10, control_y))
            control_y += control_spacing

            # Rotation buttons
            left_button = pygame.Rect(panel_x + 10, control_y, 40, 40)
            right_button = pygame.Rect(panel_x + 60, control_y, 40, 40)
            pygame.draw.rect(self.screen, (100, 100, 100), left_button)
            pygame.draw.rect(self.screen, (100, 100, 100), right_button)
            
            # Draw arrow symbols
            left_arrow = self.font.render("←", True, (255, 255, 255))
            right_arrow = self.font.render("→", True, (255, 255, 255))
            self.screen.blit(left_arrow, (left_button.x + 10, left_button.y + 5))
            self.screen.blit(right_arrow, (right_button.x + 10, right_button.y + 5))
            control_y += control_spacing

            # Save button
            save_button = pygame.Rect(panel_x + 10, control_y, 90, 40)
            pygame.draw.rect(self.screen, (0, 200, 0), save_button)
            save_text = self.font.render("Save", True, (0, 0, 0))
            self.screen.blit(save_text, (save_button.x + 20, save_button.y + 10))

            # Store button rects for click detection
            self._debug_controls = {
                'left_button': left_button,
                'right_button': right_button,
                'save_button': save_button
            }

            # Draw instructions
            instructions = [
                "Click on person to assign name",
                "Press 'Train' to learn face",
                "Use arrows to rotate",
                "Press 'Save' to keep rotation"
            ]
            
            control_y += control_spacing
            for instruction in instructions:
                text = self.font.render(instruction, True, (200, 200, 200))
                self.screen.blit(text, (panel_x + 10, control_y))
                control_y += 30

            # Update display
            pygame.display.flip()

        except Exception as e:
            logger.error(f"Error updating video debug: {e}", exc_info=True)

    def _draw_debug_overlay(self):
        """Draw debug information overlay."""
        # Draw FPS counter
        fps_text = f"FPS: {self.current_fps}"
        fps_surface = self.font.render(fps_text, True, (255, 255, 255))
        self.screen.blit(fps_surface, (10, 10))
        
        # Draw camera rotation info
        rotation_text = f"Rotation: {self.rotation}°"
        rotation_surface = self.font.render(rotation_text, True, (255, 255, 255))
        self.screen.blit(rotation_surface, (10, 40))
        
        # Draw instructions
        instructions = [
            "ESC: Exit debug mode",
            "Double-click: Toggle debug menu",
            "Click on person: Assign name",
            "CTRL+S: Toggle debug mode",
            "←/→: Rotate camera",
            "S: Save rotation"
        ]
        
        y = 70
        for instruction in instructions:
            text_surface = self.font.render(instruction, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y))
            y += 20

    def _update_audio_debug(self):
        """Update the audio debug display."""
        try:
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Draw title
            title = self.font.render("Audio Debug Mode", True, (255, 255, 255))
            title_rect = title.get_rect(center=(self.width // 2, 30))
            self.screen.blit(title, title_rect)
            
            # Draw instructions
            instructions = [
                "Press ESC to exit debug mode",
                "Speak to see audio levels",
                "Double-click to toggle debug menu"
            ]
            
            y = 60
            for instruction in instructions:
                text = self.font.render(instruction, True, (200, 200, 200))
                text_rect = text.get_rect(center=(self.width // 2, y))
                self.screen.blit(text, text_rect)
                y += 30
            
            # Draw current emotion
            emotion_text = self.font.render(f"Current Emotion: {self.current_emotion}", True, (255, 255, 255))
            emotion_rect = emotion_text.get_rect(center=(self.width // 2, y + 30))
            self.screen.blit(emotion_text, emotion_rect)
            
            # Draw FPS
            fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 255))
            fps_rect = fps_text.get_rect(center=(self.width // 2, y + 60))
            self.screen.blit(fps_text, fps_rect)
            
            # Update display
            if not self.headless_mode:
                pygame.display.flip()
                
        except Exception as e:
            self.logger.error(f"Error updating audio debug: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _draw_debug_info(self):
        """Draw debug information on the screen."""
        # Implementation of _draw_debug_info method
        pass

    def _blink_stop_event(self):
        """Stop the blink animation."""
        self._blink_stop_event.set()

    def _prompt_for_name(self):
        """Show a prompt for entering a person's name."""
        try:
            # Create input box
            input_box = pygame.Rect(self.width//2 - 100, self.height//2 - 15, 200, 30)
            color_inactive = pygame.Color('lightskyblue3')
            color_active = pygame.Color('dodgerblue2')
            color = color_inactive
            active = False
            text = ''
            done = False
            
            while not done and self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        active = input_box.collidepoint(event.pos)
                        color = color_active if active else color_inactive
                    if event.type == pygame.KEYDOWN:
                        if active:
                            if event.key == pygame.K_RETURN:
                                done = True
                            elif event.key == pygame.K_BACKSPACE:
                                text = text[:-1]
                            else:
                                text += event.unicode
                
                # Draw input box
                self.screen.fill(self.background_color)
                txt_surface = self.font.render("Enter name:", True, (255, 255, 255))
                self.screen.blit(txt_surface, (input_box.x, input_box.y - 30))
                pygame.draw.rect(self.screen, color, input_box, 2)
                txt_surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
                pygame.display.flip()
            
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Error in name prompt: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def set_camera(self, camera):
        """Set the camera reference for video debug mode."""
        self.camera = camera
        self.logger.info("Camera reference set for video debug mode")

    def set_object_detector(self, detector):
        """Set the object detector reference for video debug mode."""
        self.object_detector = detector
        self.logger.info("Object detector reference set for video debug mode") 