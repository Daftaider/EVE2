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
import cv2

from eve import config
from eve.config.display import Emotion, DisplayConfig

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
    """
    LCD display controller for rendering eye animations.
    
    This class handles the initialization of the display and
    manages the rendering of emotive eye animations.
    """
    
    def __init__(self, 
                 config: Optional[DisplayConfig] = None, 
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 fps: Optional[int] = None,
                 default_emotion: Optional[Union[str, int, Emotion]] = None,
                 background_color: Optional[Union[Tuple[int, int, int], str]] = None,
                 eye_color: Optional[Union[Tuple[int, int, int], str]] = None):
        """
        Initialize the LCD Controller.
        
        Args:
            config: Display configuration object
            width: Optional window width (overrides config)
            height: Optional window height (overrides config)
            fps: Optional frames per second (overrides config)
            default_emotion: Optional starting emotion (overrides config)
            background_color: Optional background color as RGB tuple or string (default: black)
            eye_color: Optional eye color as RGB tuple or string (default: white)
        """
        # Set environment variables for display
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        
        self.config = config or DisplayConfig
        
        # Override config values if parameters are provided
        if width is not None and height is not None:
            self.window_size = (width, height)
        else:
            self.window_size = self.config.WINDOW_SIZE
            
        self.fps = fps if fps is not None else self.config.FPS
        
        # Convert and validate emotion using the from_value method
        self._current_emotion = Emotion.from_value(default_emotion)
        
        # Handle colors
        self.background_color = self._parse_color(background_color) if background_color else (0, 0, 0)
        self.eye_color = self._parse_color(eye_color) if eye_color else (255, 255, 255)
        
        # Initialize display system
        self._init_display()
        
        # Font for debug text
        self.debug_font = pygame.font.SysFont(None, 36) # Smaller font for more info
        self.debug_ui_elements: Dict[str, pygame.Rect] = {} # Store clickable UI elements {id: rect}
        
        # Log initialization parameters
        logging.info(f"LCD Controller initialized with: size={self.window_size}, "
                    f"fps={self.fps}, default_emotion={self._current_emotion.name}, "
                    f"background_color={self.background_color}, "
                    f"eye_color={self.eye_color}")

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
                logging.warning(f"Invalid color string: {color}, using default")
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
            self.screen = pygame.display.set_mode(
                self.window_size,
                pygame.SHOWN | (pygame.FULLSCREEN if getattr(self.config, 'FULLSCREEN', False) else 0)
            )
            
            # Initialize clock
            self.clock = pygame.time.Clock()
            
            # Load images
            self._load_emotion_images()
            
            # Initial display update
            self.screen.fill(self.background_color)
            pygame.display.flip()
            
            logging.info("Display initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize display: {e}")
            self._init_fallback_mode()

    def _init_fallback_mode(self):
        """Initialize a fallback mode for headless operation."""
        logging.info("Initializing display in fallback mode")
        pygame.init()
        self.screen = pygame.Surface(self.window_size)
        self.clock = pygame.time.Clock()
        self._load_emotion_images()

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
        for emotion in Emotion:
            try:
                image_path = self.config.get_emotion_path(emotion)
                original = pygame.image.load(image_path)
                
                # Scale image if needed
                if original.get_size() != self.window_size:
                    original = pygame.transform.scale(original, self.window_size)
                
                # Apply eye color
                colored = self._apply_eye_color(original, self.eye_color)
                self.emotion_images[emotion] = colored
                
            except Exception as e:
                logging.warning(f"Failed to load emotion image for {emotion}: {e}")
                # Create a fallback surface
                surface = pygame.Surface(self.window_size)
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
            Emotion.CONFUSED: (128, 0, 128),     # Purple
        }
        return colors.get(emotion, (0, 0, 0))

    def update(self, 
               emotion: Optional[Emotion] = None, 
               debug_menu_active: bool = False,
               current_debug_view: Optional[str] = None,
               debug_frame: Optional[np.ndarray] = None,
               detections: Optional[List[Dict[str, Any]]] = None,
               camera_info: Optional[Dict[str, Any]] = None,
               camera_rotation: int = 0,
               is_correcting: bool = False,
               input_buffer: str = "",
               corrections: Optional[List[Dict[str, Any]]] = None,
               # Audio Debug Params
               audio_debug_listen_always: bool = False,
               last_recognized_text: str = "",
               available_audio_devices: Optional[List[Dict]] = None,
               selected_audio_device_index: Optional[int] = None,
               last_audio_rms: float = 0.0,
               # Sensitivity
               current_porcupine_sensitivity: float = 0.5
               ) -> None:
        """Update the display: Normal emotion, debug menu, or specific debug view."""
        # Normal Emotion Mode (if debug menu isn't active)
        if not debug_menu_active:
            if emotion is not None:
                self._current_emotion = emotion
                
                try:
                    self.screen.fill(self.background_color)
                    if self._current_emotion in self.emotion_images:
                        self.screen.blit(self.emotion_images[self._current_emotion], (0, 0))
                    pygame.display.flip()
                except Exception as e:
                    logging.error(f"Error updating display (normal mode): {e}")
                return # Exit early if not in debug mode
            
        # --- Debug Mode Active --- 
        try:
            self.screen.fill(self.background_color) # Black background for debug
            self.debug_ui_elements.clear() 
            
            # Determine what to draw based on current_debug_view
            # --- A. Main Debug Menu --- 
            if current_debug_view is None:
                 title_font = pygame.font.SysFont(None, 48)
                 title_surf = title_font.render("Debug Menu", True, (255, 255, 255))
                 title_rect = title_surf.get_rect(centerx=self.screen.get_rect().centerx, top=50)
                 self.screen.blit(title_surf, title_rect)
                 
                 button_y = title_rect.bottom + 50
                 button_font = self.debug_font # Use existing debug font
                 
                 # Video Button
                 vid_text = button_font.render("[ Video Capture ]", True, (255, 255, 255), (50, 50, 50))
                 vid_rect = vid_text.get_rect(centerx=self.screen.get_rect().centerx, top=button_y)
                 self.screen.blit(vid_text, vid_rect)
                 self.debug_ui_elements["menu_video"] = vid_rect
                 button_y += vid_rect.height + 20
                 
                 # Audio Button
                 aud_text = button_font.render("[ Audio Capture ]", True, (255, 255, 255), (50, 50, 50))
                 aud_rect = aud_text.get_rect(centerx=self.screen.get_rect().centerx, top=button_y)
                 self.screen.blit(aud_text, aud_rect)
                 self.debug_ui_elements["menu_audio"] = aud_rect
                 button_y += aud_rect.height + 20
                 
                 # Add more buttons here later...

            # --- B. Video Debug View --- 
            elif current_debug_view == 'VIDEO':
                 camera_area_rect = self.screen.get_rect() 
                 ui_panel_height = 80 
                 camera_area_rect.height -= ui_panel_height

                 if debug_frame is not None:
                     try:
                         backend = camera_info.get('backend') if camera_info else 'opencv'
                         if backend == 'opencv':
                             frame_rgb = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                         else: 
                             frame_rgb = debug_frame 
                         frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                         scaled_surface = pygame.transform.smoothscale(frame_surface, camera_area_rect.size)
                         self.screen.blit(scaled_surface, camera_area_rect.topleft)
                         if detections:
                             # Scale factor for drawing on scaled surface
                             scale_x = camera_area_rect.width / frame_rgb.shape[1]
                             scale_y = camera_area_rect.height / frame_rgb.shape[0]
                             
                             for det in detections:
                                 box = det['box']
                                 original_label = det['label'] # Get the label from the model
                                 confidence = det['confidence']
                                 det_index = det['index']
                                 
                                 # --- Check for Corrections --- 
                                 display_label = original_label
                                 is_corrected = False
                                 if corrections:
                                      for correction in reversed(corrections): # Check newest first
                                           # Check if original label matches and boxes overlap sufficiently
                                           if correction['original'] == original_label and \
                                              calculate_iou(box, correction['box']) > IOU_THRESHOLD:
                                                 display_label = correction['corrected']
                                                 is_corrected = True
                                                 logger.debug(f"Applying correction: {original_label} -> {display_label} for box {box}")
                                                 break # Apply the most recent matching correction
                                 # ---------------------------

                                 # Scale box coordinates to the display rect
                                 startX = int(box[0] * scale_x) + camera_area_rect.left
                                 startY = int(box[1] * scale_y) + camera_area_rect.top
                                 endX = int(box[2] * scale_x) + camera_area_rect.left
                                 endY = int(box[3] * scale_y) + camera_area_rect.top
                                 
                                 # Draw bounding box (maybe change color if corrected?)
                                 box_color = (255, 165, 0) if is_corrected else (0, 255, 0) # Orange if corrected, Green otherwise
                                 pygame.draw.rect(self.screen, box_color, (startX, startY, endX - startX, endY - startY), 2)
                                 
                                 # Draw label text (use display_label)
                                 label_text = f"{display_label}: {confidence:.2f}{' (C)' if is_corrected else ''}"
                                 text_surface = self.debug_font.render(label_text, True, box_color)
                                 text_offset = 5 # Small offset from box line
                                 text_y = startY - text_surface.get_height() - text_offset if startY - text_surface.get_height() - text_offset > camera_area_rect.top else startY + text_offset
                                 text_x = startX + text_offset
                                 self.screen.blit(text_surface, (text_x, text_y))
                                 
                                 # Draw "Fix?" button only if NOT currently correcting
                                 if not is_correcting:
                                     fix_text_surface = self.debug_font.render("[Fix?]", True, (255, 255, 0)) # Yellow
                                     fix_rect = fix_text_surface.get_rect(left=text_x + text_surface.get_width() + 10, top=text_y)
                                     self.screen.blit(fix_text_surface, fix_rect)
                                     self.debug_ui_elements[f"correct_det_{det_index}"] = fix_rect
                     except Exception as e:
                         logger.error(f"Error processing debug frame or detections: {e}", exc_info=True)
                 else:
                     # Show placeholder text if no frame
                     selected_idx_txt = camera_info.get('selected_index', '?') if camera_info else '?'
                     backend_txt = camera_info.get('backend', 'Unknown') if camera_info else 'Unknown'
                     no_feed_text = self.debug_font.render(f"No Feed ({backend_txt} idx:{selected_idx_txt})", True, (255, 255, 0))
                     no_feed_rect = no_feed_text.get_rect(center=camera_area_rect.center)
                     self.screen.blit(no_feed_text, no_feed_rect)

                 # Draw Correction Prompt (if correcting) - POSITIONED FOR VIDEO VIEW
                 if is_correcting:
                    prompt_text = f"Enter correct label: {input_buffer}_" 
                    prompt_surface = self.debug_font.render(prompt_text, True, (255, 255, 255), (0, 0, 0)) # White text on black background
                    prompt_rect = prompt_surface.get_rect(centerx=self.screen.get_rect().centerx, bottom=camera_area_rect.bottom - 10) # Position above UI panel
                    pygame.draw.rect(self.screen, (0, 0, 0), prompt_rect.inflate(10, 10)) # Black background rect
                    self.screen.blit(prompt_surface, prompt_rect)
                    # Maybe highlight the box being corrected?
                    # target_info = self.orchestrator... (need access or pass info)

                 # Draw UI Panel (Camera/Rotation Controls)
                 ui_panel_rect = pygame.Rect(0, camera_area_rect.bottom, self.window_size[0], ui_panel_height)
                 pygame.draw.rect(self.screen, (40, 40, 40), ui_panel_rect)
                 
                 current_x = 10
                 current_y = ui_panel_rect.top + 10
                 line_height = 30

                 # -- Camera Selection UI --
                 # Use info from camera_info dict
                 backend = camera_info.get('backend') if camera_info else 'N/A'
                 selected_index = camera_info.get('selected_index') if camera_info else -1
                 available_picam2 = camera_info.get('available_picam2', []) if camera_info else []
                 available_opencv = camera_info.get('available_opencv', []) if camera_info else []

                 cam_label = self.debug_font.render(f"Camera ({backend}):", True, (200, 200, 200))
                 self.screen.blit(cam_label, (current_x, current_y))
                 current_x += cam_label.get_width() + 10

                 # Display based on backend
                 if backend == 'picamera2':
                     if available_picam2:
                          for i, cam_info in enumerate(available_picam2):
                             is_selected = (i == selected_index)
                             text_color = (0, 255, 0) if is_selected else (255, 255, 255)
                             bg_color = (60, 60, 60) if is_selected else None
                             # Display index and maybe model?
                             cam_text_str = f"[{i}:{cam_info.get('Model', 'Unknown')[:10]}]"
                             cam_text = self.debug_font.render(cam_text_str, True, text_color, bg_color)
                             cam_rect = cam_text.get_rect(topleft=(current_x, current_y))
                             self.screen.blit(cam_text, cam_rect)
                             self.debug_ui_elements[f"select_cam_{i}"] = cam_rect # ID still based on list index
                             current_x += cam_rect.width + 15
                     else:
                          no_cam_text = self.debug_font.render("None found", True, (255, 100, 100))
                          self.screen.blit(no_cam_text, (current_x, current_y))
                          current_x += no_cam_text.get_width() + 15
                 elif backend == 'opencv':
                     if available_opencv:
                         for idx in available_opencv:
                             is_selected = (idx == selected_index)
                             text_color = (0, 255, 0) if is_selected else (255, 255, 255)
                             bg_color = (60, 60, 60) if is_selected else None
                             cam_text = self.debug_font.render(f"[{idx}]", True, text_color, bg_color)
                             cam_rect = cam_text.get_rect(topleft=(current_x, current_y))
                             self.screen.blit(cam_text, cam_rect)
                             self.debug_ui_elements[f"select_cam_{idx}"] = cam_rect # ID based on OpenCV index
                             current_x += cam_rect.width + 15
                     else:
                         no_cam_text = self.debug_font.render("None found", True, (255, 100, 100))
                         self.screen.blit(no_cam_text, (current_x, current_y))
                         current_x += no_cam_text.get_width() + 15
                 else: # Backend unknown or N/A
                     no_cam_text = self.debug_font.render("N/A", True, (150, 150, 150))
                     self.screen.blit(no_cam_text, (current_x, current_y))
                     current_x += no_cam_text.get_width() + 15

                 # -- Rotation Selection UI --
                 current_y += line_height # Move to next line
                 current_x = 10
                 rot_label = self.debug_font.render("Rotation:", True, (200, 200, 200))
                 self.screen.blit(rot_label, (current_x, current_y))
                 current_x += rot_label.get_width() + 10

                 for angle in [0, 90, 180, 270]:
                     is_selected = (angle == camera_rotation)
                     text_color = (0, 255, 0) if is_selected else (255, 255, 255)
                     bg_color = (60, 60, 60) if is_selected else None
                     rot_text = self.debug_font.render(f"[{angle}°]", True, text_color, bg_color)
                     rot_rect = rot_text.get_rect(topleft=(current_x, current_y))
                     self.screen.blit(rot_text, rot_rect)
                     # Store clickable area with ID
                     self.debug_ui_elements[f"rotate_{angle}"] = rot_rect
                     current_x += rot_rect.width + 15

                 # ADD A BACK BUTTON to the panel
                 back_text = self.debug_font.render("[< Back]", True, (255, 180, 180), (60, 0, 0))
                 back_rect = back_text.get_rect(right=ui_panel_rect.right - 10, centery=ui_panel_rect.centery)
                 self.screen.blit(back_text, back_rect)
                 self.debug_ui_elements["back_to_menu"] = back_rect

            # --- C. Audio Debug View --- 
            elif current_debug_view == 'AUDIO':
                 title_font = pygame.font.SysFont(None, 48)
                 title_surf = title_font.render("Audio Debug", True, (255, 255, 255))
                 title_rect = title_surf.get_rect(centerx=self.screen.get_rect().centerx, top=50)
                 self.screen.blit(title_surf, title_rect)
                 
                 current_y = title_rect.bottom + 30
                 
                 # --- Input Device Selection --- 
                 dev_label_surf = self.debug_font.render("Input Device:", True, (200, 200, 200))
                 dev_label_rect = dev_label_surf.get_rect(left=10, top=current_y)
                 self.screen.blit(dev_label_surf, dev_label_rect)
                 current_y += dev_label_rect.height + 5
                 current_x = 20
                 
                 # Add Default option
                 is_selected = selected_audio_device_index is None
                 dev_text_str = "[ Default ]"
                 dev_color = (0, 255, 0) if is_selected else (255, 255, 255)
                 dev_bg = (60, 60, 60) if is_selected else None
                 dev_surf = self.debug_font.render(dev_text_str, True, dev_color, dev_bg)
                 dev_rect = dev_surf.get_rect(left=current_x, top=current_y)
                 self.screen.blit(dev_surf, dev_rect)
                 self.debug_ui_elements["select_audio_dev_None"] = dev_rect # ID for default
                 current_x += dev_rect.width + 15

                 if available_audio_devices:
                     for device in available_audio_devices:
                         idx = device['index']
                         name = device['name'][:30] # Truncate name
                         is_selected = (idx == selected_audio_device_index)
                         dev_text_str = f"[ {idx}: {name} ]"
                         dev_color = (0, 255, 0) if is_selected else (255, 255, 255)
                         dev_bg = (60, 60, 60) if is_selected else None
                         dev_surf = self.debug_font.render(dev_text_str, True, dev_color, dev_bg)
                         dev_rect = dev_surf.get_rect(left=current_x, top=current_y)
                         # Prevent overflowing screen width
                         if dev_rect.right > self.window_size[0] - 10:
                              current_x = 20
                              current_y += dev_rect.height + 5
                              dev_rect.topleft = (current_x, current_y)
                              
                         self.screen.blit(dev_surf, dev_rect)
                         self.debug_ui_elements[f"select_audio_dev_{idx}"] = dev_rect
                         current_x += dev_rect.width + 15
                 else:
                      no_dev_surf = self.debug_font.render("(No devices found)", True, (255,100,100))
                      no_dev_rect = no_dev_surf.get_rect(left=current_x, top=current_y)
                      self.screen.blit(no_dev_surf, no_dev_rect)
                      current_x += no_dev_rect.width + 15
                      
                 current_y += 30 # Spacing

                 # --- Sensitivity Control --- 
                 sens_label_surf = self.debug_font.render("Wake Word Sensitivity:", True, (200, 200, 200))
                 sens_label_rect = sens_label_surf.get_rect(left=10, top=current_y)
                 self.screen.blit(sens_label_surf, sens_label_rect)
                 current_y += sens_label_rect.height + 10
                 
                 control_x = 20
                 
                 # Decrease Button
                 minus_surf = self.debug_font_large.render(" - ", True, (255, 255, 255), (80, 0, 0))
                 minus_rect = minus_surf.get_rect(left=control_x, top=current_y)
                 self.screen.blit(minus_surf, minus_rect)
                 self.debug_ui_elements["sensitivity_decrease"] = minus_rect
                 control_x += minus_rect.width + 15
                 
                 # Sensitivity Value Display
                 sens_val_text = f"{current_porcupine_sensitivity:.1f}" # Format to 1 decimal place
                 sens_val_surf = self.debug_font_large.render(sens_val_text, True, (255, 255, 0))
                 sens_val_rect = sens_val_surf.get_rect(left=control_x, centery=minus_rect.centery)
                 self.screen.blit(sens_val_surf, sens_val_rect)
                 control_x += sens_val_rect.width + 15
                 
                 # Increase Button
                 plus_surf = self.debug_font_large.render(" + ", True, (255, 255, 255), (0, 80, 0))
                 plus_rect = plus_surf.get_rect(left=control_x, top=current_y)
                 self.screen.blit(plus_surf, plus_rect)
                 self.debug_ui_elements["sensitivity_increase"] = plus_rect
                 
                 current_y += plus_rect.height + 20 # Spacing after sensitivity

                 # --- Sound Level Meter --- 
                 meter_label_surf = self.debug_font.render("Input Level:", True, (200, 200, 200))
                 meter_label_rect = meter_label_surf.get_rect(left=10, top=current_y)
                 self.screen.blit(meter_label_surf, meter_label_rect)
                 current_y += meter_label_rect.height + 10
                 
                 meter_max_width = self.window_size[0] - 40 # Max width for the bar
                 meter_height = 20
                 meter_x = 20
                 meter_y = current_y
                 
                 # Scale RMS (0.0 to ~0.5 is typical range, clamp for safety)
                 # Adjust scaling factor based on observed values
                 level_fraction = min(1.0, max(0.0, last_audio_rms * 4.0)) 
                 current_meter_width = int(meter_max_width * level_fraction)
                 
                 # Draw background
                 pygame.draw.rect(self.screen, (50, 50, 50), (meter_x, meter_y, meter_max_width, meter_height))
                 # Draw level bar
                 if current_meter_width > 0:
                      pygame.draw.rect(self.screen, (0, 255, 0), (meter_x, meter_y, current_meter_width, meter_height))
                      
                 current_y += meter_height + 20

                 # --- Listen Always Toggle --- 
                 listen_text = f"[ Listen Always: {'ON' if audio_debug_listen_always else 'OFF'} ]"
                 listen_color = (0, 255, 0) if audio_debug_listen_always else (255, 255, 255)
                 listen_bg = (50, 80, 50) if audio_debug_listen_always else (50, 50, 50)
                 listen_surf = self.debug_font.render(listen_text, True, listen_color, listen_bg)
                 listen_rect = listen_surf.get_rect(centerx=self.screen.get_rect().centerx, top=current_y)
                 self.screen.blit(listen_surf, listen_rect)
                 self.debug_ui_elements["audio_toggle_listen"] = listen_rect
                 current_y += listen_rect.height + 30
                 
                 # --- Last Recognized Text --- 
                 recog_label_surf = self.debug_font.render("Last Heard:", True, (200, 200, 200))
                 recog_label_rect = recog_label_surf.get_rect(centerx=self.screen.get_rect().centerx, top=current_y)
                 self.screen.blit(recog_label_surf, recog_label_rect)
                 current_y += recog_label_rect.height + 10
                 
                 recog_text_surf = self.debug_font.render(last_recognized_text or "-", True, (255, 255, 255))
                 recog_text_rect = recog_text_surf.get_rect(centerx=self.screen.get_rect().centerx, top=current_y)
                 # Add background for readability
                 pygame.draw.rect(self.screen, (20, 20, 20), recog_text_rect.inflate(20, 10))
                 self.screen.blit(recog_text_surf, recog_text_rect)
                 
                 # Add Back button
                 back_text = self.debug_font.render("[< Back]", True, (255, 180, 180), (60, 0, 0))
                 back_rect = back_text.get_rect(left=10, bottom=self.screen.get_rect().bottom - 10)
                 self.screen.blit(back_text, back_rect)
                 self.debug_ui_elements["back_to_menu"] = back_rect

            # --- D. Unknown View (Shouldn't happen) ---
            else:
                 err_font = pygame.font.SysFont(None, 48)
                 err_surf = err_font.render(f"Unknown Debug View: {current_debug_view}", True, (255,0,0))
                 err_rect = err_surf.get_rect(center=self.screen.get_rect().center)
                 self.screen.blit(err_surf, err_rect)
                 # Add Back button here too? Maybe.

            # --- Update display for all debug states --- 
            pygame.display.flip()
            
        except Exception as e:
            logging.error(f"Error updating display (debug mode): {e}")

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
        return self._current_emotion

    def set_emotion(self, emotion: Emotion) -> None:
        """Set the current emotion."""
        if not isinstance(emotion, Emotion):
            raise ValueError(f"Expected Emotion enum, got {type(emotion)}")
        self._current_emotion = emotion
        self.update()

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        try:
            pygame.quit()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _render_loop(self):
        """Main rendering loop"""
        last_error_time = 0
        error_cooldown = 5  # seconds between error messages
        
        while self.running:
            try:
                self._render_current_emotion()
                self.clock.tick(self.fps)
            except Exception as e:
                current_time = time.time()
                if current_time - last_error_time > error_cooldown:
                    self.logger.error(f"Error in render loop: {e}")
                    last_error_time = current_time
                time.sleep(0.1)  # Prevent tight error loop

    def _render_current_emotion(self):
        """Render the current emotion to the surface"""
        try:
            # Clear screen
            self.screen.fill(self.background_color)
            
            # Get current emotion surface
            emotion_surface = self.emotion_images.get(
                'blink' if self.is_blinking else self._current_emotion,
                self.emotion_images[Emotion.NEUTRAL]
            )
            
            # Blit emotion onto screen
            self.screen.blit(emotion_surface, (0, 0))
            
            # Save the current frame to a file for external display
            pygame.image.save(self.screen, "current_display.png")
            
        except Exception as e:
            self.logger.error(f"Error rendering emotion: {e}")

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
            except Exception as e:
                self.logger.error(f"Error stopping render thread: {e}")
        
        try:
            pygame.quit()
        except Exception as e:
            self.logger.error(f"Error during pygame cleanup: {e}")
        
        self.logger.info("Display controller stopped")

    def blink(self):
        """Perform a single blink animation"""
        try:
            self.is_blinking = True
            time.sleep(self.blink_duration)
            self.is_blinking = False
            self.logger.debug("Completed blink animation")
        except Exception as e:
            self.logger.error(f"Error during blink animation: {e}")
            self.is_blinking = False 

    print("libcamera imported successfully!") 