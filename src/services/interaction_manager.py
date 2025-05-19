"""
Interaction manager for coordinating all EVE2 services.
"""
import logging
import threading
import queue
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import cv2
import os
import pygame
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

from .eye_display import EyeDisplay, Emotion
from .face_service import FaceService
from .emotion_service import EmotionService
from .voice_synth import VoiceSynth
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class InteractionManager:
    """Manages interaction between all EVE2 services."""
    
    def __init__(self, config_path: str):
        """Initialize the interaction manager."""
        self.config_path = config_path
        self.running = False
        self.services: Dict[str, Any] = {}
        self.picam2: Optional[Picamera2] = None
        self.command_queue = queue.Queue()
        self.thread = None

        # Debug mode attributes
        self.debug_mode: Optional[str] = None  # None, "main_menu", "video_debug", "voice_debug"
        self.last_mouse_click_time: float = 0.0
        self.num_mouse_clicks: int = 0
        self.double_click_interval: float = 0.3  # seconds
        self.debug_font: Optional[pygame.font.Font] = None
        self.debug_buttons: Dict[str, Dict[str, Any]] = {} # Store button rects and actions
        self.debug_rotation_angle: int = 0
        self.debug_input_text: str = ""
        self.debug_active_input_field: Optional[str] = None # e.g. "face_name"
        self.debug_recognized_face_info: Optional[Dict[str, Any]] = None # Store info for naming
        self.debug_message: Optional[str] = None # For displaying status messages in debug UI
        self.debug_message_timer: float = 0.0
        
    def _initialize_debug_font(self):
        if self.debug_font is None:
            try:
                # Pygame needs to be initialized before font can be used.
                # EyeDisplay.start() calls pygame.init(). Ensure it's called before this.
                if pygame.get_init(): # Check if Pygame is initialized
                    self.debug_font = pygame.font.Font(None, 30)  # Default font, size 30
                    if not self.debug_font: # Fallback if Font(None, ..) fails
                         self.debug_font = pygame.font.SysFont('arial', 28)
                else:
                    logger.warning("Pygame not initialized when trying to load debug font.")
            except Exception as e:
                logger.error(f"Failed to load font for debug UI: {e}")
                if pygame.get_init(): # Check again
                    try:
                        self.debug_font = pygame.font.SysFont('arial', 28) # Fallback
                    except Exception as e_sysfont:
                        logger.error(f"Failed to load system font for debug UI: {e_sysfont}")


    def start(self) -> bool:
        """Start all services and begin interaction loop."""
        try:
            self.services = {
                'display': EyeDisplay(self.config_path),
                'face': FaceService(self.config_path),
                'emotion': EmotionService(self.config_path),
                'voice': VoiceSynth(self.config_path),
                'llm': LLMService(self.config_path)
            }
            
            # Start display service first as it initializes Pygame
            if not self.services['display'].start():
                logger.error("Failed to start EyeDisplay service")
                return False
            
            # Initialize debug font after Pygame is initialized by EyeDisplay
            self._initialize_debug_font()

            try:
                logger.info(f"Attempting to initialize Picamera2...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}, # Using RGB888 for direct Pygame use
                    # lores={"size": (320, 240), "format": "YUV420"} # lores stream not strictly needed for this
                )
                self.picam2.configure(config)
                self.picam2.start()
                if not self.picam2.started:
                    logger.error("Failed to start Picamera2")
                    # Continue without camera if other services can run
                else:
                    logger.info("Picamera2 initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Picamera2: {e}. Camera will be unavailable.")
                self.picam2 = None # Ensure picam2 is None if it fails
            
            # Start other services
            for name, service in self.services.items():
                if name == 'display': continue # Already started
                if hasattr(service, 'start') and not service.start(): # Check if service has start method
                    logger.error(f"Failed to start {name} service")
                    # Decide on recovery or abort
            
            self.running = True
            self.thread = threading.Thread(target=self._interaction_loop)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Interaction manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting interaction manager: {e}")
            return False

    def _toggle_debug_mode(self):
        if self.debug_mode is None:
            self.debug_mode = "main_menu"
            logger.info("Debug mode activated: Main Menu")
            self.debug_buttons.clear() # Clear buttons for the new menu
            self.debug_message = None # Clear any previous messages
        else:
            self.debug_mode = None
            logger.info("Debug mode deactivated")
            self.services['display'].set_emotion(Emotion.NEUTRAL) # Reset emotion on exiting debug
        # When toggling, reset any active input field
        self.debug_active_input_field = None
        self.debug_input_text = ""

    def _handle_debug_keydown(self, event: pygame.event.Event):
        if self.debug_active_input_field: # If an input field is active
            if event.key == pygame.K_RETURN:
                if self.debug_active_input_field == "face_name" and self.debug_recognized_face_info:
                    self._submit_debug_face_name()
                self.debug_active_input_field = None # Deactivate after enter
                self.debug_input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.debug_input_text = self.debug_input_text[:-1]
            elif event.key == pygame.K_ESCAPE: # Allow Esc to clear input field
                 self.debug_active_input_field = None
                 self.debug_input_text = ""
            else:
                self.debug_input_text += event.unicode
        # Add other key handlers for debug mode if needed, e.g., for rotation
        elif self.debug_mode == "video_debug":
            if event.key == pygame.K_LEFT:
                self.debug_rotation_angle = (self.debug_rotation_angle + 90) % 360
                logger.info(f"Debug camera rotation set to {self.debug_rotation_angle} degrees")
            elif event.key == pygame.K_RIGHT:
                self.debug_rotation_angle = (self.debug_rotation_angle - 90 + 360) % 360
                logger.info(f"Debug camera rotation set to {self.debug_rotation_angle} degrees")


    def _handle_debug_click(self, mouse_pos: Tuple[int, int]):
        clicked_action = None
        for name, button_info in self.debug_buttons.items():
            if button_info['rect'].collidepoint(mouse_pos):
                clicked_action = button_info['action']
                logger.info(f"Debug button clicked: {name} -> Action: {clicked_action}")
                break
        
        if clicked_action:
            if clicked_action == "exit_debug":
                self._toggle_debug_mode()
            elif clicked_action == "video_debug_menu":
                self.debug_mode = "video_debug"
                self.debug_recognized_face_info = None # Clear previous face info
                self.debug_input_text = ""
            elif clicked_action == "voice_debug_menu":
                self.debug_mode = "voice_debug"
            elif clicked_action == "back_to_main_menu":
                self.debug_mode = "main_menu"
                self.debug_recognized_face_info = None # Clear face info when going back
            elif clicked_action == "rotate_cw":
                self.debug_rotation_angle = (self.debug_rotation_angle + 90) % 360
            elif clicked_action == "rotate_ccw":
                self.debug_rotation_angle = (self.debug_rotation_angle - 90 + 360) % 360
            elif clicked_action == "enrol_face_input":
                 if self.debug_recognized_face_info and self.debug_recognized_face_info.get('roi_for_enrol'):
                    self.debug_active_input_field = "face_name"
                    self.debug_input_text = "" # Clear for new input
                    logger.info("Activated face name input for enrolment.")
                 else:
                    logger.warning("Cannot enrol: No face ROI captured for enrolment.")
                    self.debug_message = "Error: No face ROI to enrol."
                    self.debug_message_timer = time.time() + 3
            # More actions will be added (like submitting enrolment)

    def _render_text(self, screen: pygame.Surface, text: str, pos: Tuple[int, int], color: Tuple[int,int,int] = (255, 255, 255), center_x: bool = False):
        if self.debug_font:
            text_surface = self.debug_font.render(text, True, color)
            text_rect = text_surface.get_rect()
            if center_x:
                text_rect.centerx = pos[0]
                text_rect.top = pos[1]
            else:
                text_rect.topleft = pos
            screen.blit(text_surface, text_rect)
            return text_rect
        return None

    def _render_button(self, screen: pygame.Surface, text: str, y_pos: int, action_name: str, center_x_pos: Optional[int] = None) -> Optional[pygame.Rect]:
        if not self.debug_font: return None
        
        screen_width = screen.get_width()
        if center_x_pos is None:
            center_x_pos = screen_width // 2

        text_surface = self.debug_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(center_x_pos, y_pos))
        
        button_padding = 10
        button_rect = pygame.Rect(text_rect.left - button_padding, text_rect.top - button_padding,
                                  text_rect.width + 2 * button_padding, text_rect.height + 2 * button_padding)
        
        pygame.draw.rect(screen, (80, 80, 80), button_rect) # Button background
        pygame.draw.rect(screen, (150, 150, 150), button_rect, 2) # Button border
        screen.blit(text_surface, text_rect)
        
        self.debug_buttons[action_name] = {'rect': button_rect, 'action': action_name}
        return button_rect

    def _render_debug_main_menu(self, screen: pygame.Surface):
        self.debug_buttons.clear()
        title_y = 50
        button_start_y = 150
        button_spacing = 70
        screen_center_x = screen.get_width() // 2

        self._render_text(screen, "EVE2 Debug Menu", (screen_center_x, title_y), center_x=True)
        self._render_button(screen, "Video Debug", button_start_y, "video_debug_menu", center_x_pos=screen_center_x)
        self._render_button(screen, "Voice Debug", button_start_y + button_spacing, "voice_debug_menu", center_x_pos=screen_center_x)
        self._render_button(screen, "Exit Debug Mode", button_start_y + 2 * button_spacing, "exit_debug", center_x_pos=screen_center_x)

    def _render_video_debug(self, screen: pygame.Surface, frame: Optional[np.ndarray]):
        self.debug_buttons.clear()
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Back button
        self._render_button(screen, "< Back", 30, "back_to_main_menu", center_x_pos=80)
        self._render_text(screen, "Video Debug", (screen_width // 2, 30), center_x=True)

        # Video display area (placeholder for now)
        video_area_rect = pygame.Rect(50, 80, screen_width - 100, screen_height - 200)
        pygame.draw.rect(screen, (50, 50, 50), video_area_rect) # Placeholder color

        if frame is not None:
            display_frame = frame.copy()
            # Apply rotation
            if self.debug_rotation_angle == 90:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.debug_rotation_angle == 180:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)
            elif self.debug_rotation_angle == 270:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Resize to fit video_area_rect, maintaining aspect ratio
            frame_h, frame_w = display_frame.shape[:2]
            target_w, target_h = video_area_rect.width, video_area_rect.height
            scale = min(target_w / frame_w, target_h / frame_h)
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized_frame = cv2.resize(display_frame, (new_w, new_h))
                
                # Convert BGR (OpenCV) to RGB (Pygame) and create surface
                try:
                    # Ensure frame is contiguous for Pygame surface creation
                    if not resized_frame.flags['C_CONTIGUOUS']:
                        resized_frame = np.ascontiguousarray(resized_frame)

                    # Check if it's BGR or RGB (Picamera2 with RGB888 should be RGB, then we convert to BGR)
                    # The frame capture logic currently does cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # So, resized_frame is BGR. For Pygame, we need RGB.
                    pygame_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    video_surface = pygame.surfarray.make_surface(pygame_frame.swapaxes(0, 1))
                    
                    # Center the video surface within video_area_rect
                    video_x = video_area_rect.left + (video_area_rect.width - new_w) // 2
                    video_y = video_area_rect.top + (video_area_rect.height - new_h) // 2
                    screen.blit(video_surface, (video_x, video_y))
                except Exception as e_surf:
                    logger.error(f"Error creating Pygame surface from frame: {e_surf}")
                    self._render_text(screen, "Error displaying frame", (video_area_rect.centerx, video_area_rect.centery), color=(255,0,0), center_x=True)
            else:
                 self._render_text(screen, "No video signal or invalid frame size", (video_area_rect.centerx, video_area_rect.centery), center_x=True)

        else: # No frame
            self._render_text(screen, "Camera not available or failed to capture frame", (video_area_rect.centerx, video_area_rect.centery), color=(255,255,0), center_x=True)

        # Rotation buttons
        bottom_y = screen_height - 50
        self._render_button(screen, "Rot CW", bottom_y, "rotate_cw", center_x_pos=screen_width // 4)
        self._render_text(screen, f"{self.debug_rotation_angle}°", (screen_width // 2, bottom_y -10 ), center_x=True) # Show current rotation
        self._render_button(screen, "Rot CCW", bottom_y, "rotate_ccw", center_x_pos=3 * screen_width // 4)
        
        # Face enrolment placeholder
        # Logic for self.debug_recognized_face_info will be added later
        # For now, just a button if a generic face is detected (hypothetically)
        # This needs detected_faces passed to _render_video_debug
        # And then logic to select a face. For now, this is just a placeholder section.
        if self.debug_active_input_field == "face_name":
            input_rect = pygame.Rect(screen_width // 2 - 150, video_area_rect.bottom + 15, 300, 40)
            pygame.draw.rect(screen, (200, 200, 200), input_rect) # Input field bg
            pygame.draw.rect(screen, (100, 100, 100), input_rect, 2) # Border
            self._render_text(screen, self.debug_input_text, (input_rect.x + 5, input_rect.y + 5), color=(0,0,0))
            self._render_text(screen, "Enter Name & Press Enter", (screen_width//2, input_rect.bottom + 5), center_x=True, color=(200,200,200))

        elif self.debug_recognized_face_info and self.debug_recognized_face_info.get('has_face'): # Placeholder condition
            self._render_button(screen, "Enrol This Face", video_area_rect.bottom + 30, "enrol_face_input", center_x_pos=screen_width//2)
            if self.debug_recognized_face_info.get('name'):
                self._render_text(screen, f"Recognized: {self.debug_recognized_face_info['name']}", (screen_width//2, video_area_rect.bottom + 60), center_x=True)

    def _render_voice_debug_placeholder(self, screen: pygame.Surface):
        self.debug_buttons.clear()
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        self._render_button(screen, "< Back", 30, "back_to_main_menu", center_x_pos=80)
        self._render_text(screen, "Voice Debug", (screen_width // 2, 30), center_x=True)
        self._render_text(screen, "Voice Debug - To be implemented", (screen_width // 2, screen_height // 2), center_x=True)

    def _render_debug_ui(self, frame: Optional[np.ndarray]):
        screen = self.services['display'].screen
        if screen is None or self.debug_font is None: # Ensure screen and font are available
            if self.debug_font is None: self._initialize_debug_font() # Try to init font again
            if screen is None or self.debug_font is None: # Still None
                 logger.error("Debug UI cannot be rendered: screen or font not available.")
                 return

        screen.fill((30, 30, 30)) # Dark background for debug mode
        self.debug_buttons.clear() # Clear buttons before each render pass

        if self.debug_mode == "main_menu":
            self._render_debug_main_menu(screen)
        elif self.debug_mode == "video_debug":
            # We need to run face detection here if we want to show boxes and enable enrolment
            detected_faces_list: List[Tuple[int, int, int, int]] = []
            if frame is not None:
                detected_faces_list = self.services['face'].detect_faces(frame)
                # For video debug, we might want to update self.debug_recognized_face_info
                # This part needs more thought on how a face is 'selected' for enrolment in debug
                if detected_faces_list: # If any face is detected
                    # For now, let's just say there's a face if one is detected.
                    # We'd need a way to click on a face or cycle through them.
                    # Simple approach: if one face, make it available for enrolment.
                    if len(detected_faces_list) == 1 and not self.debug_active_input_field:
                         # Store the ROI of the first detected face if needed for enrolment later
                         x,y,w,h = detected_faces_list[0]
                         # Ensure ROI coordinates are within frame bounds
                         y1, y2 = max(0, y), min(frame.shape[0], y + h)
                         x1, x2 = max(0, x), min(frame.shape[1], x + w)
                         face_roi = frame[y1:y2, x1:x2]
                         if face_roi.size > 0:
                            self.debug_recognized_face_info = {'has_face': True, 'roi_for_enrol': face_roi, 'box': detected_faces_list[0]}
                         else:
                            self.debug_recognized_face_info = {'has_face': False}
                    elif not detected_faces_list and not self.debug_active_input_field : # No faces, clear info
                        self.debug_recognized_face_info = None

                # Draw face boxes on the frame to be displayed (before rotation)
                # This is tricky because the frame is passed to _render_video_debug, which then rotates it.
                # We should draw boxes on the *rotated* frame or pass boxes and rotate them too.
                # For now, _render_video_debug receives the raw frame. It should handle drawing boxes after rotation.
                
            self._render_video_debug(screen, frame) # frame is the BGR cv2 frame
            
            # Draw face boxes on the *displayed* (potentially rotated and resized) frame.
            # This needs to happen inside _render_video_debug after the frame is prepared for display.
            # Let's modify _render_video_debug to accept detected_faces_list.

        elif self.debug_mode == "voice_debug":
            self._render_voice_debug_placeholder(screen)
        
        # Display timed messages
        if self.debug_message and time.time() < self.debug_message_timer:
            self._render_text(screen, self.debug_message, (screen.get_width() // 2, screen.get_height() - 30), color=(255,200,0), center_x=True)
        elif self.debug_message and time.time() >= self.debug_message_timer:
            self.debug_message = None

    def _submit_debug_face_name(self):
        if self.debug_recognized_face_info and self.debug_input_text:
            face_roi = self.debug_recognized_face_info.get('roi_for_enrol')
            if face_roi is not None and face_roi.size > 0:
                name_to_enrol = self.debug_input_text.strip()
                if name_to_enrol:
                    logger.info(f"Attempting to enrol face from debug UI. Name: {name_to_enrol}")
                    success = self.services['face'].enrol_user(name_to_enrol, face_roi)
                    if success:
                        logger.info(f"Successfully enrolled {name_to_enrol} from debug UI.")
                        self.debug_message = f"Enrolled: {name_to_enrol}"
                    else:
                        logger.error(f"Failed to enrol {name_to_enrol} from debug UI.")
                        self.debug_message = f"Enrolment failed for {name_to_enrol}"
                    self.debug_message_timer = time.time() + 3
                else: # Empty name
                    self.debug_message = "Enrolment failed: Name cannot be empty."
                    self.debug_message_timer = time.time() + 3

            else: # No ROI
                logger.warning("Debug enrolment failed: No face ROI available.")
                self.debug_message = "Enrolment failed: No face data."
                self.debug_message_timer = time.time() + 3
        self.debug_input_text = ""
        self.debug_active_input_field = None
        self.debug_recognized_face_info = None # Clear after attempt

    def _interaction_loop(self) -> None:
        """Main interaction loop."""
        # Initialize debug font (needs Pygame to be init by EyeDisplay.start())
        # Moved to self.start() to ensure EyeDisplay.start() is called first.

        last_seen_time = time.time()
        no_face_emotion = Emotion.NEUTRAL
        consecutive_camera_failures = 0
        max_consecutive_failures = 30 
        camera_reopened_attempted_this_cycle = False

        while self.running:
            # --- Event Handling ---
            # This needs to be here regardless of debug mode for QUIT and debug toggling
            # (Pygame events should be polled frequently)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        self._toggle_debug_mode()
                    elif self.debug_mode: 
                        self._handle_debug_keydown(event)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        if self.debug_mode:
                            self._handle_debug_click(event.pos)
                        else: 
                            current_time = time.time()
                            if (current_time - self.last_mouse_click_time) < self.double_click_interval:
                                self.num_mouse_clicks += 1
                                if self.num_mouse_clicks >= 2:
                                    self._toggle_debug_mode()
                                    self.num_mouse_clicks = 0 
                            else:
                                self.num_mouse_clicks = 1 
                            self.last_mouse_click_time = current_time
            if not self.running: break # Exit loop if QUIT event was processed

            # --- Camera Frame Acquisition ---
            current_cv_frame: Optional[np.ndarray] = None
            if self.picam2 and self.picam2.started:
                try:
                    # capture_array returns an RGB numpy array
                    rgb_frame_array = self.picam2.capture_array()
                    # Convert to BGR for OpenCV processing if needed by services
                    current_cv_frame = cv2.cvtColor(rgb_frame_array, cv2.COLOR_RGB2BGR)
                    consecutive_camera_failures = 0
                    camera_reopened_attempted_this_cycle = False
                except Exception as e_cap:
                    logger.warning(f"Failed to capture frame ({consecutive_camera_failures}/{max_consecutive_failures}): {e_cap}")
                    consecutive_camera_failures += 1
                    current_cv_frame = None
                    if consecutive_camera_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive camera read failures. Attempting to stop and allow reopen.")
                        if self.picam2:
                            try:
                                self.picam2.stop()
                            except Exception as e_stop:
                                logger.error(f"Exception while stopping picam2: {e_stop}")
                            self.picam2 = None # Flag for re-initialization
                            camera_reopened_attempted_this_cycle = False # Allow reopen attempt on next cycle
                        consecutive_camera_failures = 0 
            elif not self.picam2 and not camera_reopened_attempted_this_cycle : # Try to reinitialize if picam2 is None
                logger.info("Picamera2 is not available. Attempting to reinitialize...")
                camera_reopened_attempted_this_cycle = True # Avoid re-attempting in the same loop if it fails
                try:
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                    self.picam2.configure(config)
                    self.picam2.start()
                    if self.picam2.started:
                        logger.info("Picamera2 reinitialized successfully.")
                        consecutive_camera_failures = 0
                    else:
                        logger.error("Failed to restart Picamera2 after reinitialization attempt.")
                        self.picam2 = None # Ensure it stays None if start fails
                except Exception as e_reinit:
                    logger.error(f"Error reinitializing Picamera2: {e_reinit}")
                    self.picam2 = None

            # --- Main Logic & Rendering ---
            display_service = self.services.get('display')

            if self.debug_mode:
                # Pass the BGR frame to debug UI; it will handle conversion for display if needed
                self._render_debug_ui(current_cv_frame) 
            else: # Normal mode
                # --- Normal mode logic ---
                current_emotion_to_display = no_face_emotion # Default
                
                if current_cv_frame is not None:
                    faces = self.services['face'].detect_faces(current_cv_frame)
                    
                    primary_face_roi: Optional[np.ndarray] = None
                    recognized_name: Optional[str] = None
                    detected_emotion_from_face: Optional[Emotion] = None

                    if faces:
                        last_seen_time = time.time()
                        # Get first face bounding box (x, y, w, h)
                        x, y, w, h = faces[0]
                        y1, y2 = max(0, y), min(current_cv_frame.shape[0], y + h)
                        x1, x2 = max(0, x), min(current_cv_frame.shape[1], x + w)
                        primary_face_roi = current_cv_frame[y1:y2, x1:x2]

                        if primary_face_roi.size > 0:
                            recognized_name = self.services['face'].recognize_face(primary_face_roi)
                            # Pass BGR to emotion service if it expects that
                            detected_emotion_from_face = self.services['emotion'].detect_emotion(primary_face_roi) 
                        else: # Empty ROI
                            primary_face_roi = None # Ensure it's None

                        if detected_emotion_from_face:
                            current_emotion_to_display = detected_emotion_from_face
                            no_face_emotion = Emotion.NEUTRAL # Reset if face seen
                        else: # No emotion from face, use neutral for face
                            current_emotion_to_display = Emotion.NEUTRAL
                    else: # No faces detected
                        idle_time = time.time() - last_seen_time
                        sleep_threshold = self.services['face'].config.get('interaction', {}).get('sleep_after_seconds', 600) # Default 10 mins
                        if idle_time > sleep_threshold:
                             no_face_emotion = Emotion.SLEEPY
                        current_emotion_to_display = no_face_emotion
                    
                    # Pass data to LLM if voice input occurs (example)
                    if self.services['voice'].has_input():
                        text = self.services['voice'].get_input()
                        if text:
                            logger.info(f"Received voice input: {text}")
                            response = self.services['llm'].generate_response(
                                text,
                                user_name=recognized_name, # Pass recognized name
                                emotion=current_emotion_to_display.value # Pass current displayed emotion
                            )
                            if response:
                                logger.info(f"LLM Response: {response}")
                                self.services['voice'].speak(response)
                            else:
                                logger.warning("LLM failed to generate response.")
                else: # current_cv_frame is None
                    current_emotion_to_display = Emotion.NEUTRAL # Or some other default for no camera
                
                if display_service:
                    display_service.set_emotion(current_emotion_to_display)
                    display_service.update() # EyeDisplay.update() just draws

            # --- Global screen update & FPS control ---
            if display_service and display_service.screen:
                 pygame.display.flip()
                 display_service.clock.tick(display_service.config.get('display', {}).get('fps', 30))
            else: # Fallback if display service or screen is somehow not available
                time.sleep(1/30) # Basic delay

        # End of while self.running loop
            
    def stop(self) -> None:
        """Stop all services."""
        self.running = False
        if self.thread and self.thread.is_alive(): # Check if alive before join
            logger.info("Waiting for interaction manager thread to join...")
            self.thread.join(timeout=2.0) # Add timeout
            if self.thread.is_alive():
                 logger.warning("Interaction manager thread did not join in time.")

        if self.picam2:
            try:
                if self.picam2.started:
                    self.picam2.stop()
                # Picamera2 handles its own closing, no explicit close needed usually
                logger.info("Picamera2 stopped")
            except Exception as e:
                logger.error(f"Exception stopping Picamera2: {e}")
            self.picam2 = None
            
        # Stop all services (display service handles pygame.quit())
        for service_name, service in self.services.items():
            if hasattr(service, 'stop'): # Check if service has stop method
                logger.info(f"Stopping {service_name} service...")
                service.stop()
            
        logger.info("Interaction manager stopped")
        # Pygame.quit() should be called by EyeDisplay.stop()
        # If EyeDisplay isn't guaranteed to be last or if it might fail,
        # a final pygame.quit() here could be a safeguard, but generally owned by display service.
        if pygame.get_init(): # If Pygame is still initialized
            # pygame.quit() # Consider if this is needed here or strictly in EyeDisplay
            pass


    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 