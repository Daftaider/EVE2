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
import numpy as np
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
        self.pygame_initialized_in_thread: bool = False
        self.screen: Optional[pygame.Surface] = None # Master screen surface
        self.debug_font: Optional[pygame.font.Font] = None # Moved here from EyeDisplay
        self.debug_buttons: Dict[str, Dict[str, Any]] = {} # Store button rects and actions
        self.debug_rotation_angle: int = 0
        self.debug_input_text: str = ""
        self.debug_active_input_field: Optional[str] = None # e.g. "face_name"
        self.debug_recognized_face_info: Optional[Dict[str, Any]] = None # Store info for naming
        self.debug_message: Optional[str] = None # For displaying status messages in debug UI
        self.debug_message_timer: float = 0.0
        self.debug_display_infos: List[Dict[str, Any]] = [] # New: For storing detailed info of all detected faces in video debug
        self.debug_roi_for_current_enrol_action: Optional[np.ndarray] = None # New: ROI for the face currently being enrolled
        
    def _initialize_debug_font(self):
        # This now relies on self.pygame_initialized_in_thread being true
        # and pygame.font.init() having been called.
        if self.debug_font is None and self.pygame_initialized_in_thread:
            try:
                # pygame.font.init() should have been called
                self.debug_font = pygame.font.Font(None, 30)
                if not self.debug_font:
                    self.debug_font = pygame.font.SysFont('arial', 28)
                logger.info("Debug font initialized.")
            except Exception as e:
                logger.error(f"Failed to load font for debug UI: {e}")
                try:
                    self.debug_font = pygame.font.SysFont('arial', 28)
                except Exception as e_sysfont:
                    logger.error(f"Failed to load system font for debug UI: {e_sysfont}")
        elif not self.pygame_initialized_in_thread:
            logger.warning("Attempted to initialize debug font, but Pygame not yet initialized in thread.")

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
            
            # Prepare EyeDisplay resources (loads sprites) - NO Pygame init here
            if not self.services['display'].prepare_resources():
                logger.error("Failed to prepare EyeDisplay resources")
                return False
            
            # Camera initialization (remains largely the same)
            try:
                logger.info(f"Attempting to initialize Picamera2...")
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                if not self.picam2.started:
                    logger.error("Failed to start Picamera2")
                else:
                    logger.info("Picamera2 initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Picamera2: {e}. Camera will be unavailable.")
                self.picam2 = None
            
            # Start other services
            for name, service in self.services.items():
                if name == 'display': continue # Already prepared
                if hasattr(service, 'start') and not service.start():
                    logger.error(f"Failed to start {name} service")
            
            self.running = True
            self.thread = threading.Thread(target=self._interaction_loop)
            self.thread.daemon = True # Ensure thread exits when main program exits
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
                logger.info("'exit_debug' action recognized. Calling _toggle_debug_mode.")
                self._toggle_debug_mode()
            elif clicked_action == "video_debug_menu":
                self.debug_mode = "video_debug"
                self.debug_display_infos.clear() # Clear previous face infos
                self.debug_roi_for_current_enrol_action = None
                self.debug_input_text = ""
            elif clicked_action == "voice_debug_menu":
                self.debug_mode = "voice_debug"
            elif clicked_action == "back_to_main_menu":
                self.debug_mode = "main_menu"
                self.debug_display_infos.clear()
                self.debug_roi_for_current_enrol_action = None
            elif clicked_action == "rotate_cw":
                self.debug_rotation_angle = (self.debug_rotation_angle + 90) % 360
            elif clicked_action == "rotate_ccw":
                self.debug_rotation_angle = (self.debug_rotation_angle - 90 + 360) % 360
            elif clicked_action == "enrol_face_input":
                 # self.debug_roi_for_current_enrol_action should have been set when the button was rendered
                 if self.debug_roi_for_current_enrol_action is not None and self.debug_roi_for_current_enrol_action.size > 0:
                    self.debug_active_input_field = "face_name"
                    self.debug_input_text = "" # Clear for new input
                    logger.info("Activated face name input for enrolment.")
                 else:
                    logger.warning("Cannot enrol: No valid ROI for current enrolment action.")
                    self.debug_message = "Error: No face data for this action."
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

    def _render_video_debug(self, screen: pygame.Surface, frame_with_boxes: Optional[np.ndarray]):
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        self._render_button(screen, "< Back", 30, "back_to_main_menu", center_x_pos=80)
        self._render_text(screen, "Video Debug", (screen_width // 2, 30), center_x=True)
        if self.debug_display_infos:
             self._render_text(screen, f"Detections: {len(self.debug_display_infos)}", (screen_width - 100, 50), center_x=True)

        video_area_rect = pygame.Rect(50, 80, screen_width - 100, screen_height - 250)
        pygame.draw.rect(screen, (50, 50, 50), video_area_rect)

        if frame_with_boxes is not None:
            display_frame = frame_with_boxes.copy()
            # Apply rotation
            if self.debug_rotation_angle == 90:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.debug_rotation_angle == 180:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)
            elif self.debug_rotation_angle == 270:
                display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            frame_h, frame_w = display_frame.shape[:2]
            target_w, target_h = video_area_rect.width, video_area_rect.height
            scale = min(target_w / frame_w, target_h / frame_h) if frame_w > 0 and frame_h > 0 else 1
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized_frame_for_pygame = cv2.resize(display_frame, (new_w, new_h))
                try:
                    if not resized_frame_for_pygame.flags['C_CONTIGUOUS']:
                        resized_frame_for_pygame = np.ascontiguousarray(resized_frame_for_pygame)
                    pygame_frame = cv2.cvtColor(resized_frame_for_pygame, cv2.COLOR_BGR2RGB)
                    video_surface = pygame.surfarray.make_surface(pygame_frame.swapaxes(0, 1))
                    video_x = video_area_rect.left + (video_area_rect.width - new_w) // 2
                    video_y = video_area_rect.top + (video_area_rect.height - new_h) // 2
                    screen.blit(video_surface, (video_x, video_y))

                    # Draw names/unknown status on Pygame screen, relative to displayed boxes
                    # This requires transforming box coordinates to the displayed frame's scale and position
                    # This is tricky because boxes are on the cv2 frame. Text will be separate.
                    # Let's render text below the video area for now with the info.
                    for face_info in self.debug_display_infos:
                        fx, fy, fw, fh = face_info['box'] # These are for the original full-res frame
                        
                        # Apply same rotation to box points if needed (complex, easier to label on unrotated then let text rotate with frame)
                        # For simplicity, we'll draw text near original box locations scaled to the display
                        # This might not perfectly align if rotation is used.
                        # A more robust way is to draw text on the cv2 frame before rotation and resize.
                        # Let's try drawing text on the cv2 frame (frame_with_boxes) before it's processed for display.
                        # Here, we just confirm display. If text was on frame_with_boxes, it shows.
                        # Let's add the text rendering directly on the pygame surface for now, near estimated box locations.
                        # This is tricky because boxes are on the cv2 frame. Text will be separate.
                        # Let's render text below the video area for now with the info.
                        pass # Box drawing is done on CV2 frame. Names/status will be rendered below video.

                except Exception as e_surf:
                    logger.error(f"Error creating Pygame surface from frame: {e_surf}")
                    self._render_text(screen, "Error displaying frame", (video_area_rect.centerx, video_area_rect.centery), color=(255,0,0), center_x=True)
            else:
                 self._render_text(screen, "No video signal or invalid frame size", (video_area_rect.centerx, video_area_rect.centery), center_x=True)
        else:
            self._render_text(screen, "Camera not available or failed to capture frame", (video_area_rect.centerx, video_area_rect.centery), color=(255,255,0), center_x=True)

        bottom_controls_y = screen_height - 100
        
        self._render_button(screen, "Rot CW", bottom_controls_y, "rotate_cw", center_x_pos=screen_width // 4)
        self._render_text(screen, f"{self.debug_rotation_angle}°", (screen_width // 2, bottom_controls_y -10 ), center_x=True)
        self._render_button(screen, "Rot CCW", bottom_controls_y, "rotate_ccw", center_x_pos=3 * screen_width // 4)
        
        enrol_elements_y = video_area_rect.bottom + 20
        text_y_offset = 0

        # Display recognized names or "Unknown" for each detected face
        if not self.debug_active_input_field: # Only show list if not inputting a name
            for i, face_info in enumerate(self.debug_display_infos):
                name_text = face_info['name'] if face_info['name'] else "Unknown"
                display_text = f"Face {i+1}: {name_text}"
                # For simplicity, display as a list below video
                self._render_text(screen, display_text, (50, enrol_elements_y + text_y_offset), color=(220,220,220))
                text_y_offset += 25

        if self.debug_active_input_field == "face_name":
            input_rect = pygame.Rect(screen_width // 2 - 150, enrol_elements_y + text_y_offset, 300, 40)
            pygame.draw.rect(screen, (200, 200, 200), input_rect)
            pygame.draw.rect(screen, (100, 100, 100), input_rect, 2)
            self._render_text(screen, self.debug_input_text, (input_rect.x + 5, input_rect.y + 5), color=(0,0,0))
            self._render_text(screen, "Enter Name & Press Enter", (screen_width//2, input_rect.bottom + 5), center_x=True, color=(200,200,200))
            text_y_offset += 70 # Space for input field and prompt
        else:
            unknown_faces = [info for info in self.debug_display_infos if info['name'] is None and info['roi'].size > 0]
            if len(unknown_faces) == 1:
                # If button is clicked, its action handler will use this ROI
                self.debug_roi_for_current_enrol_action = unknown_faces[0]['roi'] 
                self._render_button(screen, "Enrol This Face", enrol_elements_y + text_y_offset + 15, "enrol_face_input", center_x_pos=screen_width//2)
                text_y_offset += 45
            elif len(self.debug_display_infos) > 0 : # More than one face, or one known face
                self._render_text(screen, f"{len(self.debug_display_infos)} face(s) detected.", (screen_width//2, enrol_elements_y + text_y_offset + 10), center_x=True, color=(255, 200, 0))
                if any(uf['name'] is None for uf in self.debug_display_infos):
                     self._render_text(screen, "Refined enrolment for multiple/specific faces TBD.", (screen_width//2, enrol_elements_y + text_y_offset + 30), center_x=True, color=(255, 200, 0))
                text_y_offset += 40
                self.debug_roi_for_current_enrol_action = None # No single face to enrol directly
            else: # No faces detected
                self._render_text(screen, "No face detected for enrolment.", (screen_width//2, enrol_elements_y + text_y_offset + 10), center_x=True, color=(255, 200, 0))
                self.debug_roi_for_current_enrol_action = None

    def _render_voice_debug_placeholder(self, screen: pygame.Surface):
        self.debug_buttons.clear()
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        self._render_button(screen, "< Back", 30, "back_to_main_menu", center_x_pos=80)
        self._render_text(screen, "Voice Debug", (screen_width // 2, 30), center_x=True)
        self._render_text(screen, "Voice Debug - To be implemented", (screen_width // 2, screen_height // 2), center_x=True)

    def _render_debug_ui(self, frame: Optional[np.ndarray]):
        # Now uses self.screen directly, which is initialized in _interaction_loop
        if not self.pygame_initialized_in_thread or self.screen is None or self.debug_font is None:
            logger.error("Debug UI cannot be rendered: Pygame/screen/font not ready in thread.")
            return
        
        self.screen.fill((30, 30, 30))
        self.debug_buttons.clear() # Clear buttons from previous frame
        
        if self.debug_mode == "main_menu":
            self._render_debug_main_menu(self.screen)
        elif self.debug_mode == "video_debug":
            self.debug_display_infos.clear() # Clear from previous frame's detections
            processed_frame_for_display = None # Frame with boxes drawn
            raw_detected_faces: List[Tuple[int, int, int, int]] = []

            if frame is not None:
                logger.debug(f"InteractionManager._render_debug_ui: Received frame for detection. Shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
                frame_with_boxes = frame.copy() # For drawing boxes on
                raw_detected_faces = self.services['face'].detect_faces(frame) # Get all face candidates
                logger.debug(f"InteractionManager._render_debug_ui: detect_faces returned: {raw_detected_faces}")

                for (fx, fy, fw, fh) in raw_detected_faces:
                    # Draw rectangle for every detected face candidate
                    cv2.rectangle(frame_with_boxes, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    
                    # Extract ROI for recognition
                    y1, y2 = max(0, fy), min(frame.shape[0], fy + fh)
                    x1, x2 = max(0, fx), min(frame.shape[1], fx + fw)
                    roi = frame[y1:y2, x1:x2] # Get ROI from original, undrawn frame
                    
                    recognized_name = None
                    if roi.size > 0: # Ensure ROI is not empty
                        recognized_name = self.services['face'].recognize_face(roi)
                    
                    self.debug_display_infos.append({'box': (fx,fy,fw,fh), 'roi': roi, 'name': recognized_name})
                
                processed_frame_for_display = frame_with_boxes
            
            # Pass the frame with boxes and the detailed detection infos to the rendering function
            self._render_video_debug(self.screen, processed_frame_for_display) 
        elif self.debug_mode == "voice_debug":
            self._render_voice_debug_placeholder(self.screen)
        
        if self.debug_message and time.time() < self.debug_message_timer:
            self._render_text(self.screen, self.debug_message, (self.screen.get_width() // 2, self.screen.get_height() - 30), color=(255,200,0), center_x=True)
        elif self.debug_message and time.time() >= self.debug_message_timer:
            self.debug_message = None

    def _submit_debug_face_name(self):
        if self.debug_roi_for_current_enrol_action is not None and self.debug_roi_for_current_enrol_action.size > 0 and self.debug_input_text:
            face_roi_to_enrol = self.debug_roi_for_current_enrol_action
            name_to_enrol = self.debug_input_text.strip()
            if name_to_enrol:
                logger.info(f"Attempting to enrol face from debug UI. Name: {name_to_enrol}")
                # Ensure ROI is C-contiguous for some cv2 operations if FaceService expects it
                if not face_roi_to_enrol.flags['C_CONTIGUOUS']:
                    face_roi_to_enrol = np.ascontiguousarray(face_roi_to_enrol)
                
                success = self.services['face'].enrol_user(name_to_enrol, face_roi_to_enrol)
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
        else: # No ROI or no input text
            logger.warning("Debug enrolment failed: No face ROI available or name is empty.")
            self.debug_message = "Enrolment failed: No face data or name."
            self.debug_message_timer = time.time() + 3
            
        self.debug_input_text = ""
        self.debug_active_input_field = None
        self.debug_roi_for_current_enrol_action = None # Clear after attempt
        # self.debug_display_infos.clear() # Don't clear here, let next frame's detection repopulate
        # self.debug_recognized_face_info = None # Clear after attempt

    def _interaction_loop(self) -> None:
        """Main interaction loop. This thread handles Pygame init, events, and rendering."""
        
        # --- Pygame Initialization (within this thread) ---
        if not self.pygame_initialized_in_thread:
            try:
                pygame.init() # Initialize all Pygame modules
                pygame.font.init() # Initialize font module explicitly
                
                display_config = self.services['display'].config.get('display', {})
                width = display_config.get('width', 800)
                height = display_config.get('height', 480)
                
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("EVE2 Interaction") # Caption can be more general now
                
                self.services['display'].screen = self.screen # Provide screen to EyeDisplay service
                
                self.pygame_initialized_in_thread = True
                logger.info("Pygame initialized successfully within interaction thread.")

                # Initialize debug font now that Pygame is fully initialized and flag is set
                self._initialize_debug_font()

                # Load eye sprites now that Pygame is fully initialized
                if self.services.get('display'):
                    logger.info("Attempting to load eye sprites via InteractionManager...")
                    self.services['display']._load_eye_sprites()
                
            except Exception as e_pygame_init:
                logger.error(f"Fatal error initializing Pygame in interaction thread: {e_pygame_init}")
                self.running = False # Stop the loop if Pygame can't initialize

        # ... (rest of the loop: last_seen_time, etc.)
        last_seen_time = time.time()
        no_face_emotion = Emotion.NEUTRAL
        consecutive_camera_failures = 0
        max_consecutive_failures = 30 
        camera_reopened_attempted_this_cycle = False

        while self.running:
            if not self.pygame_initialized_in_thread: # If init failed, don't proceed
                time.sleep(0.1)
                continue

            # --- Event Handling (must be in the Pygame thread) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                # ... (rest of event handling: K_d, MOUSEBUTTONDOWN, _handle_debug_keydown, _handle_debug_click)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        self._toggle_debug_mode()
                    elif self.debug_mode: 
                        self._handle_debug_keydown(event)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
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
            if not self.running: break

            # --- Camera Frame Acquisition (remains the same) ---
            current_cv_frame: Optional[np.ndarray] = None
            if self.picam2 and self.picam2.started:
                try:
                    rgb_frame_array = self.picam2.capture_array()
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
                            self.picam2 = None
                            camera_reopened_attempted_this_cycle = False
                        consecutive_camera_failures = 0 
            elif not self.picam2 and not camera_reopened_attempted_this_cycle :
                logger.info("Picamera2 is not available. Attempting to reinitialize...")
                camera_reopened_attempted_this_cycle = True
                try:
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                    self.picam2.configure(config)
                    self.picam2.start()
                    if self.picam2.started: logger.info("Picamera2 reinitialized successfully.")
                    else: logger.error("Failed to restart Picamera2."); self.picam2 = None
                except Exception as e_reinit: logger.error(f"Error reinitializing Picamera2: {e_reinit}"); self.picam2 = None

            # --- Main Logic & Rendering ---
            display_service = self.services.get('display')

            logger.debug(f"InteractionManager loop: self.debug_mode is currently: {self.debug_mode}") # Added log for state check
            if self.debug_mode:
                self._render_debug_ui(current_cv_frame) 
            else: # Normal mode
                # ... (normal mode logic as before, using display_service.update() to draw on self.screen)
                current_emotion_to_display = no_face_emotion
                if current_cv_frame is not None:
                    faces = self.services['face'].detect_faces(current_cv_frame)
                    primary_face_roi: Optional[np.ndarray] = None
                    recognized_name: Optional[str] = None
                    detected_emotion_from_face: Optional[Emotion] = None
                    if faces:
                        last_seen_time = time.time()
                        x, y, w, h = faces[0]
                        y1, y2 = max(0, y), min(current_cv_frame.shape[0], y + h)
                        x1, x2 = max(0, x), min(current_cv_frame.shape[1], x + w)
                        primary_face_roi = current_cv_frame[y1:y2, x1:x2]
                        if primary_face_roi.size > 0:
                            recognized_name = self.services['face'].recognize_face(primary_face_roi)
                            detected_emotion_from_face = self.services['emotion'].detect_emotion(primary_face_roi) 
                        else: primary_face_roi = None
                        if detected_emotion_from_face: current_emotion_to_display = detected_emotion_from_face; no_face_emotion = Emotion.NEUTRAL
                        else: current_emotion_to_display = Emotion.NEUTRAL
                    else: 
                        idle_time = time.time() - last_seen_time
                        sleep_threshold = self.services['face'].config.get('interaction', {}).get('sleep_after_seconds', 600)
                        if idle_time > sleep_threshold: no_face_emotion = Emotion.SLEEPY
                        current_emotion_to_display = no_face_emotion
                    if self.services['voice'].has_input():
                        text = self.services['voice'].get_input()
                        if text:
                            logger.info(f"Received voice input: {text}")
                            response = self.services['llm'].generate_response(text, user_name=recognized_name, emotion=current_emotion_to_display.value)
                            if response: logger.info(f"LLM Response: {response}"); self.services['voice'].speak(response)
                            else: logger.warning("LLM failed to generate response.")
                else: current_emotion_to_display = Emotion.NEUTRAL
                if display_service: display_service.set_emotion(current_emotion_to_display); display_service.update()

            # --- Global screen update & FPS control (must be in Pygame thread) ---
            if self.screen: # Check if screen is initialized
                 pygame.display.flip()
                 if display_service: # Use display_service's clock if available
                     display_service.clock.tick(display_service.config.get('display', {}).get('fps', 30))
                 else: # Fallback clock tick if display_service not available (should not happen)
                     pygame.time.Clock().tick(30)
            else:
                time.sleep(1/30) 
        # End of while self.running loop
        logger.info("Interaction loop ended.")
            
    def stop(self) -> None:
        """Stop all services and quit Pygame if initialized in thread."""
        self.running = False # Signal the loop to stop
        if self.thread and self.thread.is_alive():
            logger.info("Waiting for interaction manager thread to join...")
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                 logger.warning("Interaction manager thread did not join in time.")

        # Picamera2 stop logic (remains the same)
        if self.picam2:
            try:
                if self.picam2.started: self.picam2.stop()
                logger.info("Picamera2 stopped")
            except Exception as e: logger.error(f"Exception stopping Picamera2: {e}")
            self.picam2 = None
            
        # Stop other services (display service stop is now simpler)
        for service_name, service in self.services.items():
            if hasattr(service, 'stop'):
                logger.info(f"Stopping {service_name} service...")
                service.stop()
        
        # Quit Pygame modules if they were initialized in the thread
        if self.pygame_initialized_in_thread:
            logger.info("Quitting Pygame modules...")
            pygame.font.quit()
            pygame.quit()
            self.pygame_initialized_in_thread = False # Reset flag
            
        logger.info("Interaction manager fully stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 