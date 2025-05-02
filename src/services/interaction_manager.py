"""
Interaction manager for coordinating all EVE2 services.
"""
import logging
import threading
import queue
import time
from typing import Optional, Dict, Any
from pathlib import Path
import cv2

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
        self.camera = None
        self.command_queue = queue.Queue()
        self.thread = None
        
    def start(self) -> bool:
        """Start all services and begin interaction loop."""
        try:
            # Initialize services with the correct config path
            self.services = {
                'display': EyeDisplay(self.config_path),
                'face': FaceService(self.config_path),
                'emotion': EmotionService(self.config_path),
                'voice': VoiceSynth(self.config_path),
                'llm': LLMService(self.config_path)
            }
            
            # Initialize camera capture
            try:
                camera_index = self.services['face'].config.get('camera', {}).get('index', 0)
                # Explicitly use V4L2 backend for potentially better compatibility on Linux/RPi
                logger.info(f"Attempting to open camera {camera_index} using V4L2 backend...")
                self.camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
                
                if not self.camera.isOpened():
                    logger.error(f"Failed to open camera at index {camera_index} using V4L2. Trying default backend...")
                    # Fallback to default if V4L2 fails
                    self.camera = cv2.VideoCapture(camera_index)
                    if not self.camera.isOpened():
                         logger.error(f"Failed to open camera at index {camera_index} using default backend either.")
                         return False
                         
                logger.info(f"Camera {camera_index} opened successfully (Backend: {'V4L2' if self.camera.getBackendName() == 'V4L2' else 'Default'})")
            except Exception as e:
                logger.error(f"Error initializing camera: {e}")
                return False
            
            # Start all services
            for name, service in self.services.items():
                if not service.start():
                    logger.error(f"Failed to start {name} service")
                    return False
                    
            # Start interaction thread
            self.running = True
            self.thread = threading.Thread(target=self._interaction_loop)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Interaction manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting interaction manager: {e}")
            return False
            
    def _interaction_loop(self) -> None:
        """Main interaction loop."""
        last_seen_time = time.time()
        no_face_emotion = Emotion.NEUTRAL
        consecutive_camera_failures = 0  # Add failure counter
        max_consecutive_failures = 30 # Allow ~1 second of failures @ 30fps before trying recovery
        camera_reopened = False # Flag to prevent continuous reopening attempts

        while self.running:
            try:
                if not self.camera or not self.camera.isOpened():
                    logger.error("Camera not available or closed unexpectedly.")
                    # Try to reopen once if not already attempted
                    if not camera_reopened:
                        logger.info("Attempting to reopen camera...")
                        try:
                            camera_index = self.services['face'].config.get('camera', {}).get('index', 0)
                            # Also use V4L2 when reopening
                            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
                            if not self.camera.isOpened():
                                 logger.warning("Failed to reopen with V4L2, trying default...")
                                 self.camera = cv2.VideoCapture(camera_index)
                                 
                            if self.camera.isOpened():
                                logger.info("Camera reopened successfully.")
                                camera_reopened = True # Mark as reopened (or attempt made)
                                consecutive_camera_failures = 0 # Reset counter
                            else:
                                logger.error("Failed to reopen camera.")
                                camera_reopened = True # Mark attempt failed
                        except Exception as e:
                            logger.error(f"Error trying to reopen camera: {e}")
                            camera_reopened = True # Mark attempt failed
                    
                    time.sleep(1) # Wait before next check/attempt
                    continue

                # Capture frame-by-frame
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    consecutive_camera_failures += 1
                    logger.warning(f"Failed to grab frame ({consecutive_camera_failures}/{max_consecutive_failures})")
                    
                    if consecutive_camera_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive camera read failures. Assuming camera issue.")
                        # Optionally, try to release and flag for reopen on next loop iteration
                        if self.camera:
                             self.camera.release()
                             logger.info("Camera released due to read failures.")
                             self.camera = None # Ensure it triggers the reopen logic
                             camera_reopened = False # Allow reopen attempt
                        consecutive_camera_failures = 0 # Reset counter after handling
                        
                    time.sleep(0.1) # Small delay before retrying read
                    continue
                
                # If read was successful, reset counter
                consecutive_camera_failures = 0
                camera_reopened = False # Allow reopen attempt if it closes later

                # Process face detection
                faces = self.services['face'].detect_faces(frame)
                
                current_name: Optional[str] = None
                current_emotion: Optional[Emotion] = None

                if faces:
                    last_seen_time = time.time()
                    # Get first face bounding box (x, y, w, h)
                    x, y, w, h = faces[0]
                    
                    # Extract face ROI
                    # Ensure ROI coordinates are within frame bounds
                    y1, y2 = max(0, y), min(frame.shape[0], y + h)
                    x1, x2 = max(0, x), min(frame.shape[1], x + w)
                    face_roi = frame[y1:y2, x1:x2]

                    if face_roi.size > 0:
                        # Recognize face
                        current_name = self.services['face'].recognize_face(face_roi)
                        
                        # Detect emotion
                        current_emotion = self.services['emotion'].detect_emotion(face_roi)
                    else:
                        logger.warning("Detected face ROI was empty, skipping recognition/emotion.")

                    # Update display emotion if detected
                    if current_emotion:
                        self.services['display'].set_emotion(current_emotion)
                        no_face_emotion = Emotion.NEUTRAL
                    else:
                        self.services['display'].set_emotion(Emotion.NEUTRAL)
                        
                else:
                    # No faces detected, check for inactivity -> Sleepy?
                    idle_time = time.time() - last_seen_time
                    sleep_threshold = self.services['face'].config.get('interaction', {}).get('sleep_after_seconds', 600)
                    if idle_time > sleep_threshold:
                         no_face_emotion = Emotion.SLEEPY
                    
                    self.services['display'].set_emotion(no_face_emotion)
                    current_emotion = no_face_emotion

                # Process voice input if available (regardless of face detection state)
                if self.services['voice'].has_input():
                    text = self.services['voice'].get_input()
                    if text:
                        logger.info(f"Received voice input: {text}")
                        # Generate response
                        response = self.services['llm'].generate_response(
                            text,
                            user_name=current_name,
                            emotion=current_emotion.value if current_emotion else None
                        )
                        
                        if response:
                            logger.info(f"LLM Response: {response}")
                            # Speak response
                            self.services['voice'].speak(response)
                        else:
                            logger.warning("LLM failed to generate response.")
                            
                # Update display rendering
                self.services['display'].update()
                
                # Simple FPS control (adjust as needed)
                time.sleep(1/30)
                
            except Exception as e:
                logger.exception(f"Error in interaction loop: {e}")
                
    def stop(self) -> None:
        """Stop all services."""
        self.running = False
        if self.thread:
            self.thread.join()

        # Release camera
        if self.camera:
            self.camera.release()
            logger.info("Camera released")
            
        # Stop all services
        for service_name, service in self.services.items():
            service.stop()
            
        logger.info("Interaction manager stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 