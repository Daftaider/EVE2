"""
Main orchestrator for the EVE2 system.

This module initializes and coordinates all modules,
managing the overall workflow and event handling.
"""
import logging
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Union, Any
import queue
import importlib
from types import SimpleNamespace

from eve import config
from eve.utils import logging_utils
from eve.vision import face_detector, emotion_analyzer
from eve.display import lcd_controller
from eve.speech import speech_recorder, speech_recognizer, llm_processor, text_to_speech
from eve.communication import message_queue
# Create a mock api module
import types
api = types.SimpleNamespace()
api.initialize = lambda: None

from eve.config.communication import TOPICS
from eve.speech.speech_recorder import AudioCapture
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.text_to_speech import TextToSpeech
from eve.speech.llm_processor import LLMProcessor
from eve.display.lcd_controller import LCDController
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer
from eve.vision.display_window import VisionDisplay
from eve.config.display import Emotion, DisplayConfig
from eve.config.speech import SpeechConfig

# Import config modules directly
try:
    from eve.config import speech as speech_config
except ImportError:
    # Create fallback speech config
    class speech_config:
        SAMPLE_RATE = 16000
        CHANNELS = 1
        CHUNK_SIZE = 1024
        THRESHOLD = 0.01
        MODEL_TYPE = "google"
        MIN_CONFIDENCE = 0.6

try:
    from eve.config import display as display_config
except ImportError:
    # Create fallback display config
    class display_config:
        WIDTH = 800
        HEIGHT = 480
        FPS = 30
        DEFAULT_EMOTION = "neutral"
        BACKGROUND_COLOR = (0, 0, 0)
        EYE_COLOR = (0, 191, 255)

try:
    from eve.config import vision as vision_config
except ImportError:
    # Create fallback vision config
    class vision_config:
        CAMERA_INDEX = 0
        RESOLUTION = (640, 480)
        FPS = 30

# Add imports for OpenCV and NumPy
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class EVEOrchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the EVE orchestrator."""
        self.config_dict = config or {}
        self._running = False
        self._audio_thread = None
        self._state_lock = threading.Lock() # Lock for accessing shared state

        # State variables (protected by lock)
        self._current_emotion: Emotion = Emotion.NEUTRAL
        self._is_listening: bool = False # True after wake word, waiting for command
        self.camera = None # Add camera attribute
        self.available_camera_indices: List[int] = []
        self.selected_camera_index: int = 0 # Default to 0
        self.camera_rotation: int = 0 # Degrees (0, 90, 180, 270)

        # Initialize configurations and subsystems
        self._init_configs()
        self._discover_cameras() # Discover cameras before initializing subsystems that might need them
        self._init_subsystems()
        self._init_camera() # Initialize camera using the selected index
        logger.info("EVEOrchestrator initialized.")

    def _init_configs(self):
        """Initialize configuration objects."""
        logger.info("Initializing configurations...")
        try:
            # Initialize display config
            self.display_config = DisplayConfig.from_dict(self.config_dict.get('display', {}))
            
            # Handle emotion from config
            if 'DEFAULT_EMOTION' in self.display_config.__dict__:
                self._current_emotion = Emotion.from_value(self.display_config.DEFAULT_EMOTION)
            
            # Set other display config attributes
            for key, value in self.display_config.__dict__.items():
                if hasattr(self.display_config, key):
                    setattr(self.display_config, key, value)

            # Initialize speech config
            self.speech_config = SpeechConfig.from_dict(self.config_dict.get('speech', {}))

            # Add wake word settings
            self.wake_word = self.speech_config.WAKE_WORD_PHRASE
            self.wake_word_threshold = self.speech_config.WAKE_WORD_THRESHOLD

            logger.info("Configurations initialized successfully")

        except Exception as e:
            logger.error(f"Fatal error initializing configs: {e}", exc_info=True)
            raise

    def _init_subsystems(self):
        """Initialize all subsystems."""
        logger.info("Initializing subsystems...")
        try:
            from eve.display.lcd_controller import LCDController
            
            # Initialize display subsystem
            self.lcd_controller = LCDController(
                config=self.display_config,
                width=getattr(self.display_config, 'WINDOW_SIZE', (800, 480))[0],
                height=getattr(self.display_config, 'WINDOW_SIZE', (800, 480))[1],
                fps=getattr(self.display_config, 'FPS', 30),
                default_emotion=self._current_emotion,
                background_color=getattr(self.display_config, 'DEFAULT_BACKGROUND_COLOR', (0, 0, 0)),
                eye_color=getattr(self.display_config, 'DEFAULT_EYE_COLOR', (255, 255, 255))
            )
            
            # Initialize audio capture
            from eve.speech.audio_capture import AudioCapture
            self.audio_capture = AudioCapture(self.speech_config)

            # Initialize speech recognition
            from eve.speech.speech_recognizer import SpeechRecognizer
            self.speech_recognizer = SpeechRecognizer(self.speech_config)

            # Initialize text-to-speech
            from eve.speech.text_to_speech import TextToSpeech
            self.tts = TextToSpeech(self.speech_config)

            # Play startup sound
            self._play_startup_sound()
            
            logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            logger.error(f"Fatal error initializing subsystems: {e}", exc_info=True)
            self.cleanup()
            raise

    def _play_startup_sound(self):
        """Play startup sound and greeting."""
        try:
            self.tts.speak("EVE system online")
        except Exception as e:
            logger.warning(f"Failed to play startup sound: {e}")

    def start_audio_processing(self):
        """Starts the audio processing thread."""
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._running = True
            self._audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            self._audio_thread.start()
            logger.info("Audio processing thread started.")
        else:
            logger.warning("Audio processing thread already running.")

    def _audio_processing_loop(self):
        """Continuously processes audio input for wake word and commands."""
        logger.info("Audio loop running...")
        while self._running:
            try:
                if not self.audio_capture.has_new_audio():
                    time.sleep(0.05) # Short sleep if no new audio
                    continue

                audio_data = self.audio_capture.get_audio()
                if not audio_data:
                    continue

                with self._state_lock:
                    is_currently_listening = self._is_listening

                if not is_currently_listening:
                    # --- Wake Word Detection Phase ---
                    if self.speech_recognizer.detect_wake_word(
                        audio_data, self.speech_config.WAKE_WORD_PHRASE
                    ):
                        self._handle_wake_word() # State changes happen inside
                else:
                    # --- Command Recognition Phase ---
                    text = self.speech_recognizer.recognize(audio_data)
                    if text:
                        self._handle_speech_command(text) # State changes happen inside
                    # Optional: Add a timeout for listening phase here
                    # If timeout expires without speech, reset _is_listening

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}", exc_info=True)
                time.sleep(1) # Avoid spamming logs on persistent errors

        logger.info("Audio processing loop stopped.")

    def _handle_wake_word(self):
        """Handle wake word detection."""
        logger.info(f"Wake word '{self.speech_config.WAKE_WORD_PHRASE}' detected!")
        with self._state_lock:
            self._is_listening = True
            self._current_emotion = Emotion.SURPRISED # Show attention
        
        # Update display immediately
        self.lcd_controller.update(self._current_emotion)
        
        try:
            self.tts.speak("Yes?") # Acknowledge wake word
            # No sleep needed here, TTS blocks or handles timing
        except Exception as e:
            logger.error(f"TTS failed after wake word: {e}")
        
        # Optionally, return to NEUTRAL after a short delay or keep SURPRISED while listening
        # Let's keep SURPRISED for now while listening.

    def _handle_speech_command(self, text: str):
        """Processes recognized speech command and generates a response."""
        logger.info(f"Recognized command: '{text}'")
        response = "Sorry, I didn't understand that." # Default response
        next_emotion = Emotion.CONFUSED # Default emotion

        # Simple command parsing (replace with more sophisticated logic/LLM later)
        text_lower = text.lower()
        if "hello" in text_lower or "hi" in text_lower:
            response = "Hello there!"
            next_emotion = Emotion.HAPPY
        elif "goodbye" in text_lower or "bye" in text_lower:
            response = "Goodbye!"
            next_emotion = Emotion.SAD
        elif "how are you" in text_lower:
            response = "I am functioning optimally!"
            next_emotion = Emotion.NEUTRAL
        
        # Update state and speak response
        with self._state_lock:
            self._current_emotion = next_emotion
            self._is_listening = False # Stop listening after handling command

        self.lcd_controller.update(self._current_emotion)
        try:
            self.tts.speak(response)
        except Exception as e:
            logger.error(f"TTS failed for response: {e}")

        # Transition back to neutral after a delay
        # Consider making this configurable or based on interaction context
        time.sleep(1.5)
        self.set_emotion(Emotion.NEUTRAL)

    def set_emotion(self, emotion: Union[str, Emotion]):
        """Sets the current emotion state safely."""
        new_emotion = Emotion.from_value(emotion)
        with self._state_lock:
            if self._current_emotion != new_emotion:
                logger.debug(f"Changing emotion from {self._current_emotion.name} to {new_emotion.name}")
                self._current_emotion = new_emotion
                # Don't update LCD here, let the main update loop handle it
                # for smoother rendering, unless immediate change is needed.
        # If immediate update is desired:
        # self.lcd_controller.update(self._current_emotion)

    def get_current_emotion(self) -> Emotion:
        """Gets the current emotion state safely."""
        with self._state_lock:
            return self._current_emotion

    def _discover_cameras(self):
        """Check available camera indices using OpenCV."""
        logger.info("Discovering available cameras...")
        index = 0
        max_tested_cameras = 5 # Limit how many indices we test
        while index < max_tested_cameras:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                self.available_camera_indices.append(index)
                cap.release()
                logger.info(f"  Found camera at index {index}")
            else:
                # If index 0 fails, we likely won't find others, but let's test a few more just in case.
                if index > 0:
                    break # Stop searching if a higher index fails after finding at least one
            index += 1
        
        if not self.available_camera_indices:
            logger.warning("No cameras discovered.")
            self.selected_camera_index = -1 # Indicate no camera available
        else:
             # Ensure default selected index is valid if possible
            if self.selected_camera_index not in self.available_camera_indices:
                 self.selected_camera_index = self.available_camera_indices[0]
             logger.info(f"Available camera indices: {self.available_camera_indices}")
             logger.info(f"Initially selected camera index: {self.selected_camera_index}")

    def _init_camera(self):
        """Initialize the camera using the selected index."""
        # Release existing camera if any
        if self.camera and self.camera.isOpened():
            logger.debug(f"Releasing previous camera instance.")
            self.camera.release()
            self.camera = None
            
        if self.selected_camera_index < 0 or self.selected_camera_index not in self.available_camera_indices:
             logger.warning(f"Cannot initialize camera: Selected index {self.selected_camera_index} is invalid or unavailable.")
             self.camera = None
             return

        camera_index = self.selected_camera_index
        logger.info(f"Initializing camera with index {camera_index}...")
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera with index {camera_index}.")
                self.camera = None
                # Attempt self-correction? Maybe later.
                # If index 0 failed, try next available? 
                # available_indices = [i for i in self.available_camera_indices if i != camera_index]
                # if available_indices:
                #    logger.info(f"Retrying with next available index: {available_indices[0]}")
                #    self.selected_camera_index = available_indices[0]
                #    self._init_camera() # Recursive call - be careful
            else:
                logger.info(f"Camera index {camera_index} initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing camera index {camera_index}: {e}", exc_info=True)
            self.camera = None

    def update(self, debug_mode: bool = False):
        """Main update loop called periodically. Updates display and potentially other periodic tasks."""
        debug_frame = None # Frame to pass to display controller
        try:
            # Capture frame if in debug mode and camera is available
            if debug_mode and self.camera:
                ret, frame = self.camera.read()
                if ret:
                    debug_frame = frame
                else:
                    logger.warning("Failed to capture frame from camera.")
            
            # Get current emotion safely
            current_emotion = self.get_current_emotion()
            
            # Update display, passing debug mode status, potential frame, and camera info
            if hasattr(self, 'lcd_controller'):
                self.lcd_controller.update(
                    current_emotion,
                    debug_mode=debug_mode,
                    debug_frame=debug_frame,
                    available_cameras=self.available_camera_indices,
                    selected_camera=self.selected_camera_index,
                    camera_rotation=self.camera_rotation
                )
            
            # Process any pending audio
            if hasattr(self, 'audio_capture'):
                self.audio_capture.update() # Audio update doesn't need debug mode (yet)
                
        except Exception as e:
            logger.error(f"Error in update loop: {e}", exc_info=True)
            # Avoid raising here to prevent crashing the main loop on camera/display errors
            # raise

    def cleanup(self):
        """Stops threads and cleans up all subsystems."""
        logger.info("Starting EVEOrchestrator cleanup...")
        self._running = False # Signal thread to stop

        # Stop and join the audio thread
        if self._audio_thread and self._audio_thread.is_alive():
            logger.debug("Waiting for audio thread to finish...")
            self._audio_thread.join(timeout=2.0) # Wait for thread
            if self._audio_thread.is_alive():
                logger.warning("Audio thread did not terminate gracefully.")
        self._audio_thread = None

        # Cleanup subsystems (in reverse order of dependency if applicable)
        logger.debug("Cleaning up subsystems...")
        subsystems = ['tts', 'speech_recognizer', 'audio_capture', 'lcd_controller']
        for name in subsystems:
            try:
                subsystem = getattr(self, name, None)
                if subsystem and hasattr(subsystem, 'cleanup'):
                    logger.debug(f"Cleaning up {name}...")
                    subsystem.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}", exc_info=True)
        
        # Release camera
        if self.camera:
            logger.debug("Releasing camera...")
            try:
                self.camera.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}", exc_info=True)
            self.camera = None
        
        logger.info("EVEOrchestrator cleanup finished.")

    def __enter__(self):
        self.start_audio_processing() # Start audio thread when entering context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() # Ensure cleanup on exit

    def _process_events(self) -> None:
        """Process events from the event queue"""
        while self.running:
            try:
                # Get event with timeout to allow for clean shutdown
                try:
                    event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process the event
                if event.topic in self.event_handlers:
                    try:
                        self.event_handlers[event.topic](event)
                    except Exception as e:
                        self.logger.error(f"Error handling event {event.topic}: {e}")
                else:
                    self.logger.warning(f"No handler for event topic: {event.topic}")
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                time.sleep(0.1)  # Prevent tight error loop
    
    def _handle_speech_recognition(self, event):
        """Handle speech recognition events"""
        try:
            text = event.data.get('text', '')
            confidence = event.data.get('confidence', 0.0)
            
            self.logger.info(f"Speech recognized: '{text}' (confidence: {confidence:.2f})")
            
            # Process the recognized speech
            if confidence >= self.speech_config.MIN_CONFIDENCE:
                # Generate response using LLM
                response = self.speech_system.process(text)
                
                # Speak the response
                if response:
                    self.speech_system.speak(response)
                    
                    # Update display emotion based on response
                    emotion = self._determine_emotion_from_response(response)
                    if emotion:
                        self.lcd_controller.set_emotion(emotion)
                
        except Exception as e:
            self.logger.error(f"Error handling speech recognition: {e}")

    def _determine_emotion_from_response(self, response):
        """Determine appropriate emotion based on response content"""
        try:
            # Simple keyword-based emotion mapping
            emotion_keywords = {
                'happy': ['happy', 'glad', 'great', 'wonderful', 'excited'],
                'sad': ['sad', 'sorry', 'unfortunate', 'regret'],
                'confused': ['unsure', 'perhaps', 'maybe', 'not certain'],
                'surprised': ['wow', 'amazing', 'incredible', 'unexpected'],
                'neutral': ['okay', 'alright', 'understood', 'indeed']
            }
            
            response_lower = response.lower()
            
            # Check each emotion's keywords
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in response_lower for keyword in keywords):
                    return emotion
            
            return 'neutral'  # Default emotion
            
        except Exception as e:
            self.logger.error(f"Error determining emotion: {e}")
            return 'neutral'

    def _handle_face_detection(self, event):
        """Handle face detection events"""
        try:
            face_data = event.data
            self.logger.debug(f"Face detected: {face_data}")
            
            # Update display based on face position
            if self.lcd_controller:
                # You might want to adjust the display based on face position
                pass
                
        except Exception as e:
            self.logger.error(f"Error handling face detection: {e}")

    def _handle_face_lost(self, event):
        """Handle face lost events"""
        try:
            self.logger.debug("Face lost from view")
            
            # Return to neutral expression
            if self.lcd_controller:
                self.lcd_controller.set_emotion('neutral')
                
        except Exception as e:
            self.logger.error(f"Error handling face lost: {e}")

    def _handle_emotion_detection(self, event):
        """Handle emotion detection events"""
        try:
            emotion = event.data.get('emotion', 'neutral')
            confidence = event.data.get('confidence', 0.0)
            
            self.logger.debug(f"Emotion detected: {emotion} ({confidence:.2f})")
            
            # Update display if confidence is high enough
            if confidence >= self.speech_config.EMOTION_CONFIDENCE_THRESHOLD:
                if self.lcd_controller:
                    self.lcd_controller.set_emotion(emotion)
                
        except Exception as e:
            self.logger.error(f"Error handling emotion detection: {e}")

    def _handle_audio_level(self, event):
        """Handle audio level events"""
        try:
            level = event.data.get('level', 0.0)
            self.logger.debug(f"Audio level: {level:.2f}")
            
            # React to loud sounds
            if level > self.speech_config.REACTION_THRESHOLD:
                if self.lcd_controller:
                    self.lcd_controller.set_emotion('surprised')
                    
        except Exception as e:
            self.logger.error(f"Error handling audio level: {e}")

    def _handle_error(self, event):
        """Handle error events"""
        try:
            error_msg = event.data.get('message', 'Unknown error')
            severity = event.data.get('severity', 'ERROR')
            
            self.logger.error(f"{severity}: {error_msg}")
            
            # React to errors
            if self.lcd_controller:
                self.lcd_controller.set_emotion('confused')
                
        except Exception as e:
            self.logger.error(f"Error handling error event: {e}")

    def post_event(self, topic, data=None):
        """Post an event to the event queue"""
        try:
            event = Event(topic, data or {})
            self.event_queue.put(event)
        except Exception as e:
            self.logger.error(f"Error posting event: {e}")

    def _process_speech(self, text):
        """Process recognized speech"""
        try:
            if text:
                # Process the command through LLM
                response = self.speech_system.process_text(text)
                
                # Speak the response
                if response:
                    self.speech_system.speak(response)
                    
                    # Update display based on response sentiment
                    # This is a simple example - you might want more sophisticated emotion detection
                    if any(word in response.lower() for word in ['sorry', 'error', 'cannot']):
                        self.lcd_controller.set_emotion('sad')
                    elif any(word in response.lower() for word in ['hello', 'hi', 'hey']):
                        self.lcd_controller.set_emotion('happy')
                    else:
                        self.lcd_controller.set_emotion('neutral')
                        
        except Exception as e:
            self.logger.error(f"Error processing speech: {e}")

    def _process_frame(self, frame):
        """Process a camera frame"""
        try:
            if self.vision_display:
                action, data = self.vision_display.process_frame(frame)
                
                if action == "unknown_face":
                    # Ask for person's name
                    self.speech_system.speak("Hello! I don't recognize you. What's your name?")
                    self.lcd_controller.set_emotion("surprised")
                    
                elif action == "continue_learning":
                    count = self.vision_display.learning_faces_count
                    prompts = [
                        "Great! Now please turn your head slightly to the left.",
                        "Perfect! Now slightly to the right.",
                        "Almost done! Look up a bit.",
                        "Last one! Look down slightly."
                    ]
                    if count < len(prompts):
                        self.speech_system.speak(prompts[count])
                    
                elif action == "learning_complete":
                    self.speech_system.speak(
                        "Thank you! I've learned your face and will remember you next time!"
                    )
                    self.lcd_controller.set_emotion("happy")
                    
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def handle_debug_ui_click(self, element_id: str):
        """Handles clicks detected on the debug UI elements."""
        logger.debug(f"Handling debug UI click: {element_id}")

        if element_id.startswith("select_cam_"):
            try:
                new_index = int(element_id.split("_")[-1])
                if new_index in self.available_camera_indices:
                    if self.selected_camera_index != new_index:
                        logger.info(f"User selected camera index: {new_index}")
                        self.selected_camera_index = new_index
                        self._init_camera() # Re-initialize camera with new index
                    else:
                        logger.debug(f"Camera index {new_index} already selected.")
                else:
                    logger.warning(f"Attempted to select invalid camera index {new_index} from UI.")
            except (ValueError, IndexError):
                logger.error(f"Could not parse camera index from element ID: {element_id}")

        elif element_id.startswith("rotate_"):
            try:
                new_rotation = int(element_id.split("_")[-1])
                if new_rotation in [0, 90, 180, 270]:
                    if self.camera_rotation != new_rotation:
                        logger.info(f"User selected camera rotation: {new_rotation} degrees")
                        self.camera_rotation = new_rotation
                        # No need to re-init camera, LCD controller handles rotation display
                    else:
                        logger.debug(f"Rotation {new_rotation} already selected.")
                else:
                    logger.warning(f"Attempted to select invalid rotation {new_rotation} from UI.")
            except (ValueError, IndexError):
                logger.error(f"Could not parse rotation from element ID: {element_id}")
        else:
            logger.warning(f"Unhandled debug UI element click: {element_id}")

class Event:
    """Event class for internal communication"""
    def __init__(self, topic, data):
        self.topic = topic
        self.data = data
        self.timestamp = time.time()

def create_orchestrator():
    """Create and initialize an EVE orchestrator instance"""
    try:
        # Create orchestrator with flat configuration
        orchestrator = EVEOrchestrator()
        return orchestrator
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating orchestrator: {e}")
        raise


if __name__ == "__main__":
    # If run directly, create and start the orchestrator
    orchestrator = create_orchestrator()
    orchestrator.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        orchestrator.cleanup() 