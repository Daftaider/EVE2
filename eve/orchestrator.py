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
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import queue
import importlib
from types import SimpleNamespace
from pathlib import Path
import urllib.request
import os
import json
import sounddevice as sd
from enum import Enum # Import Enum

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

# Explicitly manage sys.path for system dependencies
dist_packages_path = '/usr/lib/python3/dist-packages'
try:
    # Check if the path exists first to avoid errors
    if os.path.isdir(dist_packages_path) and dist_packages_path not in sys.path:
        sys.path.append(dist_packages_path)
        # Use print directly as logger might not be ready
        print(f"DEBUG: Added {dist_packages_path} to sys.path for libcamera lookup.", file=sys.stderr)
except Exception as e:
    print(f"DEBUG: Error trying to add system path: {e}", file=sys.stderr)

# Initialize logger *before* potential use in except block
logger = logging.getLogger(__name__)

# Add imports for OpenCV and NumPy
import cv2
import numpy as np
# Attempt to import picamera2
try:
    from picamera2 import Picamera2
    picamera2_available = True
except ImportError:
    Picamera2 = None # Define as None if not available
    picamera2_available = False
    logger.warning("picamera2 library not found or its dependency 'libcamera' is missing. Falling back to OpenCV VideoCapture (may be less reliable on RPi).")

# Constants for Object Detection Model
MODEL_DIR = Path(__file__).parent.parent / "assets" / "models" / "mobilenet_ssd"
PROTOTXT_PATH = MODEL_DIR / "MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = MODEL_DIR / "MobileNetSSD_deploy.caffemodel"
CLASSES_PATH = MODEL_DIR / "object_detection_classes_coco.txt" # Using COCO names
CONFIDENCE_THRESHOLD = 0.4

# URLs for downloading model files
PROTOTXT_URL = "https://raw.githubusercontent.com/nikmart/pi-object-detection/master/MobileNetSSD_deploy.prototxt.txt"
# Try sourcing caffemodel from the same repo as the working prototxt
MODEL_URL = "https://raw.githubusercontent.com/nikmart/pi-object-detection/master/MobileNetSSD_deploy.caffemodel"
CLASSES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Path for storing corrections
CORRECTIONS_DIR = Path(__file__).parent.parent / "assets" / "data"
CORRECTIONS_PATH = CORRECTIONS_DIR / "object_corrections.json"

# Define Emotion Enum based on config
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    # Add any other potential emotions used internally, like ATTENTIVE?
    ATTENTIVE = "attentive" 
    TALKING = "talking" # If needed for TTS state

    @classmethod
    def from_value(cls, value: Union[str, 'Emotion']) -> 'Emotion':
        if isinstance(value, cls):
            return value
        try:
            # Match case-insensitively
            return cls[value.upper()]
        except KeyError:
            # Fallback to default if value doesn't match any enum member
            logging.warning(f"Invalid emotion value '{value}'. Defaulting to NEUTRAL.")
            return cls.NEUTRAL

class EVEOrchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the EVE orchestrator."""
        self.config_dict = config or {}
        self._running = True # Flag for main loop and threads
        self._state_lock = threading.Lock() # Lock for state variables
        self.event_queue = queue.Queue() # Queue for inter-thread communication
        self.tts_queue = queue.Queue() # Queue for TTS requests
        self.llm_queue = queue.Queue() # Queue for LLM requests
        self.event_handlers = {} # Handlers for events from queue
        self._audio_thread = None
        self._is_listening: bool = False # True after wake word, waiting for command
        self.camera = None # Holds either Picamera2 or cv2.VideoCapture instance
        self.camera_backend = None # 'picamera2' or 'opencv'
        self.available_camera_indices: List[int] = [] # For OpenCV fallback
        self.picamera2_cameras = [] # List of dicts from Picamera2.global_camera_info()
        self.selected_camera_index: int = 0 # Index for OpenCV, ID for Picamera2?
        self.camera_rotation: int = 180 # Degrees (0, 90, 180, 270)
        self.object_detection_net = None
        self.object_detection_classes = None
        self.last_detections: List[Dict[str, Any]] = [] # Store last frame's detections
        self.is_correcting_detection: bool = False
        self.correction_target_info: Optional[Dict[str, Any]] = None
        self.user_input_buffer: str = ""
        self.corrections_data: List[Dict[str, Any]] = [] # Store loaded corrections
        # Debug state
        self.debug_menu_active: bool = False # Track if menu should be processed by update
        self.current_debug_view: Optional[str] = None # None, 'VIDEO', 'AUDIO'
        # Audio Debug State
        self.audio_debug_listen_always: bool = False
        self.last_recognized_text: str = ""
        # Audio State
        self.available_audio_devices = []
        self.selected_audio_device_index: Optional[int] = None # Can be None for default
        self.last_audio_rms: float = 0.0
        self.current_porcupine_sensitivity: float = 0.5 # Track sensitivity (assuming first keyword for UI)

        # Initialize configurations and subsystems
        self._init_configs()
        self._init_object_detection() # Initialize OD model
        self._load_corrections() # Load corrections after ensuring directory exists
        self._discover_audio_devices() # Discover audio devices
        self._discover_cameras() # Discover cameras
        self._init_subsystems() # Init other subsystems
        self._init_camera() # Initialize camera using the selected index/method
        # Fetch initial sensitivity after subsystems are created
        self._update_sensitivity_from_capture()
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

    def _download_file(self, url: str, dest_path: Path):
        """Downloads a file from a URL to a destination path, adding a User-Agent header."""
        logger.info(f"Downloading {dest_path.name} from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        try:
            # Ensure parent directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with headers
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                if response.status == 200:
                    data = response.read() # Read entire file content
                    out_file.write(data)
                    logger.info(f"Successfully downloaded {dest_path.name}.")
                    return True
                else:
                    logger.error(f"Failed to download {dest_path.name}. Status code: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to download {dest_path.name} from {url}: {e}")
            # Clean up potentially partially downloaded file
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError as oe:
                    logger.error(f"Error cleaning up partial file {dest_path}: {oe}")
            return False

    def _init_object_detection(self):
        """Load or download the object detection model and class labels."""
        logger.info("Initializing Object Detection...")
        
        # Ensure model directory exists
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create model directory {MODEL_DIR}: {e}")
            self.object_detection_net = None
            self.object_detection_classes = None
            return
            
        # Check and download prototxt if missing
        if not PROTOTXT_PATH.exists():
            logger.warning(f"Prototxt file not found at {PROTOTXT_PATH}.")
            if not self._download_file(PROTOTXT_URL, PROTOTXT_PATH):
                 logger.error("Cannot proceed with object detection without prototxt.")
                 self.object_detection_net = None
                 self.object_detection_classes = None
                 return

        # Check and download model weights if missing
        if not MODEL_PATH.exists():
            logger.warning(f"Model weights file not found at {MODEL_PATH}.")
            if not self._download_file(MODEL_URL, MODEL_PATH):
                 logger.error("Cannot proceed with object detection without model weights.")
                 self.object_detection_net = None
                 self.object_detection_classes = None
                 return
                 
        # Check and download class labels if missing
        if not CLASSES_PATH.exists():
            logger.warning(f"Class labels file not found at {CLASSES_PATH}.")
            if not self._download_file(CLASSES_URL, CLASSES_PATH):
                 logger.error("Cannot proceed with object detection without class labels.")
                 self.object_detection_net = None
                 self.object_detection_classes = None
                 return

        # --- All files should exist now, proceed with loading --- 
        try:
            # Load class labels
            with open(CLASSES_PATH, 'rt') as f:
                self.object_detection_classes = f.read().rstrip('\n').split('\n')
            logger.info(f"Loaded {len(self.object_detection_classes)} object classes.")

            # Load the neural network
            logger.info(f"Loading object detection model from {MODEL_DIR}...")
            self.object_detection_net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(MODEL_PATH))
            logger.info("Object detection model loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading object detection model or classes after ensuring files exist: {e}", exc_info=True)
            self.object_detection_net = None
            self.object_detection_classes = None

    def _load_corrections(self):
        """Loads saved object detection corrections from the JSON file."""
        logger.info(f"Attempting to load corrections from {CORRECTIONS_PATH}...")
        if CORRECTIONS_PATH.exists():
            try:
                with open(CORRECTIONS_PATH, 'r') as f:
                    self.corrections_data = json.load(f)
                logger.info(f"Loaded {len(self.corrections_data)} corrections.")
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.error(f"Error loading corrections file {CORRECTIONS_PATH}: {e}. Starting with empty list.")
                self.corrections_data = [] # Reset if file is corrupt
        else:
            logger.info("Corrections file not found. Starting with empty list.")
            self.corrections_data = []

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
            
            # Initialize audio capture with selected device index
            from eve.speech.audio_capture import AudioCapture
            # Update speech_config in memory before passing (or pass index directly)
            # self.speech_config.AUDIO_DEVICE_INDEX = self.selected_audio_device_index 
            # Or modify AudioCapture to accept index directly? Let's pass updated config.
            temp_speech_config = SimpleNamespace(**vars(self.speech_config)) # Create a mutable copy
            temp_speech_config.AUDIO_DEVICE_INDEX = self.selected_audio_device_index
            self.audio_capture = AudioCapture(temp_speech_config)

            # Initialize speech recognition
            from eve.speech.speech_recognizer import SpeechRecognizer
            self.speech_recognizer = SpeechRecognizer(self.speech_config) # Recognizer doesn't need device index directly

            # Initialize text-to-speech
            from eve.speech.text_to_speech import TextToSpeech
            # Pass config values explicitly using keyword arguments
            self.tts = TextToSpeech(
                engine=getattr(self.speech_config, 'TTS_ENGINE', 'pyttsx3'),
                voice=getattr(self.speech_config, 'TTS_VOICE', 'english'),
                rate=getattr(self.speech_config, 'TTS_RATE', 150),
                volume=getattr(self.speech_config, 'TTS_VOLUME', 1.0)
            )

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
            self._running = True # Ensure running flag is set
            # --- Start Audio Capture --- 
            if hasattr(self, 'audio_capture') and self.audio_capture:
                try:
                     self.audio_capture.start_recording()
                except Exception as e:
                     logger.error(f"Failed to start audio capture stream: {e}", exc_info=True)
                     # Decide if this is fatal? Maybe EVE can run without audio?
                     return # Don't start thread if capture failed
            else:
                 logger.warning("AudioCapture subsystem not available, cannot start audio processing.")
                 return
            # -------------------------
            self._audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            self._audio_thread.start()
            logger.info("Audio processing thread started.")
        else:
            logger.warning("Audio processing thread is already running.")

    def _audio_processing_loop(self):
        """Dedicated thread for processing audio input."""
        logger.info("Audio loop running...")
        try:
            while self._running:
                if hasattr(self, 'audio_capture'):
                     self.last_audio_rms = self.audio_capture.get_last_rms()
                     
                     if self.audio_capture.has_new_audio():
                        audio_data = self.audio_capture.get_audio_data()
                        
                        if not self._running:
                            logger.debug("_running flag is False, breaking audio loop before processing chunk.")
                            break
                        
                        if audio_data:
                            if hasattr(self, 'speech_recognizer') and self.speech_recognizer:
                                listen_for_command = self._is_listening
                                if self.current_debug_view == 'AUDIO' and self.audio_debug_listen_always:
                                     listen_for_command = True
                                     
                                self.speech_recognizer.process_audio_chunk(
                                     audio_data,
                                     listen_for_command=listen_for_command,
                                     wake_word_callback=self._handle_wake_word_detected,
                                     command_callback=self._handle_command_recognized
                                )
                            else:
                                 logger.warning("SpeechRecognizer not available, skipping audio processing.")
                     else:
                          if not self._running:
                               break
                          time.sleep(0.01)
                else:
                    if not self._running:
                         break
                    time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}", exc_info=True)
            time.sleep(1.0) # Add a sleep to prevent tight loop on error
        finally:
        logger.info("Audio processing loop stopped.")

    def _handle_wake_word_detected(self, event: Event):
        """Handles WAKE_WORD_DETECTED event."""
        with self._state_lock:
            if not self._is_listening:
                self.logger.info("Wake word detected! Entering listening state.")
                self._is_listening = True
                self._last_interaction_time = time.time()
                # Optional: Change emotion, provide audio feedback
                self.set_emotion(Emotion.ATTENTIVE) # Use the defined Enum
                if self.tts: self.tts.speak("Yes?")
                # TODO: Add timer to exit listening state after timeout
            else:
                 self.logger.debug("Wake word detected while already listening - resetting timer.")
                 self._last_interaction_time = time.time() # Reset timeout

    def _handle_command_recognized(self, text: str, confidence: float):
        # Store last recognized text for debug display
        self.last_recognized_text = text 
        
        # Check if we are in Audio Debug listen always mode
        is_audio_debug_listening = self.current_debug_view == 'AUDIO' and self.audio_debug_listen_always
        
        with self._state_lock:
            if self._is_listening or is_audio_debug_listening:
                 logger.info(f"Command recognized: '{text}' (Confidence: {confidence:.2f}) (Mode: {'Normal' if self._is_listening else 'Audio Debug'})")
                 
                 # --- Speak back if in audio debug --- 
                 if is_audio_debug_listening:
                      # TODO: Maybe add a toggle for speaking back?
                      self.tts.speak(text)
                 # ------------------------------------
                 
                 # --- Normal command processing / Stop listening check ---
                 # Only process stop commands etc. if in normal listening mode
                 if self._is_listening:
                     stop_phrases = ["stop listening", "never mind", "cancel", "go away", "shut up"] 
                     if any(phrase in text.lower() for phrase in stop_phrases):
                          logger.info("Stop listening command detected. Exiting listening state.")
                          self.tts.speak("Okay.")
                          self._is_listening = False
                          return 
                     
                     # TODO: Process actual command (LLM etc.)
                     self.tts.speak(f"I heard {text}") # Placeholder
                     logger.info("Exiting listening state after command recognition.")
                     self._is_listening = False 
                 # ------------------------------------------------------
            else:
                 logger.warning(f"Command '{text}' recognized but not in listening state or audio debug.")

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
        """Discover cameras using picamera2 if available, else OpenCV."""
        self.available_camera_indices.clear()
        self.picamera2_cameras.clear()

        if picamera2_available:
            logger.info("Discovering cameras using picamera2...")
            try:
                self.picamera2_cameras = Picamera2.global_camera_info()
                if self.picamera2_cameras:
                    logger.info(f"Found {len(self.picamera2_cameras)} camera(s) via picamera2:")
                    for i, cam_info in enumerate(self.picamera2_cameras):
                         logger.info(f"  Camera {i}: ID={cam_info.get('Id')}, Model={cam_info.get('Model')}, Location={cam_info.get('Location')}")
                    # Select the first camera found by default
                    self.selected_camera_index = 0 # Use index into self.picamera2_cameras list
                    self.camera_backend = 'picamera2'
                else:
                    logger.warning("picamera2 found no cameras. Falling back to OpenCV.")
                    self._discover_cameras_opencv()
            except Exception as e:
                 logger.error(f"Error discovering cameras with picamera2: {e}. Falling back to OpenCV.", exc_info=True)
                 self._discover_cameras_opencv()
        else:
            self._discover_cameras_opencv()

    def _discover_cameras_opencv(self):
        """Fallback camera discovery using OpenCV."""
        self.camera_backend = 'opencv'
        logger.info("Discovering available cameras (using OpenCV default backend)...")
        index = 0
        max_tested_cameras = 5 # Limit how many indices we test
        while index < max_tested_cameras:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                self.available_camera_indices.append(index)
                cap.release()
                logger.info(f"  Found camera at index {index}")
            else:
                if index > 0 and self.available_camera_indices:
                    break
                elif index == 0 and not self.available_camera_indices:
                    pass
            index += 1
        
        if not self.available_camera_indices:
            logger.warning("OpenCV found no cameras.")
            self.selected_camera_index = -1
        else:
            if self.selected_camera_index not in self.available_camera_indices:
                 self.selected_camera_index = self.available_camera_indices[0]
            logger.info(f"Available OpenCV camera indices: {self.available_camera_indices}")
            logger.info(f"Initially selected OpenCV camera index: {self.selected_camera_index}")

    def _init_camera(self):
        """Initialize the camera using the selected backend and index/ID."""
        # Release existing camera if any
        if self.camera:
             logger.debug(f"Releasing previous camera instance ({self.camera_backend}).")
             try:
                 if self.camera_backend == 'picamera2' and hasattr(self.camera, 'close'):
                     self.camera.close()
                 elif self.camera_backend == 'opencv' and hasattr(self.camera, 'release'):
                     self.camera.release()
             except Exception as e:
                 logger.error(f"Error releasing previous camera: {e}")
             self.camera = None

        if self.camera_backend == 'picamera2':
            self._init_camera_picamera2()
        elif self.camera_backend == 'opencv':
             self._init_camera_opencv()
        else:
            logger.error("No valid camera backend determined.")

    def _init_camera_picamera2(self):
         """Initialize camera using Picamera2."""
         if not self.picamera2_cameras or self.selected_camera_index >= len(self.picamera2_cameras):
             logger.error("Cannot initialize Picamera2: No cameras found or invalid index selected.")
             return
         
         cam_index_to_use = self.selected_camera_index
         logger.info(f"Initializing camera using picamera2 (camera index {cam_index_to_use})...")
         try:
             self.camera = Picamera2(camera_num=cam_index_to_use)
             
             # Configure the main stream for RGB888 format suitable for Pygame
             # Get available modes if needed to select resolution dynamically
             # sensor_modes = self.camera.sensor_modes
             # logger.debug(f"Available sensor modes: {sensor_modes}")
             main_stream_config = {"format": "RGB888", "size": (640, 480)} # Use a common default size
             # TODO: Get size from vision_config if available
             preview_config = self.camera.create_preview_configuration(main=main_stream_config)
             self.camera.configure(preview_config)
             logger.info(f"Picamera2 configured with: {preview_config}")

             self.camera.start()
             logger.info("Camera started successfully using picamera2.")
             time.sleep(1.0) # Longer delay for picamera2 startup
             logger.info(f"Picamera2 camera initialized successfully.")
         except Exception as e:
             logger.error(f"Error initializing camera with picamera2: {e}", exc_info=True)
             if self.camera and hasattr(self.camera, 'close'):
                 try:
                     self.camera.close()
                 except Exception as close_e:
                      logger.error(f"Error closing picamera2 after init failure: {close_e}")
             self.camera = None

    def _init_camera_opencv(self):
        """Initialize camera using OpenCV (fallback)."""
        if self.selected_camera_index < 0 or self.selected_camera_index not in self.available_camera_indices:
             logger.warning(f"OpenCV: Cannot initialize camera: Selected index {self.selected_camera_index} is invalid or unavailable.")
             self.camera = None
             return

        camera_index = self.selected_camera_index
        logger.info(f"Initializing camera with OpenCV index {camera_index} (using default backend)...")
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                logger.error(f"OpenCV: Failed to open camera with index {camera_index} using default backend.")
                self.camera = None
                return
            logger.info("OpenCV: Camera opened. Adding short delay...")
            time.sleep(0.5)
            logger.info(f"OpenCV: Camera index {camera_index} initialized successfully.")
        except Exception as e:
            logger.error(f"OpenCV: Error during camera initialization for index {camera_index}: {e}", exc_info=True)
            if self.camera and hasattr(self.camera, 'release'):
                 self.camera.release()
            self.camera = None

    def update(self):
        """
        Main update method, called periodically.
        Handles state updates, checks timeouts, updates display.
        """
        # --- State Checks / Timeouts ---
        with self._state_lock:
             if self._is_listening:
                  # Check if listening state should time out
                  timeout_duration = 10.0 # seconds
                  if time.time() - self._last_interaction_time > timeout_duration:
                       self.logger.info(f"Listening timed out after {timeout_duration}s of inactivity.")
                       self._is_listening = False
                       self.set_emotion(Emotion.NEUTRAL) # Use the defined Enum
                       if self.tts: 
                           try:
                               self.tts.speak("Never mind.") # Optional feedback
                           except Exception as tts_err:
                               self.logger.warning(f"Error calling TTS during listening timeout: {tts_err}")

        # --- Update Display Controller (Outside the state lock) ---
        if self.lcd_controller and hasattr(self.lcd_controller, 'update'):
             try:
                  # Gather necessary state for the display
                  emotion = self.get_current_emotion() # Get state safely
                  is_listening = False # Get state safely
                  with self._state_lock:
                      is_listening = self._is_listening
                  
                  # Get debug frame if needed (this part seems complex and might need review later)
                  debug_frame = None
                  if self.debug_menu_active and self.current_debug_view == 'VIDEO':
                       if self.camera: 
                           ret, frame = self.camera.read() # Get latest frame
                           if ret: debug_frame = frame
                       # Get detections to draw on debug frame? Requires storing them.

                  display_state = {
                       "emotion": emotion,
                       "is_listening": is_listening, # Use safe value
                       "debug_menu_active": self.debug_menu_active,
                       "current_debug_view": self.current_debug_view,
                       "debug_frame": debug_frame,
                       # Add other necessary state...
                  }
                  # Call display update
                  self.lcd_controller.update(**display_state) 
             except Exception as e:
                  self.logger.error(f"Error updating display controller: {e}", exc_info=True)

        # --- Other periodic checks? ---
        # Example: Check if subsystems are still running?
        # if self.face_detector and not self.face_detector.is_running():
        #     self.logger.warning("Face detector thread seems to have stopped unexpectedly.")
        #     # Attempt restart? Post error event?

    def cleanup(self):
        """Stops threads and cleans up all subsystems."""
        logger.info("Starting EVEOrchestrator cleanup...")

        # --- 1. Stop Audio Capture Stream Callback --- 
        # Tells sounddevice to stop calling the audio callback
        if hasattr(self, 'audio_capture') and self.audio_capture:
            logger.debug("Cleanup Step 1: Stopping AudioCapture stream callback...")
            try:
                # Use the new method to only stop the callback
                self.audio_capture.stop_stream() 
                logger.info("Cleanup Step 1: Audio capture stream callback stopped.")
            except Exception as e:
                 logger.error(f"Cleanup Step 1: Error stopping AudioCapture stream callback: {e}", exc_info=True)
            logger.debug("Cleanup Step 1: Finished stopping stream callback.")
        # --------------------------------------------

        # --- 2. Signal Processing Loop to Stop --- 
        logger.debug("Cleanup Step 2: Setting self._running = False")
        self._running = False 
        # --------------------------------------- 

        # --- 3. Wait for Processing Loop to Finish --- 
        if self._audio_thread and self._audio_thread.is_alive():
            logger.debug("Cleanup Step 3: Joining audio processing thread (timeout=5.0s)...")
            self._audio_thread.join(timeout=5.0) 
            if self._audio_thread.is_alive():
                logger.warning("Cleanup Step 3: Audio processing thread did NOT terminate gracefully after join timeout.")
            else:
                 logger.info("Cleanup Step 3: Audio processing thread joined successfully.")
        self._audio_thread = None
        logger.debug("Cleanup Step 3: Finished joining thread.")
        # ------------------------------------------- 

        # --- 4. Cleanup Subsystems --- 
        logger.debug("Cleanup Step 4: Cleaning up subsystems...")
        # Call AudioCapture cleanup (which now closes the stream) AFTER thread join
        if hasattr(self, 'audio_capture') and self.audio_capture and hasattr(self.audio_capture, 'cleanup'):
             logger.debug("Cleanup Step 4a: Calling audio_capture.cleanup() to close stream...")
             try:
                  self.audio_capture.cleanup()
                  logger.debug("Cleanup Step 4a: Finished audio_capture.cleanup().")
             except Exception as e:
                  logger.error(f"Cleanup Step 4a: Error during final AudioCapture cleanup (close_stream): {e}", exc_info=True)
                 
        subsystems = ['tts', 'speech_recognizer', 'lcd_controller'] 
        for name in subsystems:
            logger.debug(f"Cleanup Step 4b: Checking subsystem '{name}'...")
            try:
                subsystem = getattr(self, name, None)
                if subsystem and hasattr(subsystem, 'cleanup'):
                    logger.debug(f"Cleanup Step 4b: Calling {name}.cleanup()...")
                    subsystem.cleanup()
                    logger.debug(f"Cleanup Step 4b: Finished {name}.cleanup().")
                else:
                     logger.debug(f"Cleanup Step 4b: Subsystem '{name}' not found or has no cleanup method.")
            except Exception as e:
                logger.error(f"Cleanup Step 4b: Error cleaning up {name}: {e}", exc_info=True)
        logger.debug("Cleanup Step 4: Finished cleaning up subsystems.")
        # ------------------------------
        
        # --- 5. Release Camera --- 
        logger.debug("Cleanup Step 5: Releasing camera...")
        if self.camera:
            logger.debug(f"Releasing camera instance ({self.camera_backend})...")
            try:
                if self.camera_backend == 'picamera2' and hasattr(self.camera, 'close'):
                    self.camera.close()
                    logger.info("Picamera2 camera closed.")
                elif self.camera_backend == 'opencv' and hasattr(self.camera, 'release'):
                    self.camera.release()
                    logger.info("OpenCV camera released.")
            except Exception as e:
                logger.error(f"Error releasing/closing camera: {e}", exc_info=True)
            self.camera = None
        logger.debug("Cleanup Step 5: Finished releasing camera.")
        # ------------------------
        
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
        logger.debug(f"Handling debug UI click: {element_id}")

        # --- New Menu Navigation Clicks --- 
        if element_id == "menu_video":
             self.set_debug_view('VIDEO')
             return
        if element_id == "menu_audio":
             self.set_debug_view('AUDIO')
             return
        if element_id == "back_to_menu":
             self.set_debug_view(None)
             return
        # ----------------------------------

        # Existing handlers (only active if appropriate view is shown)
        if self.current_debug_view == 'VIDEO':
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
            
            elif element_id.startswith("correct_det_"):
                if self.is_correcting_detection:
                     logger.warning("Already in correction mode. Please finish current correction.")
                     return # Ignore click if already correcting
                     
                try:
                    detection_index = int(element_id.split("_")[-1])
                    # Find the corresponding detection from the last frame
                    target_detection = None
                    for det in self.last_detections:
                         if det['index'] == detection_index:
                              target_detection = det
                              break
                              
                    if target_detection:
                         logger.info(f"Initiating correction for detection index: {detection_index}, label: '{target_detection['label']}', box: {target_detection['box']}")
                         # --- Enter Correction Mode --- 
                         self.is_correcting_detection = True
                         self.correction_target_info = target_detection # Store full detection info
                         self.user_input_buffer = "" # Clear input buffer
                         self.tts.speak(f"What is the correct label?") # Prompt user
                         # ---------------------------
                    else:
                         logger.warning(f"Could not find detection details for index {detection_index} from last frame.")
                     
                except (ValueError, IndexError):
                     logger.error(f"Could not parse detection index from element ID: {element_id}")

        elif self.current_debug_view == 'AUDIO':
             if element_id == "audio_toggle_listen":
                  self.audio_debug_listen_always = not self.audio_debug_listen_always
                  logger.info(f"Audio Debug: Listen Always set to {self.audio_debug_listen_always}")
                  # Reset last text when toggling
                  self.last_recognized_text = ""
                  return # Handled
             elif element_id.startswith("select_audio_dev_"):
                  try:
                       new_index_str = element_id.split("_")[-1]
                       new_index = int(new_index_str) if new_index_str != 'None' else None
                       
                       # Validate selection
                       is_valid = False
                       if new_index is None:
                            is_valid = True # Default is always valid
                       else:
                            is_valid = any(dev['index'] == new_index for dev in self.available_audio_devices)
                            
                       if is_valid:
                            if self.selected_audio_device_index != new_index:
                                 logger.info(f"User selected audio device index: {new_index if new_index is not None else 'Default'}")
                                 self.selected_audio_device_index = new_index
                                 # Need to re-initialize audio capture subsystem
                                 logger.info("Re-initializing Audio Capture with new device...")
                                 if hasattr(self, 'audio_capture'):
                                      self.audio_capture.cleanup() # Stop existing stream
                                 try:
                                      temp_speech_config = SimpleNamespace(**vars(self.speech_config))
                                      temp_speech_config.AUDIO_DEVICE_INDEX = self.selected_audio_device_index
                                      self.audio_capture = AudioCapture(temp_speech_config)
                                      # Restart recording if it was running (it should be if audio thread is running)
                                      self.audio_capture.start_recording() 
                                      logger.info("Audio Capture re-initialized successfully.")
                                 except Exception as e:
                                      logger.error(f"Failed to re-initialize Audio Capture: {e}", exc_info=True)
                                      # TODO: Maybe disable audio debug / display error?
                            else:
                                 logger.debug(f"Audio device {new_index} already selected.")
                       else:
                            logger.warning(f"Attempted to select invalid audio device index {new_index}")
                  except (ValueError, IndexError):
                       logger.error(f"Could not parse audio device index from element ID: {element_id}")
                  return # Handled
             elif element_id == "sensitivity_increase":
                  if hasattr(self, 'audio_capture') and self.audio_capture:
                       current_list = self.audio_capture.get_porcupine_sensitivity()
                       if current_list:
                            new_val = round(min(1.0, current_list[0] + 0.1), 1)
                            # Assume we modify only the first sensitivity for now
                            new_list = [new_val] + current_list[1:] 
                            self.audio_capture.set_porcupine_sensitivity(new_list)
                            self._update_sensitivity_from_capture() # Update orchestrator state
                  return # Handled
                  
             elif element_id == "sensitivity_decrease":
                  if hasattr(self, 'audio_capture') and self.audio_capture:
                       current_list = self.audio_capture.get_porcupine_sensitivity()
                       if current_list:
                            new_val = round(max(0.0, current_list[0] - 0.1), 1)
                            # Assume we modify only the first sensitivity for now
                            new_list = [new_val] + current_list[1:] 
                            self.audio_capture.set_porcupine_sensitivity(new_list)
                            self._update_sensitivity_from_capture() # Update orchestrator state
                  return # Handled
        
        # Fall through for unhandled clicks
        logger.warning(f"Unhandled debug UI element click: {element_id} (Current view: {self.current_debug_view})")

    def submit_correction(self, corrected_label: str):
        """Handles the submitted correction from the user."""
        if not self.is_correcting_detection or not self.correction_target_info:
             logger.warning("Submit correction called but not in correction mode.")
             return
             
        original_label = self.correction_target_info.get('label')
        box = self.correction_target_info.get('box')
        timestamp = time.time()
        
        # Format the correction record
        correction_record = {
            "timestamp": timestamp,
            "box": box,
            "original": original_label,
            "corrected": corrected_label.strip() # Remove leading/trailing whitespace
        }
        
        logger.info(f"Correction submitted: {correction_record}")
        
        # --- Save Correction --- 
        try:
            # Ensure directory exists
            CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
            # Load existing corrections
            current_corrections = []
            if CORRECTIONS_PATH.exists():
                try:
                    with open(CORRECTIONS_PATH, 'r') as f:
                        current_corrections = json.load(f)
                    if not isinstance(current_corrections, list):
                         logger.warning(f"Corrections file {CORRECTIONS_PATH} was not a list. Resetting.")
                         current_corrections = []
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading existing corrections file {CORRECTIONS_PATH}: {e}. Resetting.")
                    current_corrections = []
            
            # Append new correction
            current_corrections.append(correction_record)
            
            # Save updated list back to file
            with open(CORRECTIONS_PATH, 'w') as f:
                json.dump(current_corrections, f, indent=4) # Save with indentation
            logger.info(f"Correction saved to {CORRECTIONS_PATH}")
            
            # Update in-memory corrections
            self.corrections_data = current_corrections
            
        except (IOError, OSError) as e:
            logger.error(f"Error saving correction to {CORRECTIONS_PATH}: {e}")
        # --------------------------

        # --- Exit Correction Mode --- 
        self.tts.speak(f"Okay, noted.")
        self.is_correcting_detection = False
        self.correction_target_info = None
        self.user_input_buffer = ""
        # --------------------------
        
    def cancel_correction(self):
        """Cancels the current correction input."""
        if self.is_correcting_detection:
            logger.info("Correction cancelled.")
            self.is_correcting_detection = False
            self.correction_target_info = None
            self.user_input_buffer = ""
            self.tts.speak("Correction cancelled.")

    def _perform_object_detection(self, frame: np.ndarray, rotation: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Performs object detection, rotates frame if needed, and returns results."""
        detections_list = []
        processed_frame = frame.copy() # Work on a copy

        if not self.object_detection_net or not self.object_detection_classes:
            return processed_frame, detections_list # Return original frame and empty list

        try:
            # --- Rotate frame BEFORE processing if needed --- 
            if rotation == 180:
                processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
            # ------------------------------------------------
            
            (h, w) = processed_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(processed_frame, (300, 300)), 0.007843, (300, 300), 127.5)

            self.object_detection_net.setInput(blob)
            detections = self.object_detection_net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Explicitly convert numpy ints to Python ints for JSON serialization
                    py_startX, py_startY, py_endX, py_endY = int(startX), int(startY), int(endX), int(endY)

                    label = "Unknown"
                    if idx < len(self.object_detection_classes):
                        label = self.object_detection_classes[idx]
                    
                    # Store detection info using Python ints for the box
                    detections_list.append({
                        "index": i, 
                        "label": label,
                        "confidence": float(confidence),
                        "box": (py_startX, py_startY, py_endX, py_endY) # Use Python ints
                    })
                    
                    # --- REMOVE DRAWING LOGIC --- 
                    # label_text = f"{label}: {confidence:.2f}"
                    # cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    # text_offset = 15
                    # text_y = startY - text_offset if startY - text_offset > text_offset else startY + text_offset
                    # cv2.putText(processed_frame, label_text, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # ---------------------------

        except Exception as e:
            logger.error(f"Error during object detection processing: {e}", exc_info=True)

        # Return the processed (potentially rotated) frame and the list of detections
        return processed_frame, detections_list

    def set_debug_view(self, view_name: Optional[str]):
        """Sets the active debug view."""
        logger.info(f"Setting debug view to: {view_name}")
        if view_name not in [None, 'VIDEO', 'AUDIO']:
             logger.warning(f"Invalid debug view name: {view_name}. Setting to None.")
             self.current_debug_view = None
        else:
             self.current_debug_view = view_name
        # Reset correction state when changing views
        self.cancel_correction() 

    def _discover_audio_devices(self):
         """Queries and lists available audio input devices."""
         logger.info("Discovering audio input devices...")
         try:
             devices = sd.query_devices()
             self.available_audio_devices = []
             # Filter for input devices
             for i, device in enumerate(devices):
                 # Check if it has input channels (max_input_channels > 0)
                 if device['max_input_channels'] > 0:
                     self.available_audio_devices.append({'index': i, 'name': device['name']})
                     logger.info(f"  Found Input Device {i}: {device['name']}")
             
             # Get default index from config if specified, otherwise use system default (None)
             self.selected_audio_device_index = getattr(self.speech_config, 'AUDIO_DEVICE_INDEX', None)
             # Basic validation - check if selected index exists in our list
             if self.selected_audio_device_index is not None:
                  found = any(dev['index'] == self.selected_audio_device_index for dev in self.available_audio_devices)
                  if not found:
                      logger.warning(f"Configured audio device index {self.selected_audio_device_index} not found or not an input device. Using default.")
                      self.selected_audio_device_index = None
                      
             logger.info(f"Selected audio device: {self.selected_audio_device_index if self.selected_audio_device_index is not None else 'Default'}")

         except Exception as e:
             logger.error(f"Error querying audio devices: {e}. Cannot select specific device.", exc_info=True)
             self.available_audio_devices = []
             self.selected_audio_device_index = None

    def _update_sensitivity_from_capture(self):
         """Updates the orchestrator's sensitivity state from AudioCapture."""
         if hasattr(self, 'audio_capture') and self.audio_capture:
              sens_list = self.audio_capture.get_porcupine_sensitivity()
              if sens_list:
                   # Store only the first sensitivity for the UI for now
                   self.current_porcupine_sensitivity = sens_list[0]
                   logger.debug(f"Updated orchestrator sensitivity state: {self.current_porcupine_sensitivity:.2f}")
              else:
                   # Handle case where sensitivity list might be empty (e.g., init failed)
                   self.current_porcupine_sensitivity = 0.5 # Default
                   logger.warning("Could not get sensitivities from AudioCapture, using default.")
         else:
              self.current_porcupine_sensitivity = 0.5 # Default

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