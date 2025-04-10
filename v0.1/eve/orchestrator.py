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
import traceback

# --- Corrected Config Import --- 
# Import the main SystemConfig and potentially needed nested types directly
from eve.config import (
    SystemConfig, 
    load_config, 
    DisplayConfig, 
    SpeechConfig, 
    # Add others if specifically needed for type hints within this file
)
# from eve import config # Keep specific imports
# ------------------------------- 

from eve.utils import logging_utils
# Removed direct submodule imports, rely on SystemConfig
# from eve.vision import face_detector, emotion_analyzer 
# from eve.display import lcd_controller
# from eve.speech import speech_recorder, speech_recognizer, llm_processor, text_to_speech
from eve.communication import message_queue # Keep if used directly

# Create a mock api module (If needed, otherwise remove)
import types
api = types.SimpleNamespace()
api.initialize = lambda: None

from eve.config.communication import TOPICS # Keep if TOPICS defined here

# --- Subsystem Imports --- (Ensure these are correct)
from eve.speech.audio_capture import AudioCapture # Corrected from speech_recorder
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.text_to_speech import TextToSpeech
from eve.speech.llm_processor import LLMProcessor
from eve.display.lcd_controller import LCDController # Assuming still used
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer
from eve.vision.display_window import VisionDisplay
from eve.vision.camera import Camera # Added Camera import
from eve.vision.object_detector import ObjectDetector # Added ObjectDetector import
from eve.vision.rpi_ai_camera import RPiAICamera # <<< ADD Import
# ------------------------

# Removed fallback config class definitions
# try:
#     from eve.config import speech as speech_config
# ... etc ...

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
# Note: If DisplayConfig defines Emotion, import it instead of redefining
# from eve.config import Emotion # Try importing first
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    ATTENTIVE = "attentive" 
    TALKING = "talking"

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

# Placeholder Event class - NEEDS TO BE DEFINED BEFORE EVEOrchestrator
class Event:
    """Simple Event class for internal communication."""
    def __init__(self, topic: str, data: Optional[Dict] = None):
        self.topic = topic
        self.data = data if data is not None else {}
        self.timestamp = time.time()

    def __repr__(self):
        return f"Event(topic='{self.topic}', data={self.data}, timestamp={self.timestamp})"

class EVEOrchestrator:
    """
    Coordinates EVE subsystems, manages state, and handles interaction flows.
    Relies on injected, pre-initialized subsystems.
    """
    # CORRECTED __init__ method using Dependency Injection
    def __init__(self,
                 config: SystemConfig,
                 camera: Union[Camera, RPiAICamera], # <<< UPDATE Type Hint
                 # Inject optional subsystems
                 face_detector: Optional[FaceDetector] = None,
                 object_detector: Optional[ObjectDetector] = None,
                 emotion_analyzer: Optional[EmotionAnalyzer] = None,
                 audio_capture: Optional[AudioCapture] = None,
                 speech_recognizer: Optional[SpeechRecognizer] = None,
                 llm_processor: Optional[LLMProcessor] = None,
                 tts: Optional[TextToSpeech] = None,
                 display_controller: Optional[Any] = None,
                 post_event_callback: Optional[Callable[[str, Optional[Dict]], None]] = None
                 ):
        """
        Initialize the EVE orchestrator with pre-initialized subsystems.

        Args:
            config: The main SystemConfig object.
            camera: Initialized Camera instance.
            face_detector: Optional initialized FaceDetector instance.
            object_detector: Optional initialized ObjectDetector instance.
            emotion_analyzer: Optional initialized EmotionAnalyzer instance.
            audio_capture: Optional initialized AudioCapture instance.
            speech_recognizer: Optional initialized SpeechRecognizer instance.
            llm_processor: Optional initialized LLMProcessor instance.
            tts: Optional initialized TextToSpeech instance.
            display_controller: Optional initialized display controller instance.
            post_event_callback: Function to call to post events (e.g., to a message queue).
        """
        self.logger = logging.getLogger(__name__)
        self.config = config # Store the main config object
        self._running = False # Start in non-running state
        self._state_lock = threading.Lock() # Lock for critical state variables

        # Injected Subsystems
        self.camera = camera
        self.face_detector = face_detector
        self.object_detector = object_detector
        self.emotion_analyzer = emotion_analyzer
        self.audio_capture = audio_capture
        self.speech_recognizer = speech_recognizer
        self.llm_processor = llm_processor
        self.tts = tts
        self.display_controller = display_controller # Primary display interface

        # Event Handling
        self.post_event = post_event_callback or self._default_post_event # Use provided or default
        self.event_queue = queue.Queue() # Internal queue for events posted to orchestrator
        self.event_handlers = self._register_event_handlers() # Map topics to handler methods
        self._event_thread: Optional[threading.Thread] = None

        # Application State Variables (Protected by _state_lock)
        # Use the Emotion enum defined globally or imported if needed here
        # Initialize based on the DEFAULT_EMOTION string from DisplayConfig
        default_emotion_str = getattr(config.display, 'DEFAULT_EMOTION', 'neutral')
        self._current_emotion = Emotion.from_value(default_emotion_str) # Use from_value correctly
        self._is_listening: bool = False # True after wake word, waiting for command
        self._last_interaction_time: float = time.time()
        self._detected_persons: Dict[str, Any] = {} # Track recognized people?

        # Object Detection Correction State (Keep if correction logic remains internal)
        self.is_correcting_detection: bool = False
        self.correction_target_info: Optional[Dict[str, Any]] = None
        self.user_input_buffer: str = ""
        # self.corrections_data: List[Dict[str, Any]] = [] # Now loaded externally if needed
        # If _load_corrections is kept, initialize here
        self.corrections_data: List[Dict[str, Any]] = self._load_corrections() 

        # Debug / UI State
        self.debug_menu_active: bool = False
        self.current_debug_view: Optional[str] = None
        self.audio_debug_listen_always: bool = False
        self.last_recognized_text: str = ""
        self.last_audio_rms: float = 0.0
        # Store sensitivity state if needed by UI - get initial from config
        # Ensure wake_word_sensitivity exists and is not empty
        sens_config = getattr(config.speech, 'porcupine_sensitivity', [0.5])
        # Ensure it's treated as a list
        if isinstance(sens_config, float):
            sens_list = [sens_config]
        elif isinstance(sens_config, list):
            sens_list = sens_config
        else:
            self.logger.warning(f"Invalid Porcupine sensitivity config: {sens_config}. Using default [0.5].")
            sens_list = [0.5]
            
        self.current_porcupine_sensitivity: float = sens_list[0] if sens_list else 0.5
        self.logger.info(f"Orchestrator initialized. Sensitivity set to {self.current_porcupine_sensitivity}.")

        # Initialization Steps
        # Removed: self._load_corrections() - Called above now
        # Removed: Other internal _init calls

        # Configure callbacks for subsystems that detect things
        self._configure_subsystem_callbacks()

        self.logger.info("EVEOrchestrator initialized with injected subsystems.")

    # --- COMPLETELY REMOVED OBSOLETE INTERNAL INIT METHODS --- 
    # (No code should be here for: _init_configs, _discover_cameras, _discover_cameras_opencv,
    #  _init_camera, _init_camera_picamera2, _init_camera_opencv, _init_subsystems, 
    #  _discover_audio_devices, _update_sensitivity_from_capture)
    
    # --- KEEP Necessary Helper/Internal Methods --- 

    # Keep _download_file if needed for internal OD model downloading (though OD is now injected)
    # If ObjectDetector handles its own downloads, this can be removed too.
    # def _download_file(self, url: str, dest_path: Path): ...

    # Keep _load_corrections as it was called from the new __init__
    def _load_corrections(self) -> List[Dict[str, Any]]:
        """Loads saved object detection corrections from the JSON file."""
        logger.info(f"Attempting to load corrections from {CORRECTIONS_PATH}...")
        # Ensure directory exists first
        try:
            CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             self.logger.error(f"Could not create corrections directory {CORRECTIONS_DIR}: {e}")
             return []

        if CORRECTIONS_PATH.exists():
            try:
                with open(CORRECTIONS_PATH, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if not isinstance(loaded_data, list):
                     self.logger.warning(f"Corrections file {CORRECTIONS_PATH} content is not a list. Resetting.")
                     return []
                else:
                     self.logger.info(f"Loaded {len(loaded_data)} corrections.")
                     return loaded_data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error loading corrections file {CORRECTIONS_PATH}: {e}. Starting with empty list.")
                return [] # Return empty on error
        else:
            self.logger.info("Corrections file not found. Starting with empty list.")
            return [] # Return empty if not found

    # --- Event Handling Methods --- 
    def _default_post_event(self, topic: str, data: Optional[Dict] = None):
       """Default event poster if none provided - puts on internal queue."""
       event = Event(topic, data)
       self.event_queue.put(event)

    def _register_event_handlers(self) -> Dict[str, Callable]:
        """Maps event topics to handler methods within the orchestrator."""
        # Use lowercase keys as defined in communication config
        handlers = {
            # --- Vision Events ---
            TOPICS.get('face_detected', 'face_detected'): self._handle_face_detected,
            TOPICS.get('face_recognized', 'face_recognized'): self._handle_face_recognized,
            TOPICS.get('face_learned', 'face_learned'): self._handle_face_learned,
            TOPICS.get('face_lost', 'face_lost'): self._handle_face_lost,
            TOPICS.get('emotion_detected', 'emotion_detected'): self._handle_emotion_detected,
            TOPICS.get('object_detected', 'object_detected'): self._handle_object_detected, 
             # --- Speech Events ---
            # Assuming these keys exist in TOPICS dict
            TOPICS.get('wake_word_detected', 'wake_word_detected'): self._handle_wake_word_detected,
            TOPICS.get('speech_recognized', 'speech_recognized'): self._handle_speech_recognized,
            TOPICS.get('tts_start', 'tts_start'): self._handle_tts_start, # Assuming these exist
            TOPICS.get('tts_done', 'tts_done'): self._handle_tts_done, # Assuming these exist
            # --- System Events ---
            # Assuming these keys exist in TOPICS dict
            TOPICS.get('system_error', 'error'): self._handle_system_error, # Map to 'error' if system_error doesn't exist
            TOPICS.get('system_learning_started', 'system_learning_started'): self._handle_learning_started,
            TOPICS.get('system_learning_cancelled', 'system_learning_cancelled'): self._handle_learning_cancelled,
        }
        # Filter out entries where the key lookup might have failed (though .get avoids KeyError)
        # handlers = {k: v for k, v in handlers.items() if k is not None} # Optional: Stricter check
        self.logger.info(f"Registered {len(handlers)} event handlers using TOPICS keys: {list(handlers.keys())}")
        return handlers

    def _configure_subsystem_callbacks(self):
        """Sets the orchestrator's post_event method as the callback for subsystems."""
        self.logger.info("Configuring subsystem callbacks...")
        if self.face_detector:
              self.face_detector.post_event = self.post_event
              self.logger.debug("Set post_event callback for FaceDetector.")
        if self.speech_recognizer:
              self.speech_recognizer.wake_word_callback = self._internal_wake_word_callback
              self.speech_recognizer.command_callback = self._internal_command_callback
              self.logger.debug("Set wake_word/command callbacks for SpeechRecognizer.")
        if self.tts and hasattr(self.tts, 'set_callbacks'): 
              # Use .get with defaults for safety
              tts_start_topic = TOPICS.get('tts_start', 'tts_start')
              tts_done_topic = TOPICS.get('tts_done', 'tts_done')
              on_start_cb = lambda text: self.post_event(tts_start_topic, {'text': text})
              on_done_cb = lambda text: self.post_event(tts_done_topic, {'text': text})
              try: 
                  self.tts.set_callbacks(on_start=on_start_cb, on_done=on_done_cb)
                  self.logger.debug("Set TTS start/done callbacks using keywords.")
              except TypeError: 
                  try: 
                      self.tts.set_callbacks(on_start_cb, on_done_cb)
                      self.logger.debug("Set TTS start/done callbacks using positional args.")
                  except Exception as cb_err:
                       self.logger.error(f"Failed to set TTS callbacks: {cb_err}")

    def _internal_wake_word_callback(self):
        """Internal callback from SpeechRecognizer for wake word."""
        # Use .get with defaults for safety
        wake_topic = TOPICS.get('wake_word_detected', 'wake_word_detected')
        self.post_event(wake_topic)

    def _internal_command_callback(self, text: str, confidence: float):
        """Internal callback from SpeechRecognizer for command."""
        # Use .get with defaults for safety
        rec_topic = TOPICS.get('speech_recognized', 'speech_recognized')
        self.post_event(TOPICS['SPEECH_RECOGNIZED'], {'text': text, 'confidence': confidence})

    def _process_event_queue_loop(self) -> None:
        """Dedicated thread to process events from the internal queue."""
        self.logger.info("Event processing loop started.")
        while self._running:
            try:
                event: Event = self.event_queue.get(timeout=0.5) 
                if event.topic == "system.shutdown": # Check for shutdown signal
                    self.logger.debug("Shutdown event received in queue loop.")
                    break
                handler = self.event_handlers.get(event.topic)
                if handler:
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(f"Error in event handler for {event.topic}: {e}", exc_info=True)
                else:
                    self.logger.warning(f"No handler registered for event topic: {event.topic}")
                self.event_queue.task_done()
            except queue.Empty:
                continue # Normal timeout, check _running flag
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}", exc_info=True)
                time.sleep(0.1) # Avoid tight loop on error
        self.logger.info("Event processing loop stopped.")

    # --- Event Handlers --- 
    def _handle_wake_word_detected(self, event: Event):
        """Handles WAKE_WORD_DETECTED event."""
        with self._state_lock:
            if not self._is_listening:
                self.logger.info("Wake word detected! Entering listening state.")
                self._is_listening = True
                self._last_interaction_time = time.time()
                self.set_emotion(Emotion.ATTENTIVE)
                if self.tts: self.tts.speak("Yes?")
            else:
                 self.logger.debug("Wake word detected while already listening - resetting timer.")
                 self._last_interaction_time = time.time()

    def _handle_speech_recognized(self, event: Event):
        """Handles SPEECH_RECOGNIZED event."""
        text = event.data.get('text', '').lower()
        confidence = event.data.get('confidence', 0.0)
        self.last_recognized_text = text 

        should_process = False
        with self._state_lock:
            if self._is_listening:
                 should_process = True
            elif self.audio_debug_listen_always:
                 should_process = True 

        if should_process:
            self.logger.info(f"Command recognized: '{text}' (Conf: {confidence:.2f}) (Processing: {should_process})")
            self._last_interaction_time = time.time()

            stop_phrases = ["stop listening", "never mind", "cancel", "go away", "shut up"]
            if any(phrase in text for phrase in stop_phrases):
                self.logger.info("Stop listening command detected. Exiting listening state.")
                if self.tts: self.tts.speak("Okay.")
                with self._state_lock:
                    self._is_listening = False 
                self.set_emotion(Emotion.NEUTRAL)
                return

            if self.llm_processor:
                self.logger.debug(f"Sending to LLM: '{text}'")
                try:
                    llm_response = self.llm_processor.process(text)
                    if llm_response:
                         self.logger.info(f"LLM Response: '{llm_response}'")
                         if self.tts: self.tts.speak(llm_response)
                    else:
                         self.logger.warning("LLM returned empty response.")
                         if self.tts: self.tts.speak("I don't have a response for that.")
                except Exception as e:
                     self.logger.error(f"Error processing text with LLM: {e}", exc_info=True)
                     if self.tts: self.tts.speak("Sorry, I encountered an error.")
            else:
                self.logger.warning("LLM Processor not available, cannot process command.")
                if self.tts: self.tts.speak(f"I heard {text}, but cannot process it.")

            # Exit Listening State after processing (if we were listening)
            with self._state_lock:
                 if self._is_listening: # Only turn off if it was on
                     self._is_listening = False
                     self.logger.debug("Exited listening state after command processing.")
            self.set_emotion(Emotion.NEUTRAL)

        else:
            self.logger.debug(f"Command '{text}' recognized but ignored (not in listening state or forced listen mode).")

    # ... Other _handle_... methods - ensure they have content or 'pass' ...
    def _handle_tts_start(self, event: Event):
        """Handles TTS_START event."""
        self.logger.debug(f"TTS Started: {event.data.get('text', '')[:30]}...")
        # self.set_emotion(Emotion.TALKING)

    def _handle_tts_done(self, event: Event):
        """Handles TTS_DONE event."""
        self.logger.debug(f"TTS Done: {event.data.get('text', '')[:30]}...")
        # self.set_emotion(Emotion.NEUTRAL)

    def _handle_face_detected(self, event: Event):
        """Handles FACE_DETECTED event."""
        count = event.data.get('count', 0)
        # locations = event.data.get('locations', [])
        self.logger.debug(f"Face(s) detected: count={count}")
        pass # Add logic if needed

    def _handle_face_recognized(self, event: Event):
        """Handles FACE_RECOGNIZED event."""
        name = event.data.get('name', 'Unknown')
        self.logger.info(f"Face recognized: {name}")
        if name != "Unknown" and name != "Error":
             if self.tts and name not in self._detected_persons:
                  self.tts.speak(f"Hello {name}")
                  self._detected_persons[name] = time.time()
             self._last_interaction_time = time.time()
             self.set_emotion(Emotion.HAPPY)

    def _handle_face_learned(self, event: Event):
        """Handles FACE_LEARNED event."""
        name = event.data.get('name', 'Unknown')
        self.logger.info(f"Face learned and saved: {name}")
        if self.tts: self.tts.speak(f"Okay, I've learned your face, {name}.")
        self.set_emotion(Emotion.HAPPY)

    def _handle_face_lost(self, event: Event):
         """Handles FACE_LOST event."""
         self.logger.debug("Face lost.")
         pass # Add logic if needed

    def _handle_emotion_detected(self, event: Event):
         """Handles EMOTION_DETECTED event."""
         emotion = event.data.get('emotion', self.config.display.DEFAULT_EMOTION)
         confidence = event.data.get('confidence', 0.0)
         self.logger.debug(f"Emotion detected by analyzer: {emotion} (Conf: {confidence:.2f})")
         self.set_emotion(emotion) # Update display emotion

    def _handle_object_detected(self, event: Event):
        """Handles OBJECT_DETECTED event."""
        detections = event.data.get('detections', [])
        self.logger.debug(f"Object(s) detected: {len(detections)}")
        pass # Add logic if needed

    def _handle_system_error(self, event: Event):
        """Handles SYSTEM_ERROR event."""
        message = event.data.get('message', 'Unknown error')
        subsystem = event.data.get('subsystem', 'Unknown')
        self.logger.error(f"System Error reported from {subsystem}: {message}")
        self.set_emotion(Emotion.SAD)

    def _handle_learning_started(self, event: Event):
        """Handles SYSTEM_LEARNING_STARTED event."""
        name = event.data.get('name', '')
        self.logger.info(f"System event: Face learning started for {name}.")
        pass # Add logic if needed

    def _handle_learning_cancelled(self, event: Event):
        """Handles SYSTEM_LEARNING_CANCELLED event."""
        self.logger.info("System event: Face learning cancelled.")
        pass # Add logic if needed

    def _handle_wake_word(self):
        """Handles the wake word detection."""
        # Avoid triggering if already listening
        with self._state_lock:
            if self._is_listening:
                self.logger.debug("Wake word detected, but already listening.")
                return

            self.logger.info("Wake word detected! Now listening for command.")
            self._is_listening = True
            self._last_interaction_time = time.time() # Reset listening timeout
        
        self.set_emotion(Emotion.ATTENTIVE) # Change emotion to show listening
        
        # Optional: Play an acknowledgement sound
        if self.tts:
             try:
                 # Use async speak so it doesn't block command recognition
                 self.tts.speak("Yes?") 
             except Exception as e:
                 self.logger.warning(f"Error playing wake word acknowledgement sound: {e}")

    def _handle_command(self, text: str, confidence: float):
        """Handles a recognized command."""
        # ... existing _handle_command logic ...

    # --- Start / Stop / Cleanup --- 
    def start(self):
        """Start the EVE Orchestrator and all its subsystems."""
        if self._running:
            self.logger.warning("Orchestrator already running.")
            return True

        self.logger.info("Starting EVE Orchestrator...")
        self._running = True

        try:
            # Start Camera (Assumes camera object has start method if needed)
            if self.camera and hasattr(self.camera, 'start'):
                if not self.camera.start():
                    self.logger.error("Failed to start Camera subsystem! Vision may not work.")
                    return False

            # Start Face Detector Thread
            if self.face_detector and hasattr(self.face_detector, 'start'):
                if not self.face_detector.start():
                    self.logger.error("Failed to start FaceDetector thread!")
                    return False

            # Start Object Detector Thread (if it runs threaded)
            if self.object_detector and hasattr(self.object_detector, 'start'):
                if not self.object_detector.start():
                    self.logger.error("Failed to start ObjectDetector thread!")
                    return False

            # Start Audio Capture and Processing Thread
            if self.audio_capture:
                # Start recognizer thread if it runs separately
                if self.speech_recognizer and hasattr(self.speech_recognizer, 'start'):
                    self.logger.debug("Starting Speech Recognizer thread (if applicable)...")
                    if not self.speech_recognizer.start():
                        self.logger.error("Failed to start Speech Recognizer!")
                        return False
            elif not self.config.hardware.audio_input_enabled:
                self.logger.info("Audio input is explicitly disabled in config.")
            else:
                self.logger.warning("AudioCapture not available or not started externally. Audio input disabled.")

            # Start Display Controller/Thread (if applicable)
            if self.display_controller and hasattr(self.display_controller, 'start'):
                if not self.display_controller.start():
                    self.logger.error("Failed to start Display Controller!")
                    return False

            # Start the internal event processing loop
            if not self._event_thread or not self._event_thread.is_alive():
                self._event_thread = threading.Thread(target=self._process_event_queue_loop, daemon=True)
                self._event_thread.start()
                self.logger.info("Orchestrator event processing thread started.")

            self.logger.info("EVE Orchestrator started successfully.")
            # Play startup sound after starting TTS subsystem if needed
            if self.tts and hasattr(self.tts, 'play_startup_sound'):
                self.tts.play_startup_sound()
            elif self.tts and hasattr(self.tts, 'speak_sync'): # Fallback
                self.tts.speak_sync("System initialized.")
            return True

        except Exception as e:
            self.logger.error(f"Error starting orchestrator: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._running = False
            return False

    def stop(self):
        # ... (Refactored implementation using injected subsystems) ...
        if not self._running:
            self.logger.warning("Orchestrator already stopped.")
            return

        self.logger.info("Stopping EVE Orchestrator...")
        self._running = False # Signal loops to stop

        # Stop Event Loop First
        if self._event_thread and self._event_thread.is_alive():
            self.logger.debug("Joining event thread...")
            # Add a dummy event to wake up the queue.get timeout
            try: self.event_queue.put(Event("system.shutdown")) 
            except Exception: pass
            self._event_thread.join(timeout=2.0)
            if self._event_thread.is_alive(): self.logger.warning("Event thread did not stop.")
        self._event_thread = None

        # --- Stop Vision Subsystems FIRST (Detectors then Display then Camera) ---
        # Stop detectors first, as display loop might call them
        if self.face_detector and hasattr(self.face_detector, 'stop'):
            self.logger.debug("Stopping Face Detector...")
            self.face_detector.stop()
        if self.object_detector and hasattr(self.object_detector, 'stop'):
            self.logger.debug("Stopping Object Detector...")
            self.object_detector.stop()

        # Stop Display Controller AFTER detectors
        if self.display_controller and hasattr(self.display_controller, 'stop'):
            self.logger.debug("Stopping Display Controller...")
            self.display_controller.stop()

        # Stop Camera AFTER display (which uses it)
        if self.camera and hasattr(self.camera, 'stop'):
            self.logger.debug("Stopping Camera...")
            self.camera.stop()
        # -------------------------------------------------------------------------

        # --- Stop Speech Subsystems --- 
        if self.speech_recognizer and hasattr(self.speech_recognizer, 'stop'):
             self.logger.debug("Stopping Speech Recognizer...")
             self.speech_recognizer.stop()
        if self.audio_capture and hasattr(self.audio_capture, 'stop'): 
            self.logger.debug("Stopping Audio Capture...")
            self.audio_capture.stop()
        if self.tts and hasattr(self.tts, 'stop'):
             self.logger.debug("Stopping TTS...")
             self.tts.stop()
        if self.llm_processor and hasattr(self.llm_processor, 'stop'):
             self.logger.debug("Stopping LLM Processor...")
             self.llm_processor.stop()

        self.logger.info("EVE Orchestrator stopped.")

    def cleanup(self):
        """Alias for stop() for consistency."""
        self.stop()

    def __enter__(self):
        # Start is now typically called externally via main()
        # self.start() 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # --- State Management --- 
    def set_emotion(self, emotion: Union[str, Emotion]):
        """Sets the current emotion state safely."""
        # Restoring correct indentation and body
        try:
            new_emotion = Emotion.from_value(emotion) 
            with self._state_lock:
                if self._current_emotion != new_emotion:
                    self.logger.debug(f"Changing emotion from {self._current_emotion.name} to {new_emotion.name}")
                    self._current_emotion = new_emotion
            # Update display controller immediately? 
            if self.display_controller and hasattr(self.display_controller, 'set_emotion'):
                 self.display_controller.set_emotion(new_emotion)
        except ValueError:
             self.logger.warning(f"Attempted to set invalid emotion: {emotion}")

    def get_current_emotion(self) -> Emotion:
        """Gets the current emotion state safely."""
        # Restoring correct indentation and body
        with self._state_lock:
            return self._current_emotion

    # --- Update Loop --- 
    def update(self):
        """
        Main update method, called periodically.
        Handles state updates, checks timeouts, processes audio, updates display.
        """
        # --- State Checks / Timeouts ---
        with self._state_lock:
            is_listening = self._is_listening # Get current state under lock
            if is_listening:
                timeout_duration = 10.0
                if time.time() - self._last_interaction_time > timeout_duration:
                    self.logger.info(f"Listening timed out after {timeout_duration}s of inactivity.")
                    self._is_listening = False
                    self.set_emotion(Emotion.NEUTRAL)
                    if self.tts: # Check if tts exists
                        try: self.tts.speak("Never mind.")
                        except Exception as tts_err: self.logger.warning(f"Error calling TTS during listening timeout: {tts_err}")
            current_emotion = self._current_emotion # Get current emotion under lock

        # --- Check for Wake Word Detection (from AudioCapture) ---
        if self.audio_capture and not is_listening: # Only check if not already listening
            detected_word = self.audio_capture.check_for_wake_word()
            if detected_word:
                 self.logger.info(f"Orchestrator received wake word: {detected_word}")
                 self._handle_wake_word() # Call the handler
                 # Update state immediately for next check
                 is_listening = True 

        # --- Vision Processing ---
        latest_frame = None
        if self.camera and self.camera.is_running():
            latest_frame = self.camera.get_latest_frame()

        detected_objects = [] # Store results here
        detected_faces = []
        detected_emotions = {}

        if latest_frame is not None:
            current_time = time.time()

            # --- BRANCH BASED ON CAMERA TYPE --- 
            if isinstance(self.camera, RPiAICamera):
                # --- RPi AI Camera Pathway ---
                self.logger.debug("Using RPi AI Camera pathway for detections.")
                ai_results = self.camera.get_latest_ai_results()
                if ai_results:
                    # Assuming ai_results is a list of dicts like:
                    # {"label": str, "score": float, "bbox": [xmin, ymin, xmax, ymax]}
                    # Filter by confidence threshold from config
                    od_conf = getattr(self.config.vision.object_detection, 'confidence', 0.5)
                    detected_objects = [
                        res for res in ai_results 
                        if res.get('score', 0.0) >= od_conf
                    ]
                    # TODO: The RPi AI Camera might also provide face/emotion data
                    # in the metadata. If so, parse and populate detected_faces
                    # and detected_emotions here instead of using separate modules.
                    # Example placeholder:
                    # detected_faces = parse_faces_from_ai_metadata(ai_results)
                    # detected_emotions = parse_emotions_from_ai_metadata(ai_results)
                    if detected_objects: self.logger.debug(f"RPi AI detected objects: {len(detected_objects)}")
                
            else:
                # --- Host-based Detection Pathway (CPU/Other Accelerator) ---
                self.logger.debug("Using host-based detection pathway.")
                # Face Detection
                if self.face_detector: # Check if detector exists
                    # Add interval check if needed
                    # detected_faces = self.face_detector.detect(latest_frame)
                    # Assuming detect method returns list of face bounding boxes or similar
                    # Placeholder call, adjust based on actual FaceDetector method
                    detected_faces = [] # Replace with actual call

                # Object Detection
                if self.object_detector: # Check if detector exists
                    # Add interval check if needed
                    od_results = self.object_detector.detect(latest_frame) # Assuming detect returns list of dicts
                    if od_results:
                         detected_objects = od_results # Use results directly
                    if detected_objects: self.logger.debug(f"Host detector found objects: {len(detected_objects)}")

                # Emotion Analysis (if face detected)
                if self.emotion_analyzer and detected_faces:
                    # Placeholder - requires detected_faces to be populated correctly
                    # detected_emotions = self.emotion_analyzer.analyze(latest_frame, detected_faces)
                    pass
            # --- END BRANCH --- 

            # --- Process Detections (Common Logic) ---
            # Update display with frame and detections
            if self.display_controller and hasattr(self.display_controller, 'update_overlays'):
                self.display_controller.update_overlays(latest_frame, detected_faces, detected_objects, detected_emotions)
            
            # Post events based on detections (example for objects)
            if detected_objects: 
                 self.post_event(TOPICS.VISION_OBJECTS_DETECTED, {"objects": detected_objects})
            # Post events for faces, emotions etc. similarly

        # --- Audio RMS Update (Moved from removed STT block) ---
        if self.audio_capture:
             self.last_audio_rms = self.audio_capture.get_last_rms()

        # --- Update Display Controller (Main Update) ---
        if self.display_controller and hasattr(self.display_controller, 'update'):
            try:
                # Get latest state again in case it changed during processing
                with self._state_lock: 
                    is_listening_now = self._is_listening
                    current_emotion_now = self._current_emotion
                
                # Pass necessary state to display update
                display_state = {
                    "emotion": current_emotion_now,
                    "is_listening": is_listening_now,
                    "last_rms": self.last_audio_rms,
                    # Add other relevant state: last_recognized_text, debug flags etc.
                    # "last_recognized_text": self.last_recognized_text,
                    # "debug_menu_active": self.debug_menu_active, 
                    # "current_debug_view": self.current_debug_view,
                }
                self.display_controller.update(display_state)
                
            except Exception as display_err:
                 self.logger.error(f"Error in display_controller.update: {display_err}", exc_info=True)

        # --- End of Update Loop --- 

    # --- Debug / Correction Methods --- 
    def handle_debug_ui_click(self, element_id: str):
        """Handles clicks on debug UI elements."""
        # Restoring correct indentation and body
        self.logger.debug(f"Handling debug UI click: {element_id}")
        # (Add the previous implementation for handling menu navigation, 
        #  camera selection, rotation, correction initiation, audio toggles etc.)
        # --- Menu Navigation Clicks --- 
        if element_id == "menu_video":
             self.set_debug_view('VIDEO')
             return
        if element_id == "menu_audio":
             self.set_debug_view('AUDIO')
             return
        if element_id == "back_to_menu":
             self.set_debug_view(None)
             return
        # --- Other handlers based on current_debug_view ---
        # (Code for VIDEO view clicks: select_cam, rotate, correct_det)
        # (Code for AUDIO view clicks: audio_toggle_listen, select_audio_dev, sensitivity)
        # ... (Full implementation needed here)
        else:
             self.logger.warning(f"Unhandled debug UI element click: {element_id} (Current view: {self.current_debug_view})")

    def submit_correction(self, corrected_label: str):
        """Handles the submitted object detection correction."""
        # Restoring correct indentation and body
        if not self.is_correcting_detection or not self.correction_target_info:
             self.logger.warning("Submit correction called but not in correction mode.")
             return
             
        original_label = self.correction_target_info.get('label')
        box = self.correction_target_info.get('box')
        timestamp = time.time()
        
        correction_record = {
            "timestamp": timestamp,
            "box": box,
            "original": original_label,
            "corrected": corrected_label.strip()
        }
        self.logger.info(f"Correction submitted: {correction_record}")
        
        try:
            # Load existing, append, save
            current_corrections = self._load_corrections() # Reload fresh list
            current_corrections.append(correction_record)
            with open(CORRECTIONS_PATH, 'w', encoding='utf-8') as f:
                json.dump(current_corrections, f, indent=4)
            self.logger.info(f"Correction saved to {CORRECTIONS_PATH}")
            self.corrections_data = current_corrections # Update memory
        except (IOError, OSError) as e:
            self.logger.error(f"Error saving correction to {CORRECTIONS_PATH}: {e}")

        if self.tts: self.tts.speak(f"Okay, noted.")
        self.is_correcting_detection = False
        self.correction_target_info = None
        self.user_input_buffer = ""
        
    def cancel_correction(self):
        """Cancels the current correction input."""
        # Restoring correct indentation and body
        if self.is_correcting_detection:
            self.logger.info("Correction cancelled.")
            self.is_correcting_detection = False
            self.correction_target_info = None
            self.user_input_buffer = ""
            if self.tts: self.tts.speak("Correction cancelled.")

    def set_debug_view(self, view_name: Optional[str]):
        """Sets the active debug view."""
        # Restoring correct indentation and body
        self.logger.info(f"Setting debug view to: {view_name}")
        if view_name not in [None, 'VIDEO', 'AUDIO']:
             self.logger.warning(f"Invalid debug view name: {view_name}. Setting to None.")
             self.current_debug_view = None
        else:
             self.current_debug_view = view_name
        self.cancel_correction() # Reset correction state when changing views

# --- Helper Functions / Main Execution --- 
# ... (main() and setup_logging() remain the same as refactored previously) ...

def create_orchestrator():
    """Create and initialize an EVE orchestrator instance"""
    try:
        # Create orchestrator with flat configuration
        orchestrator = EVEOrchestrator()
        return orchestrator
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating orchestrator: {e}")
        raise

def setup_logging(log_config: Any): # Use LoggingConfig type hint if possible
    """Configures logging based on the config object."""
    # Adjusted to use attributes from LoggingConfig dataclass
    level = getattr(log_config, 'level', 'INFO').upper()
    log_file = getattr(log_config, 'file', None)
    max_bytes = getattr(log_config, 'max_size_mb', 10) * 1024 * 1024
    backup_count = getattr(log_config, 'backup_count', 3)
    log_to_console = getattr(log_config, 'console', True)
    log_format = getattr(log_config, 'format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = getattr(log_config, 'date_format', "%Y-%m-%d %H:%M:%S")

    logging_utils.setup_logging(level, log_file, max_bytes, backup_count, log_to_console, log_format, date_format)

def main():
    """Main function to initialize and run the EVE Orchestrator."""

    # 1. Load Configuration using the new load_config function
    config = load_config() # Load from default config.yaml path

    # 2. Setup Logging (using the loaded config.logging section)
    # Pass the logging section of the SystemConfig object
    setup_logging(config.logging) 
    logger.info("--- EVE System Starting ---")
    # Use repr for potentially cleaner logging of the dataclass structure
    logger.info(f"Loaded configuration: {config!r}") 

    # 3. Initialize Subsystems (using loaded config sections)
    #    Use try/except blocks for non-critical subsystems
    camera = None
    face_detector = None
    object_detector = None
    emotion_analyzer = None
    audio_capture = None
    speech_recognizer = None
    llm_processor = None
    tts = None
    display_controller = None

    # Initialize Camera (needs hardware and vision config sections?)
    if config.hardware.camera_enabled:
        try:
            # Assuming Camera now takes the SystemConfig object directly
            camera = Camera(config)
            if not camera.is_open():
                 logger.warning("Camera failed to initialize or is disabled. Vision features may be limited.")
                 camera = None # Ensure camera is None if not opened
        except Exception as e:
            logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
            camera = None # Ensure camera is None on error
    else:
        logger.info("Camera subsystem disabled by config.")

    # Initialize Vision Subsystems only if camera is available
    if camera: 
        if config.vision.face_detection_enabled:
             try:
                  # FaceDetector likely needs the main SystemConfig
                  face_detector = FaceDetector(config, camera) # Pass full config and camera instance
             except Exception as e: logger.error(f"Failed FaceDetector init: {e}", exc_info=True)
        else:
             logger.info("Face Detector disabled by config.")

        if config.vision.object_detection_enabled:
             try:
                  # ObjectDetector likely needs the main SystemConfig
                  object_detector = ObjectDetector(config=config) # Pass full config
             except Exception as e: logger.error(f"Failed ObjectDetector init: {e}", exc_info=True)
        else:
            logger.info("Object Detector disabled by config.")

        if config.vision.emotion_detection_enabled:
             try:
                  # EmotionAnalyzer likely needs the main SystemConfig
                  emotion_analyzer = EmotionAnalyzer(config)
             except Exception as e: logger.error(f"Failed EmotionAnalyzer init: {e}", exc_info=True)
        else:
            logger.info("Emotion Analyzer disabled by config.")

    # Initialize VisionDisplay if display is enabled
    if config.hardware.display_enabled:
        try:
            # VisionDisplay needs the full config and initialized components
            display_controller = VisionDisplay(config, camera, face_detector, object_detector)
        except Exception as e: 
            logger.error(f"Failed VisionDisplay init: {e}", exc_info=True)
            display_controller = None # Ensure None on error
    else:
         logger.info("VisionDisplay disabled by config.")

    # Initialize Speech Subsystems
    if config.hardware.audio_input_enabled:
        try:
            # AudioCapture likely needs the speech config section
            audio_capture = AudioCapture(config.speech) 
        except Exception as e: logger.error(f"Failed AudioCapture init: {e}", exc_info=True)
    else:
        logger.info("Audio Input disabled by config.")

    # Dependent speech components
    if audio_capture and config.speech.recognition_enabled: 
        try:
            # SpeechRecognizer likely needs the speech config section
            speech_recognizer = SpeechRecognizer(config.speech)
        except Exception as e: logger.error(f"Failed SpeechRecognizer init: {e}", exc_info=True)
    elif not audio_capture:
        logger.warning("Cannot initialize SpeechRecognizer: AudioCapture failed or disabled.")
    else: # audio_capture exists but recognition disabled
        logger.info("Speech Recognition disabled by config.")

    if config.speech.tts_enabled:
        try: 
             # TTS likely needs the speech config section
             tts = TextToSpeech(config.speech)
        except Exception as e: logger.error(f"Failed TextToSpeech init: {e}", exc_info=True)
    else:
        logger.info("TextToSpeech disabled by config.")

    if config.speech.llm_enabled:
        try: 
             # LLMProcessor likely needs the speech config section
             llm_processor = LLMProcessor(config.speech)
        except Exception as e: logger.error(f"Failed LLMProcessor init: {e}", exc_info=True)
    else:
        logger.info("LLM Processor disabled by config.")

    # Initialize non-vision display (e.g., LCD) if not using VisionDisplay
    # Example: Assuming an LCDController exists and uses config.display
    # if config.hardware.display_enabled and not display_controller:
    #      try:
    #           from eve.display.lcd_controller import LCDController # Import here if not at top
    #           display_controller = LCDController(config.display) 
    #      except Exception as e: logger.error(f"Failed LCDController init: {e}", exc_info=True)
    # elif not config.hardware.display_enabled:
    #      logger.info("Display hardware disabled by config.")

    # 4. Create Orchestrator with Injected Subsystems
    try:
        orchestrator = EVEOrchestrator(
            config=config, # Pass the loaded SystemConfig
            camera=camera,
            face_detector=face_detector,
            object_detector=object_detector,
            emotion_analyzer=emotion_analyzer,
            audio_capture=audio_capture,
            speech_recognizer=speech_recognizer,
            llm_processor=llm_processor,
            tts=tts,
            display_controller=display_controller
            # post_event_callback= # Add if using external message queue
        )
    except Exception as e:
        logger.critical(f"Fatal error creating Orchestrator: {e}", exc_info=True)
        sys.exit(1) # Exit if orchestrator fails

    # 5. Start Orchestrator and Main Loop
    try:
        orchestrator.start() # Start subsystem threads/loops

        # --- Main Application Loop ---
        while orchestrator._running: # Check orchestrator running flag
            orchestrator.update()
            time.sleep(0.05) # Example: ~20 FPS loop rate

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up orchestrator...")
        if 'orchestrator' in locals():
             orchestrator.stop() # Ensure stop is called
        logger.info("--- EVE System Shutdown Complete ---")

if __name__ == "__main__":
    main() 