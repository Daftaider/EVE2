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
from eve.vision.face_detector import FaceDetector
from eve.display.lcd_controller import LCDController

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

logger = logging.getLogger(__name__)

class EVEOrchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self):
        """Initialize the EVE orchestrator"""
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.event_queue = queue.Queue()
        
        # Initialize state
        self.current_emotion = "neutral"
        self.last_face_detected = 0
        self.last_speech_detected = 0
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Define event handlers
        self.event_handlers = {
            config.communication.TOPICS['SPEECH_RECOGNIZED']: self._handle_speech_recognition,
            config.communication.TOPICS['FACE_DETECTED']: self._handle_face_detection,
            config.communication.TOPICS['FACE_LOST']: self._handle_face_lost,
            config.communication.TOPICS['EMOTION_DETECTED']: self._handle_emotion_detection,
            config.communication.TOPICS['AUDIO_LEVEL']: self._handle_audio_level,
            config.communication.TOPICS['ERROR']: self._handle_error,
        }

    def _init_subsystems(self) -> None:
        """Initialize all subsystems with proper error handling"""
        # Initialize speech subsystem
        try:
            # Create speech configuration dictionary
            speech_params = {
                'sample_rate': getattr(speech_config, 'SAMPLE_RATE', 16000),
                'channels': getattr(speech_config, 'CHANNELS', 1),
                'chunk_size': getattr(speech_config, 'CHUNK_SIZE', 1024),
                'threshold': getattr(speech_config, 'THRESHOLD', 0.01)
            }
            
            # Initialize audio capture
            self.audio_capture = AudioCapture(**speech_params)
            
            # Get model type if available
            model_type = getattr(speech_config, 'MODEL_TYPE', 'google')
            
            # Initialize speech recognizer
            self.speech_recognizer = SpeechRecognizer(
                config=speech_config,
                post_event_callback=self.post_event,
                model_type=model_type
            )
            
            self.logger.info("Speech subsystem initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize speech subsystem: {e}")
            raise

        # Initialize display subsystem
        try:
            # Create display configuration dictionary
            display_params = {
                'width': getattr(display_config, 'WIDTH', 800),
                'height': getattr(display_config, 'HEIGHT', 480),
                'fps': getattr(display_config, 'FPS', 30),
                'default_emotion': getattr(display_config, 'DEFAULT_EMOTION', 'neutral'),
                'background_color': getattr(display_config, 'BACKGROUND_COLOR', (0, 0, 0)),
                'eye_color': getattr(display_config, 'EYE_COLOR', (0, 191, 255))
            }
            
            # Initialize LCD controller
            self.lcd_controller = LCDController(**display_params)
            
            self.logger.info("Display subsystem initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize display subsystem: {e}")
            raise

        # Initialize vision subsystem
        try:
            # Create vision configuration dictionary
            vision_params = {
                'camera_index': getattr(vision_config, 'CAMERA_INDEX', 0),
                'resolution': getattr(vision_config, 'RESOLUTION', (640, 480)),
                'fps': getattr(vision_config, 'FPS', 30)
            }
            
            # Initialize face detector
            self.face_detector = FaceDetector(
                config=vision_params,
                post_event_callback=self.post_event
            )
            
            self.logger.info("Vision subsystem initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize vision subsystem: {e}")
            raise
    
    def start(self) -> None:
        """Start all subsystems and the main event loop."""
        if self.running:
            logger.warning("EVE2 system is already running")
            return
        
        logger.info("Starting EVE2 system")
        self.running = True
        
        # Start face detector if enabled
        if self.face_detector:
            threading.Thread(target=self.face_detector.start, daemon=True).start()
        
        # Start audio capture if enabled
        if self.audio_capture:
            threading.Thread(target=self.audio_capture.start, daemon=True).start()
        
        # Start LCD controller if enabled
        if self.lcd_controller:
            threading.Thread(target=self.lcd_controller.start, daemon=True).start()
            # Set initial emotion
            self.lcd_controller.set_emotion(self.current_emotion)
        
        # Start the main event processing loop
        threading.Thread(target=self._process_events, daemon=True).start()
        
        logger.info("EVE2 system started successfully")
    
    def stop(self) -> None:
        """Stop all subsystems and the main event loop."""
        if not self.running:
            logger.warning("EVE2 system is not running")
            return
        
        logger.info("Stopping EVE2 system")
        self.running = False
        
        # Stop face detector if enabled
        if self.face_detector:
            self.face_detector.stop()
        
        # Stop audio capture if enabled
        if self.audio_capture:
            self.audio_capture.stop()
        
        # Stop LCD controller if enabled
        if self.lcd_controller:
            self.lcd_controller.stop()
        
        logger.info("EVE2 system stopped successfully")
    
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
            if confidence >= self.config.speech.MIN_CONFIDENCE:
                # Generate response using LLM
                response = self.llm_processor.process(text)
                
                # Speak the response
                if response:
                    self.text_to_speech.speak(response)
                    
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
            if confidence >= self.config.vision.EMOTION_CONFIDENCE_THRESHOLD:
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
            if level > self.config.audio.REACTION_THRESHOLD:
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

class Event:
    """Event class for internal communication"""
    def __init__(self, topic, data):
        self.topic = topic
        self.data = data
        self.timestamp = time.time()

def create_orchestrator() -> EVEOrchestrator:
    """Create and return an Orchestrator instance."""
    return EVEOrchestrator()


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
        orchestrator.stop() 