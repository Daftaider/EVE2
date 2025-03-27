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

logger = logging.getLogger(__name__)

class EVEOrchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EVE orchestrator.
        
        Args:
            config: Optional configuration dictionary with subsystem configs
        """
        self.logger = logging.getLogger(__name__)
        
        self.config = config or {}
        self._init_configs()
        self._init_subsystems()
        self._last_update = time.time()
        self._current_emotion = Emotion.NEUTRAL

    def _init_configs(self):
        """Initialize configuration objects for each subsystem."""
        try:
            # Initialize display config
            display_dict = self.config.get('display', {})
            self.display_config = DisplayConfig()
            for key, value in display_dict.items():
                if hasattr(self.display_config, key):
                    setattr(self.display_config, key, value)

            # Initialize speech config
            speech_dict = self.config.get('speech', {})
            self.speech_config = SpeechConfig.from_dict(speech_dict)

        except Exception as e:
            self.logger.error(f"Error initializing configs: {e}")
            raise

    def _init_subsystems(self):
        """Initialize all subsystems with their respective configs."""
        try:
            # Initialize display subsystem
            self.lcd_controller = LCDController(
                config=self.display_config,
                width=getattr(self.display_config, 'WINDOW_SIZE', (800, 480))[0],
                height=getattr(self.display_config, 'WINDOW_SIZE', (800, 480))[1],
                fps=getattr(self.display_config, 'FPS', 30),
                default_emotion=getattr(self.display_config, 'DEFAULT_EMOTION', Emotion.NEUTRAL),
                background_color=getattr(self.display_config, 'DEFAULT_BACKGROUND_COLOR', (0, 0, 0)),
                eye_color=getattr(self.display_config, 'DEFAULT_EYE_COLOR', (255, 255, 255))
            )

            # Initialize speech subsystem (if you have one)
            self.speech_system = self._init_speech_system()

            logging.info("All subsystems initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize subsystems: {e}")
            raise

    def _init_speech_system(self):
        """Initialize the speech subsystem."""
        try:
            # Your speech system initialization code here
            # Use self.speech_config for configuration
            pass
        except Exception as e:
            logging.error(f"Failed to initialize speech system: {e}")
            raise

    def start(self):
        """Start all subsystems and perform initialization sequence"""
        try:
            self.running = True
            
            # Start subsystems
            if self.lcd_controller:
                self.lcd_controller.start()
                self.logger.info("Display started")
            
            if self.speech_system:
                self.speech_system.start()
                self.logger.info("Speech system started")
            
            # Perform initialization sequence
            self._perform_init_sequence()
            
            self.logger.info("All subsystems started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting subsystems: {e}")
            self.cleanup()
            raise

    def _perform_init_sequence(self):
        """Perform initialization sequence with visual and audio feedback"""
        try:
            # Short delay to ensure all systems are ready
            time.sleep(0.5)
            
            # Perform blink animation
            if self.lcd_controller:
                self.lcd_controller.blink()
            
            # Play startup sound
            if self.speech_system:
                self.speech_system.play_startup_sound()
            
            self.logger.info("Initialization sequence completed")
        except Exception as e:
            self.logger.error(f"Error during initialization sequence: {e}")

    def stop(self):
        """Stop all subsystems gracefully"""
        if self.stopped:
            return
        
        self.logger.info("Stopping EVE orchestrator...")
        self.running = False
        
        try:
            # Stop subsystems
            if hasattr(self, 'lcd_controller') and self.lcd_controller:
                self.lcd_controller.stop()
                self.logger.info("Display stopped")
            
            if hasattr(self, 'speech_system') and self.speech_system:
                self.speech_system.stop()
                self.logger.info("Speech system stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.stopped = True
            self.cleanup()
            self.logger.info("EVE orchestrator stopped")

    def update(self):
        """Update all subsystems."""
        try:
            current_time = time.time()
            
            # Update display
            if hasattr(self, 'lcd_controller'):
                self.lcd_controller.update(self._current_emotion)
            
            # Update other subsystems as needed
            # Add your update logic here
            
            # Example: cycle through emotions every 5 seconds
            if current_time - self._last_update > 5:
                self._cycle_emotion()
                self._last_update = current_time
                
        except Exception as e:
            logging.error(f"Error in update loop: {e}")
            raise

    def _cycle_emotion(self):
        """Cycle through emotions for testing."""
        emotions = list(Emotion)
        current_index = emotions.index(self._current_emotion)
        next_index = (current_index + 1) % len(emotions)
        self._current_emotion = emotions[next_index]
        logging.info(f"Switching to emotion: {self._current_emotion.name}")

    def cleanup(self):
        """Cleanup all subsystems."""
        try:
            if hasattr(self, 'lcd_controller'):
                self.lcd_controller.cleanup()
            # Add cleanup for other subsystems
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

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