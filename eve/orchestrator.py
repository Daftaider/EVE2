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
        """Initialize the EVE orchestrator"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration as SimpleNamespace
        self.config = config or {}
        self.config.SPEECH = speech_config
        self.config.VISION = vision_config
        self.config.DISPLAY = display_config
        
        # Initialize all attributes to None first
        self.audio_capture = None
        self.speech_recognizer = None
        self.text_to_speech = None
        self.llm_processor = None
        self.lcd_controller = None
        self.face_detector = None
        self.emotion_analyzer = None
        self.vision_display = None
        self.running = False
        self.stopped = False
        
        # Initialize subsystems
        try:
            self._init_subsystems()
        except Exception as e:
            self.logger.error(f"Error creating orchestrator: {e}")
            raise

    def _init_subsystems(self):
        """Initialize all subsystems"""
        try:
            # Initialize audio capture first
            audio_config = getattr(self.config.SPEECH, 'AUDIO_CAPTURE', {})
            self.audio_capture = AudioCapture(
                sample_rate=getattr(audio_config, 'sample_rate', 16000),
                channels=getattr(audio_config, 'channels', 1),
                chunk_size=getattr(audio_config, 'chunk_size', 1024),
                format=getattr(audio_config, 'format', 'int16')
            )
            self.logger.info("Audio capture initialized successfully")

            # Initialize speech recognition
            self.speech_recognizer = SpeechRecognizer(self.config.SPEECH)
            self.logger.info("Speech recognition initialized successfully")

            # Initialize text to speech
            tts_config = getattr(self.config.SPEECH, 'TEXT_TO_SPEECH', {})
            self.text_to_speech = TextToSpeech(
                engine=getattr(tts_config, 'engine', 'pyttsx3'),
                voice=getattr(tts_config, 'voice', 'english'),
                rate=getattr(tts_config, 'rate', 150),
                volume=getattr(tts_config, 'volume', 1.0)
            )
            self.logger.info("Text to speech initialized successfully")
            
            # Initialize LLM processor
            self.llm_processor = LLMProcessor(self.config.SPEECH)
            self.logger.info("LLM processor initialized successfully")

            # Initialize display subsystem with proper emotion enum
            display_config = self.config.get('display', {})
            
            # Convert emotion value to proper Enum if it's an integer
            default_emotion = display_config.get('default_emotion')
            if isinstance(default_emotion, int):
                # Map integer to Emotion enum (adjust mapping as needed)
                emotion_map = {
                    0: Emotion.NEUTRAL,
                    1: Emotion.HAPPY,
                    2: Emotion.SAD,
                    3: Emotion.ANGRY,
                    4: Emotion.SURPRISED,
                    5: Emotion.CONFUSED
                }
                default_emotion = emotion_map.get(default_emotion, Emotion.NEUTRAL)
            elif isinstance(default_emotion, str):
                # Convert string to Emotion enum
                try:
                    default_emotion = Emotion[default_emotion.upper()]
                except (KeyError, AttributeError):
                    default_emotion = Emotion.NEUTRAL
            elif default_emotion is not None and not isinstance(default_emotion, Emotion):
                default_emotion = Emotion.NEUTRAL

            self.lcd_controller = LCDController(
                width=display_config.get('width'),
                height=display_config.get('height'),
                fps=display_config.get('fps'),
                default_emotion=default_emotion,
                background_color=display_config.get('background_color'),
                eye_color=display_config.get('eye_color')
            )
            self.logger.info("Display subsystem initialized successfully")

            # Initialize vision subsystems
            self.face_detector = FaceDetector()
            self.emotion_analyzer = EmotionAnalyzer()
            self.logger.info("Vision subsystems initialized successfully")

            # Initialize vision display
            self.vision_display = VisionDisplay(self.config)
            self.vision_display.start()

        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            self.cleanup()
            raise

    def start(self):
        """Start all subsystems and perform initialization sequence"""
        try:
            self.running = True
            
            # Start subsystems
            if self.audio_capture:
                self.audio_capture.start()
                self.logger.info("Audio capture started")
            
            if self.lcd_controller:
                self.lcd_controller.start()
                self.logger.info("Display started")
            
            if self.face_detector:
                self.face_detector.start()
                self.logger.info("Face detection started")
            
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
            if self.text_to_speech:
                self.text_to_speech.play_startup_sound()
            
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
            # Stop audio subsystems
            if hasattr(self, 'audio_capture') and self.audio_capture:
                self.audio_capture.stop()
                self.logger.info("Audio capture stopped")

            # Stop display
            if hasattr(self, 'lcd_controller') and self.lcd_controller:
                self.lcd_controller.stop()
                self.logger.info("Display stopped")

            # Stop vision subsystems
            if hasattr(self, 'face_detector') and self.face_detector:
                self.face_detector.stop()
                self.logger.info("Face detector stopped")

            # Stop any other active components
            if hasattr(self, 'text_to_speech') and self.text_to_speech:
                self.text_to_speech.stop()
                self.logger.info("Text to speech stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.stopped = True
            self.cleanup()
            self.logger.info("EVE orchestrator stopped")

    def cleanup(self):
        """Clean up resources and perform final shutdown tasks"""
        try:
            # Release any remaining resources
            self.audio_capture = None
            self.speech_recognizer = None
            self.text_to_speech = None
            self.llm_processor = None
            self.lcd_controller = None
            self.face_detector = None
            self.emotion_analyzer = None
            self.vision_display = None
            
            # Clean up pygame
            try:
                import pygame
                pygame.quit()
            except:
                pass

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
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

    def _process_speech(self, text):
        """Process recognized speech"""
        try:
            if text:
                # Process the command through LLM
                response = self.llm_processor.process_text(text)
                
                # Speak the response
                if response:
                    self.text_to_speech.speak(response)
                    
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
                    self.text_to_speech.speak("Hello! I don't recognize you. What's your name?")
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
                        self.text_to_speech.speak(prompts[count])
                    
                elif action == "learning_complete":
                    self.text_to_speech.speak(
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