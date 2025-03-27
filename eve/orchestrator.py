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
from eve.speech.text_to_speech import TextToSpeech
from eve.speech.llm_processor import LLMProcessor
from eve.display.lcd_controller import LCDController
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer

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
        
        # Initialize all attributes to None first
        self.audio_capture = None
        self.speech_recognizer = None
        self.text_to_speech = None
        self.llm_processor = None
        self.lcd_controller = None
            self.face_detector = None
            self.emotion_analyzer = None
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
            audio_config = getattr(speech_config, 'AUDIO_CAPTURE', {})
            self.audio_capture = AudioCapture(
                sample_rate=audio_config.get('sample_rate', 16000),
                channels=audio_config.get('channels', 1),
                chunk_size=audio_config.get('chunk_size', 1024),
                format=audio_config.get('format', 'int16')
            )
            self.logger.info("Audio capture initialized successfully")

            # Initialize speech recognition
            self.speech_recognizer = SpeechRecognizer(speech_config)
            self.logger.info("Speech recognition initialized successfully")

            # Initialize text to speech
            tts_config = getattr(speech_config, 'TEXT_TO_SPEECH', {})
            self.text_to_speech = TextToSpeech(
                engine=tts_config.get('engine', 'pyttsx3'),
                voice=tts_config.get('voice', 'english'),
                rate=tts_config.get('rate', 150),
                volume=tts_config.get('volume', 1.0)
            )
            self.logger.info("Text to speech initialized successfully")
            
            # Initialize LLM processor
            self.llm_processor = LLMProcessor(speech_config)
            self.logger.info("LLM processor initialized successfully")

            # Initialize display
            display_params = {
                'width': getattr(display_config, 'WIDTH', 800),
                'height': getattr(display_config, 'HEIGHT', 480),
                'fps': getattr(display_config, 'FPS', 30),
                'default_emotion': getattr(display_config, 'DEFAULT_EMOTION', 'neutral'),
                'background_color': getattr(display_config, 'BACKGROUND_COLOR', (0, 0, 0)),
                'eye_color': getattr(display_config, 'EYE_COLOR', (0, 191, 255))
            }
            self.lcd_controller = LCDController(**display_params)
            self.logger.info("Display subsystem initialized successfully")

            # Initialize vision subsystems
            self.face_detector = FaceDetector()
            self.emotion_analyzer = EmotionAnalyzer()
            self.logger.info("Vision subsystems initialized successfully")

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