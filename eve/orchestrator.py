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

logger = logging.getLogger(__name__)

class EVEOrchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self, config=None):
        """Initialize the EVE orchestrator"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
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

    def _get_default_config(self):
        """Create default configuration if none provided"""
        class DefaultConfig:
            class speech:
                SAMPLE_RATE = 16000
                CHANNELS = 1
                CHUNK_SIZE = 1024
                THRESHOLD = 0.01
                MIN_CONFIDENCE = 0.6
            
            class display:
                WIDTH = 800
                HEIGHT = 480
                FPS = 30
                DEFAULT_EMOTION = "neutral"
                BACKGROUND_COLOR = (0, 0, 0)
                EYE_COLOR = (0, 191, 255)
            
            class vision:
                CAMERA_INDEX = 0
                RESOLUTION = (640, 480)
                FPS = 30
                
        return DefaultConfig()

    def _init_subsystems(self) -> None:
        """Initialize all subsystems based on configuration."""
        role = config.hardware.ROLE.lower()
        
        # Check if we're in distributed mode
        if config.hardware.DISTRIBUTED_MODE:
            logger.info(f"Initializing in distributed mode with role: {role}")
            # Initialize only the subsystems relevant to our role
            if role in ["all", "master"]:
                self._init_api_server()
            
            if role in ["all", "vision"]:
                self._init_vision_subsystem()
            
            if role in ["all", "speech"]:
                self._init_speech_subsystem()
            
            if role in ["all", "display"]:
                self._init_display_subsystem()
                
            # If not master, initialize API client to communicate with master
            if role not in ["all", "master"]:
                self._init_api_client()
        else:
            # Initialize all subsystems for standalone mode
            logger.info("Initializing in standalone mode")
            self._init_vision_subsystem()
            self._init_speech_subsystem()
            self._init_display_subsystem()
    
    def _init_vision_subsystem(self) -> None:
        """Initialize the vision subsystem (camera, face detection, emotion analysis)."""
        if not config.hardware.CAMERA_ENABLED:
            logger.info("Camera disabled in configuration, skipping vision subsystem")
            self.face_detector = None
            self.emotion_analyzer = None
            return
        
        logger.info("Initializing vision subsystem")
        try:
            # Initialize face detector
            vision_config = getattr(self.config, 'vision', None)
            self.face_detector = face_detector.FaceDetector(
                config=vision_config,
                post_event_callback=self.post_event
            )
            
            # Initialize emotion analyzer if enabled
            if config.vision.EMOTION_DETECTION_ENABLED:
                self.emotion_analyzer = emotion_analyzer.EmotionAnalyzer(
                    confidence_threshold=config.vision.EMOTION_CONFIDENCE_THRESHOLD
                )
            else:
                self.emotion_analyzer = None
            
            logger.info("Vision subsystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vision subsystem: {e}")
            self.face_detector = None
            self.emotion_analyzer = None
    
    def _init_speech_subsystem(self) -> None:
        """Initialize the speech subsystem (audio, ASR, LLM, TTS)."""
        if not (config.hardware.AUDIO_INPUT_ENABLED or config.hardware.AUDIO_OUTPUT_ENABLED):
            logger.info("Audio disabled in configuration, skipping speech subsystem")
            self.audio_capture = None
            self.speech_recognizer = None
            self.llm_processor = None
            self.text_to_speech = None
            return
        
        logger.info("Initializing speech subsystem")
        try:
            # Initialize audio capture if input is enabled
            if config.hardware.AUDIO_INPUT_ENABLED:
                self.audio_capture = speech_recorder.AudioCapture(
                    device_index=config.hardware.AUDIO_INPUT_DEVICE,
                    sample_rate=config.hardware.AUDIO_SAMPLE_RATE
                )
                
                # Initialize speech recognizer
                self.speech_recognizer = speech_recognizer.SpeechRecognizer(
                    model_type=config.speech.SPEECH_RECOGNITION_MODEL,
                    vosk_model_path=config.speech.VOSK_MODEL_PATH,
                    whisper_model_name=config.speech.WHISPER_MODEL_NAME
                )
            else:
                self.audio_capture = None
                self.speech_recognizer = None
            
            # Initialize LLM processor
            self.llm_processor = llm_processor.LLMProcessor(
                model_type=config.speech.LLM_MODEL_TYPE,
                model_path=config.speech.LLM_MODEL_PATH,
                context_length=config.speech.LLM_CONTEXT_LENGTH,
                max_tokens=config.speech.LLM_MAX_TOKENS,
                temperature=config.speech.LLM_TEMPERATURE
            )
            
            # Initialize text-to-speech if output is enabled
            if config.hardware.AUDIO_OUTPUT_ENABLED:
                self.text_to_speech = text_to_speech.TextToSpeech(
                    engine=config.speech.TTS_ENGINE,
                    voice_id=config.speech.TTS_VOICE_ID,
                    rate=config.speech.TTS_RATE,
                    coqui_model_path=config.speech.COQUI_MODEL_PATH
                )
            else:
                self.text_to_speech = None
            
            logger.info("Speech subsystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize speech subsystem: {e}")
            self.audio_capture = None
            self.speech_recognizer = None
            self.llm_processor = None
            self.text_to_speech = None
    
    def _init_display_subsystem(self) -> None:
        """Initialize the display subsystem (LCD controller)."""
        if not config.hardware.DISPLAY_ENABLED:
            logger.info("Display disabled in configuration, skipping display subsystem")
            self.lcd_controller = None
            return
        
        logger.info("Initializing display subsystem")
        try:
            # Initialize LCD controller
            self.lcd_controller = lcd_controller.LCDController(
                resolution=config.hardware.DISPLAY_RESOLUTION,
                fullscreen=config.hardware.FULLSCREEN,
                fps=config.display.ANIMATION_FPS,
                default_emotion=config.display.DEFAULT_EMOTION,
                background_color=config.display.BACKGROUND_COLOR,
                eye_color=config.display.EYE_COLOR,
                transition_time_ms=config.display.EMOTION_TRANSITION_TIME_MS
            )
            
            logger.info("Display subsystem initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize display subsystem: {e}")
            self.lcd_controller = None
    
    def _init_api_server(self) -> None:
        """Initialize the API server for distributed mode."""
        logger.info("Initializing API server")
        try:
            self.api_server = api.APIServer(
                host=config.communication.API_HOST,
                port=config.communication.API_PORT,
                debug=config.communication.API_DEBUG,
                message_queue=self.message_queue
            )
            logger.info("API server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API server: {e}")
            self.api_server = None
    
    def _init_api_client(self) -> None:
        """Initialize the API client for distributed mode."""
        logger.info("Initializing API client")
        try:
            self.api_client = api.APIClient(
                host=config.hardware.MASTER_IP,
                port=config.hardware.MASTER_PORT
            )
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            self.api_client = None
    
    def start(self) -> None:
        """Start all subsystems and the main event loop."""
        if self.running:
            logger.warning("EVE2 system is already running")
            return
        
        logger.info("Starting EVE2 system")
        self.running = True
        
        # Start API server if in distributed mode
        if config.hardware.DISTRIBUTED_MODE and hasattr(self, 'api_server') and self.api_server:
            threading.Thread(target=self.api_server.start, daemon=True).start()
        
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
        
        # Stop API server if in distributed mode
        if hasattr(self, 'api_server') and self.api_server:
            self.api_server.stop()
        
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
    return EVEOrchestrator(config)


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