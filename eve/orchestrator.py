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

from eve import config
from eve.utils import logging_utils
from eve.vision import face_detector, emotion_analyzer
from eve.display import lcd_controller
from eve.speech import speech_recorder as audio_capture, speech_recognizer, llm_processor, text_to_speech
from eve.communication import message_queue, api

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Main coordinator for the EVE2 system.
    
    This class initializes and manages all subsystems, handles events,
    and coordinates the flow of data between modules.
    """
    
    def __init__(self) -> None:
        """Initialize the orchestrator and all subsystems."""
        # Initialize logging
        logging_utils.setup_logging()
        logger.info("Initializing EVE2 system orchestrator")
        
        # State tracking
        self.running: bool = False
        self.current_emotion: str = config.display.DEFAULT_EMOTION
        self.current_face_id: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize the message queue for inter-module communication
        self.message_queue = message_queue.MessageQueue(config.communication.MESSAGE_QUEUE_SIZE)
        
        # Initialize subsystems based on configuration
        self._init_subsystems()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("EVE2 orchestrator initialized successfully")
    
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
            self.face_detector = face_detector.FaceDetector(
                camera_index=config.hardware.CAMERA_INDEX,
                resolution=config.hardware.CAMERA_RESOLUTION,
                fps=config.hardware.CAMERA_FPS,
                detection_model=config.vision.FACE_DETECTION_MODEL,
                known_faces_dir=config.vision.KNOWN_FACES_DIR,
                recognition_enabled=config.vision.FACE_RECOGNITION_ENABLED,
                recognition_tolerance=config.vision.FACE_RECOGNITION_TOLERANCE
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
                self.audio_capture = audio_capture.AudioCapture(
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
        """Main event processing loop."""
        logger.info("Starting event processing loop")
        
        while self.running:
            try:
                # Process vision events
                if self.face_detector and self.face_detector.has_new_frame():
                    self._process_vision_events()
                
                # Process speech events
                if self.audio_capture and self.audio_capture.has_new_audio():
                    self._process_speech_events()
                
                # Process message queue events
                while not self.message_queue.empty():
                    message = self.message_queue.get()
                    self._handle_message(message)
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
        
        logger.info("Event processing loop stopped")
    
    def _process_vision_events(self) -> None:
        """Process vision events (face detection and emotion analysis)."""
        # Get the latest frame from the face detector
        frame, faces = self.face_detector.get_latest_frame()
        
        # Skip if no faces are detected
        if not faces:
            return
        
        # Process the first detected face for simplicity
        # In a more advanced implementation, you might want to track multiple faces
        face = faces[0]
        face_id = face.get('id', 'unknown')
        face_location = face.get('location')
        
        # Update current face ID if changed
        if face_id != self.current_face_id:
            self.current_face_id = face_id
            logger.info(f"New face detected: {face_id}")
            
            # Publish face detected event
            self.message_queue.put({
                'topic': config.communication.TOPICS['face_detected'],
                'data': {
                    'face_id': face_id,
                    'timestamp': time.time()
                }
            })
        
        # Analyze emotion if enabled
        if self.emotion_analyzer and face_location:
            # Extract face region from frame
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Analyze emotion
            emotion = self.emotion_analyzer.analyze(face_image)
            
            if emotion and emotion != self.current_emotion:
                self.current_emotion = emotion
                logger.info(f"Emotion detected: {emotion}")
                
                # Update LCD display if enabled
                if self.lcd_controller:
                    self.lcd_controller.set_emotion(emotion)
                
                # Publish emotion detected event
                self.message_queue.put({
                    'topic': config.communication.TOPICS['emotion_detected'],
                    'data': {
                        'face_id': face_id,
                        'emotion': emotion,
                        'timestamp': time.time()
                    }
                })
    
    def _process_speech_events(self) -> None:
        """Process speech events (speech recognition, LLM, TTS)."""
        # Get the latest audio from the audio capture
        audio = self.audio_capture.get_latest_audio()
        
        # Skip if no audio is available
        if audio is None:
            return
        
        # Recognize speech
        if self.speech_recognizer:
            text = self.speech_recognizer.recognize(audio)
            
            # Skip if no speech was recognized
            if not text:
                return
            
            logger.info(f"Speech recognized: {text}")
            
            # Publish speech recognized event
            self.message_queue.put({
                'topic': config.communication.TOPICS['speech_recognized'],
                'data': {
                    'text': text,
                    'timestamp': time.time()
                }
            })
            
            # Add user message to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': text
            })
            
            # Generate response with LLM
            if self.llm_processor:
                response = self.llm_processor.generate_response(self.conversation_history)
                
                if response:
                    logger.info(f"LLM response: {response}")
                    
                    # Add assistant message to conversation history
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    # Publish LLM response event
                    self.message_queue.put({
                        'topic': config.communication.TOPICS['llm_response'],
                        'data': {
                            'text': response,
                            'timestamp': time.time()
                        }
                    })
                    
                    # Convert to speech if TTS is enabled
                    if self.text_to_speech:
                        self.text_to_speech.speak(response)
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle messages from the message queue."""
        topic = message.get('topic')
        data = message.get('data', {})
        
        if not topic:
            logger.warning("Received message with no topic")
            return
        
        logger.debug(f"Handling message for topic: {topic}")
        
        # Handle different message topics
        if topic == config.communication.TOPICS['face_detected']:
            # Handle face detected event
            face_id = data.get('face_id')
            if face_id:
                self.current_face_id = face_id
                
        elif topic == config.communication.TOPICS['emotion_detected']:
            # Handle emotion detected event
            emotion = data.get('emotion')
            if emotion and emotion != self.current_emotion:
                self.current_emotion = emotion
                
                # Update LCD display if enabled and local
                if self.lcd_controller:
                    self.lcd_controller.set_emotion(emotion)
                
        elif topic == config.communication.TOPICS['speech_recognized']:
            # Handle speech recognized event
            text = data.get('text')
            if text and self.llm_processor:
                # Add user message to conversation history
                self.conversation_history.append({
                    'role': 'user',
                    'content': text
                })
                
                # Generate response with LLM
                response = self.llm_processor.generate_response(self.conversation_history)
                
                if response:
                    # Add assistant message to conversation history
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    # Publish LLM response event
                    self.message_queue.put({
                        'topic': config.communication.TOPICS['llm_response'],
                        'data': {
                            'text': response,
                            'timestamp': time.time()
                        }
                    })
                    
        elif topic == config.communication.TOPICS['llm_response']:
            # Handle LLM response event
            text = data.get('text')
            if text and self.text_to_speech:
                self.text_to_speech.speak(text)
                
        elif topic == config.communication.TOPICS['system_status']:
            # Handle system status event
            logger.info(f"System status update: {data}")
            
        else:
            logger.warning(f"Unknown message topic: {topic}")
    
    def _signal_handler(self, sig: int, frame: Any) -> None:
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down")
        self.stop()
        sys.exit(0)


def create_orchestrator() -> Orchestrator:
    """Create and return an Orchestrator instance."""
    return Orchestrator()


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