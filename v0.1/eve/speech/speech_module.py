"""
Main speech module for the EVE2 system.

This module integrates speech recognition, language model processing,
and text-to-speech components to provide complete speech interaction
capabilities for the EVE2 system.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, Callable

from eve.config import config
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.llm_processor import LLMProcessor
from eve.speech.text_to_speech import TextToSpeech

class SpeechModule:
    """
    Speech module integrating speech recognition, LLM, and TTS.
    
    This class coordinates the speech components and manages the flow of
    information between them, from recognizing user speech to generating
    and speaking responses.
    """
    
    def __init__(
        self,
        message_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        speech_config=None,
        hardware_config=None
    ):
        """
        Initialize the speech module.
        
        Args:
            message_callback: Callback function to publish messages.
            speech_config: Configuration for speech components. If None, uses default config.
            hardware_config: Configuration for hardware. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.speech_config = speech_config or config.speech
        self.hardware_config = hardware_config or config.hardware
        
        # Communication
        self.message_callback = message_callback
        
        # State
        self.is_running = False
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.lock = threading.RLock()
        
        # Components
        self.speech_recognizer = None
        self.llm_processor = None
        self.text_to_speech = None
        
        # Create components
        self._init_components()
    
    def _init_components(self):
        """Initialize the speech components."""
        try:
            # Create speech recognizer with callback
            self.speech_recognizer = SpeechRecognizer(
                callback=self._on_speech_recognized
            )
            
            # Create LLM processor with callback
            self.llm_processor = LLMProcessor(
                callback=self._on_llm_response
            )
            
            # Create TTS processor with callback
            self.text_to_speech = TextToSpeech(
                callback=self._on_speech_complete
            )
            
            self.logger.info("Speech components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing speech components: {e}")
    
    def start(self) -> bool:
        """
        Start the speech module and its components.
        
        Returns:
            True if successful, False otherwise.
        """
        if self.is_running:
            self.logger.warning("Speech module is already running")
            return False
        
        with self.lock:
            self.is_running = True
            success = True
            
            # Start components
            if self.hardware_config.audio_input_enabled:
                if not self.speech_recognizer.start():
                    self.logger.error("Failed to start speech recognizer")
                    success = False
            else:
                self.logger.info("Speech recognition disabled in config")
            
            if not self.llm_processor.start():
                self.logger.error("Failed to start LLM processor")
                success = False
            
            if self.hardware_config.audio_output_enabled:
                if not self.text_to_speech.start():
                    self.logger.error("Failed to start text-to-speech")
                    success = False
            else:
                self.logger.info("Text-to-speech disabled in config")
            
            if not success:
                self.stop()
                return False
            
            self.logger.info("Speech module started")
            
            # Publish state
            self._publish_state("started")
            
            return True
    
    def stop(self):
        """Stop the speech module and its components."""
        if not self.is_running:
            return
        
        with self.lock:
            self.is_running = False
            
            # Stop components
            if self.speech_recognizer:
                self.speech_recognizer.stop()
            
            if self.llm_processor:
                self.llm_processor.stop()
            
            if self.text_to_speech:
                self.text_to_speech.stop()
            
            # Reset state
            self.is_listening = False
            self.is_processing = False
            self.is_speaking = False
            
            self.logger.info("Speech module stopped")
            
            # Publish state
            self._publish_state("stopped")
    
    def process_text(self, text: str):
        """
        Process text input directly (bypassing speech recognition).
        
        Args:
            text: The text to process.
        """
        if not self.is_running:
            self.logger.warning("Cannot process text: Speech module is not running")
            return
        
        self.logger.info(f"Processing text input: '{text}'")
        
        # Publish recognized text
        self._publish_message({
            "type": "speech_recognized",
            "text": text,
            "confidence": 1.0
        })
        
        # Process with LLM
        with self.lock:
            self.is_processing = True
        
        self.llm_processor.process_query(text)
    
    def say(self, text: str, emotion: str = "neutral"):
        """
        Speak text directly (bypassing LLM).
        
        Args:
            text: The text to speak.
            emotion: The emotion to express.
        """
        if not self.is_running:
            self.logger.warning("Cannot speak: Speech module is not running")
            return
        
        if not self.hardware_config.audio_output_enabled:
            self.logger.warning("Cannot speak: Audio output is disabled")
            return
        
        self.logger.info(f"Speaking direct text: '{text}' with emotion '{emotion}'")
        
        # Publish response
        self._publish_message({
            "type": "llm_response",
            "text": text,
            "emotion": emotion
        })
        
        # Speak the text
        with self.lock:
            self.is_speaking = True
        
        self.text_to_speech.say(text)
    
    def is_busy(self) -> bool:
        """
        Check if the speech module is busy.
        
        Returns:
            True if any component is busy, False otherwise.
        """
        with self.lock:
            return self.is_listening or self.is_processing or self.is_speaking
    
    def reset_conversation(self):
        """Reset the conversation history in the LLM processor."""
        if self.llm_processor:
            self.llm_processor.reset_conversation()
            self.logger.info("Conversation reset")
    
    def _on_speech_recognized(self, text: str, confidence: float):
        """
        Callback for when speech is recognized.
        
        Args:
            text: The recognized text.
            confidence: The confidence score.
        """
        self.logger.info(f"Speech recognized: '{text}' (confidence: {confidence:.2f})")
        
        # Publish recognized text
        self._publish_message({
            "type": "speech_recognized",
            "text": text,
            "confidence": confidence
        })
        
        # Process with LLM
        with self.lock:
            self.is_listening = False
            self.is_processing = True
        
        self.llm_processor.process_query(text)
    
    def _on_llm_response(self, response_text: str, metadata: Dict[str, Any]):
        """
        Callback for when LLM generates a response.
        
        Args:
            response_text: The response text.
            metadata: Metadata including emotion.
        """
        emotion = metadata.get("emotion", "neutral")
        self.logger.info(f"LLM response: '{response_text}' with emotion '{emotion}'")
        
        # Publish response
        self._publish_message({
            "type": "llm_response",
            "text": response_text,
            "emotion": emotion
        })
        
        with self.lock:
            self.is_processing = False
            
            # Speak the response if audio output is enabled
            if self.hardware_config.audio_output_enabled:
                self.is_speaking = True
                self.text_to_speech.say(response_text)
    
    def _on_speech_complete(self, text: str):
        """
        Callback for when speech is complete.
        
        Args:
            text: The text that was spoken.
        """
        self.logger.debug(f"Speech complete: '{text}'")
        
        # Publish event
        self._publish_message({
            "type": "speech_complete",
            "text": text
        })
        
        with self.lock:
            self.is_speaking = False
    
    def _publish_message(self, message: Dict[str, Any]):
        """
        Publish a message via the callback.
        
        Args:
            message: The message to publish.
        """
        if self.message_callback:
            message["timestamp"] = time.time()
            message["source"] = "speech_module"
            self.message_callback(message)
    
    def _publish_state(self, state: str):
        """
        Publish the module state.
        
        Args:
            state: The state to publish.
        """
        self._publish_message({
            "type": "module_state",
            "module": "speech",
            "state": state
        }) 