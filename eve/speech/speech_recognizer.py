"""
Speech recognition module for the EVE2 system.

This module provides speech recognition functionality using Whisper models
to convert audio input to text.
"""

import os
import logging
import numpy as np
import threading
import queue
import time
import sounddevice as sd
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
from faster_whisper import WhisperModel
import eve.config as config
import random
from eve.config.communication import TOPICS
import speech_recognition as sr

logger = logging.getLogger(__name__)

class MockSpeechRecognition:
    """Mock speech recognition for testing"""
    class UnknownValueError(Exception):
        pass
    
    class RequestError(Exception):
        pass
    
    def recognize_google(self, audio_data):
        """Mock google recognition"""
        # Randomly fail sometimes to simulate real behavior
        if random.random() < 0.1:  # 10% chance of not understanding
            raise self.UnknownValueError()
        if random.random() < 0.05:  # 5% chance of request error
            raise self.RequestError("Mock request error")
            
        # Generate mock responses
        responses = [
            "Hello",
            "How are you",
            "What time is it",
            "Tell me a story",
            "That's interesting",
            "I like that",
            "Can you help me",
            "What's the weather like",
            "Good morning",
            "Good evening"
        ]
        return random.choice(responses)

class SpeechRecognitionError(Exception):
    """Base class for speech recognition errors"""
    pass

class UnknownValueError(SpeechRecognitionError):
    """Raised when speech is not understood"""
    pass

class RequestError(SpeechRecognitionError):
    """Raised when there's an error with the recognition service"""
    pass

class SpeechRecognizer:
    """
    Speech recognition using WhisperModel.
    
    This class provides functionality for capturing audio and converting
    it to text using the Whisper speech recognition model.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Get configuration settings with defaults
        speech_config = getattr(config, 'SPEECH_RECOGNITION', {})
        self.model_type = 'google'  # Force Google recognition for now
        self.wake_word = speech_config.get('WAKE_WORD', 'eve').lower()
        self.conversation_timeout = speech_config.get('CONVERSATION_TIMEOUT', 10.0)
        self.language = speech_config.get('language', 'en-US')
        
        # Get audio settings
        audio_config = getattr(config, 'AUDIO_CAPTURE', {})
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        
        # Initialize state
        self.is_listening = False
        self.in_conversation = False
        self.last_interaction = 0
        self.conversation_thread = None
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size
        
        # Initialize recognizer
        self._init_recognizer()
        
        self.logger.info(f"Speech recognizer initialized with wake word: {self.wake_word}")
        self.logger.info(f"Using model type: {self.model_type}")

    def _init_recognizer(self):
        """Initialize the speech recognizer"""
        try:
            self.recognizer = sr.Recognizer()
            # Adjust recognition parameters
            self.recognizer.energy_threshold = 300  # minimum audio energy to consider for recording
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
            self.recognizer.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase
            self.recognizer.non_speaking_duration = 0.5  # seconds of non-speaking audio to keep on both sides of the recording
            
        except Exception as e:
            self.logger.error(f"Error initializing recognizer: {e}")
            raise

    def process_audio(self, audio_data):
        """Process audio data and check for wake word or conversation"""
        try:
            # Convert audio data to AudioData object
            audio = sr.AudioData(audio_data, self.sample_rate, self.channels)
            
            # Try to recognize the speech
            try:
                text = self.recognizer.recognize_google(audio, language=self.language).lower()
                self.logger.debug(f"Recognized text: {text}")
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                self.logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                return None
            
            # Check if we're in a conversation or heard the wake word
            if self.in_conversation:
                if time.time() - self.last_interaction > self.conversation_timeout:
                    self.logger.info("Conversation timed out")
                    self.in_conversation = False
                else:
                    self.last_interaction = time.time()
                    return text
            
            # Check for wake word
            if self.wake_word in text:
                self.logger.info("Wake word detected!")
                self.in_conversation = True
                self.last_interaction = time.time()
                # Remove wake word from response
                return text.replace(self.wake_word, '').strip()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

    def start_listening(self):
        """Start the listening thread"""
        if not self.is_listening:
            self.is_listening = True
            self.conversation_thread = threading.Thread(target=self._listen_loop)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()
            self.logger.info("Started listening for wake word and conversations")

    def stop_listening(self):
        """Stop the listening thread"""
        self.is_listening = False
        if self.conversation_thread:
            self.conversation_thread.join(timeout=1.0)
        self.in_conversation = False
        self.logger.info("Stopped listening")

    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                # Get audio from queue if available
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the audio
                text = self.process_audio(audio_data)
                
                if text:
                    self.logger.info(f"Recognized: {text}")
                    # Here you would typically send the text to your processing pipeline
                    
            except Exception as e:
                self.logger.error(f"Error in listening loop: {e}")
                time.sleep(0.1)

    def add_audio(self, audio_data):
        """Add audio data to the processing queue"""
        try:
            if self.audio_queue.qsize() < 100:  # Prevent queue from growing too large
                self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            self.logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            self.logger.error(f"Error adding audio to queue: {e}")

    def recognize_file(self, file_path: str) -> Tuple[str, float]:
        """
        Recognize speech from an audio file.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Tuple of (transcript, confidence).
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return "", 0.0
        
        if self.recognizer is None:
            logger.error("Cannot recognize speech: Recognizer not initialized")
            return "", 0.0
        
        try:
            # Transcribe the audio file
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=self.language).lower()
                text = text.strip()
                
                logger.info(f"Recognized from file: '{text}'")
                return text, 1.0
            
        except sr.UnknownValueError:
            logger.debug("Speech not understood")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return "", 0.0

    def _check_model_exists(self, model_type):
        """Check if the specified model files exist"""
        if model_type == "google":
            # Google doesn't require local model files
            return True
        return False

    def reset(self):
        """Reset the recognizer state"""
        if hasattr(self, 'recognizer'):
            self.recognizer = sr.Recognizer()
