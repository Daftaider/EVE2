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
    Speech recognition using faster-whisper.
    Handles wake word detection (basic, text-based) and command recognition via callbacks.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Wake word and language from config
        self.wake_word = getattr(self.config, 'WAKE_WORD_PHRASE', 'eve').lower()
        self.language = getattr(self.config, 'SPEECH_RECOGNITION_LANGUAGE', 'en') # Whisper uses language codes like 'en'
        
        # Audio format parameters
        self.sample_rate = getattr(self.config, 'AUDIO_SAMPLE_RATE', 16000)
        # Whisper works internally with mono
        self.channels = 1 # Force mono for Whisper processing?
        self.sample_width = 2 # Bytes per sample (int16)
        
        # faster-whisper specific config
        # Recommend starting small: tiny.en or base.en
        self.model_size = getattr(self.config, 'WHISPER_MODEL_SIZE', 'tiny.en') 
        self.device = getattr(self.config, 'WHISPER_DEVICE', 'cpu')
        # Use int8 for faster CPU inference
        self.compute_type = getattr(self.config, 'WHISPER_COMPUTE_TYPE', 'int8') 
        
        self.model: Optional[WhisperModel] = None
        self._init_recognizer()
        
        self.logger.info(f"Speech recognizer initialized. Wake Word='{self.wake_word}', Lang='{self.language}', Model='{self.model_size}'")
        self.logger.warning("Current wake word detection uses text check after STT and is not efficient or reliable.")

    def _init_recognizer(self):
        """Initialize the faster-whisper model."""
        self.logger.info(f"Loading faster-whisper model: {self.model_size} (Device: {self.device}, Compute: {self.compute_type})")
        try:
            # Model is downloaded automatically by faster-whisper on first use
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self.logger.info(f"faster-whisper model '{self.model_size}' loaded successfully.")
            # Perform a dummy inference to potentially speed up the first real one
            # dummy_audio = np.zeros(self.sample_rate * 1, dtype=np.float32) # 1 second of silence
            # _, _ = self.model.transcribe(dummy_audio, language=self.language)
            # logger.info("Performed dummy whisper inference.")
            
        except Exception as e:
            self.logger.error(f"Error loading faster-whisper model '{self.model_size}': {e}", exc_info=True)
            self.model = None

    def process_audio_chunk(
        self,
        audio_data: bytes,
        listen_for_command: bool,
        wake_word_callback: Callable[[], None],
        command_callback: Callable[[str, float], None]
    ):
        """Process a chunk of audio data using faster-whisper."""
        if not self.model:
             self.logger.error("Whisper model not loaded. Cannot process audio.")
             return
        if not audio_data:
            return
             
        try:
            # Convert int16 bytes back to float32 numpy array
            # Ensure data length is multiple of sample width
            if len(audio_data) % self.sample_width != 0:
                 # Handle potentially incomplete chunk if necessary
                 # For now, log warning and potentially skip/truncate
                 self.logger.warning(f"Audio data length ({len(audio_data)}) not multiple of sample width ({self.sample_width}). Truncating.")
                 audio_data = audio_data[:len(audio_data) - (len(audio_data) % self.sample_width)]
                 
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Transcribe using faster-whisper
            # beam_size=1 can be faster but less accurate, beam_size=5 is default
            # vad_filter=True can help filter silence (requires additional setup/libs)
            self.logger.debug("Transcribing audio chunk with faster-whisper...")
            segments, info = self.model.transcribe(audio_float32, language=self.language, beam_size=1)
            
            text = ""
            confidence = 0.0 # Whisper doesn't provide a simple overall confidence easily
            
            # Concatenate text from segments
            recognized_texts = [segment.text for segment in segments]
            if recognized_texts:
                 text = " ".join(recognized_texts).strip()
                 self.logger.info(f"Whisper recognized: '{text}' (Lang: {info.language}, Prob: {info.language_probability:.2f})")
            else:
                 self.logger.debug("Whisper produced no text segments.")
                 
            # --- Process recognized text --- 
            if text:
                text_lower = text.lower()
                if listen_for_command:
                     self.logger.debug(f"Calling command callback for: '{text}'")
                     command_callback(text, confidence) # Pass 0.0 confidence for now
                else:
                     # Check for wake word (case-insensitive)
                     self.logger.debug(f"Checking for wake word '{self.wake_word}' in '{text_lower}'...")
                     if self.wake_word in text_lower:
                          self.logger.info(f"Wake word FOUND in text: '{text}'")
                          wake_word_callback()
                     else:
                          self.logger.debug("Wake word NOT found in text.")
                          
        except Exception as e:
            self.logger.error(f"Unexpected error processing audio chunk with Whisper: {e}", exc_info=True)

    # recognize_file can potentially remain if needed elsewhere
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
        
        if self.model is None:
            logger.error("Cannot recognize speech: Model not initialized")
            return "", 0.0
        
        try:
            # Transcribe the audio file
            with open(file_path, "rb") as f:
                audio_data = f.read()
                text, confidence = self.process_audio_chunk(audio_data, False, lambda: None, lambda t, c: None)
                text = text.strip()
                
                logger.info(f"Recognized from file: '{text}'")
                return text, confidence
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return "", 0.0

    # reset might be useful if recognizer state needs clearing
    def reset(self):
        """Reset the recognizer state (if applicable in the future)."""
        self.logger.info("Resetting SpeechRecognizer state (currently no action).")
        # Potentially re-initialize self.model if needed
        pass

    # Remove _check_model_exists as we are forcing Google for now
    # def _check_model_exists(self, model_type):
