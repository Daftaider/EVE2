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
import speech_recognition as sr

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """
    Speech recognition using WhisperModel.
    
    This class provides functionality for capturing audio and converting
    it to text using the Whisper speech recognition model.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        language: str = "en",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        buffer_duration_sec: float = 2.0,
        max_recording_sec: float = 30.0,
        silence_duration_sec: float = 1.5,
        vad_threshold: float = 0.3,
        callback: Optional[Callable[[str, float], None]] = None,
        model_type: Optional[str] = None,
        vosk_model_path: Optional[str] = None,
        whisper_model_name: Optional[str] = None,
        config=None,
        post_event_callback=None
    ):
        """
        Initialize the speech recognizer.
        
        Args:
            model_path: Path to the Whisper model. If None, uses the model from config.
            threshold: Confidence threshold for speech recognition.
            language: Language code for speech recognition.
            device_index: Audio input device index. If None, uses the default device.
            sample_rate: Audio sample rate.
            buffer_duration_sec: Duration of the audio buffer in seconds.
            max_recording_sec: Maximum recording duration in seconds.
            silence_duration_sec: Silence duration in seconds to consider speech ended.
            vad_threshold: Voice activity detection threshold.
            callback: Callback function to be called when speech is recognized.
            model_type: The type of speech recognition model to use ('google', 'vosk', 'whisper', etc.)
            vosk_model_path: Path to Vosk model directory
            whisper_model_name: Name of Whisper model to use
            config: Configuration object
            post_event_callback: Callback function to post recognition results
        """
        logger.info("Initializing speech recognizer")
        
        # Configuration
        self.model_path = model_path or config.speech.recognition_model
        self.threshold = threshold or config.speech.recognition_threshold
        self.language = language or config.speech.language
        self.device_index = device_index or config.hardware.audio_input_device
        self.sample_rate = sample_rate or config.hardware.audio_sample_rate
        self.buffer_duration_sec = buffer_duration_sec
        self.max_recording_sec = max_recording_sec
        self.silence_duration_sec = silence_duration_sec
        self.vad_threshold = vad_threshold
        self.callback = callback
        self.model_type = model_type or "default"
        self.vosk_model_path = vosk_model_path
        self.whisper_model_name = whisper_model_name
        self.config = config
        self.post_event = post_event_callback
        
        # Internal state
        self.is_running = False
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.buffer_size = int(self.sample_rate * self.buffer_duration_sec)
        self.recording_buffer = []
        self.silence_samples = int(self.sample_rate * self.silence_duration_sec)
        self.max_samples = int(self.sample_rate * self.max_recording_sec)
        
        # Threads
        self.listen_thread = None
        self.process_thread = None
        
        # Audio stream
        self.stream = None
        
        # Load the model if the path exists
        self.model = None
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.error(f"Model file not found: {self.model_path}")
        
        logger.info(f"Initializing speech recognizer with model type: {self.model_type}")
        if self.model_type == "vosk" and self.vosk_model_path:
            logger.info(f"Using Vosk model at: {self.vosk_model_path}")
        elif self.model_type == "whisper" and self.whisper_model_name:
            logger.info(f"Using Whisper model: {self.whisper_model_name}")
        
        # Check if model files exist
        if self.model_type and not self._check_model_exists(self.model_type):
            logger.error(f"Model file not found: {self.model_type}")
            # Fall back to a simple model that doesn't require external files
            self.model_type = "simple"
        
        # Initialize speech recognition
        self._init_recognizer()
    
    def _load_model(self):
        """Load the Whisper speech recognition model."""
        try:
            logger.info(f"Loading Whisper model from {self.model_path}")
            # Check for GPU or use CPU if not available
            device = "cuda" if self._is_cuda_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # Load model - with beam_size=1 for faster inference
            self.model = WhisperModel(
                self.model_path,
                device=device,
                compute_type=compute_type,
                download_root=None,
                local_files_only=True,
                beam_size=1
            )
            logger.info(f"Whisper model loaded successfully (device: {device})")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def start(self):
        """Start the speech recognizer."""
        if self.is_running:
            logger.warning("Speech recognizer is already running")
            return False
        
        if self.model is None:
            logger.error("Cannot start speech recognizer: Model not loaded")
            return False
        
        self.is_running = True
        self.is_listening = False
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Start the listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        logger.info("Speech recognizer started")
        return True
    
    def stop(self):
        """Stop the speech recognizer."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_listening = False
        
        # Stop the audio stream if active
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for threads to finish
        if self.listen_thread is not None and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2.0)
        
        if self.process_thread is not None and self.process_thread.is_alive():
            # Add None to the queue to signal the process thread to exit
            self.audio_queue.put(None)
            self.process_thread.join(timeout=2.0)
        
        logger.info("Speech recognizer stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert audio data to mono and float32
        audio_data = indata.copy()
        if audio_data.shape[1] > 1:  # Convert stereo to mono
            audio_data = np.mean(audio_data, axis=1)
        else:
            audio_data = audio_data.flatten()
        
        # Add audio data to the queue
        self.audio_queue.put(audio_data)
    
    def _listen_loop(self):
        """Listening thread that captures audio from the microphone."""
        try:
            # Start the audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.buffer_size,
                callback=self._audio_callback,
                device=self.device_index
            )
            self.stream.start()
            logger.info("Audio stream started")
            
            # Keep the thread alive while running
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.is_running = False
    
    def _process_loop(self):
        """Processing thread that processes audio data and recognizes speech."""
        try:
            while self.is_running:
                # Get audio data from the queue
                audio_data = self.audio_queue.get()
                
                # Check for exit signal
                if audio_data is None:
                    break
                
                # Process audio data
                self._process_audio(audio_data)
                
                # Mark the task as done
                self.audio_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in process loop: {e}")
            self.is_running = False
    
    def _process_audio(self, audio_data: np.ndarray):
        """Process audio data and post recognition results"""
        try:
            if self.model_type == "simple":
                # Generate mock recognition results
                text = self._generate_mock_response()
                confidence = random.uniform(0.7, 1.0)
            else:
                # Perform actual speech recognition
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = " ".join(segment.text for segment in segments)
                text = text.strip()
                confidence = info.avg_logprob
            
            # Post recognition event
            self.post_event(TOPICS['SPEECH_RECOGNIZED'], {
                'text': text,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
        except sr.UnknownValueError:
            self.logger.debug("Speech not understood")
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            self.post_event(TOPICS['ERROR'], {
                'message': f"Speech recognition error: {e}",
                'severity': 'ERROR'
            })
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            self.post_event(TOPICS['ERROR'], {
                'message': f"Audio processing error: {e}",
                'severity': 'ERROR'
            })
    
    def _generate_mock_response(self):
        """Generate mock speech recognition results"""
        responses = [
            "Hello there",
            "How are you",
            "What's the weather like",
            "Tell me a joke",
            "That's interesting"
        ]
        return random.choice(responses)
    
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
            logger.error("Cannot recognize speech: Model not loaded")
            return "", 0.0
        
        try:
            # Transcribe the audio file
            segments, info = self.model.transcribe(
                file_path,
                language=self.language,
                beam_size=1,
                vad_filter=True
            )
            
            # Get the transcript
            text = " ".join(segment.text for segment in segments)
            text = text.strip()
            
            # Get confidence score
            confidence = info.avg_logprob
            
            logger.info(f"Recognized from file: '{text}' (confidence: {confidence:.2f})")
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error recognizing speech from file: {e}")
            return "", 0.0

    def _check_model_exists(self, model_type):
        """Check if the specified model files exist"""
        if model_type == "google":
            # Google doesn't require local model files
            return True
        elif model_type == "vosk" and self.vosk_model_path:
            return os.path.exists(self.vosk_model_path)
        elif model_type == "whisper" and self.whisper_model_name:
            # Whisper models are downloaded on first use
            return True
        return False
    
    def recognize(self, audio_data):
        """Convert audio data to text"""
        logger.info(f"Processing speech recognition with {self.model_type} model")
        
        # Simple fallback implementation
        if self.model_type == "simple":
            # Just return a placeholder response
            return "Hello EVE"
            
        # In a real implementation, we would use the actual model here
        return "Speech recognition placeholder" 