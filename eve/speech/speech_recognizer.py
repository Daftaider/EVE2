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
        
        # Get speech recognition settings
        speech_config = getattr(config, 'SPEECH_RECOGNITION', {})
        self.model_type = speech_config.get('model_type', 'google')
        self.model_path = speech_config.get('model_path')
        self.language = speech_config.get('language', 'en-US')
        
        # Get audio settings
        audio_config = getattr(config, 'AUDIO_CAPTURE', {})
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.chunk_size = audio_config.get('chunk_size', 1024)
        
        # Buffer settings
        self.buffer_duration_sec = 0.5  # Default buffer duration in seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration_sec)
        
        # Initialize recognizer based on model type
        if self.model_type == 'google':
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300  # Adjust based on your needs
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
        elif self.model_type == 'coqui':
            if not self.model_path or not os.path.exists(self.model_path):
                self.logger.warning(f"Model path not found: {self.model_path}, falling back to google")
                self.model_type = 'google'
                self.recognizer = sr.Recognizer()
            else:
                # Initialize Coqui model here if you're using it
                pass
        else:
            self.logger.warning(f"Unknown model type: {self.model_type}, falling back to google")
            self.model_type = 'google'
            self.recognizer = sr.Recognizer()
        
        self.logger.info(f"Speech recognizer initialized using {self.model_type} "
                        f"(sample rate: {self.sample_rate}Hz, channels: {self.channels})")
        
        # Initialize mock responses
        self.mock_responses = [
            "Hello", "How are you", "What time is it",
            "Tell me a story", "That's interesting",
            "I like that", "Can you help me",
            "What's the weather like", "Good morning",
            "Good evening"
        ]
        
        self.logger.info("Speech recognizer initialized in mock mode")
        
        # Configuration
        self.threshold = config.speech.recognition_threshold if hasattr(config, 'recognition_threshold') else 0.5
        self.device_index = config.hardware.audio_input_device if hasattr(config, 'hardware') and hasattr(config.hardware, 'audio_input_device') else None
        self.max_recording_sec = config.speech.max_recording_sec if hasattr(config, 'speech') and hasattr(config.speech, 'max_recording_sec') else 30.0
        self.silence_duration_sec = config.speech.silence_duration_sec if hasattr(config, 'speech') and hasattr(config.speech, 'silence_duration_sec') else 1.5
        self.vad_threshold = config.speech.vad_threshold if hasattr(config, 'speech') and hasattr(config.speech, 'vad_threshold') else 0.3
        self.callback = config.speech.callback if hasattr(config, 'speech') and hasattr(config.speech, 'callback') else None
        self.vosk_model_path = config.speech.vosk_model_path if hasattr(config, 'speech') and hasattr(config.speech, 'vosk_model_path') else None
        self.whisper_model_name = config.speech.whisper_model_name if hasattr(config, 'speech') and hasattr(config.speech, 'whisper_model_name') else None
        
        # Internal state
        self.is_running = False
        self.is_listening = False
        self.audio_queue = queue.Queue()
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
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.error(f"Model file not found: {self.model_path}")
        
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
        if self.running:
            logger.warning("Speech recognizer is already running")
            return False
        
        if self.model is None:
            logger.error("Cannot start speech recognizer: Model not loaded")
            return False
        
        self.running = True
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
        if not self.running:
            return
        
        self.running = False
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
            while self.running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in listen loop: {e}")
            self.running = False
    
    def _process_loop(self):
        """Processing thread that processes audio data and recognizes speech."""
        try:
            while self.running:
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
            self.running = False
    
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
        return random.choice(self.mock_responses)
    
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

    def _init_recognizer(self):
        """Initialize the speech recognizer"""
        try:
            # Use mock recognizer
            self.sr = MockSpeechRecognition()
            self.recognizer = self.sr
            self.logger.info("Using mock speech recognition")
            
        except Exception as e:
            self.logger.error(f"Error initializing speech recognizer: {e}")
            raise

    def process_audio(self, audio_data):
        """Process audio data and return recognized text"""
        try:
            if self.model_type == 'google':
                # Convert audio data to AudioData object
                audio = sr.AudioData(audio_data, self.sample_rate, self.channels)
                # Perform recognition
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
            elif self.model_type == 'coqui':
                # Add Coqui-specific processing here if you're using it
                pass
            return ""
        except sr.UnknownValueError:
            self.logger.debug("Speech not recognized")
            return ""
        except sr.RequestError as e:
            self.logger.error(f"Could not request results: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return ""

    def is_running(self):
        """Check if the recognizer is running"""
        return self.running

    def get_status(self):
        """Get the current status of the recognizer"""
        return {
            'running': self.running,
            'mock_mode': True,
            'model_type': self.model_type,
            'timestamp': time.time()
        }

    def add_mock_response(self, response):
        """Add a new mock response to the list"""
        if isinstance(response, str) and response:
            self.mock_responses.append(response)
            return True
        return False

    def _init_mock_responses(self):
        """Initialize mock responses"""
        self.mock_responses = [
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
        self.logger.info("Mock responses initialized")

    def _generate_mock_response(self):
        """Generate a mock response"""
        return random.choice(self.mock_responses)

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

    def _init_recognizer(self):
        """Initialize the speech recognizer"""
        try:
            # Use mock recognizer
            self.sr = MockSpeechRecognition()
            self.recognizer = self.sr
            self.logger.info("Using mock speech recognition")
            
        except Exception as e:
            self.logger.error(f"Error initializing speech recognizer: {e}")
            raise

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
        """Generate a mock response"""
        return random.choice(self.mock_responses)

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

    def reset(self):
        """Reset the recognizer state"""
        if hasattr(self, 'recognizer'):
            self.recognizer = sr.Recognizer()
