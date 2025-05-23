"""
Speech system configuration for EVE2
"""

import os
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

# LLM Processing
LLM_MODEL_TYPE = "simple"
LLM_MODEL_PATH = os.path.join('models', 'llm', 'simple_model.json')
LLM_API_KEY = ""
LLM_MAX_TOKENS = 100
LLM_TEMPERATURE = 0.7
LLM_MODEL_NAME = "text-davinci-003"
LLM_SYSTEM_PROMPT = "You are EVE, an intelligent assistant."
LLM_CONTEXT_LENGTH = 512

# General speech settings
SPEECH_ENABLED = True
SPEECH_DEBUG = False

# Audio recording settings
AUDIO_RECORDING_ENABLED = True
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "int16"
AUDIO_DEVICE_INDEX = None  # None = default device

# Speech recognition settings
SPEECH_RECOGNITION_ENABLED = True
RECOGNITION_LANGUAGE = "en-US"
RECOGNITION_TIMEOUT_SEC = 5.0
SPEECH_RECOGNITION_MODEL = "google"  # Options: "google", "vosk", "whisper", "deepspeech"
recognition_model = "google"  # Add this line - lowercase version for compatibility
SPEECH_RECOGNITION_THRESHOLD = 0.7
CONTINUOUS_LISTENING = True
RECOGNITION_ENERGY_THRESHOLD = 300

# Vosk speech recognition
VOSK_MODEL_PATH = "models/vosk/vosk-model-small-en-us"

# Whisper speech recognition
WHISPER_MODEL_NAME = "tiny"  # Options: "tiny", "base", "small", "medium", "large"
WHISPER_LANGUAGE = "en"
WHISPER_TRANSLATE = False

# DeepSpeech recognition
DEEPSPEECH_MODEL_PATH = "models/deepspeech/deepspeech-0.9.3-models.pbmm"
DEEPSPEECH_SCORER_PATH = "models/deepspeech/deepspeech-0.9.3-models.scorer"

# Wake word detection
WAKE_WORD_ENABLED = True
WAKE_WORD = "hey eve"
WAKE_WORD_SENSITIVITY = 0.7
WAKE_WORD_MODEL = "porcupine"  # Options: "porcupine", "snowboy", "custom"
WAKE_WORD_PATH = "models/wake_word/hey_eve.ppn"

# Text-to-speech settings
TTS_ENABLED = True
TTS_VOICE = "en-US-Standard-C"
TTS_VOICE_ID = None
TTS_RATE = 1.0
TTS_PITCH = 1.0
TTS_ENGINE = "pyttsx3"  # Options: "pyttsx3", "google", "espeak", "coqui"
TTS_VOLUME = 1.0
TTS_CACHE_DIR = "cache/tts"

# Coqui TTS settings (open-source TTS)
COQUI_MODEL_PATH = "models/tts/coqui/tts_model.pth"
COQUI_CONFIG_PATH = "models/tts/coqui/config.json"
COQUI_VOCODER_PATH = "models/tts/coqui/vocoder.pth"
COQUI_VOCODER_CONFIG = "models/tts/coqui/vocoder_config.json"

# Audio processing
NOISE_REDUCTION_ENABLED = True
AUDIO_ENERGY_THRESHOLD = 300
VAD_ENABLED = True  # Voice Activity Detection
VAD_MODE = 3  # 0-3, higher is more aggressive

# Audio capture settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
THRESHOLD = 0.01

# Speech recognition settings
MODEL_TYPE = "google"
MIN_CONFIDENCE = 0.6

# Speech recognition settings
SPEECH_RECOGNITION = {
    'WAKE_WORD': 'eve',
    'CONVERSATION_TIMEOUT': 10.0,  # seconds
    'language': 'en-US'
}

# Audio capture settings
AUDIO_CAPTURE = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'format': 'int16'
}

# Text to speech settings
TEXT_TO_SPEECH = {
    'engine': 'pyttsx3',
    'voice': 'english',
    'rate': 150,
    'volume': 1.0
}

# Speech processing parameters
NOISE_THRESHOLD = 300
MAX_SILENCE_TIME = 2.0
MIN_PHRASE_TIME = 0.5

@dataclass
class SpeechConfig:
    """Configuration for the speech subsystem."""
    # Recognition
    SPEECH_RECOGNITION_MODEL: str = "google"
    SPEECH_RECOGNITION_LANGUAGE: str = "en-US"
    WAKE_WORD_PHRASE: str = "hey eve"
    WAKE_WORD_THRESHOLD: float = 0.5
    
    # TTS
    TTS_ENGINE: str = "pyttsx3"
    TTS_VOICE: str = "english"
    TTS_RATE: int = 150
    TTS_VOLUME: float = 1.0
    COQUI_MODEL_PATH: Optional[str] = None # Set path if using Coqui
    
    # Audio Capture
    AUDIO_DEVICE_INDEX: Optional[int] = None
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    NOISE_THRESHOLD: float = 0.1
    
    # LLM (Optional)
    LLM_MODEL_PATH: Optional[str] = None
    LLM_CONTEXT_LENGTH: int = 1024

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SpeechConfig':
        """Create SpeechConfig from a dictionary, applying defaults."""
        # Get all field names defined in the dataclass
        known_keys = {f.name for f in fields(cls)}
        
        # Filter the input dict to only include known keys
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}

        # Create instance using filtered dict; dataclass handles defaults for missing keys
        try:
            return cls(**filtered_dict)
        except TypeError as e:
            logger.error(f"Error creating SpeechConfig from dict: {e}. Config: {config_dict}")
            # Return default config on error
            return cls() 