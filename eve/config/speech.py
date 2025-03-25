"""
Speech system configuration for EVE2
"""

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
TTS_RATE = 1.0
TTS_PITCH = 1.0
TTS_ENGINE = "pyttsx3"  # Options: "pyttsx3", "google", "espeak"
TTS_VOLUME = 1.0
TTS_CACHE_DIR = "cache/tts"

# Audio processing
NOISE_REDUCTION_ENABLED = True
AUDIO_ENERGY_THRESHOLD = 300
VAD_ENABLED = True  # Voice Activity Detection
VAD_MODE = 3  # 0-3, higher is more aggressive 