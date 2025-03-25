"""
Speech system configuration
"""

# Speech recognition settings
RECOGNITION_LANGUAGE = "en-US"
RECOGNITION_TIMEOUT_SEC = 5.0
SPEECH_RECOGNITION_MODEL = "google"  # Options: "google", "vosk", "whisper", "deepspeech"
SPEECH_RECOGNITION_THRESHOLD = 0.7

# Wake word detection
WAKE_WORD_ENABLED = True
WAKE_WORD = "hey eve"
WAKE_WORD_SENSITIVITY = 0.7

# Text-to-speech settings
TTS_VOICE = "en-US-Standard-C"
TTS_RATE = 1.0
TTS_PITCH = 1.0
TTS_ENGINE = "pyttsx3"  # Options: "pyttsx3", "google", "espeak"

# Audio processing
NOISE_REDUCTION_ENABLED = True
AUDIO_ENERGY_THRESHOLD = 300 