import yaml
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# Helper function to recursively convert dicts to dataclasses
def _dataclass_from_dict(cls, data):
    if not isinstance(data, dict):
        # If data is not a dict, return it directly if it matches the type hint,
        # otherwise log a warning and potentially return a default or raise error.
        # This simplified version assumes data is compatible or None.
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    valid_keys = {f.name for f in fields(cls)}

    for key, value in data.items():
        if key not in valid_keys:
            # Corrected f-string logging (if uncommented)
            # logger.warning(f"Ignoring unknown key '{key}' in config data for {cls.__name__}")
            continue # Skip unknown keys gently

        field_type = field_types.get(key)
        if field_type:
            # Check if the field type is itself a dataclass
            origin = getattr(field_type, '__origin__', None)
            args = getattr(field_type, '__args__', ())
            
            # Handle Optional[Dataclass] or simple Dataclass
            target_cls = None
            if origin is Union and type(None) in args: # Handles Optional[SomeDataclass]
                # Find the non-None type argument which should be the dataclass
                 non_none_args = [arg for arg in args if arg is not type(None)]
                 if len(non_none_args) == 1 and hasattr(non_none_args[0], '__dataclass_fields__'):
                      target_cls = non_none_args[0]
            elif hasattr(field_type, '__dataclass_fields__'): # Handles simple SomeDataclass
                target_cls = field_type

            if target_cls and isinstance(value, dict):
                 kwargs[key] = _dataclass_from_dict(target_cls, value)
            # Handle List[Dataclass] - simplistic, assumes list of dicts
            elif origin is list and args and hasattr(args[0], '__dataclass_fields__') and isinstance(value, list):
                 item_cls = args[0]
                 kwargs[key] = [_dataclass_from_dict(item_cls, item) for item in value if isinstance(item, dict)]
            # Basic type conversion or assignment
            else:
                # Add basic type casting if needed (e.g., int(value)), but be careful
                try:
                    # Example: handle tuples specified as lists in YAML
                    if origin is tuple and isinstance(value, list):
                         kwargs[key] = tuple(value)
                    else:
                        # Directly assign - YAML loader often handles basic types
                        kwargs[key] = value
                except (TypeError, ValueError) as e:
                     # Corrected f-string logging
                     logger.warning(f"Type mismatch or conversion error for key '{key}' in {cls.__name__}. Expected {field_type}, got {type(value)}. Error: {e}")
                     # Decide on fallback: skip key, use default (harder here), or assign raw value
                     kwargs[key] = value # Assign raw value as fallback

    # Instantiate the dataclass with potentially missing fields (defaults will apply)
    try:
        return cls(**kwargs)
    except TypeError as e:
        # Corrected f-string logging
        logger.error(f"Error creating {cls.__name__} instance: {e}. Provided kwargs: {kwargs}. Raw data: {data}")
        # Return default instance on catastrophic error
        return cls()


# --- Nested Dataclasses based on existing config files ---

@dataclass
class DisplayConfig:
    # From display.py
    WINDOW_SIZE: Tuple[int, int] = (800, 480)
    FPS: int = 30
    FULLSCREEN: bool = False
    USE_HARDWARE_DISPLAY: bool = True  # Use hardware LCD if available
    DISPLAY_ROTATION: int = 0  # 0, 90, 180, 270 degrees
    DEFAULT_EMOTION: str = "neutral" # Store as string, maybe convert to Enum later if needed
    ASSET_DIR: str = "assets/emotions"
    TRANSITION_SPEED: float = 0.5
    # From display.py (additional)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    EYE_COLOR: Tuple[int, int, int] = (0, 191, 255)
    BLINK_INTERVAL_SEC: float = 3.0
    BLINK_DURATION: float = 0.15  # Duration of blink animation in seconds
    ENABLE_BLINKING: bool = True
    ENABLE_EMOTIONS: bool = True
    EMOTION_TRANSITION_TIME_MS: int = 500
    FONT_NAME: str = "Arial"
    FONT_SIZE: int = 24
    DEBUG_MENU_ENABLED: bool = True  # Enable debug menu by default
    DEBUG_FONT_SIZE: int = 24  # Font size for debug menu
    CURRENT_FRAME_PATH: str = "current_display.png"  # Path to save current frame

@dataclass
class LoggingConfig:
    # From logging.py
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    console: bool = True
    file: Optional[str] = "logs/eve.log"
    max_size_mb: int = 10
    backup_count: int = 5
    # LOGGER_LEVELS: Dict[str, str] = field(default_factory=lambda: {
    #     "eve.vision": "INFO",
    #     "eve.speech": "INFO",
    #     "eve.display": "INFO",
    #     "eve.orchestrator": "INFO",
    #     "eve.communication": "INFO"
    # })
    debug: bool = False # Simplified from DEBUG_LOGGING

@dataclass
class VisionConfig:
    # From vision.py
    enabled: bool = True
    debug: bool = True # Merged VISION_DEBUG
    overlay: bool = True # VISION_OVERLAY
    processing_interval_sec: float = 0.1

    # Face Detection
    face_detection_enabled: bool = True
    face_detection_interval_sec: float = 0.5
    face_detection_model: str = "haarcascade"
    face_detection_confidence: float = 0.5
    face_detection_min_size: Tuple[int, int] = (30, 30)

    # Face Recognition
    face_recognition_enabled: bool = False
    face_recognition_model: str = "facenet"
    face_recognition_tolerance: float = 0.6
    known_faces_dir: str = "data/known_faces"
    face_recognition_interval_sec: float = 1.0

    # Emotion Detection
    emotion_detection_enabled: bool = True
    emotion_detection_model: str = "fer"
    emotion_detection_interval_sec: float = 1.0
    emotion_confidence_threshold: float = 0.5

    # Object Detection
    object_detection_enabled: bool = True
    object_detection_model: str = "yolov8n.pt"
    object_detection_confidence: float = 0.5
    object_detection_interval_sec: float = 0.2
    object_detection_classes: List[str] = field(default_factory=lambda: ["person", "cat", "dog", "bird", "car", "bicycle", "motorcycle", "bus", "truck"])

@dataclass
class SpeechConfig:
    # From speech.py
    enabled: bool = True
    debug: bool = False

    # Audio Recording/Capture
    audio_device_index: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    energy_threshold: float = 300 # Merged AUDIO_ENERGY_THRESHOLD & RECOGNITION_ENERGY_THRESHOLD
    noise_reduction_enabled: bool = True
    vad_enabled: bool = True
    vad_mode: int = 3

    # Recognition
    recognition_enabled: bool = True
    recognition_model: str = "google" # SPEECH_RECOGNITION_MODEL
    recognition_language: str = "en-US"
    recognition_timeout_sec: float = 5.0
    # Paths for specific models (optional, based on recognition_model)
    vosk_model_path: Optional[str] = "models/vosk/vosk-model-small-en-us"
    whisper_model_name: Optional[str] = "tiny"
    deepspeech_model_path: Optional[str] = "models/deepspeech/deepspeech-0.9.3-models.pbmm"
    deepspeech_scorer_path: Optional[str] = "models/deepspeech/deepspeech-0.9.3-models.scorer"

    # Wake Word
    wake_word_enabled: bool = True
    wake_word_phrase: str = "hey eve" # Placeholder, actual model handles detection
    # wake_word_sensitivity: float = 0.7 # OWW uses inference threshold
    wake_word_model: str = "openwakeword" # Change default model type
    # wake_word_model_path: Optional[str] = None # OWW models usually downloaded/cached
    # picovoice_access_key: Optional[str] = None # Remove Picovoice key
    # OpenWakeWord specific (optional, defaults are often okay)
    openwakeword_inference_threshold: float = 0.7 # Confidence threshold for activation

    # Text-to-Speech (TTS)
    tts_enabled: bool = True
    tts_engine: str = "pyttsx3"
    tts_voice: Optional[str] = "english" # Adapted from various examples
    tts_rate: float = 1.0 # Rate multiplier
    tts_pitch: float = 1.0
    tts_volume: float = 1.0
    # Paths/IDs for specific engines (optional)
    coqui_model_path: Optional[str] = None # Simplified from multiple coqui paths
    google_voice_id: Optional[str] = "en-US-Standard-C" # Example

    # LLM Processing
    llm_enabled: bool = False # Assuming default off unless configured
    llm_model_type: str = "simple"
    llm_model_path: Optional[str] = os.path.join('models', 'llm', 'simple_model.json')
    llm_api_key: Optional[str] = None
    llm_max_tokens: int = 100
    llm_temperature: float = 0.7
    llm_model_name: Optional[str] = "text-davinci-003"
    llm_system_prompt: str = "You are EVE, an intelligent assistant."
    llm_context_length: int = 512

@dataclass
class SystemSubConfig: # Renamed from SystemConfig to avoid clash with top-level
    # From system.py
    mode: str = "normal" # SYSTEM_MODE
    debug_mode: bool = False # DEBUG_MODE
    main_loop_interval_sec: float = 0.1
    worker_threads: int = 4
    enable_resource_monitoring: bool = True
    resource_check_interval_sec: int = 60
    # File paths (relative to project root or absolute)
    data_dir: str = "data/"
    cache_dir: str = "cache/"
    log_dir: str = "logs/"
    models_dir: str = "models/"
    # Update/Maintenance
    auto_update_enabled: bool = False
    backup_enabled: bool = True
    # Performance
    performance_logging: bool = False
    optimize_for_raspberry_pi: bool = True
    # Security
    secure_mode: bool = False

@dataclass
class CommunicationConfig:
    # From communication.py
    # Message Queue (Simplified - assuming internal queue for now)
    # message_queue_size: int = 100

    # API
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_debug: bool = False
    api_token_required: bool = False
    api_token: Optional[str] = "eve2_default_token"

    # WebSocket
    websocket_enabled: bool = True
    websocket_port: int = 5001
    websocket_host: str = "0.0.0.0"
    websocket_path: str = "/ws"

@dataclass
class HardwareConfig:
    # From hardware.py
    display_enabled: bool = True
    camera_enabled: bool = True
    audio_input_enabled: bool = True
    audio_output_enabled: bool = True
    gpio_enabled: bool = True

    # Display Specific
    display_resolution: Tuple[int, int] = (800, 480)
    fullscreen: bool = False
    display_type: str = "lcd"
    display_rotation: int = 0

    # Camera Specific
    camera_type: str = "picamera" # 'picamera', 'opencv'
    camera_index: int = 0 # Used if camera_type is 'opencv'
    camera_rotation: int = 0
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_framerate: int = 30
    # camera_fps: int = 30 # Redundant with framerate

    # Audio Specific
    audio_input_device: Optional[str] = "default" # Name or index
    audio_output_device: Optional[str] = "default" # Name or index
    audio_volume: int = 80 # percentage

    # Power Management
    enable_power_management: bool = True
    low_power_mode: bool = False
    battery_monitor_enabled: bool = True


# --- Main SystemConfig Dataclass ---

@dataclass
class SystemConfig:
    # Top-level settings
    project_name: str = "EVE2"
    version: str = "2.0.0"
    debug: bool = False # Overarching debug flag

    # Nested configuration sections
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    system: SystemSubConfig = field(default_factory=SystemSubConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Load configuration from a dictionary, handling nested structures."""
        # Use the helper to recursively populate dataclasses
        return _dataclass_from_dict(cls, data)


# --- Configuration Loading Function ---

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> SystemConfig:
    """
    Loads configuration from a YAML file into the SystemConfig dataclass.

    Args:
        config_path: Path to the YAML configuration file. Defaults to 'config.yaml'
                     in the project root.

    Returns:
        A SystemConfig instance populated with settings from the file,
        or a default SystemConfig instance if the file is not found or invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.warning(f"Configuration file not found at '{path}'. Using default settings.")
        return SystemConfig() # Return default config

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            logger.error(f"Invalid configuration format in '{path}'. Expected a dictionary (YAML map). Using default settings.")
            return SystemConfig()

        # Use the classmethod to load from the dictionary
        return SystemConfig.from_dict(config_data)

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file '{path}': {e}. Using default settings.")
        return SystemConfig()
    except IOError as e:
        logger.error(f"Error reading configuration file '{path}': {e}. Using default settings.")
        return SystemConfig()
    except Exception as e:
        logger.error(f"Unexpected error loading configuration from '{path}': {e}. Using default settings.", exc_info=True)
        return SystemConfig()

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Create a dummy config.yaml for testing
    dummy_config_content = {
        'project_name': 'EVE2-Test',
        'debug': True,
        'hardware': {
            'camera_enabled': False,
            'display_resolution': [1024, 768]
        },
        'speech': {
            'tts_engine': 'espeak',
            'wake_word_phrase': 'hello computer'
        },
        'vision': {
             'object_detection_confidence': 0.6
        },
        'unknown_section': { # Test handling of unknown keys
             'some_key': 'some_value'
        }
    }
    dummy_path = Path('./temp_config_test.yaml')
    with open(dummy_path, 'w', encoding='utf-8') as f:
        yaml.dump(dummy_config_content, f, default_flow_style=False)

    print(f"Loading config from: {dummy_path.resolve()}")
    loaded_config = load_config(dummy_path)

    print("\nLoaded Configuration:")
    # Basic check
    print(f"Project Name: {loaded_config.project_name}")
    print(f"Debug Mode: {loaded_config.debug}")
    # Nested check
    print(f"Camera Enabled: {loaded_config.hardware.camera_enabled}")
    print(f"Display Resolution: {loaded_config.hardware.display_resolution}")
    print(f"TTS Engine: {loaded_config.speech.tts_engine}")
    print(f"Wake Word: {loaded_config.speech.wake_word_phrase}")
    print(f"OD Confidence: {loaded_config.vision.object_detection_confidence}")
    # Default check
    print(f"API Enabled (default): {loaded_config.communication.api_enabled}")
    print(f"System Mode (default): {loaded_config.system.mode}")

    # Clean up dummy file
    os.remove(dummy_path) 