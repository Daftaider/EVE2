"""
Configuration module for the EVE2 system.

This module defines the configuration settings for all components of the EVE2 system,
including hardware settings, module-specific settings, and system-wide parameters.
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

# Get root directory
ROOT_DIR = Path(__file__).parent.parent
CONFIG_FILE = ROOT_DIR / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: Optional[str] = str(LOGS_DIR / "eve.log")
    max_size_mb: int = 10
    backup_count: int = 3
    console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class HardwareConfig:
    """Configuration for hardware components."""
    # Camera settings
    camera_enabled: bool = True
    camera_index: int = 0 # Default for OpenCV if picamera fails
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_fps: int = 15
    camera_rotation: int = 0 # 0, 90, 180, 270
    
    # Display settings
    display_enabled: bool = True
    display_resolution: Tuple[int, int] = (800, 480) # Target resolution for UI window
    display_fps: int = 30
    fullscreen: bool = False
    
    # Audio settings
    audio_input_enabled: bool = True
    audio_output_enabled: bool = True
    audio_input_device: Optional[int] = None # None for default
    audio_output_device: Optional[int] = None # None for default
    audio_sample_rate: int = 16000
    
    # Network settings (for distributed mode)
    # master_ip: str = "127.0.0.1"
    # master_port: int = 5000


@dataclass
class VisionConfig:
    """Configuration for vision module."""
    # Face detection settings
    detection_model: str = "hog"  # 'hog' (faster) or 'cnn' (more accurate)
    recognition_enabled: bool = True
    recognition_tolerance: float = 0.6
    known_faces_dir: str = str(DATA_DIR / "known_faces")
    
    # Emotion detection settings
    emotion_enabled: bool = True
    emotion_confidence_threshold: float = 0.4
    emotions: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "confused"
    ])

    # --- Face Detection / Recognition ---
    # Model: 'haar', 'hog' (face_recognition lib), 'cnn' (face_recognition lib)
    face_detection_model: str = "hog"
    # Haar cascade specific parameters (only used if face_detection_model='haar')
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    haar_min_face_size: Tuple[int, int] = (30, 30)
    # Face Recognition
    face_recognition_enabled: bool = True
    face_recognition_tolerance: float = 0.6
    label_map_path: Optional[str] = None # Optional: Path to label map for detector models

    # --- Emotion Detection ---
    emotion_confidence_threshold: float = 0.4
    # List of emotions used/mapped by the system (e.g., for display)
    emotions: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "confused", "disgusted", "fearful" # Expanded list
    ])

    # --- Object Detection (YOLOv8/Hailo via ObjectDetector) ---
    object_detection_enabled: bool = True
    # Primary model path for CPU fallback (e.g., yolov8n.pt)
    object_detection_cpu_model_path: str = "yolov8n.pt"
    object_detection_confidence: float = 0.5
    object_detection_nms_threshold: float = 0.45
    # Hailo specific settings
    object_detection_use_hailo: bool = True
    object_detection_hailo_hef_path: str = "models/yolov8n_hailo.hef"
    # Optional: Specify network group name if HEF has multiple or default is wrong
    object_detection_hailo_network_name: Optional[str] = None 
    # Interval between detections if running OD in a loop (e.g. VisionDisplay - potentially remove if detectors run per-frame)
    object_detection_interval: float = 0.5 # seconds 

    # --- General Vision ---
    vision_debug_mode: bool = False # Enable saving/displaying debug frames


@dataclass
class DisplayConfig:
    """Configuration for display module."""
    assets_dir: str = str(ASSETS_DIR / "emotions")
    default_emotion: str = "neutral"
    background_color: Tuple[int, int, int] = (0, 0, 0)  # RGB
    eye_color: Tuple[int, int, int] = (0, 191, 255)  # RGB (DeepSkyBlue)
    blink_interval_min_sec: float = 2.0  # Minimum time between blinks
    blink_interval_max_sec: float = 8.0  # Maximum time between blinks
    blink_duration_sec: float = 0.2  # Duration of a blink
    emotion_transition_time_ms: int = 500  # Time to transition between emotions
    # Add UI related display settings if needed
    font_path: Optional[str] = None # Path to custom font if needed
    font_size: int = 18


@dataclass
class SpeechConfig:
    """Configuration for speech module."""
    # Speech recognition settings
    recognition_model: str = str(MODELS_DIR / "whisper-small.en.gguf")
    recognition_threshold: float = 0.5
    language: str = "en"
    
    # LLM settings
    llm_model: str = str(MODELS_DIR / "llama-3-8b-instruct.Q5_K_M.gguf")
    llm_context_length: int = 2048
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    
    # Text-to-speech settings
    tts_model: str = str(MODELS_DIR / "tts-piper-en")
    tts_voice: str = "en_US/ljspeech_low"
    tts_speaking_rate: float = 1.0


@dataclass
class CommunicationConfig:
    """Configuration for communication module."""
    queue_max_size: int = 1000
    distributed: bool = False
    role: str = "standalone"  # standalone, master, vision, speech, display
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    connection_timeout_sec: int = 5


@dataclass
class SystemConfig:
    """Master configuration for the entire system."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Default singleton configuration instance
config = SystemConfig()


def load_config(config_file: Optional[str] = None) -> SystemConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to the configuration file. If None, uses default path.
        
    Returns:
        The loaded configuration.
    """
    global config
    
    if config_file is None:
        config_file = CONFIG_FILE
    
    config_path = Path(config_file)
    
    # Start with default config
    current_config = SystemConfig()
    
    if not config_path.exists():
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        config = current_config # Update global instance with defaults
        return config
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            if config_data is None: # Handle empty YAML file
                 config_data = {}

        # Helper to update dataclass from dict robustly
        def _update_section(cfg_section: Any, data: Optional[Dict]):
             if data and isinstance(data, dict):
                 # Get field names and types from the dataclass
                 fields = cfg_section.__dataclass_fields__
                 valid_data = {}
                 for key, value in data.items():
                      if key in fields:
                           # Optional: Add type checking/conversion here if needed
                           valid_data[key] = value
                      else:
                           logging.warning(f"Ignoring unknown config key '{key}' in section '{cfg_section.__class__.__name__}'")
                 # Update existing dataclass instance
                 for key, value in valid_data.items():
                      setattr(cfg_section, key, value)
             return cfg_section # Return potentially updated section

        # Update each section using the helper
        current_config.hardware = _update_section(current_config.hardware, config_data.get('hardware'))
        current_config.vision = _update_section(current_config.vision, config_data.get('vision'))
        current_config.display = _update_section(current_config.display, config_data.get('display'))
        current_config.speech = _update_section(current_config.speech, config_data.get('speech'))
        current_config.communication = _update_section(current_config.communication, config_data.get('communication'))
        current_config.logging = _update_section(current_config.logging, config_data.get('logging'))

        logging.info(f"Loaded configuration from {config_path}")
        config = current_config # Update global instance with loaded/merged config

    except yaml.YAMLError as e:
         logging.error(f"Error parsing configuration file {config_path}: {e}")
         # Keep default config if loading fails
    except Exception as e:
        logging.error(f"Unexpected error loading configuration from {config_path}: {e}", exc_info=True)
        # Keep default config if loading fails

    return config


def save_config(config_file: Optional[str] = None) -> bool:
    """
    Save the current configuration to a YAML file.
    
    Args:
        config_file: Path to save the configuration file. If None, uses default path.
        
    Returns:
        True if successful, False otherwise.
    """
    global config # Use the potentially updated global config
    if config_file is None:
        config_file = CONFIG_FILE
    
    config_path = Path(config_file)
    
    try:
        config_dict = asdict(config) # Convert the whole SystemConfig object
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logging.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving configuration to {config_path}: {e}")
        return False


def update_config(section_name: str, key: str, value: Any) -> bool:
    """
    Update a specific configuration value dynamically and save.
    Note: Basic type conversion might be needed depending on source of 'value'.
    """
    global config
    
    if not hasattr(config, section_name):
        logging.error(f"Invalid configuration section: '{section_name}'")
        return False
        
    section_obj = getattr(config, section_name)
    
    if not hasattr(section_obj, key):
        logging.error(f"Invalid key '{key}' in section '{section_name}'")
        return False
        
    try:
        # Optional: Attempt type conversion based on existing field type hint
        field_type = section_obj.__dataclass_fields__[key].type
        # Handle Optional[T] types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
             # Get the non-None type from Optional[T] which is Union[T, NoneType]
             non_none_type = next((t for t in field_type.__args__ if t is not type(None)), None)
             if non_none_type:
                  field_type = non_none_type
                  
        if field_type and value is not None:
             try:
                  # Handle common cases like tuples from strings if needed
                  if field_type == Tuple[int, int] and isinstance(value, str):
                       parts = value.strip('() ').split(',')
                       if len(parts) == 2:
                            value = (int(parts[0]), int(parts[1]))
                  elif field_type == List[float] and isinstance(value, str):
                       value = [float(x.strip()) for x in value.strip('[] ').split(',')]
                  # General conversion
                  value = field_type(value)
             except (ValueError, TypeError) as e:
                  logging.warning(f"Could not convert value '{value}' to type {field_type} for {section_name}.{key}. Using original value. Error: {e}")
                  # Keep original value type from the input

        setattr(section_obj, key, value)
        logging.info(f"Updated config: {section_name}.{key} = {value}")
        
        # Save the updated configuration immediately
        save_config() 
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating config {section_name}.{key}: {e}", exc_info=True)
        return False


# Initialize configuration on module import
load_config() 