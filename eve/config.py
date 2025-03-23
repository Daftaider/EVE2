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
    camera_index: int = 0
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_fps: int = 15
    
    # Display settings
    display_enabled: bool = True
    display_resolution: Tuple[int, int] = (800, 480)
    display_fps: int = 30
    fullscreen: bool = False
    
    # Audio settings
    audio_input_enabled: bool = True
    audio_output_enabled: bool = True
    audio_input_device: Optional[int] = None
    audio_output_device: Optional[int] = None
    audio_sample_rate: int = 16000
    
    # Network settings (for distributed mode)
    master_ip: str = "127.0.0.1"
    master_port: int = 5000


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
    
    if not config_path.exists():
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        return config
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update each section if present in the loaded data
        if 'hardware' in config_data:
            config.hardware = HardwareConfig(**config_data['hardware'])
        
        if 'vision' in config_data:
            config.vision = VisionConfig(**config_data['vision'])
        
        if 'display' in config_data:
            config.display = DisplayConfig(**config_data['display'])
        
        if 'speech' in config_data:
            config.speech = SpeechConfig(**config_data['speech'])
        
        if 'communication' in config_data:
            config.communication = CommunicationConfig(**config_data['communication'])
        
        if 'logging' in config_data:
            config.logging = LoggingConfig(**config_data['logging'])
        
        logging.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
    
    return config


def save_config(config_file: Optional[str] = None) -> bool:
    """
    Save the current configuration to a YAML file.
    
    Args:
        config_file: Path to save the configuration file. If None, uses default path.
        
    Returns:
        True if successful, False otherwise.
    """
    if config_file is None:
        config_file = CONFIG_FILE
    
    config_path = Path(config_file)
    
    try:
        # Create a dictionary from the configuration dataclasses
        config_dict = {
            'hardware': asdict(config.hardware),
            'vision': asdict(config.vision),
            'display': asdict(config.display),
            'speech': asdict(config.speech),
            'communication': asdict(config.communication),
            'logging': asdict(config.logging),
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logging.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving configuration to {config_path}: {e}")
        return False


def update_config(section: str, key: str, value: Any) -> bool:
    """
    Update a specific configuration value.
    
    Args:
        section: The configuration section (hardware, vision, display, speech, communication, logging).
        key: The configuration key to update.
        value: The new value.
        
    Returns:
        True if successful, False otherwise.
    """
    global config
    
    try:
        if section == 'hardware':
            setattr(config.hardware, key, value)
        elif section == 'vision':
            setattr(config.vision, key, value)
        elif section == 'display':
            setattr(config.display, key, value)
        elif section == 'speech':
            setattr(config.speech, key, value)
        elif section == 'communication':
            setattr(config.communication, key, value)
        elif section == 'logging':
            setattr(config.logging, key, value)
        else:
            logging.warning(f"Unknown configuration section: {section}")
            return False
        
        logging.debug(f"Updated configuration: {section}.{key} = {value}")
        return True
    except Exception as e:
        logging.error(f"Error updating configuration {section}.{key}: {e}")
        return False


# Initialize configuration on module import
load_config() 