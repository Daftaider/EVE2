"""
EVE2 Configuration Module
Contains all global configuration settings for EVE2
"""

# Import all configuration submodules
# from . import display # Deprecated
from . import logging
from . import vision
from . import speech
from . import system
from . import communication
from . import hardware

# Import the new config structures and loading function
from .config import (
    SystemConfig, 
    load_config,
    # Also expose nested config types for type hinting
    DisplayConfig,
    LoggingConfig,
    VisionConfig,
    SpeechConfig,
    SystemSubConfig,
    CommunicationConfig,
    HardwareConfig
)

# Global configuration constants - directly accessible at config level
DEBUG = True
VERSION = "2.0.0"
LOG_LEVEL = "INFO"
PROJECT_NAME = "EVE2"
SAVE_PATH = "data/"
ASSETS_DIR = "assets/"  # Add this as a top-level constant

# Create a config object that contains all the configuration
# so it can be imported as: from eve.config import config
# DEPRECATED: The old ConfigContainer is no longer the primary way to access config.
#             Use load_config() to get a SystemConfig instance instead.
# class ConfigContainer:
#     def __init__(self):
#         # Core modules
#         self.display = display
#         self.logging = logging
#         self.vision = vision
#         self.speech = speech
#         self.system = system
#         self.communication = communication
#         self.hardware = hardware
#         
#         # Global configuration settings - accessible via the container
#         self.DEBUG = DEBUG
#         self.VERSION = VERSION
#         self.LOG_LEVEL = LOG_LEVEL
#         self.PROJECT_NAME = PROJECT_NAME
#         self.SAVE_PATH = SAVE_PATH
#         self.ASSETS_DIR = ASSETS_DIR

# config = ConfigContainer() # Deprecated instance

# Also expose the submodules directly for backward compatibility?
# This might be useful during transition, but ideally code should
# use the SystemConfig object obtained from load_config().
# from . import display
# from . import logging
# from . import vision
# from . import speech
# from . import system
# from . import communication
# from . import hardware

# Also expose the submodules directly for backward compatibility 