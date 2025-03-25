"""
EVE2 Configuration Module
Contains all global configuration settings for EVE2
"""

# Import all configuration submodules
from . import display
from . import logging
from . import vision
from . import speech
from . import system
from . import communication
from . import hardware

# Global configuration constants - directly accessible at config level
DEBUG = True
VERSION = "2.0.0"
LOG_LEVEL = "INFO"
PROJECT_NAME = "EVE2"
SAVE_PATH = "data/"
ASSETS_DIR = "assets/"  # Add this as a top-level constant

# Create a config object that contains all the configuration
# so it can be imported as: from eve.config import config
class ConfigContainer:
    def __init__(self):
        # Core modules
        self.display = display
        self.logging = logging
        self.vision = vision
        self.speech = speech
        self.system = system
        self.communication = communication
        self.hardware = hardware
        
        # Global configuration settings - accessible via the container
        self.DEBUG = DEBUG
        self.VERSION = VERSION
        self.LOG_LEVEL = LOG_LEVEL
        self.PROJECT_NAME = PROJECT_NAME
        self.SAVE_PATH = SAVE_PATH
        self.ASSETS_DIR = ASSETS_DIR

# Create the config instance
config = ConfigContainer()

# Also expose the submodules directly for backward compatibility 