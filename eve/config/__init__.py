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
        
        # Global configuration settings
        self.DEBUG = True
        self.VERSION = "2.0.0"
        self.LOG_LEVEL = "INFO"
        self.PROJECT_NAME = "EVE2"
        self.SAVE_PATH = "data/"
        self.ASSETS_DIR = "assets/"

# Create the config instance
config = ConfigContainer()

# Also expose the submodules directly for backward compatibility 