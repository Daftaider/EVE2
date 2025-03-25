"""
EVE Configuration Module
Contains all global configuration settings for EVE
"""

# Import all configuration submodules
from . import display
from . import logging
from . import vision
from . import speech
from . import system
from . import communication

# Create a config object that contains all the configuration
# so it can be imported as: from eve.config import config
class ConfigContainer:
    def __init__(self):
        self.display = display
        self.logging = logging
        self.vision = vision
        self.speech = speech
        self.system = system
        self.communication = communication
        
        # Global configuration settings
        self.DEBUG = True
        self.VERSION = "2.0.0"
        self.LOG_LEVEL = "INFO"

# Create the config instance
config = ConfigContainer()

# Also expose the submodules directly for backward compatibility 