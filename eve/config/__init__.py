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

# Global configuration settings
DEBUG = True
VERSION = "2.0.0"
LOG_LEVEL = "INFO" 