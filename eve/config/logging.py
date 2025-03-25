"""
Logging configuration settings
"""

# Logging level
LOG_LEVEL = "INFO"

# Log file path (if any)
LOG_FILE = "eve.log"

# Logger levels for specific modules
LOGGER_LEVELS = {
    "eve.vision": "INFO",
    "eve.speech": "INFO",
    "eve.display": "INFO",
    "eve.orchestrator": "INFO"
} 