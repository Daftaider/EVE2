"""
Logging configuration settings for EVE2
"""

# Logging level
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_TO_CONSOLE = True

# Log file settings
LOG_TO_FILE = True
LOG_FILE = "logs/eve.log"
LOG_FILE_MAX_SIZE = 10485760  # 10MB
LOG_FILE_BACKUP_COUNT = 5
LOG_FILE_ENCODING = "utf-8"

# Module-specific logging levels
LOGGER_LEVELS = {
    "eve.vision": "INFO",
    "eve.speech": "INFO",
    "eve.display": "INFO",
    "eve.orchestrator": "INFO",
    "eve.communication": "INFO"
}

# Debug settings
DEBUG_LOGGING = False  # Set to True for more verbose logging
PERFORMANCE_LOGGING = False
LOG_SYSTEM_METRICS = False 