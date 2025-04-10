"""
System-wide configuration for EVE2
"""

# Core system settings
SYSTEM_NAME = "EVE2"
SYSTEM_VERSION = "2.0.0"
SYSTEM_MODE = "normal"  # Options: "normal", "debug", "demo", "test"
DEBUG_MODE = False

# Main loop timing
MAIN_LOOP_INTERVAL_SEC = 0.1
WORKER_THREADS = 4
BACKGROUND_TASKS_ENABLED = True

# Resource limits
MAX_CPU_PERCENT = 90
MAX_MEMORY_PERCENT = 80
ENABLE_RESOURCE_MONITORING = True
RESOURCE_CHECK_INTERVAL = 60  # seconds

# File paths
DATA_DIR = "data/"
CACHE_DIR = "cache/"
LOG_DIR = "logs/"
MODELS_DIR = "models/"
CONFIG_FILE = "config/eve.yaml"

# Updates and maintenance
AUTO_UPDATE_ENABLED = False
UPDATE_CHECK_INTERVAL = 86400  # seconds (24 hours)
BACKUP_ENABLED = True
BACKUP_INTERVAL = 604800  # seconds (7 days)

# Performance
PERFORMANCE_LOGGING = False
ENABLE_PROFILING = False
OPTIMIZE_FOR_RASPBERRY_PI = True

# Security
SECURE_MODE = False
ENCRYPTION_ENABLED = False
REQUIRE_AUTHENTICATION = False 