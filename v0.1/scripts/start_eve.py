#!/usr/bin/env python3
"""
EVE2 main startup script.

This script initializes and starts the EVE2 system.
"""
import argparse
import logging
import sys
import os
import time
import signal
import pygame
from pathlib import Path
from eve.orchestrator import EVEOrchestrator
from eve.config.display import Emotion
from eve.config import SystemConfig, load_config
from eve.utils import logging_utils
from eve.vision.camera import Camera
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer
from eve.vision.object_detector import ObjectDetector
from eve.vision.display_window import VisionDisplay
from eve.vision.rpi_ai_camera import RPiAICamera
from eve.speech.audio_capture import AudioCapture
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.llm_processor import LLMProcessor
from eve.speech.text_to_speech import TextToSpeech
from typing import Any, Optional, Union # Import Optional and Union
import traceback

# Add the project root to the Python path
# Ensure this runs before other eve imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Add at the top of the file, before importing pygame
os.environ['SDL_VIDEODRIVER'] = 'fbcon'  # Use framebuffer console
os.environ['SDL_FBDEV'] = '/dev/fb0'     # Primary framebuffer device
os.environ['SDL_VIDEO_CURSOR_HIDDEN'] = '1'  # Hide cursor
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# --- Initial Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# Start with INFO level, will be potentially overridden by --debug or config
logging.basicConfig(level=logging.INFO, format=log_format, force=True)
logger = logging.getLogger(__name__) # Get logger for this script

# --- EVE Imports --- 
from eve.orchestrator import EVEOrchestrator
from eve.config.display import Emotion
from eve.config import SystemConfig, load_config
from eve.utils import logging_utils
from eve.vision.camera import Camera
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer
from eve.vision.object_detector import ObjectDetector
from eve.vision.display_window import VisionDisplay
from eve.speech.audio_capture import AudioCapture
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.llm_processor import LLMProcessor
from eve.speech.text_to_speech import TextToSpeech

# --- Logging Configuration Function ---
def setup_logging(log_config: Any, debug_override: bool):
    """Configures logging based on the config object and debug flag."""
    level_from_config = getattr(log_config, 'level', 'INFO').upper()
    final_level = 'DEBUG' if debug_override else level_from_config
    log_file = getattr(log_config, 'file', None)
    max_bytes = getattr(log_config, 'max_size_mb', 10) * 1024 * 1024
    backup_count = getattr(log_config, 'backup_count', 3)
    log_to_console = getattr(log_config, 'console', True)
    log_format_cfg = getattr(log_config, 'format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = getattr(log_config, 'date_format', "%Y-%m-%d %H:%M:%S")
    logging_utils.setup_logging(final_level, log_file, max_bytes, backup_count, log_to_console, log_format_cfg, date_format)
    logger.info(f"Logging configured. Level set to: {final_level} (Debug override: {debug_override}, Config level: {level_from_config})")
# ------------------------------------

class EVEApplication:
    """Manages the main application lifecycle."""
    def __init__(self):
        self._running = True
        self.orchestrator: Optional[EVEOrchestrator] = None
        self.debug_menu_active = False
        self.last_click_time = 0
        self.double_click_threshold = 0.5
        self.config: Optional[SystemConfig] = None
        self.args = None # Store parsed args from main()
        self._setup_signal_handlers()
        self.clock = pygame.time.Clock()
        self.display_controller = None
        self.logger = logging.getLogger(__name__)  # Add logger attribute

    def _setup_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Sets the running flag to False and attempts graceful shutdown."""
        if self._running:
             logger.info(f"Received signal {signum}. Initiating shutdown...")
             self._running = False
             if self.orchestrator:
                  logger.info("Signaling orchestrator to stop...")
                  self.orchestrator.stop()
             else:
                  logger.warning("Orchestrator not yet initialized during shutdown signal.")

    def run(self, args):
        """Initializes orchestrator and runs the main application loop."""
        self.args = args # Store args passed from main()
        logger.info("--- Starting EVE Main Application ---")
        exit_code = 0
        try:
            # 1. Load Configuration
            self.config = load_config()
            if not self.config:
                 logger.critical("Failed to load configuration. Exiting.")
                 sys.exit(1)

            # 2. Setup Logging (using loaded config and args.debug)
            setup_logging(self.config.logging, self.args.debug)
            logger.debug(f"Loaded configuration: {self.config!r}")

            # 3. Initialize Pygame (if display enabled)
            pygame_initialized = False
            if self.config.hardware.display_enabled:
                logger.info("Initializing display subsystem...")
                try:
                    # Clear any existing SDL environment variables
                    for var in ['SDL_VIDEODRIVER', 'SDL_FBDEV', 'SDL_VIDEO_CURSOR_HIDDEN']:
                        if var in os.environ:
                            del os.environ[var]
                    
                    # Initialize Pygame
                    pygame.init()
                    pygame_initialized = True
                    self.clock = pygame.time.Clock()
                    
                    # Initialize display controller
                    from eve.display.lcd_controller import LCDController
                    self.display_controller = LCDController(self.config.display)
                    if not self.display_controller.start():
                        logger.error("Failed to start display controller")
                        self.display_controller = None
                        pygame_initialized = False
                    else:
                        logger.info("Display subsystem initialized successfully")
                        
                except Exception as pg_err:
                    logger.error(f"Display initialization failed: {pg_err}")
                    logger.error(traceback.format_exc())
                    pygame_initialized = False
                    if self.display_controller:
                        try:
                            self.display_controller.cleanup()
                        except:
                            pass
                        self.display_controller = None

            # 4. Initialize Subsystems
            camera: Optional[Union[Camera, RPiAICamera]] = None # Type hint for flexibility
            if self.config.hardware.camera_enabled:
                if self.config.hardware.camera_type == 'rpi_ai':
                    logger.info("Initializing RPi AI Camera...")
                    camera = RPiAICamera(self.config)
                elif self.config.hardware.camera_type == 'picamera':
                    logger.info("Initializing legacy Picamera...")
                    camera = Camera(self.config)
                elif self.config.hardware.camera_type == 'opencv':
                    logger.info("Initializing OpenCV Camera...")
                    camera = Camera(self.config)
                else:
                    logger.warning(f"Unsupported camera_type: {self.config.hardware.camera_type}")
                
                if camera and not camera.start(): 
                    logger.error(f"Failed to start {self.config.hardware.camera_type} camera!")
                    camera = None # Ensure camera is None if start failed

            # --- Conditional Detector Initialization --- 
            # Only initialize host-based detectors if NOT using RPi AI Camera
            face_detector = None
            object_detector = None
            emotion_analyzer = None

            if camera and self.config.hardware.camera_type != 'rpi_ai':
                 logger.info("Initializing host-based vision detectors (CPU/Other Accelerator)...")
                 if self.config.vision.face_detection_enabled:
                     face_detector = FaceDetector(self.config, camera) 
                     if face_detector and not face_detector.start(): face_detector = None # Start and check
                 
                 if self.config.vision.object_detection_enabled:
                     object_detector = ObjectDetector(config=self.config)
                     if object_detector and not object_detector.start(): object_detector = None # Start and check
                 
                 if self.config.vision.emotion_detection_enabled:
                     emotion_analyzer = EmotionAnalyzer(config=self.config)
                     if emotion_analyzer and not emotion_analyzer.start(): emotion_analyzer = None # Start and check

            # Set up camera and object detector for display controller
            if self.display_controller:
                if camera:
                    self.display_controller.set_camera(camera)
                if object_detector:
                    self.display_controller.set_object_detector(object_detector)

            # 5. Initialize Orchestrator
            self.orchestrator = EVEOrchestrator(
                config=self.config,
                camera=camera,
                face_detector=face_detector,
                object_detector=object_detector,
                emotion_analyzer=emotion_analyzer,
                display_controller=self.display_controller
            )
            
            # 6. Start Orchestrator
            if not self.orchestrator.start():
                logger.critical("Failed to start orchestrator. Exiting.")
                sys.exit(1)
            
            # 7. Main Event Loop
            self._pygame_event_loop()
            
        except Exception as e:
            logger.critical(f"Unhandled exception in run(): {e}", exc_info=True)
            exit_code = 1
        finally:
            # 8. Cleanup
            if self.orchestrator:
                self.orchestrator.stop()
            if self.display_controller:
                self.display_controller.cleanup()
            pygame.quit()
            logger.info("Application shutdown complete")
            return exit_code

    def _pygame_event_loop(self):
        """Handle Pygame events in the main loop."""
        try:
            while self._running:
                # Process all pending events
                for event in pygame.event.get():
                    # Log key events for debugging
                    if event.type == pygame.KEYDOWN:
                        key_name = pygame.key.name(event.key)
                        mod_keys = []
                        if event.mod & pygame.KMOD_CTRL: mod_keys.append('CTRL')
                        if event.mod & pygame.KMOD_SHIFT: mod_keys.append('SHIFT')
                        if event.mod & pygame.KMOD_ALT: mod_keys.append('ALT')
                        mod_str = '+'.join(mod_keys) if mod_keys else 'NO_MOD'
                        try:
                            self.logger.debug(f"Key event: {key_name}, Modifiers: {mod_str}")
                        except AttributeError:
                            logger.debug(f"Key event: {key_name}, Modifiers: {mod_str}")
                    
                    # Log mouse events for debugging
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        try:
                            self.logger.debug(f"Mouse button {event.button} clicked at {event.pos}")
                        except AttributeError:
                            logger.debug(f"Mouse button {event.button} clicked at {event.pos}")
                    
                    # Pass event to LCD controller if available
                    if self.display_controller:
                        try:
                            # Process event in LCD controller
                            self.display_controller._process_events()
                        except Exception as e:
                            self.logger.error(f"Error processing event in LCD controller: {str(e)}")
                            self.logger.error(traceback.format_exc())
                    
                    # Handle quit event
                    if event.type == pygame.QUIT:
                        self._running = False
                        break
                
                # Update display controller
                if self.display_controller:
                    try:
                        self.display_controller.update()
                    except Exception as e:
                        self.logger.error(f"Error updating display controller: {str(e)}")
                        self.logger.error(traceback.format_exc())
                
                # Update orchestrator
                if self.orchestrator:
                    self.orchestrator.update()
                
                # Cap frame rate
                self.clock.tick(60)
                
        except Exception as e:
            try:
                self.logger.error(f"Error in Pygame event loop: {str(e)}")
                self.logger.error(traceback.format_exc())
            except AttributeError:
                logger.error(f"Error in Pygame event loop: {str(e)}")
                logger.error(traceback.format_exc())
        finally:
            try:
                self.logger.info("Pygame event loop stopped")
            except AttributeError:
                logger.info("Pygame event loop stopped")
            self._running = False

# cleanup() method is no longer needed as pygame quit moved to run() finally
# def cleanup(self): ...

def main():
    """Entry point for the EVE application."""

    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Run the EVE2 application.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output.")
    args = parser.parse_args()

    # 2. Set Root Logger Level IMMEDIATELY if --debug is used
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("--- Root logger level set to DEBUG via Command Line ---")

    # 3. Initialize and Run Application
    app = EVEApplication()
    exit_code = 1 # Default to error exit code
    try:
        # Pass parsed args to run() and get exit code back
        exit_code = app.run(args)
    except SystemExit as se:
        logger.info(f"Application exited via SystemExit with code: {se.code}")
        exit_code = se.code
    except Exception as main_err:
         logger.critical(f"Unhandled exception escaping app.run(): {main_err}", exc_info=True)
         exit_code = 1
    finally:
        logger.info(f"Application main() finished. Final exit code: {exit_code}.")
        # Exit the process with the determined code
        sys.exit(exit_code)

if __name__ == "__main__":
    main() 