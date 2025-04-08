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

            # 3. Initialize Pygame (if display enabled) - RE-ENABLING FOR TESTING
            pygame_initialized = False
            if self.config.hardware.display_enabled:
                logger.info("Initializing Pygame...")
                try:
                    pygame.init()
                    pygame_initialized = True
                    logger.info("Pygame initialized.")
                except Exception as pg_err:
                    logger.error(f"Pygame initialization failed: {pg_err}. Display might not work.")
            # logger.warning("Pygame initialization explicitly disabled for shutdown hang test.")

            # 4. Initialize Subsystems
            camera: Optional[Union[Camera, RPiAICamera]] = None # Type hint for flexibility
            if self.config.hardware.camera_enabled:
                if self.config.hardware.camera_type == 'rpi_ai':
                    logger.info("Initializing RPi AI Camera...")
                    camera = RPiAICamera(self.config)
                elif self.config.hardware.camera_type == 'picamera':
                    logger.info("Initializing legacy Picamera...")
                    # camera = Camera(self.config) # Assuming old Camera class handles picamera
                    logger.warning("Legacy picamera type selected, but Camera class might need update for Picamera2 compatibility if not already done.")
                    # Fallback to RPiAICamera for now if legacy isn't updated?
                    # Or use OpenCV for generic Pi cam access?
                    camera = Camera(self.config) # Keep using old Camera class for now
                elif self.config.hardware.camera_type == 'opencv':
                    logger.info("Initializing OpenCV Camera...")
                    camera = Camera(self.config) # Assuming old Camera class handles opencv
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
                     # Add start() if object_detector needs it later
                 
                 if self.config.vision.emotion_detection_enabled:
                     emotion_analyzer = EmotionAnalyzer(self.config)
            elif camera and self.config.hardware.camera_type == 'rpi_ai':
                 logger.info("RPi AI Camera selected. Skipping initialization of host-based detectors (Face, Object, Emotion).")
            # -----------------------------------------

            # --- RE-ENABLE VisionDisplay --- 
            from eve.display.lcd_controller import LCDController
            display_controller = LCDController(
                config=self.config.display
            ) if pygame_initialized else None
            if display_controller and not display_controller.start(): display_controller = None # Start and check
            # -----------------------------

            # --- Audio Subsystems --- 
            audio_capture = AudioCapture(self.config.speech) if self.config.hardware.audio_input_enabled else None
            speech_recognizer = None # Initialize as None
            if audio_capture:
                audio_capture.start_recording() # Start stream
                if self.config.speech.recognition_enabled:
                    logger.info("Initializing SpeechRecognizer...")
                    try:
                        # Create recognizer, passing the audio queue
                        speech_recognizer = SpeechRecognizer(self.config.speech, audio_capture.audio_queue)
                        logger.info("SpeechRecognizer initialized.")
                    except Exception as e:
                        logger.error(f"Failed SpeechRecognizer init: {e}", exc_info=True)
                        speech_recognizer = None # Ensure it's None on error
                else:
                    logger.info("Speech Recognition disabled by config.")
                    speech_recognizer = None
            else:
                 logger.warning("AudioCapture not available. Speech Recognition disabled.")
                 speech_recognizer = None

            tts = TextToSpeech(self.config.speech) if self.config.speech.tts_enabled else None
            if tts and not tts.start(): tts = None # Start and check

            llm_processor = LLMProcessor(self.config.speech) if self.config.speech.llm_enabled else None

            # 5. Create and Run Orchestrator
            logger.info("Creating EVE Orchestrator...")
            with EVEOrchestrator(
                config=self.config, camera=camera, face_detector=face_detector,
                object_detector=object_detector, emotion_analyzer=emotion_analyzer,
                audio_capture=audio_capture, speech_recognizer=speech_recognizer,
                llm_processor=llm_processor, tts=tts, display_controller=display_controller
            ) as self.orchestrator:
                logger.info("Starting Orchestrator...")
                
                # Set the command callback AFTER orchestrator is created
                if speech_recognizer and hasattr(self.orchestrator, '_handle_command'):
                    logger.debug("Setting SpeechRecognizer command callback to Orchestrator._handle_command")
                    speech_recognizer.set_command_callback(self.orchestrator._handle_command)
                
                self.orchestrator.start()

                # 6. Run Main Loop (Pygame events or wait)
                logger.info("EVE application running. Press Ctrl+C to exit.")
                if display_controller and pygame_initialized:
                    self._pygame_event_loop()
                else:
                    # Fallback: Call update() periodically if no display/pygame
                    logger.info("No display or Pygame, running basic update loop...")
                    while self._running:
                        if self.orchestrator:
                             self.orchestrator.update() # Call orchestrator update
                        # Sleep for a short interval (e.g., 100ms)
                        # Avoid sleeping too long, otherwise audio queue might fill
                        time.sleep(0.1)

        except SystemExit as se:
             logger.info(f"SystemExit caught with code: {se.code}")
             exit_code = se.code
        except KeyboardInterrupt:
             logger.info("KeyboardInterrupt caught in run method.")
             self._signal_handler(signal.SIGINT, None)
             exit_code = 0 # Treat as graceful shutdown request
        except Exception as e:
            logger.critical(f"Fatal error during EVE initialization or runtime: {e}", exc_info=True)
            exit_code = 1
            # Attempt orchestrator stop even on fatal error before finally block
            if self.orchestrator and hasattr(self.orchestrator, 'stop'):
                 try: self.orchestrator.stop()
                 except Exception as stop_err:
                      logger.error(f"Error during orchestrator cleanup attempt after fatal error: {stop_err}")
        finally:
            logger.info(f"--- EVE Application Run Method Reached Finally Block (Exit Code Hint: {exit_code}) ---")
            # Orchestrator stop is handled by 'with' statement's __exit__
            # Pygame quit should happen here if initialized
            if pygame_initialized: # This should now be true if display enabled
                logger.info("Attempting Pygame quit...")
                try:
                     pygame.quit()
                     logger.info("Pygame quit successful.")
                except Exception as e:
                     logger.error(f"Error during pygame quit: {e}")
            else:
                 logger.info("Pygame was not initialized (or disabled for test), skipping quit.")

            logger.info("--- EVE Main Application Shutdown Complete --- Returning from run() ---")
            # Return exit code to main()
            return exit_code

    def _pygame_event_loop(self):
        """Runs the Pygame event loop for UI interaction."""
        logger.info("Starting Pygame event loop...")
        while self._running:
            try:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.info("Pygame QUIT event received.")
                        self._signal_handler(signal.SIGTERM, None) # Treat QUIT as shutdown signal
                        return # Exit loop
                    
                    # Pass events to LCD controller if available
                    if self.orchestrator and self.orchestrator.display_controller:
                        # Log the event for debugging
                        if event.type == pygame.KEYDOWN:
                            key_name = pygame.key.name(event.key)
                            mod_keys = []
                            if event.mod & pygame.KMOD_CTRL: mod_keys.append('CTRL')
                            if event.mod & pygame.KMOD_SHIFT: mod_keys.append('SHIFT')
                            if event.mod & pygame.KMOD_ALT: mod_keys.append('ALT')
                            mod_str = '+'.join(mod_keys) if mod_keys else 'NO_MOD'
                            logger.debug(f"Passing key event to LCD controller: {key_name}, Modifiers: {mod_str}")
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            logger.debug(f"Passing mouse event to LCD controller: Button {event.button} at {event.pos}")
                        
                        # Pass the event to the LCD controller
                        self.orchestrator.display_controller.handle_event(event)
                
                # Update orchestrator
                if self.orchestrator:
                    self.orchestrator.update()
                
                # Optional sleep to prevent busy-waiting
                time.sleep(0.01)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received in pygame loop.")
                self._signal_handler(signal.SIGINT, None)
                return # Exit loop
            except Exception as e:
                logger.error(f"Error in pygame event loop: {e}", exc_info=True)
                time.sleep(0.5)
        logger.info("Pygame event loop finished.")

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