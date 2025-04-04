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
from eve.speech.audio_capture import AudioCapture
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.llm_processor import LLMProcessor
from eve.speech.text_to_speech import TextToSpeech
from typing import Any, Optional # Import Optional

# Add the project root to the Python path
# Ensure this runs before other eve imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Add at the top of the file, before importing pygame
# os.environ['SDL_VIDEODRIVER'] = 'dummy'
# os.environ['SDL_RENDERER_DRIVER'] = 'software'
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
            camera = Camera(self.config) if self.config.hardware.camera_enabled else None
            if camera and not camera.start(): camera = None # Start and check

            face_detector = FaceDetector(self.config, camera) if camera and self.config.vision.face_detection_enabled else None
            if face_detector and not face_detector.start(): face_detector = None # Start and check

            object_detector = ObjectDetector(config=self.config) if self.config.vision.object_detection_enabled else None
            # Add start if object_detector needs it

            emotion_analyzer = EmotionAnalyzer(self.config) if self.config.vision.emotion_detection_enabled else None

            # --- Disable VisionDisplay --- 
            display_controller = None
            logger.warning("VisionDisplay explicitly disabled for shutdown hang test.")
            # display_controller = VisionDisplay(self.config, camera, face_detector, object_detector) if pygame_initialized and camera else None
            # if display_controller and not display_controller.start(): display_controller = None # Start and check
            # -----------------------------

            audio_capture = AudioCapture(self.config.speech) if self.config.hardware.audio_input_enabled else None
            if audio_capture: audio_capture.start_recording() # Start stream

            speech_recognizer = SpeechRecognizer(self.config.speech) if audio_capture and self.config.speech.recognition_enabled else None

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
                self.orchestrator.start()

                # 6. Run Main Loop (Pygame events or wait)
                logger.info("EVE application running. Press Ctrl+C to exit.")
                if display_controller and pygame_initialized:
                    self._pygame_event_loop()
                else:
                    while self._running:
                        time.sleep(1)

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
                    # ... (Keep existing event handling logic for keys/mouse) ...

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