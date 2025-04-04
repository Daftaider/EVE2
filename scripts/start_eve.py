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
from typing import Any # Import Any for type hinting

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add at the top of the file, before importing pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_RENDERER_DRIVER'] = 'software'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Adjust path to import from eve module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

# Import necessary components from the refactored config and subsystems
from eve.config import SystemConfig, load_config # Use new config loading
from eve.utils import logging_utils
from eve.vision.camera import Camera
from eve.vision.face_detector import FaceDetector
from eve.vision.emotion_analyzer import EmotionAnalyzer
from eve.vision.object_detector import ObjectDetector
from eve.vision.display_window import VisionDisplay # Or LCDController if preferred
from eve.speech.audio_capture import AudioCapture
from eve.speech.speech_recognizer import SpeechRecognizer
from eve.speech.llm_processor import LLMProcessor
from eve.speech.text_to_speech import TextToSpeech
# Import the Orchestrator itself
from eve.orchestrator import EVEOrchestrator

# Setup basic logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Copied setup_logging function from orchestrator.py --- 
# TODO: Consider moving this to eve.utils.logging_utils if not already there
def setup_logging(log_config: Any): # Use specific LoggingConfig type hint if available
    """Configures logging based on the config object."""
    level = getattr(log_config, 'level', 'INFO').upper()
    log_file = getattr(log_config, 'file', None)
    max_bytes = getattr(log_config, 'max_size_mb', 10) * 1024 * 1024
    backup_count = getattr(log_config, 'backup_count', 3)
    log_to_console = getattr(log_config, 'console', True)
    log_format = getattr(log_config, 'format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format = getattr(log_config, 'date_format', "%Y-%m-%d %H:%M:%S")
    # Assuming logging_utils exists and has setup_logging
    logging_utils.setup_logging(level, log_file, max_bytes, backup_count, log_to_console, log_format, date_format)
# --------------------------------------------------------

class EVEApplication:
    """Manages the main application lifecycle."""
    def __init__(self):
        self._running = True
        self.orchestrator = None
        self.debug_menu_active = False 
        self.last_click_time = 0
        self.double_click_threshold = 0.5
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Sets the running flag to False and attempts graceful shutdown."""
        if self._running: # Prevent multiple shutdowns
             logger.info(f"Received signal {signum}. Initiating shutdown...")
             self._running = False
             # Directly signal the orchestrator to stop if it exists
             if self.orchestrator:
                  logger.info("Signaling orchestrator to stop...")
                  self.orchestrator.stop()

    def run(self):
        """Initializes orchestrator and runs the main application loop."""
        
        logger.info("--- Starting EVE Main Application ---")
        try:
            # 1. Load Configuration
            config = load_config() 

            # 2. Setup Logging (using loaded config)
            setup_logging(config.logging)
            logger.info("Logging configured.")
            logger.info(f"Loaded configuration: {config!r}")

            # 3. Initialize Subsystems (Copied/Adapted from orchestrator.main)
            camera = None
            face_detector = None
            object_detector = None
            emotion_analyzer = None
            audio_capture = None
            speech_recognizer = None
            llm_processor = None
            tts = None
            display_controller = None

            if config.hardware.camera_enabled:
                try:
                    camera = Camera(config)
                    if not camera.is_open():
                         logger.warning("Camera failed to initialize or is disabled.")
                         camera = None
                except Exception as e:
                    logger.error(f"Failed to initialize Camera: {e}", exc_info=True)
                    camera = None
            else:
                logger.info("Camera subsystem disabled by config.")

            if camera:
                if config.vision.face_detection_enabled:
                     try: face_detector = FaceDetector(config, camera)
                     except Exception as e: logger.error(f"Failed FaceDetector init: {e}", exc_info=True)
                else: logger.info("Face Detector disabled.")
                
                if config.vision.object_detection_enabled:
                     try: object_detector = ObjectDetector(config=config)
                     except Exception as e: logger.error(f"Failed ObjectDetector init: {e}", exc_info=True)
                else: logger.info("Object Detector disabled.")

                if config.vision.emotion_detection_enabled:
                     try: emotion_analyzer = EmotionAnalyzer(config)
                     except Exception as e: logger.error(f"Failed EmotionAnalyzer init: {e}", exc_info=True)
                else: logger.info("Emotion Analyzer disabled.")

            if config.hardware.display_enabled:
                try:
                    # Use VisionDisplay or LCDController based on preference/availability
                    display_controller = VisionDisplay(config, camera, face_detector, object_detector)
                except Exception as e: 
                    logger.error(f"Failed VisionDisplay init: {e}", exc_info=True)
                    display_controller = None
            else: logger.info("Display disabled by config.")

            if config.hardware.audio_input_enabled:
                try: audio_capture = AudioCapture(config.speech)
                except Exception as e: logger.error(f"Failed AudioCapture init: {e}", exc_info=True)
            else: logger.info("Audio Input disabled.")

            if audio_capture and config.speech.recognition_enabled:
                try: speech_recognizer = SpeechRecognizer(config.speech)
                except Exception as e: logger.error(f"Failed SpeechRecognizer init: {e}", exc_info=True)
            elif not audio_capture: logger.warning("Cannot init Recognizer: AudioCapture failed/disabled.")
            else: logger.info("Speech Recognition disabled.")

            if config.speech.tts_enabled:
                try: tts = TextToSpeech(config.speech)
                except Exception as e: logger.error(f"Failed TextToSpeech init: {e}", exc_info=True)
            else: logger.info("TextToSpeech disabled.")

            if config.speech.llm_enabled:
                try: llm_processor = LLMProcessor(config.speech)
                except Exception as e: logger.error(f"Failed LLMProcessor init: {e}", exc_info=True)
            else: logger.info("LLM Processor disabled.")

            # 4. Create and Run Orchestrator (Pass initialized subsystems)
            logger.info("Creating EVE Orchestrator...")
            # Use context manager for automatic stop/cleanup
            with EVEOrchestrator(
                config=config,
                camera=camera, # Pass initialized camera
                face_detector=face_detector,
                object_detector=object_detector,
                emotion_analyzer=emotion_analyzer,
                audio_capture=audio_capture,
                speech_recognizer=speech_recognizer,
                llm_processor=llm_processor,
                tts=tts,
                display_controller=display_controller
            ) as self.orchestrator:
                
                logger.info("Starting Orchestrator...")
                self.orchestrator.start() # Start the orchestrator's internal threads/loops

                # 5. Keep the main thread alive (e.g., wait for events or sleep)
                logger.info("EVE application running. Press Ctrl+C to exit.")
                while self.orchestrator._running: # Check orchestrator state
                    # Main thread can optionally do some work or just sleep
                    # orchestrator.update() might be called internally by its own thread
                    # or called here if it doesn't manage its own loop fully.
                    # Let's assume orchestrator manages its core loops.
                    time.sleep(1) 

        except Exception as e:
            logger.critical(f"Fatal error during EVE initialization or runtime: {e}", exc_info=True)
            # Ensure cleanup if orchestrator was partially created
            if self.orchestrator and hasattr(self.orchestrator, 'stop'):
                 try: self.orchestrator.stop()
                 except Exception as stop_err:
                      logger.error(f"Error during final cleanup attempt: {stop_err}")
            sys.exit(1)
        finally:
            # This block executes after the 'with' block finishes (orchestrator.stop() called by __exit__)
            # or after an exception escapes the try block (after orchestrator.stop() is attempted).
            logger.info("--- EVE Main Application Shutdown Complete ---")
            # Try calling pygame.quit() here to isolate hangs
            # try:
            #      pygame.quit()
            #      logger.info("Pygame quit called from run() finally block.")
            # except Exception as e:
            #      logger.error(f"Error during pygame quit in run() finally: {e}")
            logger.info("Skipping pygame.quit() for testing hang...")

    def _main_loop(self):
        """The main application loop handling events and updates."""
        while self._running:
            try:
                current_time = time.time()
                
                # Get current state from orchestrator if available
                is_correcting = self.orchestrator and self.orchestrator.is_correcting_detection
                current_debug_view = self.orchestrator.current_debug_view if self.orchestrator else None
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        break 
                    
                    # --- Correction Mode Input Handling --- 
                    if is_correcting:
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN:
                                logger.info("Enter pressed during correction.")
                                # Submit the correction
                                if self.orchestrator:
                                    self.orchestrator.submit_correction(self.orchestrator.user_input_buffer)
                                # is_correcting will be set to False by submit_correction
                            elif event.key == pygame.K_BACKSPACE:
                                # Remove last character from buffer
                                if self.orchestrator:
                                    self.orchestrator.user_input_buffer = self.orchestrator.user_input_buffer[:-1]
                            elif event.key == pygame.K_ESCAPE:
                                # Cancel correction on Escape key
                                if self.orchestrator:
                                     self.orchestrator.cancel_correction()
                            else:
                                # Append typed character to buffer
                                if self.orchestrator:
                                     self.orchestrator.user_input_buffer += event.unicode
                        # Ignore other events like mouse clicks while correcting
                        continue # Skip normal event processing

                    # --- Normal / Debug Menu Event Handling --- 
                    # Toggle Debug Menu (CTRL+S)
                    if event.type == pygame.KEYDOWN:
                        mods = pygame.key.get_mods()
                        if event.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
                            self.debug_menu_active = not self.debug_menu_active
                            logger.info(f"Debug Menu toggled {'ON' if self.debug_menu_active else 'OFF'} via CTRL+S")
                            # If turning off menu, tell orchestrator to clear view
                            if not self.debug_menu_active and self.orchestrator:
                                self.orchestrator.set_debug_view(None) 

                    # Debug Menu Toggle & UI Clicks (Mouse)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            clicked_element_id = None
                            # Only check UI elements if menu is active
                            if self.debug_menu_active and self.orchestrator and hasattr(self.orchestrator, 'lcd_controller'):
                                if hasattr(self.orchestrator.lcd_controller, 'get_debug_ui_element_at'):
                                     clicked_element_id = self.orchestrator.lcd_controller.get_debug_ui_element_at(event.pos)
                                
                            if clicked_element_id:
                                if hasattr(self.orchestrator, 'handle_debug_ui_click'):
                                    self.orchestrator.handle_debug_ui_click(clicked_element_id)
                                self.last_click_time = 0 # Prevent double-click toggle
                            else:
                                # Double-click toggles menu
                                if current_time - self.last_click_time < self.double_click_threshold:
                                    self.debug_menu_active = not self.debug_menu_active
                                    logger.info(f"Debug Menu toggled {'ON' if self.debug_menu_active else 'OFF'} via double-click")
                                    # If turning off menu, clear view
                                    if not self.debug_menu_active and self.orchestrator:
                                         self.orchestrator.set_debug_view(None)
                                    self.last_click_time = 0
                                else:
                                    self.last_click_time = current_time
                
                if not self._running:
                    break 

                # --- Update Orchestrator --- 
                if self.orchestrator:
                    # Pass menu active state, orchestrator handles current_view internally
                    self.orchestrator.update(debug_menu_active=self.debug_menu_active)

                # --- Frame Limiter --- 
                # Control loop timing - sleep based on desired FPS
                # This prevents high CPU usage in the main thread.
                # The orchestrator's update frequency depends on this.
                # Ensure orchestrator and its config are available before accessing FPS
                fps = 30 # Default FPS if orchestrator not ready
                if self.orchestrator and self.orchestrator.display_config:
                     fps = self.orchestrator.display_config.FPS
                time.sleep(1.0 / fps)

            except KeyboardInterrupt: 
                logger.info("KeyboardInterrupt received in main loop.")
                self._running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Decide if error is critical. Maybe add a counter to exit after N errors?
                time.sleep(1) # Prevent spamming logs on continuous errors

    def cleanup(self):
        """Explicit cleanup call, mainly for safety."""
        logger.debug("EVEApplication cleanup called.")
        # Pygame quit is now called earlier in run() finally.
        # This method might become redundant or just log.
        # try:
        #     pygame.quit()
        #     logger.info("Pygame shut down.")
        # except Exception as e:
        #     logger.error(f"Error during pygame quit: {e}")


def main():
    """Entry point for the EVE application."""
    app = EVEApplication()
    # The run() method's finally block now handles pygame.quit()
    # The main() finally block calling app.cleanup() may no longer be needed
    # for pygame.quit(), but can be kept for other potential cleanup.
    try:
        app.run()
    finally:
        # Optionally call app.cleanup() for any other future cleanup tasks
        # app.cleanup()
        logger.info("Application main() finished.")


if __name__ == "__main__":
    main() 