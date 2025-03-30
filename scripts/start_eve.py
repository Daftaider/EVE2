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

class EVEApplication:
    """Manages the main application lifecycle."""
    def __init__(self):
        self._running = True
        self.orchestrator = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Sets the running flag to False on receiving SIGINT or SIGTERM."""
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self._running = False

    def run(self):
        """Initializes orchestrator and runs the main application loop."""
        
        # --- Configuration ---
        # Consider loading from a file (e.g., YAML, JSON) instead of hardcoding
        config = {
            'display': {
                'WINDOW_SIZE': (800, 480),
                'FPS': 30,
                'DEFAULT_EMOTION': Emotion.NEUTRAL,
                'DEFAULT_BACKGROUND_COLOR': 'black',
                'DEFAULT_EYE_COLOR': (0, 255, 255)
            },
            'speech': {
                'SPEECH_RECOGNITION_MODEL': 'google',
                'TTS_ENGINE': 'pyttsx3',
                'WAKE_WORD_PHRASE': 'hey eve',
                'AUDIO_SAMPLE_RATE': 16000,
                'NOISE_THRESHOLD': 0.05
            }
        }

        try:
            # Initialize orchestrator within a context manager
            with EVEOrchestrator(config) as self.orchestrator:
                logger.info("EVE system started. Press Ctrl+C to exit.")
                self._main_loop()

        except Exception as e:
            logger.critical(f"Fatal error during EVE initialization or runtime: {e}", exc_info=True)
            # Cleanup might have already been called by __exit__, but call again just in case.
            self.cleanup() 
            sys.exit(1) # Exit with error code

        finally:
            # Ensure cleanup happens even if context manager fails somehow
            self.cleanup() 
            logger.info("EVE application has shut down.")

    def _main_loop(self):
        """The main application loop handling events and updates."""
        while self._running:
            try:
                # Handle Pygame events (essential for window responsiveness)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.info("Received Pygame QUIT event.")
                        self._running = False
                        break # Exit event loop
                
                if not self._running:
                    break # Exit main loop if running flag changed

                # Update the orchestrator (which updates subsystems)
                if self.orchestrator:
                    self.orchestrator.update()

                # Control loop timing - sleep based on desired FPS
                # This prevents high CPU usage in the main thread.
                # The orchestrator's update frequency depends on this.
                time.sleep(1.0 / self.orchestrator.display_config.FPS if self.orchestrator else 0.1)

            except KeyboardInterrupt: # Redundant if signals handled, but safe
                logger.info("KeyboardInterrupt received in main loop.")
                self._running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Decide if error is critical. Maybe add a counter to exit after N errors?
                time.sleep(1) # Prevent spamming logs on continuous errors

    def cleanup(self):
        """Explicit cleanup call, mainly for safety."""
        logger.debug("EVEApplication cleanup called.")
        # Orchestrator cleanup is handled by its __exit__ method
        # Pygame quit is crucial here if not handled elsewhere reliably
        try:
            pygame.quit()
            logger.info("Pygame shut down.")
        except Exception as e:
            logger.error(f"Error during pygame quit: {e}")


def main():
    """Entry point for the EVE application."""
    app = EVEApplication()
    app.run()

if __name__ == "__main__":
    main() 