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
        self.debug_mode = False  # Add debug mode flag
        self.last_click_time = 0 # For double-click detection
        self.double_click_threshold = 0.5 # Seconds for double-click
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
                current_time = time.time()
                
                # --- Handle Events --- 
                is_correcting = self.orchestrator and self.orchestrator.is_correcting_detection
                
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

                    # --- Normal Mode Input Handling --- 
                    # Debug Mode Toggling (CTRL+S)
                    if event.type == pygame.KEYDOWN:
                        mods = pygame.key.get_mods()
                        if event.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
                            self.debug_mode = not self.debug_mode
                            logger.info(f"Debug mode toggled {'ON' if self.debug_mode else 'OFF'} via CTRL+S")
                            # If entering correction mode while debug is off, turn debug on
                            if is_correcting and not self.debug_mode:
                                 self.debug_mode = True 

                    # Debug Mode Toggling & UI Clicks (Mouse)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Check for UI element click FIRST if in debug mode
                        clicked_element_id = None
                        if self.debug_mode and self.orchestrator and hasattr(self.orchestrator, 'lcd_controller'):
                            if hasattr(self.orchestrator.lcd_controller, 'get_debug_ui_element_at'):
                                 clicked_element_id = self.orchestrator.lcd_controller.get_debug_ui_element_at(event.pos)
                            
                        if clicked_element_id:
                            # Handle click on a specific UI element
                            logger.debug(f"Debug UI element clicked: {clicked_element_id}")
                            if hasattr(self.orchestrator, 'handle_debug_ui_click'):
                                self.orchestrator.handle_debug_ui_click(clicked_element_id)
                            # Prevent double-click toggle if a UI element was clicked
                            self.last_click_time = 0 
                        else:
                            # If not clicking a UI element, check for double-click to toggle debug mode
                            if current_time - self.last_click_time < self.double_click_threshold:
                                self.debug_mode = not self.debug_mode
                                logger.info(f"Debug mode toggled {'ON' if self.debug_mode else 'OFF'} via double-click")
                                # If turning off debug mode while correcting, cancel correction
                                if is_correcting and not self.debug_mode:
                                     if self.orchestrator:
                                         self.orchestrator.cancel_correction()
                                self.last_click_time = 0 # Reset after double-click detected
                            else:
                                self.last_click_time = current_time
                
                if not self._running:
                    break 

                # --- Update Orchestrator --- 
                if self.orchestrator:
                    self.orchestrator.update(debug_mode=self.debug_mode)

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