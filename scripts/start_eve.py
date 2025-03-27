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

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eve import config
from eve.orchestrator import EVEOrchestrator
from eve.utils import logging_utils
from eve.config.display import Emotion  # Add this import

# Add at the top of the file, before importing pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_RENDERER_DRIVER'] = 'software'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EVEApplication:
    def __init__(self):
        self.running = True
        self.eve = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, cleaning up...")
        self.running = False

    def run(self):
        """Run the EVE2 application."""
        # Example configuration
        config = {
            'display': {
                'WINDOW_SIZE': (800, 480),
                'FPS': 30,
                'DEFAULT_EMOTION': Emotion.NEUTRAL,
                'DEFAULT_BACKGROUND_COLOR': (0, 0, 0),
                'DEFAULT_EYE_COLOR': (255, 255, 255)
            },
            'speech': {
                'SPEECH_RECOGNITION_MODEL': 'google',
                'TTS_ENGINE': 'pyttsx3',
                'AUDIO_SAMPLE_RATE': 16000,
                'LLM_MODEL_PATH': 'models/llm/simple_model',
                'LLM_CONTEXT_LENGTH': 1024,
                'COQUI_MODEL_PATH': 'models/tts/coqui'
            }
        }

        try:
            with EVEOrchestrator(config) as self.eve:
                logger.info("EVE2 system started, press Ctrl+C to exit")
                
                # Main loop
                while self.running:
                    try:
                        # Handle pygame events in the main loop
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                break
                        
                        # Update display
                        self.eve.update()
                        
                        # Add a small delay to prevent high CPU usage
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        if not self.running:
                            break
                        continue

        except Exception as e:
            logger.error(f"Error starting EVE2: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.eve:
                self.eve.cleanup()
            pygame.quit()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("EVE2 shutdown complete")

def main():
    app = EVEApplication()
    app.run()

if __name__ == "__main__":
    main() 