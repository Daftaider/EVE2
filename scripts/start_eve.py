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

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("Shutdown signal received, cleaning up...")
    running = False

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parsing.
    
    Returns:
        An ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="EVE2 Interactive Robot System")
    
    # Logging options
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        help="Path to log file"
    )
    
    # Configuration options
    parser.add_argument(
        "--role", 
        type=str, 
        choices=["all", "vision", "speech", "display", "master"],
        help="Set the role of this instance"
    )
    
    parser.add_argument(
        "--distributed", 
        action="store_true", 
        help="Enable distributed mode"
    )
    
    parser.add_argument(
        "--master-ip", 
        type=str, 
        help="IP address of the master node (for distributed mode)"
    )
    
    parser.add_argument(
        "--master-port", 
        type=int, 
        help="Port of the master node (for distributed mode)"
    )
    
    # Hardware options
    parser.add_argument(
        "--camera-index", 
        type=int, 
        help="Camera device index"
    )
    
    parser.add_argument(
        "--no-camera", 
        action="store_true", 
        help="Disable camera"
    )
    
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Disable display"
    )
    
    parser.add_argument(
        "--fullscreen", 
        action="store_true", 
        help="Run display in fullscreen mode"
    )
    
    parser.add_argument(
        "--no-audio-input", 
        action="store_true", 
        help="Disable audio input"
    )
    
    parser.add_argument(
        "--no-audio-output", 
        action="store_true", 
        help="Disable audio output"
    )
    
    parser.add_argument(
        "--audio-input-device", 
        type=int, 
        help="Audio input device index"
    )
    
    parser.add_argument(
        "--audio-output-device", 
        type=int, 
        help="Audio output device index"
    )
    
    return parser

def apply_cli_options(args: argparse.Namespace) -> None:
    """
    Apply command-line options to the configuration.
    
    Args:
        args: Parsed command-line arguments
    """
    # Apply logging options
    if args.log_level:
        config.logging.LOG_LEVEL = args.log_level
    
    if args.log_file:
        config.logging.LOG_FILE = Path(args.log_file)
    
    # Apply role and distribution options
    if args.role:
        config.hardware.ROLE = args.role
    
    if args.distributed:
        config.hardware.DISTRIBUTED_MODE = True
    
    if args.master_ip:
        config.hardware.MASTER_IP = args.master_ip
    
    if args.master_port:
        config.hardware.MASTER_PORT = args.master_port
    
    # Apply hardware options
    if args.camera_index is not None:
        config.hardware.CAMERA_INDEX = args.camera_index
    
    if args.no_camera:
        config.hardware.CAMERA_ENABLED = False
    
    if args.no_display:
        config.hardware.DISPLAY_ENABLED = False
    
    if args.fullscreen:
        config.hardware.FULLSCREEN = True
    
    if args.no_audio_input:
        config.hardware.AUDIO_INPUT_ENABLED = False
    
    if args.no_audio_output:
        config.hardware.AUDIO_OUTPUT_ENABLED = False
    
    if args.audio_input_device is not None:
        config.hardware.AUDIO_INPUT_DEVICE = args.audio_input_device
    
    if args.audio_output_device is not None:
        config.hardware.AUDIO_OUTPUT_DEVICE = args.audio_output_device

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Example configuration
    config = {
        'display': {
            'WINDOW_SIZE': (800, 480),
            'FPS': 30,
            'DEFAULT_EMOTION': Emotion.NEUTRAL,  # Use enum directly
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
        with EVEOrchestrator(config) as eve:
            logger.info("EVE2 system started, press Ctrl+C to exit")
            
            # Main loop
            while running:
                try:
                    # Update display
                    eve.update()
                    
                    # Add a small delay to prevent high CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    if not running:
                        break
                    # Continue running unless shutdown was requested
                    continue

    except Exception as e:
        logger.error(f"Error starting EVE2: {e}")
    finally:
        logger.info("EVE2 shutdown complete")

if __name__ == "__main__":
    main()
    # Ensure pygame is properly quit
    try:
        import pygame
        pygame.quit()
    except:
        pass 