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
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eve import config
from eve.orchestrator import create_orchestrator
from eve.utils import logging_utils

# Add at the top of the file, before importing pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_RENDERER_DRIVER'] = 'software'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

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
    orchestrator = None
    try:
        # Initialize and start the orchestrator
        orchestrator = create_orchestrator()
        orchestrator.start()
        
        # Main loop
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error starting EVE2: {e}")
    finally:
        # Ensure proper shutdown
        if orchestrator is not None:
            try:
                orchestrator.stop()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
            
        logger.info("EVE2 shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        # Ensure pygame is properly quit
        try:
            import pygame
            pygame.quit()
        except:
            pass 