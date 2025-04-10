"""
Main orchestrator for EVE2.
"""
import logging
import threading
import time
from typing import Optional
from pathlib import Path

from vision.camera import Camera
from vision.object_detector import ObjectDetector
from display.lcd_controller import LCDController, Emotion

logger = logging.getLogger(__name__)

class EVEOrchestrator:
    """Main orchestrator for EVE2."""
    
    def __init__(self, width: int = 800, height: int = 480):
        """Initialize EVE2 orchestrator."""
        self.width = width
        self.height = height
        self.running = False
        
        # Initialize components
        self.camera = Camera(resolution=(width, height))
        self.object_detector = ObjectDetector()
        self.display = LCDController(width=width, height=height)
        
        # Thread control
        self.update_thread = None
        self.thread_lock = threading.Lock()
        
    def start(self) -> None:
        """Start EVE2."""
        logger.info("Starting EVE2...")
        
        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            return
            
        # Load object detection model
        if not self.object_detector.load_model():
            logger.error("Failed to load object detection model")
            self.camera.stop()
            return
            
        # Start display
        if not self.display.start():
            logger.error("Failed to start display")
            self.camera.stop()
            return
            
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("EVE2 started successfully")
        
    def stop(self) -> None:
        """Stop EVE2."""
        logger.info("Stopping EVE2...")
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
            
        self.camera.stop()
        self.display.stop()
        
        logger.info("EVE2 stopped successfully")
        
    def _update_loop(self) -> None:
        """Main update loop."""
        while self.running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Run object detection
                detections = self.object_detector.detect(frame)
                
                # Update display
                self.display.update_frame(frame, detections)
                
                # Update display state
                if not self.display.handle_events():
                    self.running = False
                    break
                    
                self.display.update()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(0.1)  # Prevent rapid error loops
                
    def set_emotion(self, emotion: Emotion) -> None:
        """Set EVE's emotion."""
        self.display.set_emotion(emotion)
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

def main():
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and start EVE2
        with EVEOrchestrator() as eve:
            # Main loop
            try:
                while True:
                    time.sleep(0.1)  # Small sleep to prevent CPU hogging
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                
    except Exception as e:
        logger.error(f"Error running EVE2: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 