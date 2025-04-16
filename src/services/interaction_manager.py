"""
Interaction manager for coordinating all EVE2 services.
"""
import logging
import threading
import queue
import time
from typing import Optional, Dict, Any
from pathlib import Path

from .eye_display import EyeDisplay, Emotion
from .face_service import FaceService
from .emotion_service import EmotionService
from .voice_synth import VoiceSynth
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class InteractionManager:
    """Manages interaction between all EVE2 services."""
    
    def __init__(self, config_path: str):
        """Initialize the interaction manager."""
        self.config_path = config_path
        self.running = False
        self.services: Dict[str, Any] = {}
        self.command_queue = queue.Queue()
        self.thread = None
        
    def start(self) -> bool:
        """Start all services and begin interaction loop."""
        try:
            # Initialize services with the correct config path
            self.services = {
                'display': EyeDisplay(self.config_path),
                'face': FaceService(self.config_path),
                'emotion': EmotionService(self.config_path),
                'voice': VoiceSynth(self.config_path),
                'llm': LLMService(self.config_path)
            }
            
            # Start all services
            for name, service in self.services.items():
                if not service.start():
                    logger.error(f"Failed to start {name} service")
                    return False
                    
            # Start interaction thread
            self.running = True
            self.thread = threading.Thread(target=self._interaction_loop)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Interaction manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting interaction manager: {e}")
            return False
            
    def _interaction_loop(self) -> None:
        """Main interaction loop."""
        while self.running:
            try:
                # Process face detection
                faces = self.services['face'].detect_faces()
                if faces:
                    # Get first face
                    face = faces[0]
                    
                    # Recognize face
                    name = self.services['face'].recognize_face(face)
                    
                    # Detect emotion
                    emotion = self.services['emotion'].detect_emotion(face)
                    
                    # Update display
                    self.services['display'].set_emotion(emotion)
                    
                    # Process voice input if available
                    if self.services['voice'].has_input():
                        text = self.services['voice'].get_input()
                        
                        # Generate response
                        response = self.services['llm'].generate_response(
                            text,
                            user_name=name,
                            emotion=emotion.value if emotion else None
                        )
                        
                        if response:
                            # Speak response
                            self.services['voice'].speak(response)
                            
                # Update display
                self.services['display'].update()
                
                # Sleep to control update rate
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                logger.error(f"Error in interaction loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
                
    def stop(self) -> None:
        """Stop all services."""
        self.running = False
        if self.thread:
            self.thread.join()
            
        # Stop all services
        for service in self.services.values():
            service.stop()
            
        logger.info("Interaction manager stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 