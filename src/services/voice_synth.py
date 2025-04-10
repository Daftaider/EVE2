"""
Voice synthesis and recognition service for EVE2.
"""
import logging
import queue
import threading
import time
from typing import Optional, Dict, Any

import pyttsx3
import speech_recognition as sr

logger = logging.getLogger(__name__)

class VoiceSynth:
    """Handles text-to-speech and speech recognition."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the voice synthesis service."""
        self.config_path = config_path
        self.running = False
        self.engine = None
        self.recognizer = None
        self.microphone = None
        self.input_queue = queue.Queue()
        self.thread = None
        
    def start(self) -> bool:
        """Start the voice synthesis service."""
        try:
            # Initialize text-to-speech engine
            self.engine = pyttsx3.init()
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                
            # Start listening thread
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Voice synthesis service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voice synthesis service: {e}")
            return False
            
    def _listen_loop(self) -> None:
        """Background thread for continuous speech recognition."""
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source)
                    
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.input_queue.put(text)
                    logger.debug(f"Recognized speech: {text}")
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Could not request results: {e}")
                    
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
                
    def has_input(self) -> bool:
        """Check if there is any speech input available."""
        return not self.input_queue.empty()
        
    def get_input(self) -> Optional[str]:
        """Get the next speech input if available."""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None
            
    def speak(self, text: str) -> None:
        """Convert text to speech."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
    def stop(self) -> None:
        """Stop the voice synthesis service."""
        self.running = False
        if self.thread:
            self.thread.join()
            
        # Clean up resources
        if self.engine:
            self.engine.stop()
            
        logger.info("Voice synthesis service stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 