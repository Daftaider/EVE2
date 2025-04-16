"""
Voice synthesis and recognition service for EVE2.
"""
import logging
import queue
import threading
import time
import os
import platform
from typing import Optional, Dict, Any

import pyttsx3
import speech_recognition as sr

logger = logging.getLogger(__name__)

class VoiceSynth:
    """Handles text-to-speech and speech recognition."""
    
    def __init__(self, config_path: str):
        """Initialize the voice synthesis service."""
        self.config_path = config_path
        self.running = False
        self.engine = None
        self.recognizer = None
        self.microphone = None
        self.input_queue = queue.Queue()
        self.thread = None
        
        # Platform-specific audio configuration
        if platform.system() == 'Windows':
            # Windows-specific settings
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame welcome message
        else:
            # Linux/Unix settings
            os.environ['ALSA_CARD'] = 'Generic'
            os.environ['ALSA_PCM_CARD'] = '0'
            os.environ['ALSA_PCM_DEVICE'] = '0'
            
    def start(self) -> bool:
        """Start the voice synthesis service."""
        try:
            # Initialize text-to-speech engine
            self.engine = pyttsx3.init()
            
            # Set voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice on Windows
                if platform.system() == 'Windows':
                    for voice in voices:
                        if 'female' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                    else:
                        self.engine.setProperty('voice', voices[0].id)
                else:
                    self.engine.setProperty('voice', voices[0].id)
                    
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            
            # Configure recognizer settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5
            
            # Try to find a working microphone
            for i in range(3):  # Try up to 3 times
                try:
                    # List available microphones
                    mic_list = sr.Microphone.list_microphone_names()
                    logger.info(f"Available microphones: {mic_list}")
                    
                    # Try to use the first available microphone
                    device_index = None
                    for idx, name in enumerate(mic_list):
                        if platform.system() == 'Windows':
                            # On Windows, prefer the default microphone
                            if 'default' in name.lower():
                                device_index = idx
                                break
                        else:
                            # On Linux/Unix, try to find a working microphone
                            if 'pulse' in name.lower() or 'default' in name.lower():
                                device_index = idx
                                break
                                
                    if device_index is None and mic_list:
                        device_index = 0  # Use first microphone if no default found
                        
                    if device_index is not None:
                        self.microphone = sr.Microphone(device_index=device_index)
                        with self.microphone as source:
                            self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        break
                    else:
                        logger.warning("No microphones found")
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize microphone (attempt {i+1}): {e}")
                    time.sleep(1)
                    
            if not self.microphone:
                logger.error("Could not initialize microphone")
                return False
                
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