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
        
        # Set ALSA configuration for Waveshare Audio Hat
        config_dir = os.path.dirname(config_path)
        alsa_config_path = os.path.join(config_dir, 'asound.conf')
        
        # Try different ALSA card names
        alsa_card_names = ['wm8960soundcard', 'wm8960', 'default']
        identified_card = False
        for card_name in alsa_card_names:
            try:
                # os.environ['ALSA_CARD'] = card_name # Removed: Let system defaults handle card selection
                # os.environ['ALSA_PCM_CARD'] = card_name # Removed
                # os.environ['ALSA_PCM_DEVICE'] = '0' # Removed
                # os.environ['ALSA_CONFIG_PATH'] = alsa_config_path # Removed: Let ALSA use default system config
                
                # Test if the card seems available via sr listing
                if not identified_card and self._test_alsa_card(card_name):
                    logger.info(f"Identified potential working ALSA card via name: {card_name}")
                    identified_card = True # Found one, no need to keep looping/setting env vars
                    # We don't break here, just note that a potentially valid card name was found
            except Exception as e:
                logger.warning(f"Exception during ALSA card test for {card_name}: {e}")
        
        if not identified_card:
             logger.warning("Could not identify a specific working ALSA card name via testing. Relying on system defaults.")

        # Disable PulseAudio to avoid conflicts with ALSA
        os.environ['PULSE_SERVER'] = ''
        
        # Disable JACK to avoid conflicts
        os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
            
    def _test_alsa_card(self, card_name: str) -> bool:
        """Test if an ALSA card is available."""
        try:
            # Try to list available devices
            mic_list = sr.Microphone.list_microphone_names()
            if mic_list:
                logger.info(f"Found microphones with card {card_name}: {mic_list}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Error testing ALSA card {card_name}: {e}")
            return False
            
    def start(self) -> bool:
        """Start the voice synthesis service."""
        try:
            # Initialize text-to-speech engine
            try:
                # Try forcing espeak driver on Linux, might be more stable with ALSA
                if platform.system() == "Linux":
                    logger.info("Attempting to initialize pyttsx3 with espeak driver...")
                    self.engine = pyttsx3.init(driverName='espeak')
                else:
                    self.engine = pyttsx3.init()
            except Exception as e:
                logger.warning(f"Failed to init pyttsx3 with specific driver, falling back to default: {e}")
                self.engine = pyttsx3.init() # Fallback to default driver

            # Set voice properties (if engine initialized)
            if self.engine:
                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)  # Use first available voice
                self.engine.setProperty('rate', 150)  # Speed of speech
                self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            else:
                logger.error("Failed to initialize pyttsx3 engine.")
                # Decide if we should continue without TTS? For now, we will.

            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            
            # Configure recognizer settings
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 0.5
            
            # Find and initialize the target microphone
            self.microphone = None
            try:
                mic_list = sr.Microphone.list_microphone_names()
                logger.info(f"Available microphones: {mic_list}")
                
                # Try each available microphone device index until one works
                for i, mic_name in enumerate(mic_list):
                    logger.info(f"Attempting to initialize microphone: {mic_name} (index {i})")
                    try:
                        temp_mic = sr.Microphone(device_index=i)
                        with temp_mic as source:
                            logger.info(f"Adjusting for ambient noise on {mic_name} (index {i})...")
                            self.recognizer.adjust_for_ambient_noise(source, duration=1.0) # Reduced duration slightly
                        self.microphone = temp_mic # Successfully initialized
                        logger.info(f"Successfully initialized microphone: {mic_name} (index {i})")
                        break # Found a working microphone
                    except Exception as e_mic:
                        logger.warning(f"Failed to initialize {mic_name} (index {i}): {e_mic}")
                
                if not self.microphone:
                    logger.error("No suitable microphone device found after trying all available indices.")

            except Exception as e:
                logger.error(f"Failed during microphone search/initialization: {e}")
                self.microphone = None # Ensure microphone is None if init fails

            # Start listening thread ONLY if microphone was successfully initialized
            if self.microphone:
                self.running = True
                self.thread = threading.Thread(target=self._listen_loop)
                self.thread.daemon = True
                self.thread.start()
                logger.info("Voice synthesis service started successfully (with microphone)")
            else:
                logger.warning("Could not initialize microphone. Voice input disabled.")
                # Continue running for text-to-speech functionality if engine is available

            # Service start is considered successful if either TTS or STT (or both) are ready
            if self.engine or self.microphone:
                 return True
            else:
                 logger.error("Failed to initialize both TTS engine and microphone.")
                 return False

        except Exception as e:
            logger.exception(f"Critical error starting voice synthesis service: {e}") # Use exception for full traceback
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