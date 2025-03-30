"""
Text-to-speech module for the EVE2 system.

This module provides text-to-speech functionality using Piper TTS to
convert text responses to speech audio.
"""

import os
import logging
import time
import threading
import wave
import json
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import subprocess
import tempfile
import shutil

from eve.config import config

logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Text-to-speech processor using Piper TTS.
    
    This class provides functionality for converting text to speech using
    the Piper TTS system, which provides high-quality speech synthesis
    that can run efficiently on Raspberry Pi hardware.
    """
    
    def __init__(self, engine='pyttsx3', voice='english', rate=150, volume=1.0):
        self.logger = logging.getLogger(__name__)
        self.engine_type = engine
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.is_running = True
        self._init_engine()

    def _init_engine(self):
        """Initialize the text-to-speech engine with fallbacks"""
        self.logger.info("[TTS Init Trace] Entering _init_engine...")
        try:
            # Try pyttsx3 first
            if self.engine_type == 'pyttsx3':
                self.logger.info("[TTS Init Trace] Attempting pyttsx3 init...")
                try:
                    import pyttsx3
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', self.rate)
                    self.engine.setProperty('volume', self.volume)
                    self.logger.info("Initialized pyttsx3 engine")
                    self.logger.info("[TTS Init Trace] pyttsx3 Success.")
                    return # Success with pyttsx3
                except ImportError:
                    self.logger.warning("[TTS Init Trace] pyttsx3 ImportError. Falling back to espeak.")
                    self.engine_type = 'espeak'
                except Exception as e:
                    self.logger.warning(f"[TTS Init Trace] pyttsx3 Exception: {e}. Falling back to espeak.")
                    self.engine_type = 'espeak'
                self.logger.info("[TTS Init Trace] pyttsx3 check finished.")
            else:
                 self.logger.info("[TTS Init Trace] Skipping pyttsx3 (engine_type was not pyttsx3).")

            # Fallback to espeak: Try running it directly
            if self.engine_type == 'espeak':
                self.logger.info("[TTS Init Trace] Attempting espeak check...")
                try:
                    result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, check=False, timeout=2)
                    self.logger.info(f"[TTS Init Trace] espeak check subprocess finished. Return code: {result.returncode}")
                    if result.returncode == 0 or "Espeak NG" in result.stdout or "eSpeak" in result.stdout:
                         self.engine = 'espeak'
                         self.logger.info(f"Initialized espeak engine (found via subprocess check).")
                         self.logger.info("[TTS Init Trace] espeak Success.")
                         return # Success with espeak
                    else:
                         self.logger.warning(f"[TTS Init Trace] espeak check failed (RC={result.returncode}). Falling back to mock.")
                         self.engine_type = 'mock'
                except FileNotFoundError:
                    self.logger.warning("[TTS Init Trace] espeak FileNotFoundError. Falling back to mock.")
                    self.engine_type = 'mock'
                except subprocess.TimeoutExpired:
                     self.logger.warning("[TTS Init Trace] espeak TimeoutExpired. Falling back to mock.")
                     self.engine_type = 'mock'
                except Exception as e:
                    self.logger.warning(f"[TTS Init Trace] espeak Exception: {e}. Falling back to mock.")
                    self.engine_type = 'mock'
                self.logger.info("[TTS Init Trace] espeak check finished.")
            else:
                 self.logger.info("[TTS Init Trace] Skipping espeak check (engine_type was not espeak).")

            # Final fallback to mock engine
            if self.engine_type == 'mock':
                self.logger.info("[TTS Init Trace] engine_type is mock. Setting engine to mock.")
                self.engine = 'mock'
                self.logger.info("Initialized mock text-to-speech engine")
            else:
                 self.logger.warning(f"[TTS Init Trace] Reached end of _init_engine unexpectedly with engine_type={self.engine_type}. Defaulting to mock.")
                 self.engine_type = 'mock'
                 self.engine = 'mock'

        except Exception as e:
            self.logger.error(f"[TTS Init Trace] Outer Exception in _init_engine: {e}. Defaulting to mock.", exc_info=True)
            self.engine_type = 'mock'
            self.engine = 'mock'

    def speak(self, text):
        """Speak the given text using the current engine"""
        try:
            if not text:
                return

            if self.engine_type == 'pyttsx3':
                self.engine.say(text)
                self.engine.runAndWait()
            elif self.engine_type == 'espeak':
                subprocess.run(['espeak', text], check=False)
            else:  # mock engine
                self.logger.info(f"Mock TTS would say: {text}")

        except Exception as e:
            self.logger.error(f"Error during speech: {e}")

    def play_startup_sound(self):
        """Play a startup sound or message"""
        self.speak("System initialized and ready")

    def stop(self):
        """Stop the text-to-speech engine"""
        self.is_running = False
        try:
            if self.engine_type == 'pyttsx3' and hasattr(self, 'engine'):
                self.engine.stop()
        except Exception as e:
            self.logger.error(f"Error stopping text-to-speech: {e}")

    def start(self):
        """Start the text-to-speech processor."""
        if self.is_running:
            logger.warning("TTS processor is already running")
            return False
        
        if not self.model_available:
            logger.error("Cannot start TTS: Model not available")
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("TTS processor started")
        return True
    
    def say(self, text: str):
        """
        Convert text to speech asynchronously.
        
        Args:
            text: The text to convert to speech.
        """
        if not self.is_running:
            logger.warning("TTS processor is not running")
            return
        
        # Add text to the queue
        self.text_queue.append(text)
        
        logger.debug(f"Added text to TTS queue: '{text}'")
    
    def say_sync(self, text: str) -> bool:
        """
        Convert text to speech synchronously.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.model_available:
            logger.error("Cannot synthesize speech: Model not available")
            return False
        
        logger.info(f"Synthesizing speech: '{text}'")
        
        try:
            # Generate speech
            audio_data = self._synthesize_speech(text)
            if audio_data is None:
                return False
            
            # Play the audio
            self._play_audio(audio_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return False
    
    def is_busy(self) -> bool:
        """Check if the TTS processor is currently speaking or has queued text."""
        return self.is_speaking or len(self.text_queue) > 0
    
    def clear_queue(self):
        """Clear the text queue."""
        self.text_queue = []
        logger.info("TTS queue cleared")
    
    def _process_loop(self):
        """Process text from the queue."""
        try:
            while self.is_running:
                # Check if there's text in the queue
                if self.text_queue:
                    # Get the next text
                    text = self.text_queue.pop(0)
                    self.current_text = text
                    
                    # Synthesize and play speech
                    self.is_speaking = True
                    audio_data = self._synthesize_speech(text)
                    
                    if audio_data is not None:
                        # Play audio in a separate thread
                        self.audio_thread = threading.Thread(
                            target=self._play_audio,
                            args=(audio_data,),
                            daemon=True
                        )
                        self.audio_thread.start()
                        self.audio_thread.join()  # Wait for audio to finish
                    
                    self.is_speaking = False
                    self.current_text = None
                    
                    # Call callback
                    if self.callback:
                        self.callback(text)
                
                # Sleep to avoid busy waiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in process loop: {e}")
            self.is_running = False
            self.is_speaking = False
    
    def _synthesize_speech(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize speech using Piper TTS.
        
        Args:
            text: The text to synthesize.
            
        Returns:
            Audio data as a NumPy array, or None if synthesis failed.
        """
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Check if piper is installed
            piper_executable = shutil.which("piper")
            if not piper_executable:
                if os.name == "nt":
                    piper_executable = shutil.which("piper.exe")
                
                if not piper_executable:
                    logger.error("Piper executable not found in PATH. Please install piper-tts")
                    return None
                
            logger.debug(f"Using Piper executable: {piper_executable}")
                
            # Command arguments
            cmd = [
                piper_executable,
                "--model", str(self.model_file),
                "--output_file", output_path,
                "--sentence_silence", "0.25"
            ]
            
            # Add speaking rate if not 1.0
            if self.speaking_rate != 1.0:
                cmd.extend(["--length_scale", str(1.0 / self.speaking_rate)])
            
            # Run piper
            logger.debug(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send text to stdin
            stdout, stderr = process.communicate(text)
            
            if process.returncode != 0:
                logger.error(f"Piper TTS failed with return code {process.returncode}: {stderr}")
                return None
            
            # Read the WAV file
            try:
                audio_data = self._read_wav(output_path)
                return audio_data
            finally:
                # Clean up temporary file
                if os.path.exists(output_path):
                    os.unlink(output_path)
                
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
    
    def _read_wav(self, file_path: str) -> np.ndarray:
        """
        Read WAV file and return audio data.
        
        Args:
            file_path: Path to the WAV file.
            
        Returns:
            Audio data as a NumPy array.
        """
        with wave.open(file_path, 'rb') as wf:
            # Get WAV properties
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all frames
            frames = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit audio
                dtype = np.int16
            elif sample_width == 4:  # 32-bit audio
                dtype = np.int32
            else:
                dtype = np.uint8
            
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            # Reshape for multiple channels
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
            
            # Convert to float32 normalized between -1 and 1
            audio_data = audio_data.astype(np.float32) / np.iinfo(dtype).max
            
            return audio_data
    
    def _play_audio(self, audio_data: np.ndarray):
        """
        Play audio data.
        
        Args:
            audio_data: Audio data as a NumPy array.
        """
        try:
            # Start playback
            sd.play(audio_data, samplerate=self.sample_rate, device=self.device_index)
            
            # Wait for playback to finish
            while sd.get_stream().active and not self.stop_event.is_set():
                time.sleep(0.1)
            
            # Stop playback if requested
            if self.stop_event.is_set():
                sd.stop()
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            
    def set_voice(self, voice: str):
        """
        Set the voice to use for speech synthesis.
        
        Args:
            voice: The voice to use.
        """
        self.voice = voice
        logger.info(f"TTS voice set to {voice}")
    
    def set_speaking_rate(self, speaking_rate: float):
        """
        Set the speaking rate.
        
        Args:
            speaking_rate: Speaking rate multiplier (1.0 = normal speed).
        """
        self.speaking_rate = max(0.5, min(2.0, speaking_rate))  # Limit range
        logger.info(f"TTS speaking rate set to {self.speaking_rate}") 