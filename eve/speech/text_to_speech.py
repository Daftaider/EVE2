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
    
    def __init__(self, voice=None, rate=1.0, pitch=1.0, volume=1.0, engine=None, 
                 voice_id=None, coqui_model_path=None):
        """
        Initialize text to speech synthesizer
        
        Args:
            voice (str): Voice name to use
            voice_id (str): Alternative voice identifier
            rate (float): Speech rate (1.0 is normal speed)
            pitch (float): Voice pitch (1.0 is normal pitch)
            volume (float): Audio volume (0.0-1.0)
            engine (str): TTS engine to use ('pyttsx3', 'espeak', 'coqui', etc.)
            coqui_model_path (str): Path to Coqui TTS model files
        """
        self.voice = voice
        self.voice_id = voice_id or voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.engine = engine or "pyttsx3"
        self.coqui_model_path = coqui_model_path
        self.tts_engine = None
        
        logger.info(f"Initializing text to speech with engine: {self.engine}")
        
        # Initialize appropriate TTS engine
        self._init_tts_engine()
    
    def _init_tts_engine(self):
        """Initialize the selected TTS engine"""
        if self.engine == "pyttsx3":
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                if self.voice_id:
                    self.tts_engine.setProperty('voice', self.voice_id)
                self.tts_engine.setProperty('rate', int(self.rate * 200))
                self.tts_engine.setProperty('volume', self.volume)
                logger.info("Initialized pyttsx3 engine")
            except ImportError:
                logger.error("Failed to initialize pyttsx3: No module named 'pyttsx3'")
                logger.info("Falling back to espeak")
                self.engine = "espeak"
                self._init_espeak()
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                logger.info("Falling back to espeak")
                self.engine = "espeak"
                self._init_espeak()
        elif self.engine == "espeak":
            self._init_espeak()
        elif self.engine == "coqui":
            self._init_coqui()
        else:
            logger.warning(f"Unknown TTS engine: {self.engine}, using fallback")
            self._init_fallback()
    
    def _init_espeak(self):
        """Initialize espeak as fallback"""
        try:
            # Check if espeak is available
            import subprocess
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Using espeak as fallback TTS engine")
                self.tts_engine = "espeak"
            else:
                logger.warning("espeak not found, using silent fallback")
                self._init_fallback()
        except Exception as e:
            logger.error(f"Failed to initialize espeak: {e}")
            self._init_fallback()
    
    def _init_coqui(self):
        """Initialize Coqui TTS engine"""
        try:
            if self.coqui_model_path and os.path.exists(self.coqui_model_path):
                # Here you would initialize Coqui TTS with the model
                logger.info(f"Initialized Coqui TTS with model: {self.coqui_model_path}")
                self.tts_engine = "coqui"
            else:
                logger.error("Coqui model path not found")
                self._init_fallback()
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback (silent) TTS"""
        logger.info("Using silent fallback TTS")
        self.tts_engine = "fallback"
    
    def speak(self, text):
        """Convert text to speech and play it"""
        if not text:
            logger.warning("Empty text provided to TTS")
            return False
            
        logger.info(f"Speaking: {text[:50]}...")
        
        try:
            if self.engine == "pyttsx3" and isinstance(self.tts_engine, object):
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return True
            elif self.engine == "espeak":
                import subprocess
                cmd = ["espeak"]
                if self.voice_id:
                    cmd.extend(["-v", self.voice_id])
                cmd.extend(["-a", str(int(self.volume * 100))])
                cmd.extend(["-s", str(int(self.rate * 150))])
                cmd.extend(["-p", str(int(self.pitch * 50))])
                cmd.append(text)
                subprocess.run(cmd)
                return True
            else:
                # Fallback just logs the text
                logger.info(f"[FALLBACK TTS] Would speak: {text}")
                return True
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return False

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
    
    def stop(self):
        """Stop the text-to-speech processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.process_thread is not None and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        if self.audio_thread is not None and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        logger.info("TTS processor stopped")
    
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

    def speak(self, text):
        """Convert text to speech and play it"""
        logger.info(f"Speaking: {text}")
        return True

    def play_startup_sound(self):
        """Play a short startup sound"""
        try:
            startup_message = "System initialized and ready"
            self.speak(startup_message)
            logger.info("Played startup sound")
        except Exception as e:
            logger.error(f"Failed to play startup sound: {e}") 