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

class TextToSpeech:
    """
    Text-to-speech processor using Piper TTS.
    
    This class provides functionality for converting text to speech using
    the Piper TTS system, which provides high-quality speech synthesis
    that can run efficiently on Raspberry Pi hardware.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        voice: Optional[str] = None,
        speaking_rate: float = 1.0,
        device_index: Optional[int] = None,
        sample_rate: int = 22050,
        callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the text-to-speech processor.
        
        Args:
            model_path: Path to the Piper TTS model. If None, uses the model from config.
            voice: Voice to use for speech synthesis. If None, uses the voice from config.
            speaking_rate: Speaking rate multiplier (1.0 = normal speed).
            device_index: Audio output device index. If None, uses the default device.
            sample_rate: Audio sample rate for output.
            callback: Callback function to call when speech is complete.
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_path = model_path or config.speech.tts_model
        self.voice = voice or config.speech.tts_voice
        self.speaking_rate = speaking_rate or config.speech.tts_speaking_rate
        self.device_index = device_index or config.hardware.audio_output_device
        self.sample_rate = sample_rate
        self.callback = callback
        
        # State
        self.is_running = False
        self.is_speaking = False
        self.text_queue = []
        self.current_text = None
        self.process_thread = None
        self.audio_thread = None
        self.stop_event = threading.Event()
        
        # Validate the model path
        if not os.path.isdir(self.model_path):
            self.logger.error(f"TTS model directory not found: {self.model_path}")
            self.model_available = False
            return
            
        # Parse the voice format (should be "LANG/NAME")
        voice_parts = self.voice.split("/")
        if len(voice_parts) != 2:
            self.logger.error(f"Invalid voice format: {self.voice}. Expected format: 'LANG/NAME'")
            self.model_available = False
            return
            
        # Get the model file paths
        lang, name = voice_parts
        self.model_dir = Path(self.model_path)
        model_file = self.model_dir / lang / f"{name}_medium.onnx"
        config_file = self.model_dir / lang / f"{name}_medium.onnx.json"
        
        # If files don't exist with _medium suffix, try without it
        if not model_file.exists() or not config_file.exists():
            # Check if we have the files with the exact names from the download
            model_file = self.model_dir / lang / "lessac_medium.onnx"
            config_file = self.model_dir / lang / "lessac_medium.onnx.json"
        
        # Check if model files exist
        if not model_file.exists():
            self.logger.error(f"TTS model file not found: {model_file}")
            self.model_available = False
            return
            
        if not config_file.exists():
            self.logger.error(f"TTS config file not found: {config_file}")
            self.model_available = False
            return
            
        self.model_file = model_file
        self.config_file = config_file
        self.model_available = True
        
        # Try to load the config to get the sample rate
        try:
            with open(config_file, 'r') as f:
                model_config = json.load(f)
                self.sample_rate = model_config.get('audio', {}).get('sample_rate', self.sample_rate)
                self.logger.info(f"Using sample rate from model config: {self.sample_rate}")
        except Exception as e:
            self.logger.warning(f"Could not load model config: {e}")
        
        self.logger.info(f"Using TTS model at {self.model_file}")
    
    def start(self):
        """Start the text-to-speech processor."""
        if self.is_running:
            self.logger.warning("TTS processor is already running")
            return False
        
        if not self.model_available:
            self.logger.error("Cannot start TTS: Model not available")
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        self.logger.info("TTS processor started")
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
        
        self.logger.info("TTS processor stopped")
    
    def say(self, text: str):
        """
        Convert text to speech asynchronously.
        
        Args:
            text: The text to convert to speech.
        """
        if not self.is_running:
            self.logger.warning("TTS processor is not running")
            return
        
        # Add text to the queue
        self.text_queue.append(text)
        
        self.logger.debug(f"Added text to TTS queue: '{text}'")
    
    def say_sync(self, text: str) -> bool:
        """
        Convert text to speech synchronously.
        
        Args:
            text: The text to convert to speech.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.model_available:
            self.logger.error("Cannot synthesize speech: Model not available")
            return False
        
        self.logger.info(f"Synthesizing speech: '{text}'")
        
        try:
            # Generate speech
            audio_data = self._synthesize_speech(text)
            if audio_data is None:
                return False
            
            # Play the audio
            self._play_audio(audio_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error synthesizing speech: {e}")
            return False
    
    def is_busy(self) -> bool:
        """Check if the TTS processor is currently speaking or has queued text."""
        return self.is_speaking or len(self.text_queue) > 0
    
    def clear_queue(self):
        """Clear the text queue."""
        self.text_queue = []
        self.logger.info("TTS queue cleared")
    
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
            self.logger.error(f"Error in process loop: {e}")
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
                    self.logger.error("Piper executable not found in PATH. Please install piper-tts")
                    return None
                
            self.logger.debug(f"Using Piper executable: {piper_executable}")
                
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
            self.logger.debug(f"Running command: {' '.join(cmd)}")
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
                self.logger.error(f"Piper TTS failed with return code {process.returncode}: {stderr}")
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
            self.logger.error(f"Error synthesizing speech: {e}")
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
            self.logger.error(f"Error playing audio: {e}")
            
    def set_voice(self, voice: str):
        """
        Set the voice to use for speech synthesis.
        
        Args:
            voice: The voice to use.
        """
        self.voice = voice
        self.logger.info(f"TTS voice set to {voice}")
    
    def set_speaking_rate(self, speaking_rate: float):
        """
        Set the speaking rate.
        
        Args:
            speaking_rate: Speaking rate multiplier (1.0 = normal speed).
        """
        self.speaking_rate = max(0.5, min(2.0, speaking_rate))  # Limit range
        self.logger.info(f"TTS speaking rate set to {self.speaking_rate}") 