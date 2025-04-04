import os
import logging
import numpy as np
import threading
import queue # Use queue instead of list for thread safety
import time
import sounddevice as sd
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from pathlib import Path
import subprocess
import shutil
import tempfile
import wave

# Attempt to import pyttsx3, but don't fail if it's not there
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    logging.warning("pyttsx3 library not found. pyttsx3 TTS engine will be unavailable.")

from eve.config import SpeechConfig # Assuming SpeechConfig is available

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Handles text-to-speech synthesis using different engines."""

    def __init__(self, config: SpeechConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.engine_type = getattr(config, 'tts_engine', 'pyttsx3').lower()
        self.voice = getattr(config, 'tts_voice', None)
        self.speaking_rate = getattr(config, 'tts_rate', 1.0) # Rate multiplier
        self.volume = getattr(config, 'tts_volume', 1.0)
        self.pitch = getattr(config, 'tts_pitch', 1.0) # Used by pyttsx3

        # Piper specific settings (if used)
        self.model_file = getattr(config, 'coqui_model_path', None) # Re-use coqui path for piper model
        self.sample_rate = getattr(config, 'sample_rate', 16000) # Default SR for playback
        self.device_index = getattr(config, 'audio_device_index', None) # Output device

        self.engine: Any = None
        self.model_available = False # Flag for engines requiring models (like Piper)
        self.is_running = False
        self.is_speaking = False
        self.text_queue = queue.Queue() # Thread-safe queue
        self.stop_event = threading.Event()
        self.process_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        self.current_text: Optional[str] = None
        self.callback: Optional[Callable[[str], None]] = None # Callback after speech finishes

        self._init_engine()

    def _init_engine(self):
        """Initialize the selected TTS engine."""
        self.logger.info(f"Initializing TTS engine: {self.engine_type}")
        try:
            if self.engine_type == 'pyttsx3':
                if pyttsx3:
                    self.engine = pyttsx3.init()
                    # Set properties
                    self.engine.setProperty('rate', self.speaking_rate * 150) # Adjust base rate
                    self.engine.setProperty('volume', self.volume)
                    # Pitch might not be universally supported
                    try: self.engine.setProperty('pitch', self.pitch)
                    except: pass
                    # Voice selection (can be complex)
                    if self.voice:
                        available_voices = self.engine.getProperty('voices')
                        selected_voice_obj = None
                        for v in available_voices:
                            # Attempt matching by name or ID fragments
                            if self.voice.lower() in v.name.lower() or self.voice == v.id:
                                selected_voice_obj = v
                                break
                        if selected_voice_obj:
                             self.engine.setProperty('voice', selected_voice_obj.id)
                             self.logger.info(f"pyttsx3 voice set to: {selected_voice_obj.name}")
                        else:
                             self.logger.warning(f"pyttsx3 voice '{self.voice}' not found. Using default.")
                    self.model_available = True # pyttsx3 doesn't need external models
                    self.logger.info("pyttsx3 engine initialized successfully.")
                else:
                    self.logger.error("pyttsx3 library not found, cannot initialize engine.")
                    self.engine_type = 'mock' # Fallback

            elif self.engine_type == 'espeak' or self.engine_type == 'espeak-ng':
                # Check if espeak command exists
                try:
                    espeak_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
                    if espeak_cmd:
                         result = subprocess.run([espeak_cmd, '--version'], capture_output=True, text=True, timeout=2)
                         if result.returncode == 0:
                              self.logger.info(f"Found espeak/espeak-ng executable: {espeak_cmd}")
                              self.engine = espeak_cmd # Store the command path
                              self.model_available = True # espeak is self-contained
                         else:
                              self.logger.warning(f"espeak check failed (RC={result.returncode}). Falling back.")
                              self.engine_type = 'mock' # Fallback
                    else:
                         self.logger.warning("espeak/espeak-ng command not found in PATH. Falling back.")
                         self.engine_type = 'mock' # Fallback
                except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                    self.logger.warning(f"Error checking for espeak: {e}. Falling back.")
                    self.engine_type = 'mock' # Fallback

            elif self.engine_type == 'piper':
                 piper_executable = shutil.which("piper") or shutil.which("piper.exe")
                 if not piper_executable:
                      self.logger.error("Piper executable not found in PATH. Please install piper-tts.")
                      self.engine_type = 'mock'
                 elif not self.model_file or not Path(self.model_file).is_file():
                      self.logger.error(f"Piper model file not found or not specified: {self.model_file}")
                      self.engine_type = 'mock'
                 else:
                      self.engine = piper_executable # Store path to executable
                      self.model_available = True
                      self.logger.info(f"Piper TTS configured with model: {self.model_file}")

            # Add other engine initializations here (google, coqui) if needed

            else: # If engine_type is unknown or explicitly 'mock'
                 self.logger.info(f"TTS engine type '{self.engine_type}' not recognized or supported, using mock engine.")
                 self.engine_type = 'mock'

            # Final check for mock engine setting
            if self.engine_type == 'mock':
                self.engine = 'mock' # Use string 'mock' as placeholder
                self.model_available = True # Mock is always available
                self.logger.info("Initialized mock text-to-speech engine")

        except Exception as e:
            self.logger.error(f"Error during TTS engine initialization: {e}. Defaulting to mock.", exc_info=True)
            self.engine_type = 'mock'
            self.engine = 'mock'
            self.model_available = True

    def speak(self, text: str):
        """Add text to the queue to be spoken asynchronously."""
        if not self.is_running:
            self.logger.warning("TTS processor is not running, cannot queue text.")
            return
        if not text:
            return
        self.text_queue.put(text)
        self.logger.debug(f"Added text to TTS queue: '{text[:50]}...'")

    # Renamed say_sync to speak_sync for consistency
    def speak_sync(self, text: str) -> bool:
        """
        Convert text to speech synchronously. Blocks until speech is finished.

        Args:
            text: The text to convert to speech.

        Returns:
            True if successful, False otherwise.
        """
        if not self.model_available and self.engine_type != 'mock':
            self.logger.error(f"Cannot synthesize speech: Model/Engine '{self.engine_type}' not available")
            return False
        if not text:
            return True # Nothing to say

        self.logger.info(f"Synthesizing speech synchronously: '{text[:50]}...'")

        # Directly call synthesis and playback based on engine
        try:
            if self.engine_type == 'pyttsx3' and self.engine:
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            elif self.engine_type in ('espeak', 'espeak-ng') and self.engine:
                subprocess.run([self.engine, text], check=True) # Use stored command
                return True
            elif self.engine_type == 'piper' and self.engine and self.model_available:
                audio_data = self._synthesize_speech_piper(text)
                if audio_data is not None:
                    self._play_audio(audio_data)
                    return True
                else:
                    return False # Synthesis failed
            elif self.engine_type == 'mock':
                self.logger.info(f"[Sync] Mock TTS would say: {text}")
                # Simulate speech duration
                time.sleep(max(0.5, len(text) / 15.0)) # Rough estimate
                return True
            else:
                self.logger.error(f"Cannot speak_sync: Unsupported engine type '{self.engine_type}' or engine not initialized.")
                return False
        except Exception as e:
            self.logger.error(f"Error during synchronous speech for '{self.engine_type}': {e}", exc_info=True)
            return False

    def play_startup_sound(self):
        """Play a startup sound or message (uses sync method)."""
        self.speak_sync("System initialized and ready")

    def stop(self):
        """Stop the TTS processor thread and any current speech."""
        self.logger.info("Stopping TTS processor...")
        self.stop_event.set() # Signal threads to stop

        # Clear the queue to prevent processing more items
        while not self.text_queue.empty():
            try: self.text_queue.get_nowait()
            except queue.Empty: break

        # Stop pyttsx3 engine if running
        if self.engine_type == 'pyttsx3' and self.engine:
            try: self.engine.stop()
            except Exception as e: self.logger.warning(f"Error stopping pyttsx3 engine: {e}")

        # Stop sounddevice playback
        try: sd.stop()
        except Exception as e: self.logger.warning(f"Error stopping sounddevice: {e}")

        # Join threads
        if self.process_thread and self.process_thread.is_alive():
            self.logger.debug("Joining TTS process thread...")
            self.process_thread.join(timeout=2.0)
        if self.audio_thread and self.audio_thread.is_alive():
             self.logger.debug("Joining TTS audio thread...")
             self.audio_thread.join(timeout=2.0)

        self.is_running = False
        self.is_speaking = False
        self.logger.info("TTS processor stopped.")

    def start(self):
        """Start the text-to-speech processor thread for async speech."""
        if self.is_running:
            logger.warning("TTS processor is already running")
            return True

        if not self.model_available and self.engine_type != 'mock':
            logger.error(f"Cannot start TTS async processor: Model/Engine '{self.engine_type}' not available")
            return False

        self.is_running = True
        self.stop_event.clear()

        # Start the processing thread as non-daemon
        self.process_thread = threading.Thread(target=self._process_loop, daemon=False)
        self.process_thread.start()

        logger.info("TTS processor thread started")
        return True

    def is_busy(self) -> bool:
        """Check if the TTS processor is currently speaking or has queued text."""
        return self.is_speaking or not self.text_queue.empty()

    def clear_queue(self):
        """Clear the text queue."""
        with self.text_queue.mutex:
             self.text_queue.queue.clear()
        logger.info("TTS queue cleared")

    def set_callback(self, callback: Optional[Callable[[str], None]]):
        """Set a callback function to be called when speech finishes."""
        self.callback = callback

    def _process_loop(self):
        """Process text from the queue in a separate thread."""
        self.logger.debug("TTS process loop started.")
        while self.is_running and not self.stop_event.is_set():
            try:
                # Wait for text with a timeout to allow checking stop_event
                text = self.text_queue.get(timeout=0.2)
                self.current_text = text
                self.logger.debug(f"Processing TTS for: '{text[:50]}...'")

                # Synthesize and play speech based on engine type
                self.is_speaking = True
                success = False
                try:
                    if self.engine_type == 'pyttsx3' and self.engine:
                        self.engine.say(text)
                        self.engine.runAndWait()
                        success = True
                    elif self.engine_type in ('espeak', 'espeak-ng') and self.engine:
                        subprocess.run([self.engine, text], check=True)
                        success = True
                    elif self.engine_type == 'piper' and self.engine and self.model_available:
                        audio_data = self._synthesize_speech_piper(text)
                        if audio_data is not None:
                            # Play audio in a separate thread to allow stopping
                            self.audio_thread = threading.Thread(
                                target=self._play_audio,
                                args=(audio_data,),
                                daemon=True
                            )
                            self.audio_thread.start()
                            # Wait for playback, checking stop_event
                            while self.audio_thread.is_alive() and not self.stop_event.is_set():
                                self.audio_thread.join(timeout=0.1)
                            success = True # Assume success if playback started
                        else:
                            self.logger.error("Piper synthesis failed in process loop.")
                    elif self.engine_type == 'mock':
                        self.logger.info(f"[Async] Mock TTS would say: {text}")
                        # Simulate speech duration, checking stop_event
                        sleep_time = max(0.5, len(text) / 15.0)
                        start_time = time.time()
                        while time.time() - start_time < sleep_time and not self.stop_event.is_set():
                            time.sleep(0.1)
                        success = True
                    else:
                        self.logger.error(f"Unsupported engine type '{self.engine_type}' in process loop.")

                except Exception as e:
                    self.logger.error(f"Error during speech synthesis/playback in process loop: {e}", exc_info=True)
                finally:
                    self.is_speaking = False
                    self.current_text = None
                    self.text_queue.task_done()

                    # Call callback if speech completed (even if playback was interrupted by stop)
                    if success and self.callback:
                         try: self.callback(text)
                         except Exception as cb_err: self.logger.error(f"Error in TTS completion callback: {cb_err}")

            except queue.Empty:
                # Queue is empty, loop continues to check stop_event
                continue
            except Exception as e:
                # Unexpected error in the loop itself
                logger.error(f"Error in TTS process loop: {e}", exc_info=True)
                time.sleep(0.5) # Prevent tight loop on error

        logger.debug("TTS process loop finished.")
        self.is_running = False # Ensure flag is set if loop exits
        self.is_speaking = False

    # Renamed from _synthesize_speech to be specific
    def _synthesize_speech_piper(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize speech using Piper TTS executable.
        """
        if not self.engine or not self.model_file:
            self.logger.error("Piper engine or model not configured correctly.")
            return None

        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name

            # Command arguments
            cmd = [
                self.engine, # Path to piper executable
                "--model", str(self.model_file),
                "--output_file", output_path,
                "--sentence_silence", "0.2" # Small silence between sentences
            ]

            # Add speaking rate if not 1.0 (Piper uses length_scale)
            # length_scale < 1.0 = faster, > 1.0 = slower
            if self.speaking_rate != 1.0:
                length_scale = 1.0 / max(0.5, min(2.0, self.speaking_rate)) # Inverse, clamped
                cmd.extend(["--length_scale", str(length_scale)])

            # Run piper, feeding text via stdin
            self.logger.debug(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, # Use text mode for stdin/stderr
                encoding='utf-8' # Ensure correct encoding
            )

            stdout, stderr = process.communicate(text)

            if process.returncode != 0:
                self.logger.error(f"Piper TTS failed (RC {process.returncode}): {stderr.strip()}")
                if os.path.exists(output_path): os.unlink(output_path)
                return None

            # Read the generated WAV file
            try:
                audio_data = self._read_wav(output_path)
                return audio_data
            finally:
                if os.path.exists(output_path):
                    try: os.unlink(output_path)
                    except OSError as unlink_err: self.logger.warning(f"Could not delete temp WAV file {output_path}: {unlink_err}")

        except Exception as e:
            self.logger.error(f"Error synthesizing speech with Piper: {e}", exc_info=True)
            return None

    def _read_wav(self, file_path: str) -> Optional[np.ndarray]:
        """
        Read WAV file and return audio data as float32 numpy array.
        Checks sample rate against configured rate.
        """
        try:
            with wave.open(file_path, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                file_sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

                if file_sample_rate != self.sample_rate:
                    self.logger.warning(f"WAV file sample rate ({file_sample_rate}) differs from configured rate ({self.sample_rate}). Playback might be incorrect.")
                    # Basic resampling could be added here if necessary, but adds complexity/dependencies

                # Determine numpy dtype based on sample width
                if sample_width == 1: dtype = np.uint8
                elif sample_width == 2: dtype = np.int16
                elif sample_width == 4: dtype = np.int32
                else:
                    self.logger.error(f"Unsupported WAV sample width: {sample_width}")
                    return None

                audio_data = np.frombuffer(frames, dtype=dtype)

                # Handle multiple channels - Select first channel if mono needed
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)[:, 0] # Take first channel

                # Convert to float32 normalized between -1 and 1
                if dtype != np.float32:
                    max_val = np.iinfo(dtype).max if dtype != np.uint8 else 255.0
                    offset = 0 if dtype != np.uint8 else 128.0 # Center uint8
                    audio_data = (audio_data.astype(np.float32) - offset) / max_val

                return audio_data
        except wave.Error as e:
            self.logger.error(f"Error reading WAV file {file_path}: {e}")
            return None
        except Exception as e:
             self.logger.error(f"Unexpected error processing WAV file {file_path}: {e}", exc_info=True)
             return None

    def _play_audio(self, audio_data: np.ndarray):
        """
        Play audio data using sounddevice, checking stop_event.
        """
        if audio_data is None or audio_data.size == 0:
             self.logger.warning("Attempted to play empty audio data.")
             return
        try:
            self.logger.debug(f"Playing audio data ({audio_data.shape}, SR={self.sample_rate}) on device {self.device_index or 'default'}")
            sd.play(audio_data, samplerate=self.sample_rate, device=self.device_index)

            # Wait for playback to finish, checking stop_event
            # sd.wait() blocks, so use a loop with sleep
            stream = sd.get_stream()
            while stream.active and not self.stop_event.is_set():
                time.sleep(0.05) # Sleep briefly

            # If stopped early, ensure playback halts
            if self.stop_event.is_set():
                self.logger.debug("Playback interrupted by stop event.")
                sd.stop()
                stream.close() # Ensure stream resources are released
            else:
                 self.logger.debug("Finished playing audio.")

        except sd.PortAudioError as pae:
             self.logger.error(f"PortAudio error during playback: {pae}")
             # Log available devices on error?
             # self.logger.info(f"Available audio devices: {sd.query_devices()}")
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}", exc_info=True)
            # Ensure stop on any error during playback
            try: sd.stop()
            except: pass

    # Methods to modify settings (optional)
    def set_voice(self, voice: str):
        """Set the voice (engine-specific). May require re-initialization for some engines."""
        self.voice = voice
        self.logger.info(f"TTS voice set to '{voice}'. Restart may be needed for some engines.")
        # Simple engines like pyttsx3 might allow dynamic change
        if self.engine_type == 'pyttsx3' and self.engine:
            try: self.engine.setProperty('voice', voice) # May fail if voice ID invalid
            except: self.logger.warning("Failed to dynamically set pyttsx3 voice.")

    def set_speaking_rate(self, speaking_rate: float):
        """Set the speaking rate multiplier (0.5 to 2.0). Restart may be needed."""
        self.speaking_rate = max(0.5, min(2.0, speaking_rate)) # Clamp rate
        self.logger.info(f"TTS speaking rate set to {self.speaking_rate:.2f}. Restart may be needed.")
        # Simple engines like pyttsx3 might allow dynamic change
        if self.engine_type == 'pyttsx3' and self.engine:
             try: self.engine.setProperty('rate', self.speaking_rate * 150)
             except: self.logger.warning("Failed to dynamically set pyttsx3 rate.") 