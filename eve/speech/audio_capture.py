import logging
import queue
import threading
import sounddevice as sd
import numpy as np
import pvporcupine
from typing import Optional, List

logger = logging.getLogger(__name__)

class AudioCapture:
    """Audio capture module for EVE using sounddevice."""
    
    def __init__(self, speech_config):
        logger.info("Initializing audio capture")
        self.speech_config = speech_config
        self.sample_rate = getattr(self.speech_config, 'AUDIO_SAMPLE_RATE', 16000)
        self.channels = getattr(self.speech_config, 'AUDIO_CHANNELS', 1)
        self.device_index = getattr(self.speech_config, 'AUDIO_DEVICE_INDEX', None) # Allow specific device
        self.chunk_size = 1024 # Size of audio chunks to process (adjust as needed)
        
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_recording = False
        self._lock = threading.Lock()
        self.last_rms: float = 0.0 # Add attribute for RMS

        # --- Porcupine Wake Word Detection ---
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self._porcupine_buffer = bytearray()
        self._wake_word_detected_flag = False
        self.current_sensitivities: List[float] = [0.5] # Default if not in config
        self._init_porcupine() # Initializes using self.current_sensitivities
        # -------------------------------------

        logger.info(f"AudioCapture configured: Rate={self.sample_rate}, Channels={self.channels}, Device={self.device_index or 'Default'}")

    def _init_porcupine(self, sensitivities: Optional[List[float]] = None): # Accept optional sensitivities
        """Initializes the Porcupine wake word engine."""
        # If sensitivities are provided, update the instance variable
        if sensitivities is not None:
             self.current_sensitivities = sensitivities
             
        # --- Delete existing instance if any ---
        if self.porcupine:
             try:
                  self.porcupine.delete()
                  logger.info("Deleted existing Porcupine instance before re-initialization.")
                  self.porcupine = None
             except Exception as e:
                  logger.error(f"Error deleting existing Porcupine instance: {e}", exc_info=True)
                  # Attempt to continue, but Porcupine might be in a bad state
        # ---------------------------------------

        logger.info(f"Initializing Porcupine with sensitivities: {self.current_sensitivities} ...\n")
        try:
            access_key = getattr(self.speech_config, 'PICOVOICE_ACCESS_KEY', None)
            # ... (rest of access key check)

            # Keyword setup (paths or built-ins)
            keyword_paths = getattr(self.speech_config, 'PICOVOICE_KEYWORD_PATH', None)
            keywords = None
            num_keywords = 0
            if keyword_paths:
                 if isinstance(keyword_paths, str): keyword_paths = [keyword_paths]
                 num_keywords = len(keyword_paths)
                 logger.info(f"Using Porcupine custom keyword paths: {keyword_paths}")
            else:
                 keywords = getattr(self.speech_config, 'PICOVOICE_BUILTIN_KEYWORD', 'porcupine')
                 if isinstance(keywords, str): keywords = [keywords]
                 num_keywords = len(keywords)
                 logger.info(f"Using Porcupine built-in keywords: {keywords}")

            # --- Validate Sensitivities ---
            config_sensitivities = getattr(self.speech_config, 'PICOVOICE_SENSITIVITIES', None)
            if sensitivities is None and config_sensitivities is not None:
                 # Use config sensitivities on first init if provided
                  self.current_sensitivities = config_sensitivities
                  
            if not isinstance(self.current_sensitivities, list):
                 logger.warning(f"Sensitivities are not a list ({self.current_sensitivities}). Resetting to defaults for {num_keywords} keywords.")
                 self.current_sensitivities = [0.5] * num_keywords
            elif len(self.current_sensitivities) != num_keywords:
                 logger.warning(f"Number of sensitivities ({len(self.current_sensitivities)}) != number of keywords ({num_keywords}). Resetting sensitivities to defaults.")
                 self.current_sensitivities = [0.5] * num_keywords
            # Clamp values between 0.0 and 1.0
            self.current_sensitivities = [max(0.0, min(1.0, s)) for s in self.current_sensitivities]
            logger.info(f"Using final sensitivities: {self.current_sensitivities}")
            # -----------------------------

            library_path = getattr(self.speech_config, 'PICOVOICE_LIBRARY_PATH', None)
            model_path = getattr(self.speech_config, 'PICOVOICE_MODEL_PATH', None)

            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=keyword_paths,
                keywords=keywords,
                sensitivities=self.current_sensitivities, # Use the validated list
                library_path=library_path,
                model_path=model_path
            )

            # ... (rest of validation and logging)

        except pvporcupine.PorcupineError as pe:
             # ... (error handling)
             self.porcupine = None # Ensure it's None on failure
        except Exception as e:
             # ... (error handling)
             self.porcupine = None # Ensure it's None on failure

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            logger.warning(f"Audio Callback Status: {status}")
        
        # Calculate RMS for volume level
        # Ensure indata is float for RMS calculation if it isn't already (it should be float or int16 based on dtype)
        try:
            # Ensure input is numpy array for calculations
            # indata might already be numpy array depending on sounddevice version/config
            # If dtype='int16', convert to float for calculation
            if indata.dtype == np.int16:
                 float_data = indata.astype(np.float32) / 32768.0
            else:
                 float_data = indata # Assume already float
                 
            rms = np.sqrt(np.mean(float_data**2))
            # Update last_rms thread-safely
            with self._lock:
                self.last_rms = float(rms)
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            with self._lock:
                 self.last_rms = 0.0 # Reset on error

        # Add the raw audio data (int16 bytes) to the queue
        self.audio_queue.put(indata.tobytes())

    def get_last_rms(self) -> float:
         """Get the last calculated RMS value thread-safely."""
         with self._lock:
             return self.last_rms

    def start_recording(self):
        """Start recording audio using sounddevice InputStream."""
        with self._lock:
            if self.is_recording:
                logger.warning("Audio recording is already active.")
                return
            
            if self.stream is not None and self.stream.active:
                 logger.warning("Stream seems active before start. Stopping first.")
                 try:
                     self.stream.stop()
                     self.stream.close()
                 except Exception as e:
                     logger.error(f"Error stopping/closing existing stream: {e}")
                 self.stream = None
            
            try:
                logger.info("Starting audio stream...")
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=self.device_index,
                    dtype='int16', # Common format for speech recognition
                    blocksize=self.chunk_size, # Use defined chunk size
                    callback=self._audio_callback
                )
                self.stream.start()
                self.is_recording = True
                logger.info("Audio stream started successfully.")
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}", exc_info=True)
                self.stream = None
                self.is_recording = False

    def stop_stream(self):
        """Stops the audio stream from calling the callback."""
        logger.info("Stopping audio stream callback...")
        with self._lock:
            self.is_recording = False # Prevent adding to queue
        if self.stream and not self.stream.stopped:
            try:
                self.stream.stop()
                logger.debug("Sounddevice stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping sounddevice stream: {e}", exc_info=True)
                
    def close_stream(self):
        """Closes the audio stream and releases resources."""
        logger.info("Closing audio stream...")
        if self.stream:
            try:
                # Ensure stopped first, just in case
                if not self.stream.stopped:
                    self.stream.stop()
                    
                if not self.stream.closed:
                    self.stream.close()
                    logger.debug("Sounddevice stream closed.")
                self.stream = None
                logger.info("Audio stream closed successfully.")
            except Exception as e:
                logger.error(f"Error closing sounddevice stream: {e}", exc_info=True)
                self.stream = None # Ensure stream is None even on error
        # Clear queue on close
        with self._lock:
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except queue.Empty: break
            self.is_recording = False
            logger.debug("Audio queue cleared.")

    def stop_recording(self):
        """Stops and closes the audio stream. Use stop_stream() and close_stream() for finer control."""
        logger.warning("Using deprecated stop_recording(). Call stop_stream() and close_stream() instead.")
        self.stop_stream()
        self.close_stream()

    def has_new_audio(self) -> bool:
        """Check if new audio data is available in the queue."""
        return not self.audio_queue.empty()

    def get_audio_data(self) -> bytes:
        """Get all available audio data from the queue."""
        if not self.has_new_audio():
            return b''
        
        # Concatenate all available data chunks in the queue
        data = bytearray()
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait() # Non-blocking get
                data.extend(chunk)
            except queue.Empty:
                break # Should not happen with the initial check, but safe
            except Exception as e:
                 logger.error(f"Error getting data from audio queue: {e}")
                 break
        return bytes(data)

    def update(self):
        """Update audio capture state (placeholder - not needed with callback)."""
        # This method is no longer necessary as the callback handles data capture.
        pass

    def cleanup(self):
        logger.info("AudioCapture cleanup: Closing stream and deleting Porcupine...")
        self.close_stream()
        if self.porcupine:
            try:
                 self.porcupine.delete()
                 logger.info("Porcupine instance deleted.")
            except Exception as e:
                 logger.error(f"Error deleting Porcupine instance: {e}", exc_info=True)
            self.porcupine = None
        else:
             logger.debug("No Porcupine instance to delete during cleanup.")

    def get_porcupine_sensitivity(self) -> List[float]:
         """Returns the current list of Porcupine sensitivities."""
         # Return a copy to prevent external modification?
         return list(self.current_sensitivities) 

    def set_porcupine_sensitivity(self, new_sensitivities: List[float]):
         """Sets new Porcupine sensitivities and re-initializes the engine."""
         logger.info(f"Attempting to set new Porcupine sensitivities: {new_sensitivities}")
         # Re-initialize Porcupine with the new sensitivities
         # _init_porcupine handles validation, deletion, and creation
         self._init_porcupine(sensitivities=new_sensitivities)
         # Optionally: Check if self.porcupine is None after re-init and log/raise error

# Also update the __init__.py file if needed 