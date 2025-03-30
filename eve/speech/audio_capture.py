import logging
import queue
import threading
import sounddevice as sd
import numpy as np

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

        logger.info(f"AudioCapture configured: Rate={self.sample_rate}, Channels={self.channels}, Device={self.device_index or 'Default'}")

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

    def stop_recording(self):
        """Stop recording audio."""
        with self._lock:
            if not self.is_recording or self.stream is None:
                logger.warning("Audio recording is not active or stream is None.")
                self.is_recording = False # Ensure state is correct
                return
                
            try:
                logger.info("Stopping audio stream...")
                if self.stream.active:
                     self.stream.stop()
                self.stream.close()
                self.is_recording = False
                # Clear the queue after stopping?
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                logger.info("Audio stream stopped and queue cleared.")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}", exc_info=True)
                # Try to force state even on error
                self.is_recording = False 
            finally:
                self.stream = None # Ensure stream object is cleared
        
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
         """Ensure the stream is stopped and closed."""
         self.stop_recording()

# Also update the __init__.py file if needed 