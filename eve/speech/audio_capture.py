import logging
import queue
import threading
import sounddevice as sd
import numpy as np
import pvporcupine
from typing import Optional, List, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import if SpeechConfig is complex
if TYPE_CHECKING:
    from eve.config import SpeechConfig # Assuming SpeechConfig is defined here

logger = logging.getLogger(__name__)

class AudioCapture:
    """Audio capture module for EVE using sounddevice."""
    
    def __init__(self, speech_config: 'SpeechConfig'):
        logger.info("Initializing audio capture")
        self.speech_config = speech_config
        self.sample_rate = getattr(self.speech_config, 'sample_rate', 16000)
        self.channels = getattr(self.speech_config, 'channels', 1)
        self.device_index = getattr(self.speech_config, 'audio_device_index', None)

        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self._init_porcupine() # Initializes Porcupine if enabled

        # Set chunk size based on Porcupine or config
        if self.porcupine:
             self.chunk_size = self.porcupine.frame_length
             logger.info(f"Using Porcupine frame length as chunk size: {self.chunk_size}")
        else:
             # Use chunk_size from config, fall back to 1024
             self.chunk_size = getattr(self.speech_config, 'chunk_size', 1024)
             logger.info(f"Using chunk size: {self.chunk_size}")

        self.audio_queue = queue.Queue(maxsize=100) # Add a maxsize to prevent unbounded growth
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False
        self._stop_event = threading.Event() # Use an event for signaling stop
        self._lock = threading.Lock() # General purpose lock
        self.last_rms: float = 0.0

        logger.info(f"AudioCapture configured: Rate={self.sample_rate}, Channels={self.channels}, Chunk={self.chunk_size}, Device={self.device_index or 'Default'}")

    def _init_porcupine(self, sensitivities: Optional[List[float]] = None): # Accept optional sensitivities
        """Initializes the Porcupine wake word engine if enabled in config."""
        if not getattr(self.speech_config, 'wake_word_enabled', False):
             logger.info("Porcupine wake word detection disabled in config.")
             self.porcupine = None
             return

        # --- Delete existing instance if any ---
        if self.porcupine:
             try:
                  self.porcupine.delete()
                  logger.info("Deleted existing Porcupine instance before re-initialization.")
                  self.porcupine = None
             except Exception as e:
                  logger.error(f"Error deleting existing Porcupine instance: {e}", exc_info=True)

        logger.info(f"Initializing Porcupine with sensitivities: {sensitivities or 'default'} ...")
        try:
            # Use the dedicated field for the access key
            access_key = getattr(self.speech_config, 'picovoice_access_key', None)
            if not access_key:
                raise ValueError("picovoice_access_key not found in config.")

            # Keyword setup
            keyword_paths = getattr(self.speech_config, 'wake_word_model_path', None)
            keywords = None
            num_keywords = 0
            if keyword_paths:
                 if isinstance(keyword_paths, str): keyword_paths = [keyword_paths]
                 num_keywords = len(keyword_paths)
                 logger.info(f"Using Porcupine custom keyword paths: {keyword_paths}")
            else:
                 # Use wake_word_phrase from config as built-in keyword(s)
                 wake_phrases = getattr(self.speech_config, 'wake_word_phrase', 'porcupine')
                 if isinstance(wake_phrases, str): keywords = [wake_phrases.lower()] # Use lowercase standard
                 elif isinstance(wake_phrases, list): keywords = [str(p).lower() for p in wake_phrases]
                 else: keywords = ['porcupine'] # Fallback
                 num_keywords = len(keywords)
                 logger.info(f"Using Porcupine built-in keywords: {keywords}")

            # --- Validate Sensitivities ---
            final_sensitivities = sensitivities # Use passed-in value if provided
            if final_sensitivities is None:
                # Fallback to config value if not passed in
                final_sensitivities = getattr(self.speech_config, 'wake_word_sensitivity', None)

            # Ensure it's a list of the correct length
            if isinstance(final_sensitivities, float):
                 final_sensitivities = [final_sensitivities] * num_keywords
            elif not isinstance(final_sensitivities, list):
                 logger.warning(f"Sensitivities are not a list or float ({final_sensitivities}). Resetting to defaults.")
                 final_sensitivities = [0.5] * num_keywords
            elif len(final_sensitivities) != num_keywords:
                 logger.warning(f"Number of sensitivities ({len(final_sensitivities)}) != number of keywords ({num_keywords}). Adjusting...")
                 # Adjust list length: truncate or pad with last value
                 if len(final_sensitivities) > num_keywords:
                     final_sensitivities = final_sensitivities[:num_keywords]
                 else:
                     padding = [final_sensitivities[-1]] * (num_keywords - len(final_sensitivities))
                     final_sensitivities.extend(padding)

            # Clamp values and store
            self.current_sensitivities = [max(0.0, min(1.0, s)) for s in final_sensitivities]
            logger.info(f"Using final sensitivities: {self.current_sensitivities}")

            # Picovoice paths
            library_path = getattr(self.speech_config, 'porcupine_library_path', None) # Check config source name
            model_path = getattr(self.speech_config, 'porcupine_model_path', None) # Check config source name

            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=keyword_paths,
                keywords=keywords,
                sensitivities=self.current_sensitivities,
                library_path=library_path,
                model_path=model_path
            )
            logger.info(f"Porcupine initialized. Frame length: {self.porcupine.frame_length}, Sample rate: {self.porcupine.sample_rate}, Version: {self.porcupine.version}")

            # Verify expected sample rate
            if self.porcupine.sample_rate != self.sample_rate:
                 logger.warning(f"Porcupine expected sample rate {self.porcupine.sample_rate} differs from configured rate {self.sample_rate}. Audio may be processed incorrectly!")

        except pvporcupine.PorcupineError as pe:
             logger.error(f"Porcupine initialization error: {pe}", exc_info=True)
             self.porcupine = None
        except Exception as e:
             logger.error(f"Unexpected error initializing Porcupine: {e}", exc_info=True)
             self.porcupine = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """Callback function for sounddevice InputStream."""
        if status:
            logger.warning(f"Audio Callback Status: {status}")

        if self._stop_event.is_set() or not self.is_recording:
            return

        # --- RMS Calculation ---
        try:
            if indata.dtype == np.int16:
                 float_data = indata.astype(np.float32) / 32768.0
            elif indata.dtype == np.float32:
                 float_data = indata
            else:
                 logger.warning(f"Unexpected audio dtype: {indata.dtype}. Skipping RMS calculation.")
                 float_data = None

            if float_data is not None:
                rms = np.sqrt(np.mean(float_data**2))
                with self._lock:
                    self.last_rms = float(rms)
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            with self._lock:
                 self.last_rms = 0.0

        # --- Put data in queue ---
        try:
            if indata.dtype != np.int16:
                 logger.warning(f"Unexpected audio dtype {indata.dtype} in callback, expected int16.")
                 return
            audio_bytes = indata.tobytes()
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
             logger.warning("Audio queue is full. Discarding audio data.")
        except Exception as e:
             logger.error(f"Error putting data into audio queue: {e}")

    def get_last_rms(self) -> float:
         """Get the last calculated RMS value thread-safely."""
         with self._lock:
             return self.last_rms

    def start_recording(self):
        """Start the audio stream."""
        with self._lock:
            if self.is_recording:
                logger.warning("Audio recording is already active.")
                return
            if self.stream is not None:
                 logger.warning("Stream object exists before start. Attempting cleanup.")
                 self._close_stream_internal()
            try:
                logger.info("Starting audio stream...")
                self._stop_event.clear()
                self.is_recording = True
                while not self.audio_queue.empty():
                     try: self.audio_queue.get_nowait()
                     except queue.Empty: break
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=self.device_index,
                    dtype='int16',
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                    finished_callback=self._stream_finished_callback
                )
                self.stream.start()
                logger.info("Audio stream started successfully.")
            except sd.PortAudioError as pae:
                logger.error(f"PortAudioError starting audio stream: {pae}", exc_info=True)
                self._handle_stream_start_error()
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}", exc_info=True)
                self._handle_stream_start_error()

    def _handle_stream_start_error(self):
        """Helper to clean up state after a stream start fails."""
        with self._lock:
            if self.stream:
                 try:
                     if not self.stream.closed:
                          self.stream.close(ignore_errors=True)
                 except Exception: pass
            self.stream = None
            self.is_recording = False
            self._stop_event.set()

    def _stream_finished_callback(self):
        """Callback executed when the stream naturally finishes or is stopped/aborted."""
        log_func = logger.info if self._stop_event.is_set() else logger.warning
        log_func(f"Audio stream finished callback executed ({'expected' if self._stop_event.is_set() else 'unexpected'}).")
        self.is_recording = False

    def stop_stream(self):
        """Stops the audio stream from calling the callback. Does NOT close."""
        logger.info("Stopping audio stream callback...")
        self.is_recording = False
        self._stop_event.set()
        stream_to_stop = None
        with self._lock:
             stream_to_stop = self.stream
        if stream_to_stop and stream_to_stop.active:
            try:
                stream_to_stop.stop(ignore_errors=True)
                logger.debug("Sounddevice stream stop requested.")
            except Exception as e:
                logger.error(f"Error requesting sounddevice stream stop: {e}", exc_info=True)
        elif stream_to_stop:
             logger.debug("Stream was already inactive when stop_stream was called.")
        else:
             logger.debug("No stream object found to stop.")

    def _close_stream_internal(self):
        """Internal helper to close the stream, assumes lock might be held or called safely."""
        stream_to_close = self.stream
        self.stream = None
        if stream_to_close:
            try:
                if stream_to_close.active:
                    logger.warning("Closing stream that was still active. Stopping first.")
                    stream_to_close.stop(ignore_errors=True)
                if not stream_to_close.closed:
                    stream_to_close.close(ignore_errors=True)
                    logger.debug("Sounddevice stream closed.")
                else:
                    logger.debug("Stream was already closed.")
            except Exception as e:
                logger.error(f"Error closing sounddevice stream: {e}", exc_info=True)

    def close_stream(self):
        """Closes the audio stream and releases resources."""
        logger.info("Closing audio stream...")
        with self._lock:
             self._close_stream_internal()
        logger.debug("Clearing audio queue...")
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
        logger.debug("Audio queue cleared.")

    def stop(self): # Renamed from stop_recording
        """Stops and closes the audio stream and cleans up Porcupine."""
        logger.info("Stopping and cleaning up AudioCapture...")
        self.stop_stream()
        self.close_stream()
        porcupine_to_delete = None
        with self._lock:
            porcupine_to_delete = self.porcupine
            self.porcupine = None
        if porcupine_to_delete:
            try:
                logger.debug("Deleting Porcupine instance...")
                porcupine_to_delete.delete()
                logger.info("Porcupine instance deleted.")
            except Exception as e:
                logger.error(f"Error deleting Porcupine instance during cleanup: {e}")
        logger.info("AudioCapture cleanup finished.")

    def has_new_audio(self) -> bool:
        """Check if new audio data is available in the queue."""
        return not self.audio_queue.empty()

    def get_audio_data(self, max_duration_ms: Optional[int] = None) -> bytes:
        """Get audio data from the queue, optionally up to a max duration."""
        if not self.has_new_audio():
            return b''
        data = bytearray()
        bytes_per_ms = (self.sample_rate * self.channels * 2) // 1000
        max_bytes = (max_duration_ms * bytes_per_ms) if max_duration_ms else float('inf')
        while len(data) < max_bytes:
            try:
                chunk = self.audio_queue.get_nowait()
                data.extend(chunk)
            except queue.Empty:
                break
            except Exception as e:
                 logger.error(f"Error getting data from audio queue: {e}")
                 break
        return bytes(data)

    # ... (get/set porcupine sensitivity methods if needed) ...

# Also update the __init__.py file if needed 