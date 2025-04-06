import logging
import queue
import threading
import sounddevice as sd
import numpy as np
# Remove pvporcupine import
# import pvporcupine
# Add OpenWakeWord import
from openwakeword.model import Model as OpenWakeWordModel
from typing import Optional, List, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import if SpeechConfig is complex
if TYPE_CHECKING:
    from eve.config import SpeechConfig # Assuming SpeechConfig is defined here

logger = logging.getLogger(__name__)

class AudioCapture:
    """Audio capture module using sounddevice and OpenWakeWord."""
    
    def __init__(self, speech_config: 'SpeechConfig'):
        logger.info("Initializing audio capture with OpenWakeWord")
        self.speech_config = speech_config
        self.sample_rate = getattr(self.speech_config, 'sample_rate', 16000)
        self.channels = getattr(self.speech_config, 'channels', 1)
        self.device_index = getattr(self.speech_config, 'audio_device_index', None)
        self.wake_word_enabled = getattr(self.speech_config, 'wake_word_enabled', True)
        self.wake_word_model_type = getattr(self.speech_config, 'wake_word_model', 'openwakeword').lower()

        # OpenWakeWord specific settings
        self.oww_model: Optional[OpenWakeWordModel] = None
        self.oww_inference_threshold = getattr(self.speech_config, 'openwakeword_inference_threshold', 0.7)
        # OWW works on 16kHz mono audio, chunk size must be multiple of frame samples (e.g., 1280 for 80ms)
        # Get frame_samples from model if available, otherwise use default (e.g., 1280)
        # We initialize OWW first to get frame_samples if possible
        self._init_openwakeword()
        oww_frame_samples = self.oww_model.model_definition['ww_model_definition']['frame_samples'] if self.oww_model else 1280
        # Ensure chunk size is a multiple of OWW frame samples for optimal processing
        self.chunk_size = oww_frame_samples * 2 # Process 160ms chunks (adjust multiplier as needed)

        self.audio_queue = queue.Queue(maxsize=100) # For main STT
        self.wake_word_queue = queue.Queue(maxsize=20) # Separate small queue for wake word detection callbacks
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.last_rms: float = 0.0

        logger.info(f"AudioCapture configured: Rate={self.sample_rate}, Channels={self.channels}, Chunk={self.chunk_size}, Device={self.device_index or 'Default'}, WW Enabled={self.wake_word_enabled}")

    def _init_openwakeword(self):
        """Initialize the OpenWakeWord model."""
        if not self.wake_word_enabled or self.wake_word_model_type != 'openwakeword':
            logger.info("OpenWakeWord disabled or not selected.")
            self.oww_model = None
            return

        try:
            logger.info("Initializing OpenWakeWord model...")
            # Models are downloaded automatically on first use to a cache directory
            # Specify custom models via `wakeword_models` argument if needed
            self.oww_model = OpenWakeWordModel(inference_framework='onnx') # Use ONNX runtime
            logger.info("OpenWakeWord model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing OpenWakeWord model: {e}", exc_info=True)
            self.oww_model = None

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
                 float_data = None
            if float_data is not None:
                rms = np.sqrt(np.mean(float_data**2))
                with self._lock:
                    self.last_rms = float(rms)
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}")
            with self._lock: self.last_rms = 0.0

        # Ensure audio is int16 for processing
        if indata.dtype != np.int16:
            logger.warning(f"Received non-int16 audio data ({indata.dtype}), attempting conversion.")
            try:
                # Assuming input range needs scaling if it was float
                if np.issubdtype(indata.dtype, np.floating):
                     indata = (indata * 32767).astype(np.int16)
                else:
                     indata = indata.astype(np.int16)
            except Exception as conv_err:
                 logger.error(f"Failed to convert audio to int16: {conv_err}")
                 return # Skip processing if conversion fails

        # --- Wake Word Detection (OpenWakeWord) --- 
        if self.oww_model and self.wake_word_enabled:
            try:
                # Predict using the int16 numpy array
                prediction = self.oww_model.predict(indata)
                # Check scores against threshold for each ww model
                # OWW returns scores per-model; we trigger if any exceed threshold
                for ww_name, score in prediction.items():
                    if score > self.oww_inference_threshold:
                        logger.info(f"OpenWakeWord DETECTED: '{ww_name}' (Score: {score:.2f}) Threshold: {self.oww_inference_threshold}")
                        # Put a simple signal or the name onto the wake word queue
                        try:
                            self.wake_word_queue.put_nowait(ww_name) 
                        except queue.Full:
                            logger.warning("Wake word queue full, detection missed.")
                        # Optional: Stop processing further wake words in this chunk?
                        break # Trigger on first detection in chunk
            except Exception as oww_err:
                logger.error(f"Error during OpenWakeWord prediction: {oww_err}", exc_info=False) # Reduce log spam

        # --- Put data in main STT queue --- 
        try:
            audio_bytes = indata.tobytes()
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
             logger.warning("Audio (STT) queue is full. Discarding audio data.")
        except Exception as e:
             logger.error(f"Error putting data into STT audio queue: {e}")

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
        """Stops and closes the audio stream and cleans up resources."""
        logger.info("Stopping and cleaning up AudioCapture...")
        self.stop_stream()
        self.close_stream()

        # --- Clean up OpenWakeWord --- 
        # OpenWakeWord models don't have an explicit delete/cleanup method in the same way
        # as Porcupine. Python's garbage collection should handle the model object.
        self.oww_model = None
        logger.info("OpenWakeWord model reference cleared.")
        # --------------------------

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

    # Add a method to check the wake word queue
    def check_for_wake_word(self) -> Optional[str]:
        """Checks the wake word queue non-blockingly."""
        try:
            return self.wake_word_queue.get_nowait()
        except queue.Empty:
            return None

# Also update the __init__.py file if needed 