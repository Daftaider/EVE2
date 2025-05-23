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
        # Correctly get the samples per chunk required by the OWW model
        # Attempts to dynamically read chunk size failed. Use the default 1280 (80ms @ 16kHz).
        self.chunk_size = 1280
        logger.debug(f"Using fixed chunk size for OpenWakeWord: {self.chunk_size}")
        # Old attempts:
        # oww_samples_per_chunk = self.oww_model.preprocessor.window_size if self.oww_model and hasattr(self.oww_model, 'preprocessor') else 1280
        # oww_samples_per_chunk = self.oww_model.samples_per_chunk if self.oww_model else 1280
        # oww_frame_samples = self.oww_model.model_definition['ww_model_definition']['frame_samples'] if self.oww_model else 1280

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
            # --- Check available models first ---
            from openwakeword.utils import get_model_paths
            available_models = get_model_paths()
            logger.info(f"Available OpenWakeWord models: {available_models}")
            
            # Try to use a known working model
            target_model = "hey_jarvis_v0.1"  # More likely to be available
            if target_model not in available_models:
                # Fall back to any available model
                if available_models:
                    target_model = list(available_models.keys())[0]
                    logger.info(f"Using fallback model: {target_model}")
                else:
                    logger.warning("No OpenWakeWord models available, disabling wake word detection")
                    self.oww_model = None
                    return

            # --- Attempt model download ---
            from openwakeword.utils import download_models
            logger.info(f"Attempting download/cache check for model: {target_model}")
            try:
                download_models(model_names=[target_model])
                logger.info(f"Download/cache check for {target_model} completed.")
            except Exception as dl_err:
                logger.error(f"Error during model download/cache check: {dl_err}", exc_info=True)
                # Continue anyway, maybe the model exists despite the error

            # --- Initialize model ---
            logger.info(f"Initializing OpenWakeWord model ({target_model})...")
            self.oww_model = OpenWakeWordModel(
                inference_framework='onnx',
                wakeword_models=[target_model]
            )
            logger.info(f"OpenWakeWord model '{target_model}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing OpenWakeWord model: {e}", exc_info=True)
            self.oww_model = None  # Ensure model is None on failure

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """Callback function for sounddevice InputStream."""
        if status:
            logger.warning(f"Audio Callback Status: {status}")
        if self._stop_event.is_set() or not self.is_recording:
            return

        # --- RMS Calculation (using float32) ---
        float_data = None # Initialize float_data
        try:
            if indata.dtype == np.int16:
                 float_data = indata.astype(np.float32) / 32768.0
            elif indata.dtype == np.float32:
                 float_data = indata # Already float
            else:
                 logger.warning(f"Received unexpected audio data type ({indata.dtype}), attempting float conversion.")
                 float_data = indata.astype(np.float32) # Potential scaling needed depending on source

            rms = np.sqrt(np.mean(float_data**2))
            with self._lock:
                self.last_rms = float(rms)
        except Exception as e:
            logger.error(f"Error calculating RMS: {e}", exc_info=False)
            with self._lock: self.last_rms = 0.0
            # Continue processing audio even if RMS fails

        # --- Prepare int16 data for OWW and STT --- 
        int16_data = None
        try:
            if indata.dtype == np.int16:
                 int16_data = indata
            elif float_data is not None: # Use the float_data calculated above if available
                 # Scale float data assumed in range [-1.0, 1.0] to int16
                 int16_data = (float_data * 32767).astype(np.int16)
            else: # Fallback if input wasn't int16 and float conversion failed
                 logger.warning(f"Converting non-int16 audio data ({indata.dtype}) to int16 directly.")
                 int16_data = indata.astype(np.int16)

            # --- Handle different input shapes, ensuring 1D mono output --- 
            if int16_data.ndim == 1:
                 pass # Already 1D mono, proceed
            elif int16_data.ndim == 2:
                 if int16_data.shape[1] == 1 and self.channels == 1:
                     # Received (N, 1) shape for mono, squeeze to (N,)
                     int16_data = np.squeeze(int16_data, axis=1)
                 elif int16_data.shape[1] > 1 and self.channels == 1:
                     # Received (N, M>1) shape for mono, warn and take first channel
                     logger.warning(f"Received {int16_data.shape[1]} channels (shape: {int16_data.shape}) when 1 was requested. Taking first channel.")
                     int16_data = int16_data[:, 0]
                 else:
                      # Received 2D data, but multiple channels might be expected
                      # If self.channels > 1, this would require different handling (e.g., averaging)
                      # For now, assuming we only process first channel if multichannel detected unexpectedly
                      logger.warning(f"Received unexpected 2D audio data (shape: {int16_data.shape}), configured channels={self.channels}. Taking first channel.")
                      int16_data = int16_data[:, 0] 
            else:
                 # Handle unexpected dimensions (ndim > 2 or 0)
                 logger.error(f"Unexpected audio data dimensions: {int16_data.ndim}, expected 1 or 2. Skipping frame.")
                 return

            # --- VERIFY SHAPE (Now MUST be 1D) --- 
            if int16_data.ndim != 1:
                 logger.error(f"Audio data is not 1D after processing (shape: {int16_data.shape}). Skipping frame.")
                 return
            # Check number of samples
            if int16_data.shape[0] != self.chunk_size:
                 logger.error(f"Unexpected audio chunk size: {int16_data.shape[0]}, expected {self.chunk_size}. Skipping frame.")
                 return

        except Exception as conv_err:
             logger.error(f"Failed to prepare int16 audio data: {conv_err}")
             return # Skip processing if conversion/verification fails

        # --- Wake Word Detection (OpenWakeWord) --- 
        if self.oww_model and self.wake_word_enabled:
            try:
                # --- RESHAPE FOR MODEL --- 
                # Predict expects specific shape, often (batch_size, num_samples)
                # Reshape from (chunk_size,) to (1, chunk_size)
                input_chunk = int16_data.reshape(1, self.chunk_size)
                # -------------------------

                prediction = self.oww_model.predict(input_chunk)
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
            # Use the verified int16 data
            audio_bytes = int16_data.tobytes()
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
                # --- List available audio devices ---
                devices = sd.query_devices()
                logger.info("Available audio devices:")
                for i, device in enumerate(devices):
                    logger.info(f"Device {i}: {device['name']} (Input: {device['max_input_channels']} channels, Output: {device['max_output_channels']} channels)")

                # --- Select appropriate device ---
                if self.device_index is None:
                    # Find first input device
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            self.device_index = i
                            logger.info(f"Using default input device: {device['name']}")
                            break
                    if self.device_index is None:
                        raise RuntimeError("No input devices found")

                # --- Start stream ---
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

        # 1. Signal the callback thread to stop processing
        self._stop_event.set()
        logger.debug("Stop event set for audio callback.")

        # 2. Stop and close the stream object
        stream_to_stop_and_close = None
        with self._lock:
             stream_to_stop_and_close = self.stream
             self.stream = None # Prevent further access via self.stream
             self.is_recording = False # Update state under lock

        if stream_to_stop_and_close:
            logger.debug("Attempting to stop and close sounddevice stream...")
            try:
                if stream_to_stop_and_close.active:
                     stream_to_stop_and_close.stop(ignore_errors=True)
                     logger.debug("Stream stop requested.")
                if not stream_to_stop_and_close.closed:
                     stream_to_stop_and_close.close(ignore_errors=True)
                     logger.debug("Stream close requested.")
            except Exception as e:
                 logger.error(f"Error during stream stop/close: {e}", exc_info=True)
        else:
             logger.debug("No active stream object found during stop.")

        # 3. Clear audio queue
        logger.debug("Clearing audio queue...")
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
        logger.debug("Audio queue cleared.")
        
        # 4. Clear wake word queue
        logger.debug("Clearing wake word queue...")
        while not self.wake_word_queue.empty():
            try: self.wake_word_queue.get_nowait()
            except queue.Empty: break
        logger.debug("Wake word queue cleared.")

        # 5. Clear OWW model reference
        if self.oww_model:
             logger.info("Clearing OpenWakeWord model reference.")
             self.oww_model = None

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