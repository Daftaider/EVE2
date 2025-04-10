"""
Speech recognition module for the EVE2 system.

This module provides speech recognition functionality using Whisper models
to convert audio input to text.
"""

import os
import logging
import numpy as np
import threading
import queue
import time
import sounddevice as sd
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
from faster_whisper import WhisperModel
import eve.config as config
import random
from eve.config.communication import TOPICS

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """
    Speech recognition using faster-whisper.
    Consumes audio from a queue in a separate thread and calls a command callback
    when speech is recognized.
    """
    
    def __init__(self, config, audio_queue: queue.Queue):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.audio_queue = audio_queue
        self.command_callback: Optional[Callable[[str, float], None]] = None
        
        # Language from config
        self.language = getattr(config, 'speech_recognition_language', 'en')
        
        # Audio format parameters (expected from audio_queue)
        self.sample_rate = getattr(config, 'audio_sample_rate', 16000)
        self.channels = 1 # Whisper works internally with mono
        self.sample_width = 2 # Bytes per sample (int16)
        
        # faster-whisper specific config
        self.model_size = getattr(config, 'whisper_model_size', 'tiny.en')
        self.device = getattr(config, 'whisper_device', 'cpu')
        self.compute_type = getattr(config, 'whisper_compute_type', 'int8')
        
        self.model: Optional[WhisperModel] = None
        self._init_recognizer()
        
        # Threading control
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        
        self.logger.info(f"Speech recognizer initialized. Lang='{self.language}', Model='{self.model_size}'")

    def set_command_callback(self, callback: Callable[[str, float], None]):
        """Sets the callback function to be invoked when a command is recognized."""
        self.logger.debug("Setting command callback.")
        self.command_callback = callback

    def _init_recognizer(self):
        """Initialize the faster-whisper model."""
        self.logger.info(f"Loading faster-whisper model: {self.model_size} (Device: {self.device}, Compute: {self.compute_type})")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self.logger.info(f"faster-whisper model '{self.model_size}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading faster-whisper model '{self.model_size}': {e}", exc_info=True)
            self.model = None

    def start(self):
        """Starts the background processing thread."""
        if self._running:
            self.logger.warning("SpeechRecognizer processing thread already running.")
            return
        if not self.model:
            self.logger.error("Cannot start SpeechRecognizer thread: Model not loaded.")
            return

        self.logger.info("Starting SpeechRecognizer processing thread...")
        self._running = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True, name="SpeechRecogThread")
        self._processing_thread.start()

    def stop(self):
        """Stops the background processing thread."""
        if not self._running:
            # self.logger.warning("SpeechRecognizer processing thread already stopped.")
            return # Okay to call stop multiple times

        self.logger.info("Stopping SpeechRecognizer processing thread...")
        self._running = False
        # Put a sentinel value in the queue to ensure the thread wakes up and exits
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            self.logger.warning("Audio queue full while trying to stop SpeechRecognizer thread. May block briefly.")
            # If queue is full, thread might be blocked on put; stopping might take longer.

        if self._processing_thread and self._processing_thread.is_alive():
            self.logger.debug("Joining SpeechRecognizer processing thread...")
            self._processing_thread.join(timeout=2.0) # Use a shorter timeout
            if self._processing_thread.is_alive():
                 self.logger.warning("SpeechRecognizer processing thread did not stop gracefully.")
        self._processing_thread = None
        self.logger.info("SpeechRecognizer processing thread stopped.")

    def _processing_loop(self):
        """Continuously processes audio chunks from the queue."""
        self.logger.info("SpeechRecognizer processing loop started.")
        while self._running:
            try:
                # Block until an item is available or timeout occurs
                audio_data = self.audio_queue.get(timeout=0.5)

                # Check for sentinel value used for stopping
                if audio_data is None:
                    self.logger.debug("Received stop sentinel in audio queue.")
                    self.audio_queue.task_done() # Mark sentinel as done
                    break

                if not audio_data:
                    self.audio_queue.task_done() # Mark empty data as done
                    continue # Skip empty data

                # --- Perform Transcription ---
                text = ""
                try:
                    # Convert int16 bytes back to float32 numpy array
                    if len(audio_data) % self.sample_width != 0:
                         self.logger.warning(f"Audio data length ({len(audio_data)}) not multiple of sample width ({self.sample_width}). Truncating.")
                         audio_data = audio_data[:len(audio_data) - (len(audio_data) % self.sample_width)]

                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    # Check if array is empty after potential truncation
                    if audio_int16.size == 0:
                        self.logger.warning("Audio data became empty after truncation.")
                        self.audio_queue.task_done()
                        continue

                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    self.logger.debug("Transcribing audio chunk with faster-whisper...")
                    segments, info = self.model.transcribe(audio_float32, language=self.language, beam_size=1)

                    # Concatenate text from segments
                    recognized_texts = [segment.text for segment in segments]
                    if recognized_texts:
                         text = " ".join(recognized_texts).strip()
                         # Prevent logging overly long transcriptions if they occur
                         log_text = text[:100] + ('...' if len(text) > 100 else '')
                         self.logger.info(f"Whisper recognized: '{log_text}' (Lang: {info.language}, Prob: {info.language_probability:.2f})")
                    else:
                         self.logger.debug("Whisper produced no text segments.")
                         self.audio_queue.task_done() # Mark chunk as done even if no text
                         continue # No text, nothing to do

                except Exception as e:
                    self.logger.error(f"Unexpected error during Whisper transcription: {e}", exc_info=False) # Reduce log spam
                    self.audio_queue.task_done() # Mark chunk as done even on error
                    continue # Skip to next chunk on error

                # --- Process recognized text ---
                if text and self.command_callback:
                    # Use a placeholder confidence if needed
                    confidence = info.language_probability if info else 0.8 # Use lang prob as confidence proxy
                    self.logger.debug(f"Calling command callback for recognized text.")
                    try:
                        self.command_callback(text, confidence)
                    except Exception as cb_err:
                         self.logger.error(f"Error executing command callback: {cb_err}", exc_info=True)

                # Indicate that the task from the queue is done
                self.audio_queue.task_done()

            except queue.Empty:
                # Timeout occurred, just loop again and check self._running
                continue
            except Exception as e:
                # Catch unexpected errors in the loop itself
                self.logger.error(f"Unexpected error in SpeechRecognizer processing loop: {e}", exc_info=True)
                # Ensure task_done is called even if loop logic fails badly
                try:
                     self.audio_queue.task_done()
                except ValueError: # May happen if get() failed before task_done
                     pass
                time.sleep(0.1) # Avoid tight loop on error

        self.logger.info("SpeechRecognizer processing loop finished.")

    # recognize_file can potentially remain if needed elsewhere, but needs update if used
    def recognize_file(self, file_path: str) -> Tuple[str, float]:
        """
        Recognize speech from an audio file (bypasses queue).
        NOTE: Use with caution as it's not thread-safe with the main processing loop
              if they access the model simultaneously without locks (which they might).
              Best for offline tasks or when the processing loop isn't running.
        """
        if not self.model:
            self.logger.error("Cannot recognize file: Model not initialized")
            return "", 0.0

        try:
            self.logger.info(f"Recognizing speech from file: {file_path}")
            segments, info = self.model.transcribe(file_path, language=self.language, beam_size=1)
            recognized_texts = [segment.text for segment in segments]
            text = " ".join(recognized_texts).strip()
            confidence = info.language_probability if info else 0.8 # Use lang prob as confidence proxy

            log_text = text[:100] + ('...' if len(text) > 100 else '')
            self.logger.info(f"Recognized from file: '{log_text}'")
            return text, confidence

        except FileNotFoundError:
             logger.error(f"Audio file not found: {file_path}")
             return "", 0.0
        except Exception as e:
            logger.error(f"Speech recognition error on file {file_path}: {e}", exc_info=True)
            return "", 0.0

    # reset might be useful if recognizer state needs clearing
    def reset(self):
        """Reset the recognizer state (if applicable in the future)."""
        self.logger.info("Resetting SpeechRecognizer state (currently no action).")
        pass
