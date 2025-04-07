"""
Camera module implementation for the Raspberry Pi AI Camera.

Uses Picamera2 library and retrieves onboard AI results from metadata.
"""
import logging
import threading
import time
from collections import deque
from typing import Optional, Any, List, Dict, Tuple

import numpy as np
from picamera2 import Picamera2, Preview
from picamera2.controls import Controls
from libcamera import Transform # Import Transform for rotation

logger = logging.getLogger(__name__)

class RPiAICamera:
    """
    Handles video capture and onboard AI result retrieval for the RPi AI Camera.
    """
    def __init__(self, config):
        self.config = config.hardware # Assuming hardware config is passed
        self.logger = logging.getLogger(__name__)
        self.picam2: Optional[Picamera2] = None

        # Configurable parameters
        self.resolution: Tuple[int, int] = tuple(self.config.camera_resolution)
        self.framerate: int = self.config.camera_framerate
        self.rotation: int = self.config.camera_rotation

        # Frame and AI result storage
        self._frame_buffer = deque(maxlen=2) # Store last few frames
        self._metadata_buffer = deque(maxlen=2) # Store last few metadata dicts
        self._ai_results_buffer = deque(maxlen=2) # Store parsed AI results
        self._buffer_lock = threading.Lock()

        # Threading control
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self.camera_started = False

        self.logger.info(f"RPiAICamera initialized. Resolution={self.resolution}, Framerate={self.framerate}, Rotation={self.rotation}")

    def start(self) -> bool:
        """Initializes and starts the camera capture thread."""
        if self._running:
            self.logger.warning("Camera is already running.")
            return True

        self.logger.info("Starting RPi AI Camera...")
        try:
            self.picam2 = Picamera2()

            # --- Configuration ---
            transform = Transform(hflip=(self.rotation in [180]), vflip=(self.rotation in [90, 270])) # Adjust flip based on rotation? Needs testing.

            # Configure the main video stream
            # Use a format suitable for OpenCV (e.g., RGB888)
            # Ensure dimensions match config, though Picamera2 might adjust slightly
            video_config = self.picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                transform=transform,
                controls={
                    "FrameRate": float(self.framerate) # Ensure framerate is float
                }
            )
            self.logger.debug(f"Video Config: {video_config}")
            self.picam2.configure(video_config)
            self.logger.info("Picamera2 configured.")

            self.logger.info("Attempting picam2.start()...")
            self.picam2.start()
            self.camera_started = True
            self.logger.info(f"Picamera2 started successfully. Sensor resolution: {self.picam2.camera_properties['PixelArraySize']}")

            # Start the background capture thread
            self.logger.info("Starting camera capture thread...")
            self._running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="RPiAICamThread")
            self._capture_thread.start()
            self.logger.info("Camera capture thread started.")
            return True

        except ImportError as e:
             logger.error(f"Picamera2 library not found or import error: {e}. Please install it.", exc_info=True)
             self.picam2 = None
             return False
        except Exception as e:
            self.logger.error(f"Failed to initialize or start RPi AI Camera: {e}", exc_info=True)
            if self.picam2 and self.camera_started:
                try: self.picam2.stop()
                except Exception: pass
            if self.picam2:
                 try: self.picam2.close()
                 except Exception: pass
            self.picam2 = None
            self.camera_started = False
            return False

    def stop(self):
        """Stops the camera capture thread and releases the camera."""
        if not self._running:
            return

        self.logger.info("Stopping RPi AI Camera...")
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self.logger.debug("Joining camera capture thread...")
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                self.logger.warning("Camera capture thread did not stop gracefully.")
        self._capture_thread = None

        if self.picam2 and self.camera_started:
            self.logger.debug("Stopping Picamera2...")
            try:
                self.picam2.stop()
                self.camera_started = False
                self.logger.debug("Picamera2 stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping Picamera2: {e}", exc_info=True)

        if self.picam2:
            self.logger.debug("Closing Picamera2...")
            try:
                self.picam2.close()
                self.logger.debug("Picamera2 closed.")
            except Exception as e:
                self.logger.error(f"Error closing Picamera2: {e}", exc_info=True)
            self.picam2 = None

        with self._buffer_lock:
            self._frame_buffer.clear()
            self._metadata_buffer.clear()
            self._ai_results_buffer.clear()

        self.logger.info("RPi AI Camera stopped.")

    def _capture_loop(self):
        """Background thread to continuously capture frames and metadata."""
        self.logger.info("RPi AI Camera capture loop started.")
        while self._running and self.picam2:
            try:
                # capture_request() gives access to frame data AND metadata
                request = self.picam2.capture_request()
                if request is None:
                    logger.warning("Capture request returned None, camera might be stopping.")
                    time.sleep(0.01)
                    continue

                frame = request.make_array("main") # Get the main frame
                metadata = request.get_metadata() # Get the metadata dictionary
                request.release() # Release the request buffers

                if frame is not None and metadata is not None:
                    with self._buffer_lock:
                        self._frame_buffer.append(frame)
                        self._metadata_buffer.append(metadata)

                    # Process metadata for AI results (Placeholder)
                    ai_results = self._parse_ai_metadata(metadata)
                    if ai_results:
                         with self._buffer_lock:
                              self._ai_results_buffer.append(ai_results)

                # Optional: Small sleep to prevent busy-waiting if capture is very fast
                # time.sleep(0.005)

            except Exception as e:
                if self._running: # Only log errors if we are supposed to be running
                    self.logger.error(f"Error in camera capture loop: {e}", exc_info=False) # Keep logs cleaner
                    # Consider adding a longer sleep on error to prevent spamming logs
                    time.sleep(0.5)
                else:
                    # If not running, error might be expected during shutdown
                    self.logger.debug(f"Capture loop exiting due to error during shutdown: {e}")
                    break # Exit loop

        self.logger.info("RPi AI Camera capture loop finished.")

    def _parse_ai_metadata(self, metadata: Dict) -> Optional[List[Dict]]:
        """
        Parses the AI detection results from the camera metadata.

        Placeholder implementation - **This needs to be adapted based on the
        actual structure of the metadata provided by the RPi AI Camera.**

        Expected output format: A list of dictionaries, where each dictionary
        represents a detected object, e.g.:
        [
            {"label": "person", "score": 0.85, "bbox": [x_min, y_min, x_max, y_max]},
            {"label": "car", "score": 0.72, "bbox": [x_min, y_min, x_max, y_max]},
            ...
        ]
        Bounding box coordinates should ideally be relative (0.0 to 1.0).
        """
        try:
            # --- Placeholder Logic ---
            # Examine the 'metadata' dictionary structure based on documentation or testing.
            # Documentation suggests Algorithm.Output might contain results.

            # Example structure based on potential Picamera2/libcamera metadata:
            algo_output = metadata.get("Algorithm.Output", [])

            parsed_results = []
            if isinstance(algo_output, list):
                for det in algo_output:
                    # Extract label, score, bbox based on the actual format
                    # This requires inspecting the 'det' dictionary content
                    label = det.get("label", "unknown") # Key might be 'label', 'name', 'class_name'
                    score = det.get("score", det.get("confidence", 0.0)) # Key might be 'score', 'confidence'
                    # Bbox key might be 'bbox', 'rect', 'bounding_box'
                    # Format might be [xmin, ymin, width, height] or [xmin, ymin, xmax, ymax]
                    # Coordinates might be absolute pixels or relative (0.0-1.0)
                    bbox_raw = det.get("bbox", [0,0,0,0])

                    # --- TODO: Convert bbox_raw to [xmin, ymin, xmax, ymax] relative coords ---
                    # This is the most critical part needing specific adaptation.
                    # Need to know source format (absolute/relative, xywh/xyxy)
                    # Assume relative [xmin, ymin, xmax, ymax] for now if possible, otherwise conversion needed.
                    # Example conversion if bbox_raw is [xmin_abs, ymin_abs, width_abs, height_abs]:
                    # img_width, img_height = self.resolution
                    # if img_width > 0 and img_height > 0:
                    #     xmin_rel = bbox_raw[0] / img_width
                    #     ymin_rel = bbox_raw[1] / img_height
                    #     xmax_rel = (bbox_raw[0] + bbox_raw[2]) / img_width
                    #     ymax_rel = (bbox_raw[1] + bbox_raw[3]) / img_height
                    #     bbox = [max(0.0, xmin_rel), max(0.0, ymin_rel), min(1.0, xmax_rel), min(1.0, ymax_rel)]
                    # else: bbox = [0.0, 0.0, 0.0, 0.0]
                    bbox = bbox_raw # Placeholder - needs real conversion

                    parsed_results.append({
                        "label": str(label),
                        "score": float(score),
                        "bbox": bbox
                    })

            if parsed_results:
                # self.logger.debug(f"Parsed AI results: {parsed_results}") # Can be noisy
                return parsed_results
            else:
                # self.logger.debug("No AI detections found in metadata.")
                return None

        except Exception as e:
            self.logger.error(f"Error parsing AI metadata: {e}", exc_info=False)
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the most recently captured frame."""
        with self._buffer_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1] # Get the latest item
            return None

    def get_latest_metadata(self) -> Optional[Dict]:
        """Returns the most recently captured metadata dictionary."""
        with self._buffer_lock:
            if self._metadata_buffer:
                return self._metadata_buffer[-1]
            return None

    def get_latest_ai_results(self) -> Optional[List[Dict]]:
        """Returns the most recently parsed AI results."""
        with self._buffer_lock:
            if self._ai_results_buffer:
                 # Return a copy to prevent external modification
                 # Ensure items in the list are also copied if they are mutable
                return list(self._ai_results_buffer[-1]) 
            return None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Provides an OpenCV-like read() interface."""
        frame = self.get_latest_frame()
        if frame is not None:
            return True, frame
        else:
            return False, None

    # --- Add methods required by other parts of your system ---
    # e.g., is_running(), get_resolution() etc. if needed

    def is_running(self) -> bool:
        return self._running

    def get_resolution(self) -> Tuple[int, int]:
        return self.resolution

    def is_open(self) -> bool:
        """Checks if the camera stream is initialized and running."""
        # Consider adding check for self.picam2.is_open if that attribute exists
        return self.picam2 is not None and self.camera_started and self._running 