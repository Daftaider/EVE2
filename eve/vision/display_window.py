import cv2
import numpy as np
import threading
import logging
import os
from pathlib import Path
import time
import signal
from typing import List, Dict, Optional, Tuple, Any

# Import EVE components and config
from eve.config import SystemConfig
from eve.vision.camera import Camera
from eve.vision.face_detector import FaceDetector
from eve.vision.object_detector import ObjectDetector

logger = logging.getLogger(__name__)

class VisionDisplay:
    """Handles displaying the vision processing output in an OpenCV window."""

    # Corrected __init__ signature and implementation
    def __init__(self, 
                 config: SystemConfig, 
                 camera: Camera, 
                 face_detector: Optional[FaceDetector] = None, 
                 object_detector: Optional[ObjectDetector] = None):
        """
        Initialize the Vision Display.

        Args:
            config: The main SystemConfig object.
            camera: The initialized Camera instance.
            face_detector: Optional initialized FaceDetector instance.
            object_detector: Optional initialized ObjectDetector instance.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.camera = camera
        self.face_detector = face_detector
        self.object_detector = object_detector

        self.logger.info("VisionDisplay: OpenCV backend hints can be set via env vars (commented out).")

        self.window_name = "EVE Vision"
        self.display_resolution = config.hardware.display_resolution
        self.fullscreen = config.hardware.fullscreen
        
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._window_created = False # Track if window was successfully created

        self.logger.info("Vision Display initialized.")

    def start(self):
        """Start the display thread."""
        if self._running:
            self.logger.warning("Vision display thread already running.")
            return

        if not self.camera or not self.camera.is_open():
             self.logger.error("Cannot start VisionDisplay: Camera is not available or not open.")
             return

        self.logger.info("Starting vision display thread...")
        self._stop_event.clear()
        self._running = True
        self._window_created = False # Reset flag on start
        # Use non-daemon thread for explicit join
        self._display_thread = threading.Thread(target=self._display_loop, daemon=False)
        self._display_thread.start()
        self.logger.info("Vision display thread started.")

    def stop(self):
        """Stop the display thread."""
        # Add check if already stopped
        if not self._running and self._display_thread is None:
            self.logger.debug("Display stop called but already stopped/not started.")
            return

        self.logger.info("Stopping vision display...")
        self._stop_event.set() # Signal loop to stop

        if self._display_thread is not None and self._display_thread.is_alive():
            self.logger.debug("Joining display thread...")
            self._display_thread.join(timeout=3.0) # Increased timeout
            if self._display_thread.is_alive():
                 # Log error if it hangs
                 self.logger.error("Display thread did not stop gracefully after timeout!")
            else:
                 self.logger.debug("Display thread joined successfully.")
            self._display_thread = None

            # Explicitly destroy all OpenCV windows after thread join attempt
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1) # Allow time for cleanup
                self.logger.info("Explicit cv2.destroyAllWindows() called in stop().")
            except Exception as e:
                self.logger.warning(f"Error during explicit cv2.destroyAllWindows() in stop(): {e}")

        self._running = False
        self.logger.info("Vision display stopped.")

    def _display_loop(self):
        """Display loop: Reads frames, gets detection results, shows window."""
        self.logger.info("Display loop started.")
        # Reset window created flag at loop start
        self._window_created = False

        try: # Wrap entire loop for robust cleanup
            while not self._stop_event.is_set():
                start_time = time.perf_counter()
                try:
                    ret, frame = self.camera.read()
                    # Check stop event immediately after potentially blocking camera read
                    if self._stop_event.is_set(): break

                    if not ret or frame is None:
                        if hasattr(self.camera, 'is_running') and not self.camera.is_running():
                             self.logger.warning("Camera is not running, stopping display loop.")
                             break
                        time.sleep(0.05)
                        continue

                    display_frame = frame.copy()

                    # --- Get Detection Results & Draw --- 
                    # Face Detection/Recognition Drawing
                    if self.face_detector and self.face_detector.debug_mode:
                         debug_face_frame = self.face_detector.get_debug_frame()
                         if debug_face_frame is not None:
                              # Basic overlay/replace, adjust logic as needed
                              if debug_face_frame.shape == display_frame.shape:
                                   display_frame = debug_face_frame

                    # Object Detection Drawing
                    if self.object_detector and self.config.vision.object_detection_enabled:
                        try:
                             # WARNING: Synchronous call, potential hang point
                             self.logger.debug("Calling object_detector.detect()...")
                             detections = self.object_detector.detect(display_frame)
                             self.logger.debug("Finished object_detector.detect().")
                             display_frame = self.object_detector.draw_detections(display_frame, detections)
                        except Exception as od_err:
                             self.logger.error(f"Error during object detection/drawing: {od_err}", exc_info=False)

                    # --- Display Frame --- 
                    if not self._window_created:
                        # Create window only once
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                        if self.fullscreen:
                             cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.resizeWindow(self.window_name, self.display_resolution[0], self.display_resolution[1])
                        self._window_created = True # Set flag after successful creation
                        self.logger.info(f"OpenCV window '{self.window_name}' created.")

                    # Resize frame to fit display window if needed 
                    h, w = display_frame.shape[:2]
                    disp_w, disp_h = self.display_resolution
                    canvas = display_frame # Default to original frame
                    if w != disp_w or h != disp_h:
                         scale = min(disp_w / w, disp_h / h) if w > 0 and h > 0 else 1
                         resized_frame = cv2.resize(display_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                         # Create black canvas and place resized frame in center
                         canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                         rh, rw = resized_frame.shape[:2]
                         x_offset = (disp_w - rw) // 2
                         y_offset = (disp_h - rh) // 2
                         canvas[y_offset:y_offset+rh, x_offset:x_offset+rw] = resized_frame
                                      
                    cv2.imshow(self.window_name, canvas)
                    # Check stop event immediately after potentially blocking imshow
                    if self._stop_event.is_set(): break
                    
                    # --- Store Last Frame --- 
                    with self._frame_lock:
                         self._last_frame = display_frame.copy()
                     
                    # --- Handle Window Events & Exit Conditions --- 
                    key = cv2.waitKey(1) & 0xFF # Essential!
                    # Check stop event immediately after potentially blocking waitKey
                    if self._stop_event.is_set(): break

                    # 1. Check for ESC key press
                    if key == 27:  # ESC
                        self.logger.info("ESC key pressed, stopping display loop.")
                        break # Exit while loop

                    # Check if window closed (best effort)
                    try:
                        if self._window_created and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                             self.logger.info(f"Window '{self.window_name}' closed by user, stopping display loop.")
                             break # Exit while loop
                    except cv2.error:
                        self.logger.info(f"Window '{self.window_name}' likely closed abruptly.")
                        break # Exit while loop

                    # --- Loop Rate Control --- 
                    loop_duration = time.perf_counter() - start_time
                    target_interval = 1.0 / self.config.hardware.display_fps if self.config.hardware.display_fps > 0 else 0.01
                    sleep_time = target_interval - loop_duration
                    if sleep_time > 0:
                         time.sleep(sleep_time)

                except Exception as loop_err:
                    # Check if error is due to stop event
                    if self._stop_event.is_set():
                        self.logger.info("Display loop interrupted by stop event.")
                        break
                    self.logger.error(f"Error in display loop iteration: {loop_err}", exc_info=True)
                    time.sleep(0.5) # Avoid tight loop on error

        finally:
            # --- Cleanup inside the thread --- 
            self.logger.info("Display loop finished. Cleaning up window...")
            if self._window_created:
                 try:
                      cv2.destroyWindow(self.window_name)
                      # Add waitKey here to process destroy event
                      cv2.waitKey(1)
                      self.logger.info(f"OpenCV window '{self.window_name}' destroyed by display loop.")
                 except Exception as destroy_err:
                      self.logger.warning(f"Error destroying OpenCV window '{self.window_name}' in display loop cleanup: {destroy_err}")
            self._running = False # Mark as not running after loop exit
            self._window_created = False # Reset flag

    def get_last_displayed_frame(self) -> Optional[np.ndarray]:
        """Returns a copy of the last frame shown in the display window (thread-safe)."""
        with self._frame_lock:
            if self._last_frame is not None:
                return self._last_frame.copy()
            return None

    # Removed process_frame, load_known_faces, start_learning_face, _save_learned_face, _init_display_fallback

    # Removed _signal_handler, assuming main app/orchestrator handles signals.

    # Removed _capture_loop - Frame capture is handled by Camera, detection by detectors.