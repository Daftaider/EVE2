import cv2
import time
import logging
import numpy as np
import os
import queue
import threading # Added for thread safety
import random # Added random import for mock camera
from typing import Optional, Tuple, List
from eve.config import SystemConfig # Import the main config object type

# Try to import picamera2, but don't fail if it's not available
try:
    from picamera2 import Picamera2, Preview
    # For configuration
    from libcamera import controls, Transform 
    HAVE_PICAMERA = True
    logging.info("picamera2 library loaded successfully.")
except ImportError:
    Picamera2 = None
    HAVE_PICAMERA = False
    logging.warning("picamera2 library not found or unavailable. PiCamera features disabled.")
except Exception as e:
    # Catch other potential import errors (like libcamera issues)
    Picamera2 = None
    HAVE_PICAMERA = False
    logging.error(f"Error importing picamera2: {e}. PiCamera features disabled.", exc_info=True)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FPS = 15
MAX_OPENCV_PROBE_INDICES = 5 # Limit how many OpenCV indices to check

class Camera:
    """Unified camera interface handling OpenCV and PiCamera2 backends."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the camera based on configuration and available hardware.

        Args:
            config: The main SystemConfig object.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.camera_instance = None # Holds Picamera2 or cv2.VideoCapture
        self.camera_backend: Optional[str] = None # 'picamera2' or 'opencv'
        self.mock_mode: bool = False
        self.is_running: bool = False
        self.width: int = config.hardware.camera_resolution[0]
        self.height: int = config.hardware.camera_resolution[1]
        self.target_fps: int = config.hardware.camera_framerate or DEFAULT_FPS
        self.rotation: int = config.hardware.camera_rotation or 0
        self.backend_preference: str = config.hardware.camera_type.lower() # 'picamera', 'opencv'
        
        self._frame_lock = threading.Lock() # Lock for accessing frame and related stats
        self._current_frame: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event() # Signal to stop capture thread
        
        # FPS calculation related
        self._frame_count: int = 0
        self._start_time: float = time.time()
        self._actual_fps: float = 0.0
        self._last_frame_read_time: float = 0.0
        
        # Mock mode specifics
        self._mock_patterns: List[np.ndarray] = []
        self._current_mock_pattern: int = 0
        self._mock_pattern_change_time: float = time.time()

        if not config.hardware.camera_enabled:
            self.logger.warning("Camera is disabled in configuration. Using mock mode.")
            self._initialize_mock_camera()
        else:
            self._initialize_best_camera()
            
    def _initialize_best_camera(self):
        """Tries to initialize PiCamera2 first, then falls back to OpenCV."""
        initialized = False
        if HAVE_PICAMERA:
            self.logger.info("Attempting to initialize camera using picamera2...")
            if self._try_initialize_picamera2():
                self.camera_backend = 'picamera2'
                initialized = True
            else:
                 self.logger.warning("Failed to initialize with picamera2, falling back to OpenCV.")

        if not initialized:
            self.logger.info("Attempting to initialize camera using OpenCV...")
            if self._try_initialize_opencv():
                self.camera_backend = 'opencv'
                initialized = True
            else:
                 self.logger.error("Failed to initialize with OpenCV as well.")

        if not initialized:
             self.logger.warning("Could not initialize any real camera. Switching to mock mode.")
             self._initialize_mock_camera()

    def _try_initialize_picamera2(self) -> bool:
        """Attempts to initialize using Picamera2."""
        if not HAVE_PICAMERA or Picamera2 is None:
             return False
        try:
            # Check if cameras are detected
            cameras = Picamera2.global_camera_info()
            if not cameras:
                self.logger.warning("picamera2: No cameras detected.")
                return False
            self.logger.info(f"picamera2: Found cameras: {cameras}")
            
            # Use the first detected camera for now
            # TODO: Potentially allow selecting camera via config if multiple Pi cameras exist
            picam2 = Picamera2(camera_num=0)

            # Configure for desired resolution and format
            # Need BGR format for OpenCV compatibility downstream
            config_data = picam2.create_video_configuration(
                 main={"size": (self.width, self.height), "format": "BGR888"},
                 controls={
                      "FrameRate": float(self.target_fps),
                      # Add other controls if needed (e.g., exposure, awb)
                      # "AeEnable": True, "AwbEnable": True, "AwbMode": controls.AwbModeEnum.Auto
                 }
            )
            # Apply rotation using libcamera Transform
            if self.rotation != 0:
                 transform = Transform(hflip=(self.rotation >= 180), vflip=(self.rotation==180 or self.rotation==270))
                 config_data["transform"] = transform
                 
            picam2.configure(config_data)
            self.logger.info(f"picamera2: Configured with {config_data}")

            picam2.start()
            self.logger.info("picamera2: Camera started.")
            time.sleep(1.0) # Allow sensor to settle

            # Verify capture
            test_frame = picam2.capture_array("main")
            if test_frame is None or test_frame.size == 0:
                 self.logger.error("picamera2: Started but failed to capture test frame.")
                 picam2.close()
                 return False

            self.camera_instance = picam2
            self.logger.info(f"picamera2: Initialization successful. Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            return True

        except Exception as e:
            self.logger.error(f"picamera2: Error during initialization: {e}", exc_info=True)
            if 'picam2' in locals() and picam2 and hasattr(picam2, 'is_open') and picam2.is_open:
                 try:
                     picam2.close()
                 except Exception as close_e:
                     self.logger.error(f"picamera2: Error closing after failure: {close_e}")
            self.camera_instance = None
            return False
            
    def _try_initialize_opencv(self) -> bool:
        """Attempts to initialize using OpenCV."""
        preferred_index = self.config.hardware.camera_index
        indices_to_try = [preferred_index] + [i for i in range(MAX_OPENCV_PROBE_INDICES) if i != preferred_index]
        
        self.logger.info(f"OpenCV: Probing camera indices: {indices_to_try}")
        
        for index in indices_to_try:
            self.logger.debug(f"OpenCV: Trying index {index}...")
            # Suppress stderr during probe (optional)
            original_stderr = -1
            try:
                # original_stderr = os.dup(2)
                # devnull_fd = os.open(os.devnull, os.O_WRONLY)
                # os.dup2(devnull_fd, 2)
                # os.close(devnull_fd)
                
                cap = cv2.VideoCapture(index)
                
            # finally:
            #     # Restore stderr if redirection was used
            #     if original_stderr != -1:
            #         os.dup2(original_stderr, 2)
            #         os.close(original_stderr)
            # pass # Not using redirection for now

            except Exception as e:
                 self.logger.error(f"OpenCV: Exception during VideoCapture({index}): {e}")
                 continue # Try next index
                 
            if cap is not None and cap.isOpened():
                self.logger.info(f"OpenCV: Successfully opened index {index}.")
                try:
                    # Set desired properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, float(self.target_fps)) # FPS might need float
                    
                    # Verify by reading a frame
                    time.sleep(0.5) # Allow buffer to fill
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Success!
                        self.camera_instance = cap
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        self.logger.info(f"OpenCV: Initialization successful for index {index}.")
                        self.logger.info(f"OpenCV: Actual Res: {actual_w}x{actual_h}, FPS: {actual_fps:.2f} (Requested: {self.width}x{self.height} @ {self.target_fps} FPS)")
                        # Update internal resolution if camera forced a different one
                        self.width = actual_w
                        self.height = actual_h
                        return True
                    else:
                        self.logger.warning(f"OpenCV: Opened index {index} but failed to read valid frame.")
                        cap.release()
                except Exception as e_set:
                     self.logger.error(f"OpenCV: Error setting properties or reading from index {index}: {e_set}")
                     if cap.isOpened(): cap.release()
            else:
                 self.logger.debug(f"OpenCV: Index {index} could not be opened.")
                 if cap: cap.release() # Release if object exists but not opened
                 
        # If loop finishes without success
        self.camera_instance = None
        return False
        
    def _initialize_mock_camera(self):
        """Initialize mock camera mode."""
        self.mock_mode = True
        self.camera_backend = 'mock'
        self._mock_patterns = []
        width, height = self.width, self.height
        
        # Create 3 different mock face patterns
        for i in range(3):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame.fill(random.randint(30, 70)) # Vary background slightly
            
            offset_x = (i - 1) * width // 10
            center_x = width // 2 + offset_x
            center_y = height // 2
            radius = min(width, height) // 4
            
            # --- Draw simple face --- 
            face_color = (random.randint(180, 220), random.randint(170, 210), random.randint(230, 255))
            cv2.ellipse(frame, (center_x, center_y), (radius, int(radius * 1.1)), 0, 0, 360, face_color, -1)
            eye_radius = radius // 6
            eye_offset_x = radius // 2
            eye_offset_y = radius // 4
            for x_mult in [-1, 1]:
                x = center_x + x_mult * eye_offset_x
                y = center_y - eye_offset_y
                cv2.circle(frame, (x, y), eye_radius, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), eye_radius // 2, (50, 50, 50), -1)
            mouth_y = center_y + radius // 2
            cv2.line(frame, (center_x - radius // 2, mouth_y), (center_x + radius // 2, mouth_y), (150, 150, 150), 3)
            # -------------------------
            
            # Add text label
            cv2.putText(frame, f"Mock Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "MOCK CAMERA ACTIVE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self._mock_patterns.append(frame)
        
        self._current_mock_pattern = 0
        self._mock_pattern_change_time = time.time()
        self.logger.info(f"Mock camera initialized with {len(self._mock_patterns)} patterns. Resolution: {width}x{height}")

    def _get_mock_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Generate the next mock frame."""
        if not self.mock_mode or not self._mock_patterns:
             return False, None
             
        current_time = time.time()
        
        # Change pattern periodically
        if current_time - self._mock_pattern_change_time > 3.0:
            self._current_mock_pattern = (self._current_mock_pattern + 1) % len(self._mock_patterns)
            self._mock_pattern_change_time = current_time
        
        frame = self._mock_patterns[self._current_mock_pattern].copy()
        
        # Add subtle movement
        shift_x = int(np.sin(current_time * 1.5) * 4)
        shift_y = int(np.cos(current_time * 1.5) * 4)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Simulate camera FPS delay
        if self.target_fps > 0:
             time.sleep(1.0 / self.target_fps)
             
        return True, frame
        
    def _capture_loop_internal(self):
        """Internal loop run by the capture thread."""
        self.logger.info(f"Capture thread started for backend: {self.camera_backend}")
        self._start_time = time.time()
        self._frame_count = 0
        consecutive_error_count = 0
        max_consecutive_errors = 5 # Threshold before stopping
        
        while not self._stop_event.is_set():
            frame = None
            ret = False
            try:
                if self.mock_mode:
                    ret, frame = self._get_mock_frame()
                elif self.camera_instance:
                    capture_start_time = time.perf_counter()
                    if self.camera_backend == 'picamera2':
                         # capture_array blocks until frame is available
                         frame = self.camera_instance.capture_array("main") 
                         ret = frame is not None and frame.size > 0
                    elif self.camera_backend == 'opencv':
                         ret, frame = self.camera_instance.read()
                    capture_duration = time.perf_counter() - capture_start_time
                    
                    # If capture takes longer than frame interval, log warning
                    if self.target_fps > 0 and capture_duration > (1.1 / self.target_fps):
                        self.logger.debug(f"Capture took {capture_duration*1000:.1f}ms (longer than target interval {1000/self.target_fps:.1f}ms)")
                        
                else:
                    # Camera not initialized or lost
                    self.logger.warning("Camera instance not available in capture loop. Attempting re-init...")
                    time.sleep(1.0)
                    self._initialize_best_camera() # Try to recover
                    if self.mock_mode: # Switched to mock after re-init failed
                         ret, frame = self._get_mock_frame()
                    continue # Skip rest of loop iteration
                
                # Process the captured frame
                if ret and frame is not None:
                    consecutive_error_count = 0 # Reset error count on success
                    with self._frame_lock:
                        # Apply rotation for OpenCV frames if needed (PiCam handled by Transform)
                        if self.camera_backend == 'opencv' and self.rotation != 0:
                            if self.rotation == 90:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            elif self.rotation == 180:
                                frame = cv2.rotate(frame, cv2.ROTATE_180)
                            elif self.rotation == 270:
                                 frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                 
                        self._current_frame = frame
                        self._frame_count += 1
                        self._last_frame_read_time = time.time()
                else:
                    # Handle read failure
                    consecutive_error_count += 1
                    self.logger.warning(f"Failed to read frame from {self.camera_backend} (Error count: {consecutive_error_count}).")
                    if consecutive_error_count >= max_consecutive_errors:
                         self.logger.error(f"Max consecutive read errors reached. Stopping capture thread.")
                         break # Exit loop
                    time.sleep(0.5) # Wait before retrying
                    continue # Skip FPS calculation update if frame is invalid
                
                # Calculate FPS
                current_time = time.time()
                elapsed_time = current_time - self._start_time
                if elapsed_time > 1.0: # Update FPS calculation roughly every second
                    self._actual_fps = self._frame_count / elapsed_time
                    # Reset for next calculation period
                    # self._start_time = current_time 
                    # self._frame_count = 0 
                    # OR keep running average:
                    if elapsed_time > 60: # Reset periodically to avoid drift if FPS changes
                         self._start_time = current_time
                         self._frame_count = 0
                         
                # Optional delay to attempt target FPS, if capture was fast
                # This is less reliable than camera controlling FPS
                # if self.target_fps > 0:
                #     delay = (1.0 / self.target_fps) - capture_duration
                #     if delay > 0:
                #         time.sleep(delay)
                
            except Exception as e:
                # Avoid logging errors during normal shutdown
                if self._stop_event.is_set():
                     self.logger.info("Capture loop interrupted by stop event.")
                     break
                self.logger.error(f"Exception in camera capture loop ({self.camera_backend}): {e}", exc_info=True)
                consecutive_error_count += 1
                if consecutive_error_count >= max_consecutive_errors:
                         self.logger.error(f"Max consecutive exceptions reached. Stopping capture thread.")
                         break # Exit loop
                time.sleep(1.0) # Wait after error
                
        self.logger.info("Capture thread finished.")
        # Mark as not running AFTER loop exit
        self.is_running = False
        
    def start(self) -> bool:
        """Starts the camera capture thread."""
        if self.is_running:
            self.logger.warning("Camera capture thread already running.")
            return True
            
        if self.camera_instance is None and not self.mock_mode:
             self.logger.error("Cannot start capture thread, camera not initialized.")
             return False
             
        self.logger.info("Starting camera capture thread...")
        self._stop_event.clear()
        # Use non-daemon thread for explicit join
        self._capture_thread = threading.Thread(target=self._capture_loop_internal, daemon=False) # Changed daemon=False
        self._capture_thread.start()
        self.is_running = True
        return True

    def stop(self):
        """Stops the camera capture thread and releases resources."""
        if not self.is_running and self._capture_thread is None:
            self.logger.debug("Camera stop called but already stopped/not started.")
            return

        self.logger.info("Stopping camera...")
        self._stop_event.set() # Signal thread to stop

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self.logger.debug("Joining capture thread...")
            # Use a slightly longer timeout for camera thread join
            self._capture_thread.join(timeout=5.0) # Increased timeout
            if self._capture_thread.is_alive():
                 self.logger.error("Capture thread did not stop gracefully after timeout!")
            else:
                 self.logger.debug("Capture thread joined successfully.")
            self._capture_thread = None

        # Release hardware resources AFTER thread has been joined (or timed out)
        self.release()
        self.is_running = False # Set running state false after cleanup
        self.logger.info("Camera stopped and resources released.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the latest captured frame (thread-safe)."""
        # Check if thread is running; start if not? Or rely on external start call?
        # For now, assume start() was called.
        if not self.is_running and not self.mock_mode:
             self.logger.warning("Read called but camera is not running.")
             # Optionally, try starting automatically?
             # if not self.start(): 
             #    return False, None # Failed to start
             return False, None
             
        # Return immediately if in mock mode and it generates frames on read
        if self.mock_mode:
             # The mock frame logic is now in the capture loop
             # We return the latest stored mock frame
             with self._frame_lock:
                  frame = self._current_frame
             # Check if a frame has been generated yet
             if frame is None:
                  time.sleep(0.05) # Short wait if called right at start
                  with self._frame_lock:
                       frame = self._current_frame
             return frame is not None, frame.copy() if frame is not None else None

        # For real camera, return the latest frame captured by the thread
        with self._frame_lock:
            if self._current_frame is not None:
                # Check how old the frame is
                frame_age = time.time() - self._last_frame_read_time
                if frame_age > 1.0: # If frame is older than 1s, maybe thread died?
                     self.logger.warning(f"Last camera frame is {frame_age:.1f}s old. Capture thread might be stuck.")
                     # Check thread health
                     if self._capture_thread is not None and not self._capture_thread.is_alive():
                          self.logger.error("Capture thread is not alive!")
                          self.is_running = False # Update state
                          return False, None
                          
                return True, self._current_frame.copy() # Return a copy
            else:
                # No frame available yet or thread stopped
                if not self.is_running:
                    self.logger.warning("Read called, but camera thread is not running.")
                else:
                     self.logger.debug("Read called, but no frame available yet.")
                return False, None

    def get_frame(self):
        """Alias for read() for compatibility."""
        return self.read()

    def get_actual_fps(self) -> float:
        """Returns the calculated actual FPS (thread-safe)."""
        # No lock needed as float reads/writes are generally atomic
        # However, reading multiple related values (fps, frame_count, start_time) 
        # might need locking if consistency is critical.
        # For a simple FPS display, direct read is likely fine.
        return self._actual_fps 

    def is_open(self) -> bool:
        """Check if camera is initialized (real or mock)."""
        return self.mock_mode or (self.camera_instance is not None and self.is_running)

    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution (might differ from config if camera forced it)."""
        # Read resolution thread-safe? Unlikely to change after init.
        return (self.width, self.height)

    def release(self):
        """Release camera hardware resources (called by stop)."""
        self.logger.debug(f"Releasing camera hardware ({self.camera_backend})...")
        # Ensure release happens even if thread join timed out
        if self.camera_instance:
            try:
                if self.camera_backend == 'picamera2' and hasattr(self.camera_instance, 'close'):
                    self.camera_instance.close()
                    self.logger.info("picamera2 instance closed.")
                elif self.camera_backend == 'opencv' and hasattr(self.camera_instance, 'release'):
                    self.camera_instance.release()
                    self.logger.info("OpenCV VideoCapture released.")
            except Exception as e:
                 self.logger.error(f"Exception during camera release/close: {e}")
            finally:
                 self.camera_instance = None # Ensure instance is cleared

    def __enter__(self):
        # Optional: Start automatically when used as context manager
        # self.start() # Prefer explicit start
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures stop is called when exiting context manager."""
        self.stop() 