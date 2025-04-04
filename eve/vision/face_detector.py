"""
Face detection and recognition module.

Uses Haar Cascade or face_recognition library models for detection,
and face_recognition library for optional recognition.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import cv2
import face_recognition
import numpy as np

# Import the consolidated Camera class and the config object
from eve.config import SystemConfig, VisionConfig
from eve.vision.camera import Camera 

# Assume TOPICS are defined elsewhere (e.g., eve.config.communication or eve.topics)
# Placeholder:
class TOPICS:
     FACE_DETECTED = "vision.face.detected"
     FACE_RECOGNIZED = "vision.face.recognized"
     FACE_LEARNED = "vision.face.learned"
     FACE_LOST = "vision.face.lost" # Placeholder if needed

logger = logging.getLogger(__name__)

# Constants
FACE_LEARNING_SAMPLE_COUNT = 5

class FaceDetector:
    """
    Handles face detection and optional recognition using different models.
    Operates in its own thread, processing frames from a Camera instance.
    """
    
    def __init__(self, config: SystemConfig, camera: Camera, post_event_callback: Optional[Callable] = None):
        """
        Initialize the face detector.
        
        Args:
            config: The main SystemConfig object.
            camera: An initialized Camera instance.
            post_event_callback: Callback function for posting events (e.g., orchestrator.post_event).
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.camera = camera
        self.post_event = post_event_callback
        
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock() # Lock for shared resources

        # Face Detection Model
        self.vision_config: VisionConfig = self.config.vision
        self.model_name = self.vision_config.face_detection_model
        self.face_cascade = None
        if self.model_name == 'haar':
            self._load_haar_cascade()
        elif self.model_name not in ['hog', 'cnn']:
             self.logger.warning(f"Invalid face_detection_model '{self.model_name}'. Defaulting to 'hog'.")
             self.model_name = 'hog'

        # Face Recognition State
        self.recognition_enabled = self.vision_config.face_recognition_enabled
        self.known_faces_dir = Path(self.vision_config.known_faces_dir)
        self.comparison_tolerance = self.vision_config.face_recognition_tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self._load_known_faces() # Load on init

        # Face Learning State
        self._learning_face_active: bool = False
        self._learning_candidate_name: Optional[str] = None
        self._learning_samples_collected: int = 0
        self._learning_temp_encodings: List[np.ndarray] = []

        # Debugging
        self.debug_mode = self.vision_config.debug
        self._current_debug_frame: Optional[np.ndarray] = None

        self.logger.info(f"FaceDetector initialized. Detection Model: {self.model_name}, Recognition Enabled: {self.recognition_enabled}")

    def _load_haar_cascade(self):
        """Loads the Haar Cascade classifier if needed."""
        try:
            # Check if cv2.data is available
            if not hasattr(cv2, 'data') or not hasattr(cv2.data, 'haarcascades'):
                 # Fallback path or error if cv2.data is not found
                 # This path might vary depending on OpenCV installation
                 fallback_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml" 
                 if os.path.exists(fallback_path):
                      cascade_path = fallback_path
                      self.logger.warning(f"cv2.data.haarcascades not found, using fallback path: {cascade_path}")
                 else:
                      raise RuntimeError("Haar cascade file path could not be determined.")
            else:
                 cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                 
            if not os.path.exists(cascade_path):
                raise RuntimeError(f"Haar cascade file not found at: {cascade_path}")
                
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError(f"Failed to load face cascade classifier from {cascade_path}")
            self.logger.info("Haar cascade face detector loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading Haar cascade: {e}. Face detection model 'haar' will not work.")
            self.face_cascade = None # Ensure it's None on failure
            # Consider falling back to HOG if Haar fails?
            # self.model_name = 'hog' 

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects faces based on the configured model."""
        face_locations = []
        try:
            if self.model_name == 'haar':
                if self.face_cascade is None or self.face_cascade.empty():
                     self.logger.error("Haar cascade not loaded, cannot detect faces.")
                     return []
                # Convert to grayscale for Haar
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.vision_config.haar_scale_factor,
                    minNeighbors=self.vision_config.haar_min_neighbors,
                    minSize=self.vision_config.haar_min_face_size
                )
                # Convert (x, y, w, h) to (top, right, bottom, left) format
                for (x, y, w, h) in faces:
                    face_locations.append((y, x + w, y + h, x))
            
            elif self.model_name in ['hog', 'cnn']:
                 # face_recognition library expects RGB
                 rgb_frame = frame[:, :, ::-1] 
                 # number_of_times_to_upsample=1 can help find smaller faces
                 face_locations = face_recognition.face_locations(rgb_frame, model=self.model_name)
                 # Returns list of (top, right, bottom, left) tuples
            
        except Exception as e:
            self.logger.error(f"Error detecting faces using model '{self.model_name}': {e}", exc_info=True)
            return [] # Return empty list on error
            
        return face_locations

    def _get_face_encodings(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
         """Gets face encodings for the detected locations."""
         if not face_locations:
              return []
         try:
              # Expects RGB frame
              rgb_frame = frame[:, :, ::-1] 
              # known_face_locations=face_locations helps performance
              # num_jitters=1 is default, increase for more robustness but slower
              return face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
         except Exception as e:
              self.logger.error(f"Error getting face encodings: {e}", exc_info=True)
              return []
              
    def _recognize_faces(self, face_encodings: List[np.ndarray]) -> List[str]:
        """Compares detected encodings against known faces."""
        face_names = []
        if not self.recognition_enabled or not self.known_face_encodings:
             return ["Unknown"] * len(face_encodings)
             
        for face_encoding in face_encodings:
             try:
                  # Compare face against known encodings
                  matches = face_recognition.compare_faces(
                       self.known_face_encodings, 
                       face_encoding,
                       tolerance=self.comparison_tolerance
                  )
                  name = "Unknown"

                  # Find the best match
                  face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                  best_match_index = np.argmin(face_distances)
                  if matches[best_match_index]:
                       name = self.known_face_names[best_match_index]
                       self.logger.debug(f"Recognized face: {name} (Distance: {face_distances[best_match_index]:.2f})")
                  else:
                       self.logger.debug(f"Unknown face detected (Closest match: {self.known_face_names[best_match_index]}, Distance: {face_distances[best_match_index]:.2f})")
                       
                  face_names.append(name)
             except Exception as e:
                  self.logger.error(f"Error during face comparison: {e}", exc_info=True)
                  face_names.append("Error") # Indicate error during recognition
        return face_names

    def _detection_loop(self):
        """Main face detection and recognition loop running in a separate thread."""
        self.logger.info(f"FaceDetector loop started. Using model: {self.model_name}")
        frame_count = 0
        empty_frame_count = 0
        last_log_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                ret, frame = self.camera.read() # Read from the injected camera instance
                frame_count += 1
                
                if not ret or frame is None:
                    empty_frame_count += 1
                    if time.time() - last_log_time > 5.0:
                        self.logger.warning(f"Received invalid/empty frame from camera ({empty_frame_count}/{frame_count} empty). Thread still alive: {self.camera._capture_thread.is_alive() if hasattr(self.camera, '_capture_thread') else 'N/A'}")
                        last_log_time = time.time()
                    time.sleep(0.1) # Avoid busy-waiting on camera errors
                    continue
                
                # Reset counter on valid frame
                if empty_frame_count > 0: 
                    self.logger.info(f"Camera stream recovered after {empty_frame_count} empty frames.")
                    empty_frame_count = 0
                
                # --- Face Detection --- 
                face_locations = self._detect_faces(frame)
                
                face_encodings = []
                face_names = []
                detected_faces_info = [] # Store combined info
                
                if face_locations:
                    # Post basic detection event immediately
                    if self.post_event:
                         self.post_event(TOPICS.FACE_DETECTED, {
                              'count': len(face_locations),
                              'locations': face_locations, # Already (top, right, bottom, left)
                              'timestamp': time.time()
                         })
                    
                    # --- Face Encoding & Recognition (if enabled) --- 
                    if self.recognition_enabled:
                         face_encodings = self._get_face_encodings(frame, face_locations)
                         if face_encodings:
                              face_names = self._recognize_faces(face_encodings)
                         else:
                              face_names = ["Unknown"] * len(face_locations)
                    else:
                        face_names = ["DetectionOnly"] * len(face_locations)
                        
                    # --- Prepare combined info --- 
                    for i, loc in enumerate(face_locations):
                        name = face_names[i]
                        encoding = face_encodings[i] if i < len(face_encodings) else None
                        detected_faces_info.append({'location': loc, 'name': name, 'encoding': encoding})
                        # Post recognition event per recognized face
                        if self.recognition_enabled and name != "Unknown" and name != "Error" and self.post_event:
                             self.post_event(TOPICS.FACE_RECOGNIZED, {
                                  'name': name,
                                  'location': loc,
                                  'timestamp': time.time()
                             })
                             
                    # --- Handle Face Learning --- 
                    # Check if learning is active AFTER potential recognition attempt
                    if self._learning_face_active and self._learning_candidate_name:
                         # Try to find a single, preferably unknown face to learn
                         # Prioritize unknown faces, then known faces if necessary (but might be confusing)
                         candidate_face = None
                         if len(detected_faces_info) == 1: # Best case: only one face in view
                              candidate_face = detected_faces_info[0]
                         elif len(detected_faces_info) > 1:
                              # Prefer unknown faces if learning
                              unknowns = [f for f in detected_faces_info if f['name'] == "Unknown"]
                              if len(unknowns) == 1:
                                   candidate_face = unknowns[0]
                              else:
                                   # Too ambiguous (multiple faces, maybe multiple unknowns)
                                   self.logger.debug("Learning face: Multiple faces detected, cannot reliably select sample.")
                                   # Maybe provide feedback? self.post_event(...) 
                         
                         if candidate_face and candidate_face['encoding'] is not None:
                              self._collect_learning_sample(candidate_face['encoding'])
                else:
                     # Optional: Post FACE_LOST event if needed
                     # Requires tracking previously seen faces
                     pass 

                # --- Update Debug Frame (Thread Safe) --- 
                if self.debug_mode:
                    debug_frame = frame.copy()
                    for face_info in detected_faces_info:
                        (top, right, bottom, left) = face_info['location']
                        name = face_info['name']
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(debug_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                        
                    # Indicate learning state
                    if self._learning_face_active:
                        learn_text = f"Learning: {self._learning_candidate_name} ({self._learning_samples_collected}/{FACE_LEARNING_SAMPLE_COUNT})"
                        cv2.putText(debug_frame, learn_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                    with self._lock:
                        self._current_debug_frame = debug_frame
                
                # Control loop rate (optional, camera might control FPS)
                # time.sleep(0.01) 
                
            except Exception as e:
                self.logger.error(f"Error in face detection loop: {e}", exc_info=True)
                time.sleep(1.0) # Avoid tight loop on critical error

        self.logger.info("FaceDetector loop finished.")

    def start(self) -> bool:
        """Start face detection thread."""
        if self._running:
            self.logger.warning("Face detector thread already running.")
            return True
            
        if not self.camera or not self.camera.is_open():
             self.logger.error("Cannot start FaceDetector: Camera is not available or not open.")
             return False
             
        self.logger.info("Starting FaceDetector thread...")
        self._stop_event.clear()
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._running = True
        self._detection_thread.start()
        return True

    def stop(self):
        """Stop face detection thread."""
        self.logger.info("Stopping FaceDetector thread...")
        self._stop_event.set()
        if self._detection_thread is not None:
            self._detection_thread.join(timeout=2.0)
            if self._detection_thread.is_alive():
                 self.logger.warning("Face detector thread did not stop gracefully.")
            self._detection_thread = None
        self._running = False
        # Don't release camera here, camera lifecycle managed externally
        self.logger.info("Face detector stopped.")

    def is_running(self) -> bool:
        """Check if detector thread is running."""
        return self._running and self._detection_thread is not None and self._detection_thread.is_alive()

    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Get the latest debug frame (thread-safe)."""
        with self._lock:
            if self._current_debug_frame is not None:
                 return self._current_debug_frame.copy()
            return None
            
    # --- Face Recognition Methods ---
    def _load_known_faces(self):
        """Load known face encodings from the configured directory."""
        self.logger.info(f"Loading known faces from: {self.known_faces_dir}")
        loaded_encodings: List[np.ndarray] = []
        loaded_names: List[str] = []
        try:
            if not self.known_faces_dir.exists():
                 self.logger.warning(f"Known faces directory does not exist: {self.known_faces_dir}")
                 self.known_faces_dir.mkdir(parents=True, exist_ok=True)
                 return # Nothing to load
                 
            for person_dir in self.known_faces_dir.iterdir():
                if person_dir.is_dir():
                    # Expect images (jpg, png) or a single encodings.npy file
                    encodings_file = person_dir / "encodings.npy"
                    person_name = person_dir.name
                    person_encodings = []

                    if encodings_file.exists():
                        try:
                             person_encodings = list(np.load(str(encodings_file), allow_pickle=True))
                             self.logger.debug(f"Loaded {len(person_encodings)} encodings for '{person_name}' from .npy file.")
                        except Exception as e_npy:
                             self.logger.error(f"Error loading numpy encodings for '{person_name}': {e_npy}")
                    else:
                         # Process image files if no .npy file
                         img_count = 0
                         for img_file in person_dir.glob('*.jpg'): person_encodings.extend(self._encode_image_file(img_file)); img_count+=1
                         for img_file in person_dir.glob('*.png'): person_encodings.extend(self._encode_image_file(img_file)); img_count+=1
                         # TODO: Add other image types if needed
                         self.logger.debug(f"Processed {img_count} images for '{person_name}', found {len(person_encodings)} face encodings.")
                         # Optionally save computed encodings back to .npy?
                         # if person_encodings:
                         #    np.save(str(encodings_file), np.array(person_encodings))
                             
                    if person_encodings:
                         loaded_encodings.extend(person_encodings)
                         loaded_names.extend([person_name] * len(person_encodings))
            
            with self._lock:
                self.known_face_encodings = loaded_encodings
                self.known_face_names = loaded_names
            self.logger.info(f"Finished loading. Total known encodings: {len(self.known_face_encodings)}")
            
        except Exception as e:
            self.logger.error(f"Error scanning known faces directory: {e}", exc_info=True)
            # Reset state on error
            with self._lock:
                 self.known_face_encodings = []
                 self.known_face_names = []
                 
    def _encode_image_file(self, image_path: Path) -> List[np.ndarray]:
         """Loads an image file and returns face encodings found."""
         try:
              image = face_recognition.load_image_file(str(image_path))
              # Find faces (use default HOG model for loading)
              face_locations = face_recognition.face_locations(image, model='hog')
              if not face_locations:
                   self.logger.warning(f"No faces found in image: {image_path.name}")
                   return []
              # Get encodings for all found faces
              encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
              return encodings
         except Exception as e:
              self.logger.error(f"Error encoding image file {image_path.name}: {e}")
              return []

    def start_learning_face(self, name: str):
        """Initiates the process of learning a new face."""
        if not name or not isinstance(name, str):
             self.logger.error("Invalid name provided for face learning.")
             return
             
        name = name.strip().replace(" ", "_") # Sanitize name
        if not name:
             self.logger.error("Name became empty after sanitization.")
             return
             
        with self._lock:
             if self._learning_face_active:
                  self.logger.warning("Face learning already in progress. Please wait or cancel.")
                  return
             
             self.logger.info(f"Starting face learning process for name: '{name}'")
             self._learning_face_active = True
             self._learning_candidate_name = name
             self._learning_samples_collected = 0
             self._learning_temp_encodings = []
             # Optionally, provide feedback (e.g., TTS prompt)
             if self.post_event: self.post_event("system.learning.started", {'name': name})

    def cancel_learning_face(self):
         """Cancels the current face learning process."""
         with self._lock:
              if not self._learning_face_active:
                   self.logger.info("No face learning process is active to cancel.")
                   return
                   
              self.logger.info("Cancelling face learning process.")
              self._learning_face_active = False
              self._learning_candidate_name = None
              self._learning_samples_collected = 0
              self._learning_temp_encodings = []
              if self.post_event: self.post_event("system.learning.cancelled", {})

    def _collect_learning_sample(self, face_encoding: np.ndarray):
        """Collects a single face encoding sample during learning."""
        # This method assumes lock is already held or race condition is acceptable for learning phase
        if not self._learning_face_active or not self._learning_candidate_name:
            return # Not in learning mode
            
        self._learning_temp_encodings.append(face_encoding)
        self._learning_samples_collected += 1
        self.logger.info(f"Collected learning sample {self._learning_samples_collected}/{FACE_LEARNING_SAMPLE_COUNT} for '{self._learning_candidate_name}'")
        
        # Check if enough samples collected
        if self._learning_samples_collected >= FACE_LEARNING_SAMPLE_COUNT:
             self._save_learned_face()
        # else:
             # Optionally provide feedback on sample collection
             # if self.post_event: self.post_event("system.learning.sample_taken", ...)

    def _save_learned_face(self):
        """Saves the collected face encodings and updates known faces."""
        # This method assumes lock is already held or called right after sample collection
        if not self._learning_candidate_name or not self._learning_temp_encodings:
            self.logger.error("Attempted to save learned face with no name or encodings.")
            self._learning_face_active = False # Exit learning mode
            return
            
        name = self._learning_candidate_name
        encodings_to_save = np.array(self._learning_temp_encodings)
        
        self.logger.info(f"Saving {len(encodings_to_save)} learned face encodings for '{name}'")
        
        try:
            person_dir = self.known_faces_dir / name
            person_dir.mkdir(parents=True, exist_ok=True)
            encodings_file = person_dir / "encodings.npy"
            
            # Save encodings to .npy file
            np.save(str(encodings_file), encodings_to_save)
            
            # --- Update known faces in memory (Requires Lock) ---
            with self._lock:
                self.known_face_encodings.extend(list(encodings_to_save))
                self.known_face_names.extend([name] * len(encodings_to_save))
                self.logger.info(f"In-memory known faces updated. Total encodings: {len(self.known_face_encodings)}")
            # -------------------------------------------------------
                
            self.logger.info(f"Successfully saved face encodings for '{name}' to {encodings_file}")
            if self.post_event: self.post_event(TOPICS.FACE_LEARNED, {'name': name})

        except Exception as e:
            self.logger.error(f"Error saving learned face for '{name}': {e}", exc_info=True)
            # Don't update in-memory list if save failed?
            if self.post_event: self.post_event("system.error", {'message': f"Failed to save face {name}"}) 
        
        finally:
            # --- Exit Learning Mode (Requires Lock) ---
            with self._lock:
                 self._learning_face_active = False
                 self._learning_candidate_name = None
                 self._learning_samples_collected = 0
                 self._learning_temp_encodings = []
            # ----------------------------------------- 