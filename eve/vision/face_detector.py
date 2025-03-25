"""
Face detection and recognition module.

This module handles camera input, face detection, and face recognition.
"""
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque

import cv2
import face_recognition
import numpy as np

from eve import config

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detection and recognition class.
    
    This class handles video capture, face detection, and optionally face
    recognition if enabled in the configuration.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        detection_model: str = "hog",
        known_faces_dir: Optional[Path] = None,
        recognition_enabled: bool = True,
        recognition_tolerance: float = 0.6
    ) -> None:
        """
        Initialize the face detector.
        
        Args:
            camera_index: Camera device index (default: 0)
            resolution: Camera resolution as (width, height) (default: (640, 480))
            fps: Target frames per second (default: 30)
            detection_model: Model to use for face detection ("hog" or "cnn") (default: "hog")
            known_faces_dir: Directory containing known face images (default: None)
            recognition_enabled: Whether to enable face recognition (default: True)
            recognition_tolerance: Tolerance for face recognition (lower is more strict) (default: 0.6)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.detection_model = detection_model
        self.known_faces_dir = known_faces_dir
        self.recognition_enabled = recognition_enabled
        self.recognition_tolerance = recognition_tolerance
        
        # State variables
        self.running = False
        self.frame_queue: Deque[Tuple[np.ndarray, List[Dict[str, Any]]]] = deque(maxlen=10)
        self.frame_lock = threading.Lock()
        self.last_detection_time = 0
        self.video_capture = None
        
        # Face recognition data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces if directory is provided and recognition is enabled
        if self.recognition_enabled and self.known_faces_dir is not None:
            self._load_known_faces()
    
    def _load_known_faces(self) -> None:
        """Load known faces from the specified directory."""
        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"Known faces directory does not exist: {self.known_faces_dir}")
            return
        
        logger.info(f"Loading known faces from {self.known_faces_dir}")
        
        try:
            # Iterate through each image file in the directory
            for filename in os.listdir(self.known_faces_dir):
                # Skip non-image files
                if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    continue
                
                # Extract the person's name from the filename (assuming format: name.jpg)
                name = os.path.splitext(filename)[0]
                
                # Load the image and get the face encoding
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Try to find a face in the image
                face_encodings = face_recognition.face_encodings(image)
                
                if not face_encodings:
                    logger.warning(f"No face found in image: {filename}")
                    continue
                
                # Take the first face encoding (assuming one face per image)
                face_encoding = face_encodings[0]
                
                # Add the encoding and name to our lists
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
            
            logger.info(f"Loaded {len(self.known_face_names)} known faces")
        
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
    
    def start(self) -> None:
        """Start the face detector."""
        if self.running:
            logger.warning("Face detector is already running")
            return
        
        logger.info("Starting face detector")
        
        try:
            # Initialize video capture
            self.video_capture = cv2.VideoCapture(self.camera_index)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if video capture was successfully initialized
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Failed to open camera at index {self.camera_index}")
            
            # Start the detection thread
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            logger.info("Face detector started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start face detector: {e}")
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self.running = False
    
    def stop(self) -> None:
        """Stop the face detector."""
        if not self.running:
            logger.warning("Face detector is not running")
            return
        
        logger.info("Stopping face detector")
        
        # Signal the detection thread to stop
        self.running = False
        
        # Wait for the detection thread to finish
        if hasattr(self, 'detection_thread') and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Release video capture resources
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        
        logger.info("Face detector stopped successfully")
    
    def _detection_loop(self) -> None:
        """Main detection loop running in a separate thread."""
        logger.info("Face detection loop started")
        
        # Calculate frame interval based on target FPS
        frame_interval = 1.0 / self.fps
        
        # Define detection interval based on config
        detection_interval = config.vision.FACE_DETECTION_INTERVAL_SEC
        
        while self.running:
            loop_start_time = time.time()
            
            try:
                # Capture a frame
                ret, frame = self.video_capture.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Detect faces at regular intervals
                current_time = time.time()
                if current_time - self.last_detection_time >= detection_interval:
                    # Detect faces in the current frame
                    faces = self._detect_faces(frame)
                    self.last_detection_time = current_time
                else:
                    # Reuse the last detected faces
                    faces = [] if not self.frame_queue else self.frame_queue[-1][1]
                
                # Add the frame and faces to the queue
                with self.frame_lock:
                    self.frame_queue.append((frame, faces))
                
                # Calculate time elapsed and sleep if necessary to maintain target FPS
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in face detection loop: {e}")
                time.sleep(0.1)
        
        logger.info("Face detection loop stopped")
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            A list of dictionaries with face information:
            [
                {
                    'location': (top, right, bottom, left),
                    'id': 'person_name' or 'unknown',
                    'confidence': recognition_confidence
                },
                ...
            ]
        """
        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_model)
        
        # If no faces are found, return empty list
        if not face_locations:
            return []
        
        # Initialize results list
        results = []
        
        # Get face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each detected face
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # Default to unknown face
            face_id = "unknown"
            confidence = 0.0
            
            # If face recognition is enabled and we have known faces, try to recognize the face
            if self.recognition_enabled and self.known_face_encodings:
                # Compare face with known faces
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    # Get the best match if it's within tolerance
                    best_match_index = np.argmin(face_distances)
                    confidence = 1.0 - face_distances[best_match_index]
                    
                    if confidence >= 1.0 - self.recognition_tolerance:
                        face_id = self.known_face_names[best_match_index]
            
            # Add face information to results
            results.append({
                'location': face_location,
                'id': face_id,
                'confidence': float(confidence)
            })
        
        return results
    
    def has_new_frame(self) -> bool:
        """Check if there's a new frame available."""
        with self.frame_lock:
            return len(self.frame_queue) > 0
    
    def get_latest_frame(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Get the latest frame and detected faces.
        
        Returns:
            Tuple containing:
            - The video frame as a numpy array
            - A list of dictionaries with face information
        """
        with self.frame_lock:
            if not self.frame_queue:
                # Return empty frame and face list if queue is empty
                return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8), []
            
            # Get the latest frame and faces from the queue
            frame, faces = self.frame_queue.pop()
            
            # Clear the queue to avoid processing old frames
            self.frame_queue.clear()
            
            return frame, faces 