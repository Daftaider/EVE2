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
        
        # Create camera if not provided
        if self.camera_index is None:
            from eve.vision.camera import Camera
            try:
                self.camera = Camera(camera_index=config.hardware.CAMERA_INDEX)
                logger.info("Camera initialized for face detection")
            except Exception as e:
                logger.error(f"Error initializing camera: {e}")
                self.camera = MockCamera()  # Use a mock camera as fallback
        else:
            self.camera = None
    
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
        detection_interval = getattr(config.vision, "FACE_DETECTION_INTERVAL_SEC", 0.1)
        
        # Initialize camera if needed
        if not hasattr(self, "camera") or self.camera is None:
            from eve.vision.camera import Camera
            try:
                self.camera = Camera(camera_index=getattr(config.hardware, "CAMERA_INDEX", 0))
            except Exception as e:
                logger.error(f"Error creating camera: {e}")
                self.camera = MockCamera()
        
        # Reset frame counter for monitoring
        frame_count = 0
        empty_frame_count = 0
        
        while self.running:
            try:
                # Get a frame from the camera
                frame = self.camera.get_frame()
                frame_count += 1
                
                # Check if frame is valid
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    empty_frame_count += 1
                    if empty_frame_count % 10 == 0:  # Log only every 10th empty frame
                        logger.warning(f"Invalid or empty frame received ({empty_frame_count}/{frame_count} frames empty)")
                    
                    # Use a mock frame as fallback
                    frame = self._create_mock_frame()
                
                # Detect faces
                faces = self._detect_faces(frame)
                
                # Update faces
                self.faces = faces
                
                # Store current frame
                self.frame = frame
                
                # Wait for next detection interval
                time.sleep(detection_interval)
            
            except Exception as e:
                logger.error(f"Error in face detection loop: {e}")
                time.sleep(detection_interval)
        
        logger.info("Face detection loop stopped")
    
    def _create_mock_frame(self):
        """Create a mock frame when camera frame is invalid"""
        frame_size = getattr(config.vision, "CAMERA_RESOLUTION", (640, 480))
        if not isinstance(frame_size, tuple):
            frame_size = (640, 480)
        
        # Create blank frame
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Add text indicating this is a mock frame
        cv2.putText(frame, "Mock Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
                
        return frame
    
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

class MockCamera:
    """Mock camera implementation for testing"""
    def __init__(self):
        self.frame_count = 0
        logger.info("Initialized mock camera for face detection")
        
    def get_frame(self):
        """Generate a mock frame for testing"""
        self.frame_count += 1
        
        # Create a valid frame with test pattern
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving circle as a "face"
        center_x = width // 2 + int(100 * np.sin(self.frame_count / 30))
        center_y = height // 2 + int(50 * np.cos(self.frame_count / 20))
        
        # Draw face
        cv2.circle(frame, (center_x, center_y), 100, (200, 200, 200), -1)
        
        # Draw eyes
        cv2.circle(frame, (center_x - 30, center_y - 20), 20, (255, 255, 255), -1)
        cv2.circle(frame, (center_x + 30, center_y - 20), 20, (255, 255, 255), -1)
        
        # Draw mouth
        cv2.ellipse(frame, (center_x, center_y + 30), (50, 20), 0, 0, 180, (255, 255, 255), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Simulate processing delay
        time.sleep(0.05)
        
        return frame
        
    def release(self):
        """Mock release method"""
        pass 