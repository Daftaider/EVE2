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
from eve.config.communication import TOPICS
from eve.vision.camera import Camera

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detection and recognition class.
    
    This class handles video capture, face detection, and optionally face
    recognition if enabled in the configuration.
    """
    
    def __init__(self, config=None, post_event_callback=None):
        """Initialize the face detector
        
        Args:
            config: Configuration dictionary or object
            post_event_callback: Callback function for posting events
        """
        self.logger = logging.getLogger(__name__)
        self.post_event = post_event_callback
        self.running = False
        self.detection_thread = None
        self.empty_frame_count = 0
        self.total_frame_count = 0
        
        # Store the configuration
        self.config = config or {}
        
        # Extract camera configuration (handle both dict and object)
        if isinstance(config, dict):
            camera_config = config
        else:
            # Extract from object
            camera_config = {
                'camera_index': getattr(config, 'CAMERA_INDEX', 0),
                'resolution': getattr(config, 'RESOLUTION', (640, 480)),
                'fps': getattr(config, 'FPS', 30)
            }
        
        # Set default detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)
        self.debug_mode = False
        
        # Update from config if available
        if config:
            if isinstance(config, dict):
                self.scale_factor = config.get('SCALE_FACTOR', self.scale_factor)
                self.min_neighbors = config.get('MIN_NEIGHBORS', self.min_neighbors)
                self.min_face_size = config.get('MIN_FACE_SIZE', self.min_face_size)
                self.debug_mode = config.get('DEBUG', self.debug_mode)
            else:
                self.scale_factor = getattr(config, 'SCALE_FACTOR', self.scale_factor)
                self.min_neighbors = getattr(config, 'MIN_NEIGHBORS', self.min_neighbors)
                self.min_face_size = getattr(config, 'MIN_FACE_SIZE', self.min_face_size)
                self.debug_mode = getattr(config, 'DEBUG', self.debug_mode)
        
        # Initialize camera with proper parameters
        self.camera = Camera(**camera_config)
        
        # Load face cascade classifier
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load face cascade classifier")
        except Exception as e:
            self.logger.error(f"Error loading face cascade: {e}")
            raise

    def _detect_faces(self, frame):
        """Detect faces in the frame"""
        if frame is None or frame.size == 0:
            return []
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []

    def _detection_loop(self):
        """Main face detection loop"""
        last_warning_time = 0
        warning_interval = 5.0  # Seconds between warnings
        
        while self.running:
            try:
                # Get frame from camera
                ret, frame = self.camera.read()
                self.total_frame_count += 1
                
                if not ret or frame is None or frame.size == 0:
                    self.empty_frame_count += 1
                    current_time = time.time()
                    
                    # Log warning periodically
                    if current_time - last_warning_time > warning_interval:
                        self.logger.warning(
                            f"Invalid or empty frame received ({self.empty_frame_count}/{self.total_frame_count} frames empty)")
                        last_warning_time = current_time
                    
                    time.sleep(0.1)  # Prevent tight loop
                    continue
                
                # Reset empty frame counter on valid frame
                self.empty_frame_count = 0
                
                # Detect faces
                faces = self._detect_faces(frame)
                
                # Post event if faces found
                if len(faces) > 0 and self.post_event:
                    self.post_event(TOPICS['FACE_DETECTED'], {
                        'count': len(faces),
                        'locations': faces.tolist(),
                        'timestamp': time.time()
                    })
                    
                # Add debug visualization if needed
                if self.debug_mode:
                    debug_frame = frame.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Store debug frame
                    self.current_debug_frame = debug_frame
                
                # Control detection rate
                time.sleep(1.0 / 30)  # Limit to 30 FPS
                
            except Exception as e:
                self.logger.error(f"Error in face detection loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop

    def start(self):
        """Start face detection"""
        if not self.running:
            self.running = True
            self.empty_frame_count = 0
            self.total_frame_count = 0
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            self.logger.info("Face detector started")

    def stop(self):
        """Stop face detection"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        if hasattr(self, 'camera'):
            self.camera.release()
        self.logger.info("Face detector stopped")

    def is_running(self):
        """Check if detector is running"""
        return self.running and self.detection_thread and self.detection_thread.is_alive()

    def get_debug_frame(self):
        """Get the latest debug frame with face detections marked"""
        return getattr(self, 'current_debug_frame', None)

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