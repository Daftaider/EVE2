import cv2
import numpy as np
import threading
import queue
import logging
import os
from pathlib import Path
import time
import face_recognition
import signal
from eve.vision.camera_utils import CameraManager

class VisionDisplay:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Force X11 backend and disable Qt
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
        os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'  # Prefer GStreamer
        
        # Initialize flags
        self.running = False
        self.initialized = False
        
        # Initialize queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.display_thread = None
        self.capture_thread = None
        
        # Get vision settings
        vision_config = getattr(config.VISION, 'VISION', {})
        self.frame_width = getattr(vision_config, 'FRAME_WIDTH', 640)
        self.frame_height = getattr(vision_config, 'FRAME_HEIGHT', 480)
        self.fps = getattr(vision_config, 'FPS', 30)
        
        # Initialize camera
        self.camera = CameraManager(
            width=self.frame_width,
            height=self.frame_height,
            fps=self.fps
        )
        self.camera.initialize()
        
        # Face recognition settings
        default_faces_dir = os.path.join('data', 'known_faces')
        self.known_faces_dir = Path(getattr(vision_config, 'KNOWN_FACES_DIR', default_faces_dir))
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Recognition state
        self.known_face_encodings = []
        self.known_face_names = []
        self.learning_face = False
        self.current_learning_name = None
        self.learning_faces_count = 0
        self.temp_face_encodings = []
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Vision display initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _capture_loop(self):
        """Camera capture loop"""
        while self.running:
            try:
                ret, frame = self.camera.read_frame()
                if ret and frame is not None:
                    # Process frame for face detection
                    processed_frame = self.process_frame(frame)
                    
                    # Update display queue
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame, timeout=0.1)
                
                time.sleep(1.0 / self.fps)  # Control frame rate
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

    def start(self):
        """Start the vision system"""
        if self.running:
            return
        
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        self.logger.info("Vision system started")

    def stop(self):
        """Stop the vision system"""
        self.running = False
        
        # Stop threads
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        self.logger.info("Vision system stopped")

    def _display_loop(self):
        """Display loop"""
        window_created = False
        
        while self.running:
            try:
                if not window_created:
                    cv2.namedWindow('EVE Vision', cv2.WINDOW_NORMAL)
                    window_created = True
                
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if frame is not None:
                    cv2.imshow('EVE Vision', frame)
                    
                    # Process key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self.running = False
                        break
                    
            except Exception as e:
                self.logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)

    def process_frame(self, frame):
        """Process frame for face detection and recognition"""
        try:
            # Create a copy of the frame
            display_frame = frame.copy()
            
            # Convert to RGB for face_recognition library
            rgb_frame = frame[:, :, ::-1]
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Process each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Draw box around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Try to recognize face
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    if True in matches:
                        name = self.known_face_names[matches.index(True)]
                
                # Draw name
                cv2.putText(display_frame, name, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            return display_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame

    def load_known_faces(self):
        """Load known face encodings"""
        try:
            for person_dir in self.known_faces_dir.iterdir():
                if person_dir.is_dir():
                    encodings_file = person_dir / "encodings.npy"
                    if encodings_file.exists():
                        encodings = np.load(str(encodings_file))
                        self.known_face_encodings.extend(encodings)
                        self.known_face_names.extend([person_dir.name] * len(encodings))
            
            self.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
            
        except Exception as e:
            self.logger.error(f"Error loading known faces: {e}")

    def start_learning_face(self, name):
        """Start the face learning process"""
        self.learning_face = True
        self.current_learning_name = name
        self.learning_faces_count = 0
        self.temp_face_encodings = []
        self.logger.info(f"Started learning face for: {name}")

    def _save_learned_face(self):
        """Save the learned face encodings"""
        try:
            person_dir = self.known_faces_dir / self.current_learning_name
            person_dir.mkdir(exist_ok=True)
            
            # Save encodings
            encodings_file = person_dir / "encodings.npy"
            np.save(str(encodings_file), np.array(self.temp_face_encodings))
            
            # Update known faces
            self.known_face_encodings.extend(self.temp_face_encodings)
            self.known_face_names.extend([self.current_learning_name] * len(self.temp_face_encodings))
            
            self.learning_face = False
            self.current_learning_name = None
            self.temp_face_encodings = []
            self.learning_faces_count = 0
            
            self.logger.info(f"Saved face encodings for: {self.current_learning_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving learned face: {e}")

    def _init_display_fallback(self):
        """Initialize fallback display mode using basic window creation"""
        try:
            # Disable all GUI backends
            os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
            os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
            os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
            
            # Create a basic window
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.logger.info("Initialized fallback display mode")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback display: {e}")
            return False 