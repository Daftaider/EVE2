import cv2
import numpy as np
import threading
import queue
import logging
import os
from pathlib import Path
import time
import face_recognition

class VisionDisplay:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Force X11 display backend
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        # Disable Qt logging
        os.environ['QT_LOGGING_RULES'] = '*=false'
        
        # Initialize display settings
        self.window_name = "EVE Vision"
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False
        self.display_thread = None
        
        # Get vision settings from config with defaults
        vision_config = getattr(config.VISION, 'VISION', {})
        
        # Face recognition settings
        default_faces_dir = os.path.join('data', 'known_faces')
        self.known_faces_dir = Path(getattr(vision_config, 'KNOWN_FACES_DIR', default_faces_dir))
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Recognition parameters
        self.min_face_images = getattr(vision_config, 'MIN_FACE_IMAGES', 5)
        self.recognition_threshold = getattr(vision_config, 'RECOGNITION_THRESHOLD', 0.6)
        
        # State tracking
        self.learning_face = False
        self.current_learning_name = None
        self.learning_faces_count = 0
        self.temp_face_encodings = []
        
        # Load known faces
        self.load_known_faces()
        
        self.logger.info("Vision display initialized")

    def load_known_faces(self):
        """Load known face encodings from storage"""
        try:
            for person_dir in self.known_faces_dir.iterdir():
                if person_dir.is_dir():
                    name = person_dir.name
                    encodings_file = person_dir / "encodings.npy"
                    if encodings_file.exists():
                        encodings = np.load(str(encodings_file))
                        self.known_face_encodings.extend(encodings)
                        self.known_face_names.extend([name] * len(encodings))
                        self.logger.info(f"Loaded {len(encodings)} encodings for {name}")
        except Exception as e:
            self.logger.error(f"Error loading known faces: {e}")

    def start(self):
        """Start the display window"""
        if not self.running:
            try:
                # Create window with specific flags for better compatibility
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                # Set window properties
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                self.running = True
                self.display_thread = threading.Thread(target=self._display_loop)
                self.display_thread.daemon = True
                self.display_thread.start()
                self.logger.info("Vision display started")
            except Exception as e:
                self.logger.error(f"Error starting vision display: {e}")
                self.running = False
                raise

    def stop(self):
        """Stop the display window"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        self.logger.info("Vision display stopped")

    def _display_loop(self):
        """Main display loop"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    # Ensure frame is in BGR format
                    if len(frame.shape) == 2:  # If grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    try:
                        cv2.imshow(self.window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            self.running = False
                    except Exception as e:
                        self.logger.error(f"Error displaying frame: {e}")
                        time.sleep(0.1)
                        
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)

    def process_frame(self, frame):
        """Process a frame for face detection and recognition"""
        try:
            # Resize frame for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Check if face is known
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=self.recognition_threshold
                )
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    # Draw box and name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                else:
                    # Unknown face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    
                    if not self.learning_face:
                        # Trigger the learning process
                        return "unknown_face", face_encoding

                # If we're learning a face, save the encoding
                if self.learning_face and self.current_learning_name:
                    self.temp_face_encodings.append(face_encoding)
                    self.learning_faces_count += 1
                    
                    if self.learning_faces_count >= self.min_face_images:
                        self._save_learned_face()
                        return "learning_complete", None
                    else:
                        return "continue_learning", None

            # Add the processed frame to the display queue
            if self.frame_queue.qsize() < 10:
                self.frame_queue.put(frame)

            return None, None

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None, None

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