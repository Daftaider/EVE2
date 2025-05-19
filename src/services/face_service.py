"""
Face recognition service for detecting and recognizing faces.
"""
import logging
import cv2
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Literal
import yaml
import hailo_platform as hpf

logger = logging.getLogger(__name__)

class FaceService:
    """Face recognition service for detecting and recognizing faces, with support for user enrolment and multiple inference backends."""
    
    def __init__(self, config_path: str = "config/settings.yaml", backend: Literal["cpu", "hailo"] = "cpu"):
        """
        Initialize face recognition service.
        Args:
            config_path: Path to YAML config
            backend: 'cpu' (OpenCV) or 'hailo' (Hailo-8L, not yet implemented)
        """
        self.config = self._load_config(config_path)
        self.backend = backend
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.face_recognizer = None
        self.db_conn = None
        self.known_faces: Dict[str, np.ndarray] = {}
        self.known_faces_dir = Path(self.config.get('face_recognition', {}).get('known_faces_dir', 'data/known_faces'))
        self.face_database_file = self.known_faces_dir / self.config.get('face_recognition', {}).get('database_file', 'face_encodings.pkl')
        
        face_detection_config = self.config.get('face_detection', {})
        logger.debug(f"FaceService __init__: Raw face_detection_config from settings: {face_detection_config}")
        cascade_path_str = face_detection_config.get('model_path', 'config/haarcascade_frontalface_default.xml')
        logger.debug(f"FaceService __init__: cascade_path_str after get: {cascade_path_str}")
        
        # Try to load the cascade classifier
        # Ensure the path is absolute or correctly relative to the project root
        project_root = Path(__file__).resolve().parent.parent.parent 
        cascade_file_path = project_root / cascade_path_str

        logger.info(f"Attempting to load Haar cascade from: {cascade_file_path}")
        if cascade_file_path.exists():
            self.face_cascade = cv2.CascadeClassifier(str(cascade_file_path))
            if self.face_cascade.empty():
                logger.error(f"Failed to load Haar cascade from {cascade_file_path}. The CascadeClassifier is empty. Check the file integrity and path.")
                self.face_cascade = None # Ensure it's None if loading failed
            else:
                logger.info(f"Haar cascade loaded successfully from {cascade_file_path}.")
        else:
            logger.error(f"Haar cascade file not found at {cascade_file_path}. Face detection will not work.")
            self.face_cascade = None

        self.known_face_encodings: List[np.ndarray] = []
        logger.info(f"FaceService initialized with backend: {backend}")
        # TODO: Add Hailo-8L backend support
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def start(self) -> bool:
        """Start the face recognition service."""
        try:
            # Ensure face_cascade is valid after __init__ attempt.
            if self.face_cascade is None or self.face_cascade.empty():
                logger.error("Face cascade is not properly loaded. Face detection will not work. Check __init__ logs and cascade_path in settings.yaml.")
                # Optionally, you could prevent the service from starting fully or raise an error.
                # For now, it will continue, but detect_faces will return empty.

            self.hailo_ready = False
            self.hailo_context = None
            self.hailo_input_shape = None
            self.hailo_output_shape = None
            self.hailo_input_name = None
            self.hailo_output_name = None
            self.hailo_infer_pipeline = None
            self.hailo_vdevice = None
            self.hailo_network_group = None
            self.hailo_network_group_params = None
            if self.backend == "hailo":
                try:
                    hef_path = self.config.get('face_recognition', {}).get('hef_path', 'src/models/face/model.hef')
                    self.hailo_hef = hpf.HEF(hef_path)
                    self.hailo_vdevice = hpf.VDevice()
                    configure_params = hpf.ConfigureParams.create_from_hef(self.hailo_hef, interface=hpf.HailoStreamInterface.PCIe)
                    self.hailo_network_group = self.hailo_vdevice.configure(self.hailo_hef, configure_params)[0]
                    self.hailo_network_group_params = self.hailo_network_group.create_params()
                    input_vstream_info = self.hailo_hef.get_input_vstream_infos()[0]
                    output_vstream_info = self.hailo_hef.get_output_vstream_infos()[0]
                    self.hailo_input_name = input_vstream_info.name
                    self.hailo_output_name = output_vstream_info.name
                    self.hailo_input_shape = input_vstream_info.shape
                    self.hailo_output_shape = output_vstream_info.shape
                    input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(self.hailo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
                    output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.hailo_network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
                    self.hailo_infer_pipeline = hpf.InferVStreams(self.hailo_network_group, input_vstreams_params, output_vstreams_params)
                    self.hailo_ready = True
                    logger.info(f"Hailo-8L initialized: input shape {self.hailo_input_shape}, output shape {self.hailo_output_shape}")
                except Exception as e:
                    logger.error(f"Failed to initialize Hailo-8L: {e}")
                    self.hailo_ready = False
            if self.backend == "cpu":
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif self.backend == "hailo" and not self.hailo_ready:
                logger.warning("Hailo-8L backend not ready. Falling back to CPU.")
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            elif self.backend == "hailo" and self.hailo_ready:
                self.face_recognizer = None  # Not used
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            db_path = self.config.get('face_recognition', {}).get('database_path', 'data/faces.db')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_conn = sqlite3.connect(db_path)
            self._init_database()
            self._load_known_faces()
            logger.info("Face recognition service started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting face recognition service: {e}")
            return False
            
    def _init_database(self) -> None:
        """Initialize the face database."""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        self.db_conn.commit()
        
    def _load_known_faces(self) -> None:
        """Load known faces from the database."""
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT name, embedding FROM faces')
        for name, embedding_blob in cursor.fetchall():
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            self.known_faces[name] = embedding
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame."""
        if self.face_cascade is None:
            logger.warning("Face cascade is not loaded. Cannot detect faces.")
            return []
        if frame is None:
            logger.warning("Received a None frame. Cannot detect faces.")
            return []
        if frame.size == 0:
            logger.warning("Received an empty frame (size 0). Cannot detect faces.")
            return []
            
        try:
            logger.debug(f"FaceService.detect_faces: Input frame shape: {frame.shape}, dtype: {frame.dtype}, min_val: {frame.min()}, max_val: {frame.max()}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            logger.debug(f"FaceService.detect_faces: Grayscale frame shape: {gray.shape}, dtype: {gray.dtype}, min_val: {gray.min()}, max_val: {gray.max()}")

            # Get detection parameters from config, with defaults
            detection_params = self.config.get('face_detection', {})
            scale_factor = detection_params.get('scale_factor', 1.1)
            min_neighbors = detection_params.get('min_neighbors', 5)
            min_size_tuple = detection_params.get('min_size', [30, 30])
            min_size = tuple(min_size_tuple) # Ensure it's a tuple

            logger.debug(f"FaceService.detect_faces: Using detectMultiScale with scaleFactor={scale_factor}, minNeighbors={min_neighbors}, minSize={min_size}")

            faces_output = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            logger.debug(f"FaceService.detect_faces: detectMultiScale raw output: {faces_output}, type: {type(faces_output)}")

            if isinstance(faces_output, np.ndarray):
                return faces_output.tolist()
            else:
                return []
        except cv2.error as cv2_err: # Catch OpenCV specific errors
            logger.error(f"OpenCV error during face detection: {cv2_err}. Frame shape: {frame.shape}, dtype: {frame.dtype}")
            return []
        except Exception as e:
            logger.error(f"Generic error detecting faces: {e}. Frame shape: {frame.shape}, dtype: {frame.dtype}")
            return []
        
    def extract_embedding(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Extract a face embedding from the ROI using the selected backend.
        Args:
            face_roi: Face region of interest (BGR image)
        Returns:
            Embedding as a 1D np.ndarray (float32)
        """
        if self.backend == "cpu":
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            return gray.flatten().astype(np.float32)
        elif self.backend == "hailo" and self.hailo_ready:
            try:
                # Preprocess face_roi to match Hailo input shape
                resized = cv2.resize(face_roi, (self.hailo_input_shape[2], self.hailo_input_shape[1]))
                input_data = np.expand_dims(resized.astype(np.float32), axis=0)
                input_dict = {self.hailo_input_name: input_data}
                with self.hailo_network_group.activate(self.hailo_network_group_params):
                    with self.hailo_infer_pipeline as infer_pipeline:
                        results = infer_pipeline.infer(input_dict)
                        output_data = results[self.hailo_output_name]
                        embedding = output_data.flatten().astype(np.float32)
                        return embedding
            except Exception as e:
                logger.error(f"Hailo-8L inference failed: {e}")
                # Fallback to dummy embedding
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                return gray.flatten().astype(np.float32)
        elif self.backend == "hailo":
            logger.warning("Hailo-8L backend not ready. Returning dummy embedding.")
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            return gray.flatten().astype(np.float32)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def enrol_user(self, name: str, face_roi: np.ndarray) -> bool:
        """
        Enrol a new user by capturing their face embedding and storing it in the database.
        Args:
            name: User's name
            face_roi: Face region of interest (BGR image)
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding = self.extract_embedding(face_roi)
            cursor = self.db_conn.cursor()
            cursor.execute(
                'INSERT INTO faces (name, embedding) VALUES (?, ?)',
                (name, embedding.tobytes())
            )
            self.db_conn.commit()
            self.known_faces[name] = embedding
            logger.info(f"Enrolled new user: {name}")
            return True
        except Exception as e:
            logger.error(f"Error enrolling user: {e}")
            return False

    def recognize_face(self, face_roi: np.ndarray) -> Optional[str]:
        """
        Recognize a face from the region of interest by comparing embeddings.
        Args:
            face_roi: Face region of interest (BGR image)
        Returns:
            Name of recognized user, or None if not recognized
        """
        try:
            embedding = self.extract_embedding(face_roi)
            min_dist = float('inf')
            best_match = None
            for name, known_emb in self.known_faces.items():
                dist = np.linalg.norm(embedding - known_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name
            threshold = self.config.get('face_recognition', {}).get('embedding_distance_threshold', 1000.0)
            if min_dist < threshold:
                return best_match
            return None
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None
        
    def stop(self) -> None:
        """Stop the face recognition service."""
        if self.db_conn:
            self.db_conn.close()
        logger.info("Face recognition service stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 