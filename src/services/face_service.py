"""
Face recognition service for detecting and recognizing faces.
"""
import logging
import cv2
import numpy as np
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yaml

logger = logging.getLogger(__name__)

class FaceService:
    """Face recognition service for detecting and recognizing faces."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize face recognition service."""
        self.config = self._load_config(config_path)
        self.face_cascade = None
        self.face_recognizer = None
        self.db_conn = None
        self.known_faces: Dict[str, np.ndarray] = {}
        
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
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Initialize face recognition
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Initialize database
            db_path = self.config.get('face_recognition', {}).get('database_path', 'data/faces.db')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db_conn = sqlite3.connect(db_path)
            self._init_database()
            
            # Load known faces
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
            return []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
            
    def recognize_face(self, face_roi: np.ndarray) -> Optional[str]:
        """Recognize a face from the region of interest."""
        if self.face_recognizer is None:
            return None
            
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            label, confidence = self.face_recognizer.predict(gray)
            
            if confidence < self.config.get('face_recognition', {}).get('confidence_threshold', 0.6):
                return list(self.known_faces.keys())[label]
            return None
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return None
            
    def add_face(self, name: str, face_roi: np.ndarray) -> bool:
        """Add a new face to the database."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Add to recognizer
            self.face_recognizer.update([gray], [len(self.known_faces)])
            
            # Add to database
            cursor = self.db_conn.cursor()
            cursor.execute(
                'INSERT INTO faces (name, embedding) VALUES (?, ?)',
                (name, gray.tobytes())
            )
            self.db_conn.commit()
            
            # Add to known faces
            self.known_faces[name] = gray
            
            logger.info(f"Added new face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
            
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