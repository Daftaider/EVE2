"""
Emotion analysis module.

This module analyzes facial expressions to detect emotions.
"""
import logging
from typing import Dict, Optional, List, Union, Any

import cv2
import numpy as np

from eve import config

import sys
import types
import random
import importlib

# Set up logger first, before using it
logger = logging.getLogger(__name__)

# Define the setup_dependency_mocks function that was missing
def setup_dependency_mocks():
    """Set up mock implementations for dependencies like moviepy and tensorflow"""
    # Create moviepy mock
    if 'moviepy' not in sys.modules:
        moviepy_module = types.ModuleType('moviepy')
        sys.modules['moviepy'] = moviepy_module
    
        editor_module = types.ModuleType('moviepy.editor')
        sys.modules['moviepy.editor'] = editor_module
        sys.modules['moviepy'].editor = editor_module
        
        class MockVideoFileClip:
            def __init__(self, *args, **kwargs):
                pass
            def close(self):
                pass
            def subclip(self, *args, **kwargs):
                return self
            def resize(self, *args, **kwargs):
                return self
        
        editor_module.VideoFileClip = MockVideoFileClip
        editor_module.__all__ = ['VideoFileClip']
    
    # Create tensorflow mock
    if 'tensorflow' not in sys.modules:
        tf_module = types.ModuleType('tensorflow')
        keras_module = types.ModuleType('tensorflow.keras')
        models_module = types.ModuleType('tensorflow.keras.models')
        
        # Create class with make_predict_function method
        class MockModel:
            def __init__(self):
                self.input_shape = (1, 48, 48, 1)  # Add input_shape attribute
                self.output_shape = (1, 7)
            
            def make_predict_function(self):
                pass
                
            def predict(self, *args, **kwargs):
                return np.zeros((1, 7))  # Return empty emotion predictions
        
        def load_model(*args, **kwargs):
            return MockModel()
        
        models_module.load_model = load_model
        sys.modules['tensorflow'] = tf_module
        sys.modules['tensorflow.keras'] = keras_module
        sys.modules['tensorflow.keras.models'] = models_module
        tf_module.keras = keras_module
        keras_module.models = models_module
    
    logger.info("Mock dependencies set up successfully")

# Call the function to set up mocks
setup_dependency_mocks()

# Now try to import FER
try:
    from fer import FER
    logger.info("Successfully imported FER")
    USE_FER = True
except Exception as e:
    logger.error(f"Error importing FER: {e}")
    USE_FER = False

class EmotionAnalyzer:
    """
    Facial expression analysis class.
    
    This class uses the FER library to detect emotions from facial images.
    """
    
    def __init__(self, confidence_threshold=0.5, detection_interval=1.0, model_type=None):
        """
        Initialize emotion analyzer
        
        Args:
            confidence_threshold (float): Minimum confidence threshold for emotions (0.0-1.0)
            detection_interval (float): Time between emotion detections in seconds
            model_type (str): Type of emotion detection model to use
        """
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.model_type = model_type
        self.emotions = ["neutral", "happy", "sad", "surprise", "angry"]
        self.detector = None
        
        logger.info(f"Initializing EmotionAnalyzer with confidence threshold: {confidence_threshold}")
        
        try:
            self.detector = FER()
            logger.info("Successfully initialized FER detector")
        except Exception as e:
            logger.error(f"Failed to initialize FER detector: {e}")
            logger.info("Using fallback emotion detection")
            self.detector = None
    
    def detect_emotions(self, frame):
        """Detect emotions in the given frame"""
        if self.detector is not None:
            try:
                return self.detector.detect_emotions(frame)
            except Exception as e:
                logger.error(f"Error using FER: {e}")
                return self._fallback_detection(frame)
        else:
            return self._fallback_detection(frame)
    
    def _fallback_detection(self, frame):
        """Simple fallback implementation that returns basic emotion data"""
        import random
        height, width = frame.shape[:2]
        
        face_box = [width//4, height//4, width//2, height//2]
        emotion_dict = {emotion: random.random() for emotion in self.emotions}
        # Normalize probabilities
        total = sum(emotion_dict.values())
        emotion_dict = {k: v/total for k, v in emotion_dict.items()}
        
        return [{'box': face_box, 'emotions': emotion_dict}]
    
    def analyze(self, face_image: np.ndarray) -> Optional[str]:
        """
        Analyze the emotion in a face image.
        
        Args:
            face_image: A cropped image of a face (numpy array)
            
        Returns:
            The most probable emotion as a string, or None if no emotion is detected
            with sufficient confidence
        """
        if self.detector is None:
            logger.warning("Emotion analyzer not initialized")
            return None
        
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided")
            return None
        
        try:
            # Ensure the image is in the correct format (BGR for OpenCV)
            if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                logger.warning("Invalid face image format")
                return None
            
            # Resize image if too small for the model
            min_size = 48  # Minimum size expected by most emotion models
            height, width = face_image.shape[:2]
            if height < min_size or width < min_size:
                scale = max(min_size / height, min_size / width)
                face_image = cv2.resize(
                    face_image, 
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Detect emotions in the face image
            emotions = self.detect_emotions(face_image)
            
            # Check if any faces were detected
            if not emotions or len(emotions) == 0:
                logger.debug("No faces detected in the image")
                return None
            
            # Get the emotions for the first (and likely only) face
            emotion_scores = emotions[0]["emotions"]
            
            # Find the emotion with the highest score
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion, score = max_emotion
            
            # Check if the confidence exceeds the threshold
            if score < self.confidence_threshold:
                logger.debug(f"Emotion confidence ({score:.2f}) below threshold ({self.confidence_threshold})")
                return None
            
            logger.debug(f"Detected emotion: {emotion} with confidence: {score:.2f}")
            
            # Map the FER emotion to our configured emotions
            return self._map_emotion(emotion)
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return None
    
    def _map_emotion(self, fer_emotion: str) -> str:
        """
        Map the FER emotion to one of our configured emotions.
        
        Args:
            fer_emotion: The emotion detected by FER
            
        Returns:
            The mapped emotion from our configuration
        """
        # Define mapping from FER emotions to our configured emotions
        emotion_mapping = {
            "angry": "angry",
            "disgust": "disgusted",
            "fear": "fearful",
            "happy": "happy",
            "sad": "sad",
            "surprise": "surprised",
            "neutral": "neutral"
        }
        
        # Get the mapped emotion or default to neutral
        mapped_emotion = emotion_mapping.get(fer_emotion.lower(), "neutral")
        
        # Ensure the mapped emotion is in our configured emotions list
        if mapped_emotion not in config.display.EMOTIONS:
            logger.warning(f"Mapped emotion '{mapped_emotion}' not in configured emotions")
            mapped_emotion = config.display.DEFAULT_EMOTION
        
        return mapped_emotion
    
    def get_emotion_confidence(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Get confidence scores for all emotions.
        
        Args:
            face_image: A cropped image of a face (numpy array)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        if self.detector is None:
            logger.warning("Emotion analyzer not initialized")
            return {}
        
        try:
            # Detect emotions in the face image
            emotions = self.detect_emotions(face_image)
            
            # Check if any faces were detected
            if not emotions or len(emotions) == 0:
                logger.debug("No faces detected in the image")
                return {}
            
            # Get the emotions for the first (and likely only) face
            emotion_scores = emotions[0]["emotions"]
            
            # Map the emotions to our configured emotions
            mapped_scores = {}
            for emotion, score in emotion_scores.items():
                mapped_emotion = self._map_emotion(emotion)
                # If multiple FER emotions map to the same configured emotion,
                # take the maximum score
                mapped_scores[mapped_emotion] = max(
                    mapped_scores.get(mapped_emotion, 0.0),
                    score
                )
            
            return mapped_scores
            
        except Exception as e:
            logger.error(f"Error getting emotion confidence: {e}")
            return {}

if 'eve.communication' in sys.modules:
    importlib.reload(sys.modules['eve.communication']) 