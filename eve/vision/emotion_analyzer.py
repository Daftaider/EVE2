"""
Emotion analysis module.

This module analyzes facial expressions to detect emotions.
"""
import logging
from typing import Dict, Optional, List, Union, Any

import cv2
import numpy as np
from fer import FER

from eve import config

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """
    Facial expression analysis class.
    
    This class uses the FER library to detect emotions from facial images.
    """
    
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """
        Initialize the emotion analyzer.
        
        Args:
            confidence_threshold: Minimum confidence required for emotion detection (default: 0.5)
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize the FER detector
        logger.info("Initializing emotion analyzer")
        try:
            self.detector = FER(mtcnn=True)  # Use MTCNN for more accurate face detection
            logger.info("Emotion analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize emotion analyzer: {e}")
            self.detector = None
    
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
            emotions = self.detector.detect_emotions(face_image)
            
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
            emotions = self.detector.detect_emotions(face_image)
            
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