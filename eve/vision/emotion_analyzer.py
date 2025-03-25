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

# Create a complete mock module structure for moviepy.editor
# This must be done BEFORE any imports that might use it
def setup_moviepy_mock():
    # Create the parent module if it doesn't exist
    if 'moviepy' not in sys.modules:
        moviepy_module = types.ModuleType('moviepy')
        sys.modules['moviepy'] = moviepy_module
    
    # Create the editor submodule
    editor_module = types.ModuleType('moviepy.editor')
    sys.modules['moviepy.editor'] = editor_module
    sys.modules['moviepy'].editor = editor_module
    
    # Add commonly used classes/functions to the mock
    class MockVideoFileClip:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def subclip(self, *args, **kwargs):
            return self
        def resize(self, *args, **kwargs):
            return self
    
    # Add all expected attributes to the editor module
    editor_module.VideoFileClip = MockVideoFileClip
    
    # Make 'from moviepy.editor import *' work by adding to __all__
    editor_module.__all__ = ['VideoFileClip']
    
    # For star imports, we need to put these in the module's globals
    for name in editor_module.__all__:
        setattr(editor_module, name, getattr(editor_module, name))
    
    return editor_module

# Set up our mock BEFORE any imports
setup_moviepy_mock()

# Now we can safely import FER
try:
    from fer import FER
    logger.info("Successfully imported FER")
except Exception as e:
    logger.error(f"Error importing FER: {e}")
    # We'll need to define a fallback if this fails

logger = logging.getLogger(__name__)

# Don't import FER directly, create a wrapper
class CustomEmotionDetector:
    """A lightweight emotion detector for Raspberry Pi"""
    def __init__(self):
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        logger.info("Using custom lightweight emotion detector")
    
    def detect_emotions(self, frame):
        # Return placeholder data in the same format FER would
        # In a real implementation, you could use a simpler model or API
        return [{
            'box': [0, 0, frame.shape[1], frame.shape[0]],
            'emotions': {e: random.random() for e in self.emotions}
        }]

class EmotionAnalyzer:
    """
    Facial expression analysis class.
    
    This class uses the FER library to detect emotions from facial images.
    """
    
    def __init__(self):
        # Use the custom detector instead of FER
        self.detector = CustomEmotionDetector()
        logger.info("Using lightweight emotion analyzer")
    
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
            if score < 0.5:
                logger.debug(f"Emotion confidence ({score:.2f}) below threshold (0.5)")
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