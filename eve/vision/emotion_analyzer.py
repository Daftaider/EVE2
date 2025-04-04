"""
Emotion analysis module.

Uses the FER (Face Emotion Recognition) library to detect emotions.
"""
import logging
from typing import Dict, Optional, List, Union, Any, Callable

import cv2
import numpy as np

# Use the main config object
from eve.config import SystemConfig, VisionConfig 
# Removed unused Emotion import from display config

# Removed dependency mocking setup and sys imports

logger = logging.getLogger(__name__)

# Attempt to import FER - let it fail naturally if deps aren't met
FER_AVAILABLE = False
FER = None
try:
    from fer import FER
    # Optionally check specific dependencies like tensorflow if FER doesn't fail gracefully
    # import tensorflow
    FER_AVAILABLE = True
    logger.info("Successfully imported FER library.")
except ImportError as e:
    logger.warning(f"Could not import FER library: {e}. Emotion analysis via FER disabled.")
except Exception as e:
    logger.error(f"Error during FER library import (check dependencies like TensorFlow/Keras?): {e}", exc_info=True)


class EmotionAnalyzer:
    """
    Analyzes facial images to detect emotions using the FER library (if available).
    Provides a fallback mechanism if FER is unavailable.
    """
    
    def __init__(self, config: SystemConfig, post_event_callback: Optional[Callable] = None):
        """
        Initialize emotion analyzer.
        
        Args:
            config: The main SystemConfig object.
            post_event_callback: Optional callback function to post events.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vision_config: VisionConfig = self.config.vision # Store vision sub-config
        self.post_event = post_event_callback # Store the callback

        # --- Configuration ---
        self.enabled = self.vision_config.emotion_detection_enabled
        self.model_name = self.vision_config.emotion_detection_model.lower()
        self.confidence_threshold = self.vision_config.emotion_confidence_threshold
        # Correctly access emotions via vision config
        self.configured_emotions = self.vision_config.emotions # List of emotions system uses/maps to
        self.display_config = config.display # Needed for mapping
        
        self.default_display_emotion = self.display_config.default_emotion
        
        self.detector: Optional[FER] = None
        self.fer_initialized: bool = False
        
        if not self.enabled:
            self.logger.info("Emotion analysis disabled in configuration.")
            return # Don't initialize detector if disabled
            
        logger.info(f"Initializing EmotionAnalyzer (FER Available: {FER_AVAILABLE}). Confidence threshold: {self.confidence_threshold}")
        
        if FER_AVAILABLE and FER is not None:
            try:
                # Initialize FER. mtcnn=True might be more accurate but adds dependency.
                # Default uses OpenCV Haar cascade for face detection within FER.
                self.detector = FER(mtcnn=False) 
                self.fer_initialized = True
                logger.info("Successfully initialized FER detector.")
            except Exception as e:
                self.logger.error(f"Failed to initialize FER detector instance: {e}", exc_info=True)
                self.detector = None # Ensure detector is None on failure
        else:
             self.logger.warning("FER library not available or failed import. Using fallback emotion detection.")

    def analyze(self, face_image: np.ndarray) -> Optional[str]:
        """
        Analyze the primary emotion in a single pre-cropped face image.
        
        Args:
            face_image: A cropped image of a face (numpy array, BGR format).
            
        Returns:
            The most probable emotion string (from configured list) if confidence threshold is met,
            otherwise None.
        """
        if not self.enabled:
            return None # Disabled
            
        if face_image is None or face_image.size == 0:
            # self.logger.debug("Empty face image provided to analyze.")
            return None

        # Use FER if available and initialized
        if self.detector and self.fer_initialized:
            try:
                # FER expects BGR format by default when using OpenCV cascade
                # Verify input format
                if len(face_image.shape) != 3 or face_image.shape[2] != 3:
                    self.logger.warning(f"Invalid face image format for FER: shape={face_image.shape}")
                    return None
                
                # Resize image if too small for the internal model
                # FER models typically need at least 48x48
                min_size = 48 
                height, width = face_image.shape[:2]
                if height < min_size or width < min_size:
                    scale = max(min_size / height, min_size / width)
                    face_image_resized = cv2.resize(
                        face_image, 
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_CUBIC
                    )
                else:
                     face_image_resized = face_image
                
                # --- Call FER for emotion detection --- 
                # result is a list of dictionaries, one per face found in the image.
                # Since we pass a cropped face, we expect a list with zero or one elements.
                result = self.detector.detect_emotions(face_image_resized)
                # ----------------------------------------
                
                if not result: # No face detected by FER's internal detector
                    # self.logger.debug("FER did not detect a face in the provided image.")
                    return None
                
                # Get the primary (likely only) face's emotion scores
                emotion_scores = result[0].get("emotions")
                if not emotion_scores:
                     self.logger.warning("FER result missing 'emotions' dictionary.")
                     return None

                # Find the highest scoring emotion from FER's output
                if not emotion_scores:
                     return None # Handle case where FER returns empty scores
                fer_emotion = max(emotion_scores, key=emotion_scores.get)
                fer_score = emotion_scores[fer_emotion]
                
                # Check confidence threshold
                if fer_score < self.confidence_threshold:
                    # self.logger.debug(f"Emotion '{fer_emotion}' confidence ({fer_score:.2f}) below threshold ({self.confidence_threshold})")
                    return None
                
                # Map FER emotion to configured system emotion
                mapped_emotion = self._map_emotion(fer_emotion)
                self.logger.debug(f"Analyzed emotion: '{mapped_emotion}' (from FER: '{fer_emotion}' @ {fer_score:.2f})")
                return mapped_emotion
                
            except Exception as e:
                self.logger.error(f"Error during FER emotion analysis: {e}", exc_info=True)
                # Fallback or return None on error?
                # return self._fallback_analyze(face_image) # Optional fallback
                return None 
        else:
            # Use fallback if FER not available/initialized
            return self._fallback_analyze(face_image)

    def get_emotion_confidences(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Get confidence scores for all configured emotions for a single face image.
        
        Args:
            face_image: A cropped image of a face (numpy array, BGR format).
            
        Returns:
            Dictionary mapping configured emotion names to confidence scores (0.0-1.0).
            Returns empty dict if analysis fails or is disabled.
        """
        if not self.enabled:
            return {}
            
        if face_image is None or face_image.size == 0:
            return {}

        # Use FER if available and initialized
        if self.detector and self.fer_initialized:
            try:
                # Similar preprocessing as in analyze()
                if len(face_image.shape) != 3 or face_image.shape[2] != 3: return {}
                min_size = 48 
                height, width = face_image.shape[:2]
                if height < min_size or width < min_size:
                    scale = max(min_size / height, min_size / width)
                    face_image_resized = cv2.resize(face_image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
                else:
                     face_image_resized = face_image
                
                result = self.detector.detect_emotions(face_image_resized)
                
                if not result: return {}
                
                fer_emotion_scores = result[0].get("emotions")
                if not fer_emotion_scores: return {}

                # Map FER scores to our configured emotions
                mapped_scores: Dict[str, float] = {emo: 0.0 for emo in self.configured_emotions}
                for fer_emotion, fer_score in fer_emotion_scores.items():
                    mapped_emotion = self._map_emotion(fer_emotion)
                    if mapped_emotion in mapped_scores:
                        # If multiple FER emotions map to the same configured one (e.g., disgust -> confused?),
                        # take the max score for that target emotion.
                        mapped_scores[mapped_emotion] = max(mapped_scores[mapped_emotion], fer_score)
                
                return mapped_scores
                
            except Exception as e:
                self.logger.error(f"Error getting FER emotion confidences: {e}", exc_info=True)
                return {} # Return empty on error
        else:
            # Use fallback if FER not available
            return self._fallback_confidences(face_image)
            
    def _map_emotion(self, fer_emotion: str) -> str:
        """Maps an emotion name from FER library to configured emotion names."""
        # Mapping from FER default output keys
        emotion_mapping = {
            "angry": "angry",
            "disgust": "disgusted", # Map disgust if needed
            "fear": "fearful",     # Map fear if needed
            "happy": "happy",
            "sad": "sad",
            "surprise": "surprised",
            "neutral": "neutral"
        }
        
        mapped = emotion_mapping.get(fer_emotion.lower(), "neutral")
        
        # Ensure the result is one of the emotions the system actually uses/displays
        if mapped not in self.configured_emotions:
            self.logger.debug(f"Mapped emotion '{mapped}' from FER '{fer_emotion}' is not in configured list: {self.configured_emotions}. Defaulting.")
            # Fallback to the configured default display emotion
            return self.default_display_emotion 
            
        return mapped

    def _fallback_analyze(self, face_image: np.ndarray) -> Optional[str]:
        """Fallback: Returns a random configured emotion (low confidence simulation)."""
        self.logger.debug("Using fallback emotion analysis.")
        # Simulate low confidence by only sometimes returning an emotion
        if random.random() < 0.3: # 30% chance of returning *any* emotion
             return random.choice(self.configured_emotions)
        else:
             return None # Simulate confidence below threshold

    def _fallback_confidences(self, face_image: np.ndarray) -> Dict[str, float]:
         """Fallback: Returns roughly normalized random scores for configured emotions."""
         self.logger.debug("Using fallback emotion confidences.")
         scores = {emo: random.random() for emo in self.configured_emotions}
         total = sum(scores.values())
         if total > 0:
              normalized_scores = {emo: score / total for emo, score in scores.items()}
         else:
              normalized_scores = {emo: 1.0 / len(scores) for emo in self.configured_emotions}
         return normalized_scores

# Removed unused/mock methods: analyze_frame, get_current_emotion, set_emotion
# Removed importlib.reload call at the end 