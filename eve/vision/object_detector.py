"""
Simple object detector using YOLOv8.
"""
import logging
import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Simple object detector using YOLOv8."""
    
    def __init__(self, confidence_threshold: float = 0.4):
        """Initialize the object detector."""
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize the YOLOv8 model."""
        try:
            # Use YOLOv8n (nano) for better performance on Raspberry Pi
            self.model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model = None
            
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            List of detections, each with 'box', 'label', and 'confidence'
        """
        if self.model is None:
            logger.warning("Model not initialized")
            return []
            
        try:
            # Run inference
            results = self.model(frame)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Skip low confidence detections
                    if confidence < self.confidence_threshold:
                        continue
                        
                    # Get class name
                    class_id = int(box.cls[0])
                    label = r.names[class_id]
                    
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'label': label,
                        'confidence': confidence
                    })
                    
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for YOLOv8
        pass 