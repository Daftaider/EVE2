"""
Object detection module using YOLOv8.
"""
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection class using YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """Initialize the object detector
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        try:
            # Load the YOLOv8 model
            self.model = YOLO(model_path)
            self.logger.info(f"Successfully loaded YOLOv8 model from {model_path}")
            self.logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in the frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing detection information:
            {
                'class': str,  # Class name
                'confidence': float,  # Detection confidence
                'bbox': Tuple[int, int, int, int]  # (x1, y1, x2, y2)
            }
        """
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold)[0]
            
            # Process results
            detections = []
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results.names[class_id]
                
                # Log detection including class ID
                self.logger.debug(f"Detected Class ID: {class_id}, Name: {class_name} with confidence {confidence:.2f} at position ({x1}, {y1}, {x2}, {y2})")
                
                detections.append({
                    'class_id': class_id,
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
            
            if detections:
                # Create a list of strings for each detection first (including class ID)
                detection_strings = [f'{d["class"]} ({d["confidence"]:.2f})' for d in detections]
                # Join the list of strings in the final log message
                self.logger.info(f"Found {len(detections)} objects: {', '.join(detection_strings)}")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during object detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on the frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Frame with detection boxes and labels drawn
        """
        display_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det.get('class_id', -1) # Get class_id, default to -1 if missing
            class_name = det['class']
            confidence = det['confidence']
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with background (including class_id)
            label = f"ID:{class_id} {class_name} {confidence:.2f}"
            self.logger.debug(f"Drawing label: '{label}' for bbox {det['bbox']}")
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return display_frame 