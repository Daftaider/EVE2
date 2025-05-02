"""
Emotion detection service for recognizing facial expressions with modular backend (cpu, onnx, hailo).
"""
import logging
import cv2
import numpy as np
from typing import Optional, Dict
import yaml
from .eye_display import Emotion
import hailo_platform as hpf

logger = logging.getLogger(__name__)

class EmotionService:
    """Emotion detection service for recognizing facial expressions with modular backend (cpu, onnx, hailo)."""
    
    def __init__(self, config_path: str = "config/settings.yaml", backend: str = "cpu"):
        """Initialize emotion detection service."""
        self.config = self._load_config(config_path)
        self.backend = backend
        self.emotion_model = None
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy',
            'sad', 'surprise', 'neutral'
        ]
        # Hailo-8L fields
        self.hailo_ready = False
        self.hailo_input_shape = None
        self.hailo_output_shape = None
        self.hailo_input_name = None
        self.hailo_output_name = None
        self.hailo_infer_pipeline = None
        self.hailo_vdevice = None
        self.hailo_network_group = None
        self.hailo_network_group_params = None
        logger.info(f"EmotionService initialized with backend: {backend}")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def start(self) -> bool:
        """Start the emotion detection service."""
        try:
            if self.backend == "cpu":
                # TODO: Load actual CPU/DeepFace/FER model
                self.emotion_model = "dummy_model"
                logger.info("EmotionService using CPU backend (dummy model)")
            elif self.backend == "onnx":
                # TODO: Load ONNX model
                self.emotion_model = "onnx_model_placeholder"
                logger.info("EmotionService using ONNX backend (not implemented)")
            elif self.backend == "hailo":
                try:
                    hef_path = self.config.get('emotion_detection', {}).get('hef_path', 'src/models/emotion/model.hef')
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
                    logger.info(f"Hailo-8L initialized for emotion: input shape {self.hailo_input_shape}, output shape {self.hailo_output_shape}")
                except Exception as e:
                    logger.error(f"Failed to initialize Hailo-8L for emotion: {e}")
                    self.hailo_ready = False
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            logger.info("Emotion detection service started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting emotion detection service: {e}")
            return False
            
    def detect_emotion(self, face_roi: np.ndarray) -> Optional[Emotion]:
        """Detect emotion from a face region of interest using the selected backend."""
        if self.backend == "cpu":
            # TODO: Implement actual CPU/DeepFace/FER emotion detection
            return Emotion.NEUTRAL
        elif self.backend == "onnx":
            # TODO: Implement ONNX emotion detection
            logger.warning("ONNX backend not implemented, returning NEUTRAL")
            return Emotion.NEUTRAL
        elif self.backend == "hailo" and self.hailo_ready:
            try:
                resized = cv2.resize(face_roi, (self.hailo_input_shape[2], self.hailo_input_shape[1]))
                input_data = np.expand_dims(resized.astype(np.float32), axis=0)
                input_dict = {self.hailo_input_name: input_data}
                with self.hailo_network_group.activate(self.hailo_network_group_params):
                    with self.hailo_infer_pipeline as infer_pipeline:
                        results = infer_pipeline.infer(input_dict)
                        output_data = results[self.hailo_output_name]
                        # Assume output is a softmax or logits for each emotion
                        emotion_idx = int(np.argmax(output_data))
                        emotion_label = self.emotion_labels[emotion_idx]
                        return Emotion[emotion_label.upper()] if emotion_label.upper() in Emotion.__members__ else Emotion.NEUTRAL
            except Exception as e:
                logger.error(f"Hailo-8L emotion inference failed: {e}")
                return Emotion.NEUTRAL
        elif self.backend == "hailo":
            logger.warning("Hailo-8L backend not ready, returning NEUTRAL")
            return Emotion.NEUTRAL
        else:
            logger.error(f"Unknown backend: {self.backend}")
            return None
            
    def stop(self) -> None:
        """Stop the emotion detection service."""
        logger.info("Emotion detection service stopped")
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 