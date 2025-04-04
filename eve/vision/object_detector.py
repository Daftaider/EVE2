"""
Object detection module using YOLOv8 with optional Hailo-8 acceleration.
"""
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional, Any
import platform
import os
import time # Ensure time is imported
from eve.config.config import SystemConfig

# Attempt to import HailoRT libraries
_HAILO_AVAILABLE = False
try:
    # Common HailoRT imports - adjust based on your SDK structure
    from hailo_platform import Device, VDevice, HailoStreamInterface
    # Import HailoFormatType - adjust path if needed in your Hailo SDK version
    from hailo_platform.pyhailort._pyhailort import ConfigureParams, HailoFormatType 
    # Might need hailo_model_zoo for postprocessing helpers depending on the model
    # from hailo_model_zoo.postprocessing.yolov8_postprocessing import yolov8_postprocess # Example
    _HAILO_AVAILABLE = True
    logging.info("HailoRT library loaded successfully.")
except ImportError as e:
    logging.warning(f"HailoRT library import failed: {e}. Hailo acceleration will not be available.")
    # Set Hailo specific variables to None if import fails
    Device = None
    VDevice = None
    ConfigureParams = None 
    HailoStreamInterface = None
    HailoFormatType = None # Ensure this is None if import fails

# Replace placeholder Config with actual import
# Remove the placeholder Config class definition
try:
    from eve.config import config # Assumes eve.config provides the necessary config object
    logging.info("Successfully imported configuration from eve.config")
except ImportError:
     logging.critical("Failed to import configuration from eve.config. Cannot proceed.")
     # You might want to raise an error or use default values as a last resort
     # For now, define a dummy config to avoid crashing the rest of the code structure definition
     class DummyConfig:
         USE_HAILO = False
         HAILO_HEF_PATH = "dummy.hef"
         CPU_MODEL_PATH = "yolov8n.pt"
         DETECTION_CONFIDENCE = 0.5
         HAILO_NETWORK_NAME = None
     config = DummyConfig()
     logging.warning("Using dummy configuration due to import error.")

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection class using YOLOv8 with optional Hailo-8 acceleration"""
    
    def __init__(self, config: SystemConfig, model_path: Optional[str] = None, confidence_threshold: Optional[float] = None):
        """Initialize the object detector
        
        Attempts to use Hailo-8 if configured and available, otherwise falls back to CPU YOLOv8.
        
        Args:
            config: The main SystemConfig object.
            model_path: Path to the primary model file (ignored if using config).
            confidence_threshold: Minimum confidence threshold (ignored if using config).
        """
        self.config = config # Store the main config
        self.logger = logging.getLogger(__name__) # Initialize logger first
        
        # Extract relevant sub-configs or values
        # Example: Assuming object detection config is under vision
        vision_config = config.vision
        hardware_config = config.hardware
        
        # Determine confidence threshold: Use constructor arg > config > default
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = vision_config.object_detection_confidence # Use config value
            
        self.hailo_enabled = False
        self.hailo_target = None # VDevice or Device
        self.hailo_hef = None
        self.hailo_network_group = None
        self.hailo_input_vstream_infos = None
        self.hailo_output_vstream_infos = None

        self.yolo_model = None # For CPU fallback
        self.hailo_network_group_active = False # Track activation state

        # Check for Hailo config (adjust attribute name if necessary)
        hardware_config = config.hardware
        # use_hailo = hardware_config.accelerator == 'hailo' # Attribute doesn't exist in HardwareConfig
        use_hailo = False # Temporarily disable Hailo check until config is added

        # 1. Check configuration and Hailo availability
        if use_hailo and _HAILO_AVAILABLE and HailoFormatType is not None:
            self.logger.info("Hailo usage requested and HailoRT library is available.")
            
            if not hardware_config.hailo_hef_path or not os.path.exists(hardware_config.hailo_hef_path):
                self.logger.warning(f"Hailo HEF file not found or path not specified: {hardware_config.hailo_hef_path}. Falling back to CPU.")
            else:
                try:
                    devices = Device.scan()
                    if not devices:
                        self.logger.warning("No Hailo devices found. Falling back to CPU.")
                    else:
                        self.logger.info(f"Found Hailo devices: {devices}")
                        self.hailo_target = VDevice()
                        self.logger.info(f"Created Hailo VDevice target.")

                        self.logger.info(f"Loading Hailo HEF from: {hardware_config.hailo_hef_path}")
                        self.hailo_hef = self.hailo_target.load_hef(hardware_config.hailo_hef_path)
                        network_group_names = self.hailo_hef.get_network_group_names()
                        self.logger.info(f"Successfully loaded HEF. Network groups: {network_group_names}")
                        
                        # Determine network group name
                        if hardware_config.hailo_network_name and hardware_config.hailo_network_name in network_group_names:
                             network_group_name = hardware_config.hailo_network_name
                        elif network_group_names:
                             network_group_name = network_group_names[0]
                             self.logger.info(f"Using first available network group: {network_group_name}")
                        else:
                            raise RuntimeError("No network groups found in the HEF.")

                        configure_params = self.hailo_target.create_configure_params(network_group_name)
                        self.logger.info(f"Configuring network group: {network_group_name} with params: {configure_params}")
                        self.hailo_network_group = self.hailo_target.configure(self.hailo_hef, configure_params)[network_group_name] # Use name as key
                        self.logger.info(f"Network group configured successfully.")

                        self.hailo_input_vstreams = self.hailo_network_group.get_input_vstreams()
                        self.hailo_output_vstreams = self.hailo_network_group.get_output_vstreams()
                        self.logger.info(f"Got {len(self.hailo_input_vstreams)} input and {len(self.hailo_output_vstreams)} output vstreams.")

                        if not self.hailo_input_vstreams:
                             raise RuntimeError("No input vstreams found.")
                        if not self.hailo_output_vstreams:
                             raise RuntimeError("No output vstreams found.")
                             
                        # Store input details
                        input_vstream = self.hailo_input_vstreams[0]
                        self.hailo_input_vstream_infos = {
                            'name': input_vstream.name,
                            'shape': input_vstream.shape,
                            'format_type': input_vstream.format.type,
                            'quant_info': input_vstream.get_quant_info()
                        }
                        self.logger.info(f"Hailo Input Shape: {self.hailo_input_vstream_infos['shape']}, Format: {self.hailo_input_vstream_infos['format_type']}")
                        if not (len(self.hailo_input_vstream_infos['shape']) == 4 and self.hailo_input_vstream_infos['shape'][3] == 3): 
                             self.logger.warning(f"Hailo input shape {self.hailo_input_vstream_infos['shape']} might not be standard N H W C image format.")

                        # Store output details (like names) if needed for postprocessing
                        self.hailo_output_vstream_infos = []
                        for ovs in self.hailo_output_vstreams:
                            details = {
                                'name': ovs.name,
                                'shape': ovs.shape,
                                'format_type': ovs.format.type,
                                'quant_info': ovs.get_quant_info()
                            }
                            self.hailo_output_vstream_infos.append(details)
                            self.logger.info(f"Hailo Output VStream Info: Name={details['name']}, Shape={details['shape']}, Format={details['format_type']}, Quant={details['quant_info']}")

                        self.hailo_network_group.activate()
                        self.logger.info("Hailo network group activated.")
                        self.hailo_network_group_active = True
                        self.hailo_enabled = True
                        self.logger.info(f"Successfully initialized Hailo-8 accelerator.")

                except Exception as e:
                    self.logger.error(f"Failed to initialize Hailo device or model: {e}", exc_info=True)
                    self.logger.warning("Falling back to CPU YOLOv8 model.")
                    if self.hailo_network_group:
                         try: self.hailo_network_group.deactivate() # Attempt deactivation
                         except: pass # Ignore errors during cleanup
                    self.hailo_target = None 
                    self.hailo_hef = None
                    self.hailo_network_group = None
                    self.hailo_input_vstreams = None
                    self.hailo_output_vstreams = None
        else:
            if not use_hailo:
                 self.logger.info("Hailo usage is disabled in configuration. Using CPU YOLOv8.")
            elif not _HAILO_AVAILABLE:
                 self.logger.warning("Hailo usage requested, but HailoRT library not found or failed to import. Using CPU YOLOv8.")
            elif HailoFormatType is None: # Check specifically if enum failed import
                 self.logger.warning("Hailo usage requested, but HailoFormatType could not be imported. Using CPU YOLOv8.")

        # 2. Fallback to CPU YOLOv8
        if not self.hailo_enabled:
            # Use model path from constructor arg OR config
            cpu_model_path = model_path if model_path is not None else vision_config.object_model_path
            self._load_cpu_model(cpu_model_path) # Pass path to helper

    def _get_network_group_name(self, available_groups: List[str]) -> Optional[str]:
        """Determines the network group name to use."""
        if config.HAILO_NETWORK_NAME and config.HAILO_NETWORK_NAME in available_groups:
            self.logger.info(f"Using configured network group name: {config.HAILO_NETWORK_NAME}")
            return config.HAILO_NETWORK_NAME
        elif available_groups:
            name = available_groups[0]
            self.logger.info(f"Using first available network group name: {name}")
            return name
        else:
            self.logger.error("No network groups found in the HEF.")
            return None

    def _load_cpu_model(self, cpu_model_path: Optional[str]):
         """Loads the CPU YOLOv8 model."""
         if not cpu_model_path:
             self.logger.error("No CPU model path specified. Cannot load CPU model.")
             # If Hailo also failed, we need to raise an error
             if not self.hailo_enabled:
                 raise ValueError("Neither Hailo nor CPU model could be initialized: No CPU model path provided.")
             return # If Hailo worked, maybe CPU fallback isn't critical
         
         self.logger.info(f"Loading CPU YOLOv8 model from {cpu_model_path}")
         try:
             self.yolo_model = YOLO(cpu_model_path)
             self.logger.info(f"Successfully loaded CPU YOLOv8 model: {self.yolo_model.names}")
         except Exception as e:
             self.logger.error(f"Failed to load CPU YOLOv8 model: {e}", exc_info=True)
             # If Hailo also failed, we have no model. Raise critical error.
             if not self.hailo_enabled:
                raise RuntimeError(f"Failed to initialize any detection model (Hailo or CPU): {e}")

    def _cleanup_hailo_resources(self):
        """Safely cleans up Hailo resources."""
        self.logger.debug("Cleaning up Hailo resources...")
        if hasattr(self, 'hailo_network_group') and self.hailo_network_group:
             if self.hailo_network_group_active:
                 try:
                     self.logger.info("Deactivating Hailo network group.")
                     self.hailo_network_group.deactivate()
                     self.hailo_network_group_active = False
                 except Exception as e:
                     self.logger.error(f"Error deactivating Hailo network group: {e}", exc_info=True)
             # Release network group reference
             self.hailo_network_group = None 
        
        # VDevice should ideally be used with a context manager, 
        # but explicit release might be needed depending on HailoRT version/usage
        if hasattr(self, 'hailo_target') and self.hailo_target:
            try:
                self.logger.info("Releasing Hailo target (VDevice).")
                # Check documentation for proper VDevice release without context manager
                # If VDevice holds onto the configured group, releasing it might be important
                if hasattr(self.hailo_target, 'release'): self.hailo_target.release()
            except Exception as e:
                 self.logger.error(f"Error releasing Hailo target: {e}", exc_info=True)
                 
        self.hailo_target = None
        self.hailo_hef = None
        self.hailo_input_vstream_infos = None
        self.hailo_output_vstream_infos = None
        self.hailo_enabled = False
        self.logger.debug("Finished Hailo resource cleanup.")

    def _preprocess_hailo(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo input requirements using stored info."""
        if not self.hailo_input_vstream_infos:
            raise RuntimeError("Hailo input vstream info not available for preprocessing.")

        input_shape = self.hailo_input_vstream_infos['shape']
        input_format_type = self.hailo_input_vstream_infos['format_type']
        # quant_info = self.hailo_input_vstream_infos['quant_info'] # Input quantization rarely needed

        target_height, target_width = input_shape[1], input_shape[2]
        processed_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Color conversion placeholder - add if your model requires RGB
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Cast to correct data type
        dtype_map = {
            HailoFormatType.UINT8: np.uint8,
            HailoFormatType.FLOAT32: np.float32,
            HailoFormatType.UINT16: np.uint16,
            HailoFormatType.INT8: np.int8, # Add other types as needed
            HailoFormatType.INT16: np.int16,
            HailoFormatType.INT32: np.int32,
        }
        target_dtype = dtype_map.get(input_format_type)

        if target_dtype:
            processed_frame = processed_frame.astype(target_dtype)
            # Apply normalization if FLOAT32 (common practice)
            if input_format_type == HailoFormatType.FLOAT32:
                 processed_frame /= 255.0 
        else:
            self.logger.warning(f"Unsupported Hailo input format type: {input_format_type}. Using UINT8 default.")
            processed_frame = processed_frame.astype(np.uint8)

        # Add batch dimension if needed
        if len(input_shape) == 4 and input_shape[0] == 1:
             processed_frame = np.expand_dims(processed_frame, axis=0)
        elif len(input_shape) != len(processed_frame.shape):
             self.logger.error(f"Shape mismatch: Expected {len(input_shape)} dims, got {len(processed_frame.shape)}")
             raise ValueError("Frame shape mismatch after preprocessing")

        return np.ascontiguousarray(processed_frame)

    def _dequantize(self, tensor: np.ndarray, quant_info: Dict[str, Any]) -> np.ndarray:
        """Dequantizes a tensor using provided quantization info."""
        if not quant_info or not isinstance(quant_info, dict) or \
           'qp_scale' not in quant_info or 'qp_zp' not in quant_info:
            self.logger.warning(f"Invalid or missing quant_info: {quant_info}. Cannot dequantize.")
            return tensor.astype(np.float32) # Return as float, but not dequantized
        
        scale = quant_info['qp_scale']
        zp = quant_info['qp_zp']
        
        # Handle potential per-channel quantization if scale/zp are arrays
        if isinstance(scale, np.ndarray) or isinstance(zp, np.ndarray):
             # Simple broadcast, assumes scale/zp shapes match tensor's last dim
             return (tensor.astype(np.float32) - zp) * scale
        else:
             # Per-tensor quantization
             return (tensor.astype(np.float32) - zp) * scale

    def _postprocess_hailo(self, output_tensors: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess Hailo output using stored vstream details (adaptive)."""
        detections = []
        original_height, original_width = original_shape
        if not self.hailo_output_vstream_infos or len(output_tensors) != len(self.hailo_output_vstream_infos):
            self.logger.error("Output tensor count mismatch or missing details. Cannot postprocess.")
            return []
        
        # --- Adaptive Parsing Logic --- 
        # Strategy: Try to match known YOLO patterns based on tensor count and shapes.
        # Primarily supports single combined output tensor pattern for now.
        
        if len(self.hailo_output_vstream_infos) == 1:
            details = self.hailo_output_vstream_infos[0]
            tensor = output_tensors[0]
            self.logger.debug(f"Processing single output tensor: Name={details['name']}, Shape={details['shape']}, Format={details['format_type']}")
            
            # Check shape (e.g., [Batch, NumProposals, Features])
            if len(details['shape']) == 3 and details['shape'][0] == 1 and details['shape'][2] > 4:
                num_proposals = details['shape'][1]
                num_features = details['shape'][2]
                output_data = tensor.squeeze(0) # Remove batch dim

                # Dequantize if needed
                if details['format_type'] != HailoFormatType.FLOAT32:
                    self.logger.debug(f"Dequantizing output tensor {details['name']}")
                    output_data = self._dequantize(output_data, details['quant_info'])
                
                # Now output_data should be float32
                # Infer number of classes (assuming Box(4) + ClassScores or Box(4) + ObjConf(1) + ClassScores)
                num_classes = -1
                box_dim = 4 # Assume box coords always first 4
                if num_features > box_dim:
                     # Simple check: if features = 4 + N, assume N classes. If 4+1+N, assume N classes + obj score
                     # This heuristic might need refinement based on common models
                     potential_num_classes_1 = num_features - box_dim 
                     potential_num_classes_2 = num_features - box_dim - 1
                     # Choose based on typical model outputs (e.g. COCO has 80 classes)
                     # This is still a guess!
                     if potential_num_classes_1 == 80: # COCO example
                         num_classes = potential_num_classes_1
                         score_offset = box_dim
                         self.logger.debug(f"Inferred {num_classes} classes (Box + Scores format)")
                     elif potential_num_classes_2 == 80: # COCO example
                         num_classes = potential_num_classes_2
                         score_offset = box_dim + 1 # Assuming obj score at index 4
                         self.logger.debug(f"Inferred {num_classes} classes (Box + Obj + Scores format)")
                     else:
                         # Fallback or error if class count unclear
                         num_classes = potential_num_classes_1 # Default guess
                         score_offset = box_dim 
                         self.logger.warning(f"Could not reliably infer class count from features={num_features}. Assuming {num_classes} classes (Box+Scores format). Check HEF.")
                
                if num_classes <= 0:
                     self.logger.error(f"Invalid number of features ({num_features}) or failed to infer class count.")
                     return []

                # Prepare for NMS
                boxes = []
                scores = []
                class_ids = []
                input_height, input_width = self.hailo_input_vstream_infos['shape'][1:3]

                for i in range(num_proposals):
                    candidate = output_data[i]
                    class_scores = candidate[score_offset : score_offset + num_classes]
                    max_score = np.max(class_scores)
                    class_id = np.argmax(class_scores)

                    # Use objectness score if present and inferred (e.g., index 4)
                    objectness_score = 1.0 # Default if not present
                    if score_offset == box_dim + 1:
                        objectness_score = candidate[box_dim]
                    
                    confidence = max_score * objectness_score # Combine if obj score exists
                    
                    if confidence >= self.confidence_threshold:
                        scores.append(confidence)
                        class_ids.append(class_id)
                        # Box coords (assuming cx, cy, w, h relative to input size)
                        cx, cy, w, h = candidate[:box_dim]
                        x1 = (cx - w / 2) 
                        y1 = (cy - h / 2)
                        boxes.append([int(x1 * input_width), int(y1 * input_height), int(w * input_width), int(h * input_height)]) # NMS needs absolute coords
                
                if not boxes:
                    return []

                # NMS
                nms_threshold = 0.45
                indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, nms_threshold)
                if isinstance(indices, np.ndarray): indices = indices.flatten()
                else: indices = [] # Handle empty tuple case

                # Scale and format final detections
                scale_x = original_width / input_width
                scale_y = original_height / input_height
                for i in indices:
                    box = boxes[i] # Box is already absolute to input size (x1, y1, w, h)
                    x1_in, y1_in, w_in, h_in = box
                    
                    x1 = int(x1_in * scale_x)
                    y1 = int(y1_in * scale_y)
                    x2 = int((x1_in + w_in) * scale_x)
                    y2 = int((y1_in + h_in) * scale_y)
                    
                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(original_width - 1, x2), min(original_height - 1, y2)
                    
                    class_id = class_ids[i]
                    confidence = scores[i]
                    class_name = f"ID:{class_id}" # Add label mapping later if needed
                    
                    detections.append({
                        'class_id': int(class_id),
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': (x1, y1, x2, y2)
                    })
            else:
                 self.logger.warning(f"Unsupported shape for single output tensor: {details['shape']}. Cannot parse.")

        elif len(self.hailo_output_vstream_infos) > 1:
            # Placeholder for handling multiple output tensors (e.g., separate box/score/class tensors)
            # Would need logic to identify tensor roles based on names/shapes/etc.
            self.logger.warning(f"Postprocessing for {len(self.hailo_output_vstream_infos)} output tensors is not implemented. Check HEF structure.")
            # Example: Find tensors named like 'boxes' and 'scores'
            # boxes_tensor, scores_tensor = None, None
            # for i, details in enumerate(self.hailo_output_vstream_infos):
            #    if 'box' in details['name'].lower(): boxes_tensor = output_tensors[i]
            #    if 'score' in details['name'].lower(): scores_tensor = output_tensors[i]
            # ... then parse, dequantize, and apply NMS using these tensors ...

        else: # No output details
            self.logger.error("No Hailo output details available.")

        # --- End Adaptive Parsing Logic --- 

        if detections:
             detection_strings = [f'{d["class"]} ({d["confidence"]:.2f})' for d in detections]
             self.logger.info(f"Hailo Found {len(detections)} objects after NMS: {', '.join(detection_strings)}")
        return detections

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in the frame using Hailo or CPU"""
        if self.hailo_enabled and self.hailo_target and self.hailo_network_group:
            return self._detect_hailo(frame)
        elif self.yolo_model:
            return self._detect_cpu(frame)
        else:
            self.logger.error("No detection model loaded.")
            return []

    def _detect_hailo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using the Hailo accelerator"""
        if not self.hailo_network_group_active: # Check if group is active
             self.logger.error("Hailo network group is not active. Cannot run inference.")
             return []
        if not self.hailo_output_vstream_infos: # Check if output details were stored
             self.logger.error("Hailo output details missing. Cannot run inference.")
             return []
             
        try:
            prep_start = time.perf_counter()
            input_data = self._preprocess_hailo(frame)
            original_shape = frame.shape[:2]
            prep_end = time.perf_counter()
            
            infer_start = time.perf_counter()
            # Assuming single input stream
            input_vstream = self.hailo_network_group.get_input_vstreams()[0] 
            output_vstreams = self.hailo_network_group.get_output_vstreams()
            
            input_vstream.write(input_data)
            output_tensors = [ovs.read() for ovs in output_vstreams]
            infer_end = time.perf_counter()

            post_start = time.perf_counter()
            detections = self._postprocess_hailo(output_tensors, original_shape)
            post_end = time.perf_counter()

            self.logger.debug(
                f"Hailo Timings: Prep: {(prep_end - prep_start)*1000:.1f}ms, "
                f"Infer: {(infer_end - infer_start)*1000:.1f}ms, "
                f"Post: {(post_end - post_start)*1000:.1f}ms"
            )
            return detections

        except Exception as e:
            self.logger.error(f"Error during Hailo detection pipeline: {e}", exc_info=True)
            # Consider attempting to reset/reconfigure Hailo on specific errors
            return []

    def _detect_cpu(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using the CPU YOLOv8 model"""
        if not self.yolo_model: 
            self.logger.error("CPU YOLO model not loaded.")
            return []
        try:
            start_time = time.perf_counter()
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)[0]
            inference_time = time.perf_counter() - start_time
            self.logger.debug(f"CPU Inference time: {inference_time * 1000:.2f} ms")

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo_model.names.get(class_id, f"CPU_ID:{class_id}") 
                
                detections.append({
                    'class_id': class_id,
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
            
            if detections:
                detection_strings = [f'{d["class"]} ({d["confidence"]:.2f})' for d in detections]
                self.logger.info(f"CPU Found {len(detections)} objects: {', '.join(detection_strings)}")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during CPU object detection: {e}", exc_info=True)
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on the frame"""
        display_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det.get('class_id', -1)
            class_name = det['class'] # Should have name from CPU or Hailo postprocessing
            confidence = det['confidence']
            
            color = (0, 255, 0) if self.hailo_enabled else (255, 0, 0) 
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}" 
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        mode_text = "Detector: Hailo-8" if self.hailo_enabled else "Detector: CPU YOLOv8"
        cv2.putText(display_frame, mode_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display_frame

    def __del__(self):
        # Ensure logger exists before trying to use it
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("ObjectDetector destructor called. Cleaning up resources.")
        self._cleanup_hailo_resources()

# Ensure necessary imports are present at the top
# import time, os, platform, logging, cv2, numpy as np etc.

# (rest of the original code remains unchanged) 