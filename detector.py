"""
Real-Time YOLO Object Detector using ONNX Runtime.
Supports Apple Silicon CoreML acceleration and CPU execution.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import requests

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_URL = "https://github.com/ultralytics/assets/releases/latest/download/yolo11n.onnx"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "yolo11n.onnx"


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the ONNX model if not already present."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        print(f"Downloading YOLO11n ONNX model to {model_path}...")
        resp = requests.get(MODEL_URL, stream=True, timeout=60)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded successfully ({model_path.stat().st_size / (1024*1024):.1f} MB)")
    return model_path


class ObjectDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        import onnxruntime as ort

        self.model_path = ensure_model(model_path)
        available_providers = ort.get_available_providers()
        
        # Select best execution provider
        providers = []
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
            
        print(f"Initializing ONNXRuntime with providers: {providers}")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Query input shape metadata
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # e.g. [1, 3, 640, 640]
        self.img_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.img_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640
        
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize image to model input size (640x640) and normalize."""
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Compute ratio and padding
        scale = min(self.img_width / orig_w, self.img_height / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (self.img_width - new_w) // 2
        pad_y = (self.img_height - new_h) // 2
        
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_height, self.img_width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB, transpose to [1, 3, H, W], normalize to [0, 1]
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, (pad_x, pad_y)

    def detect(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a BGR image frame.
        Returns list of detections with labels, confidences, and bbox coordinates.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        tensor, scale, (pad_x, pad_y) = self.preprocess(img_bgr)
        
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        
        # Output shape is [1, 84, 8400] -> transpose to [8400, 84]
        preds = np.squeeze(outputs, axis=0).T
        
        boxes = preds[:, :4]     # cx, cy, w, h
        scores = preds[:, 4:]    # 80 class scores
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert cx, cy, w, h to original image coordinates [x1, y1, w, h]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Remove padding and scale back to original image
        x1 = np.maximum(0, (cx - w / 2 - pad_x) / scale)
        y1 = np.maximum(0, (cy - h / 2 - pad_y) / scale)
        w_orig = w / scale
        h_orig = h / scale
        
        nms_boxes = [[int(x), int(y), int(width), int(height)] for x, y, width, height in zip(x1, y1, w_orig, h_orig)]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, iou_threshold)
        
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = nms_boxes[idx]
                bx1 = max(0, min(orig_w - 1, box[0]))
                by1 = max(0, min(orig_h - 1, box[1]))
                bx2 = max(0, min(orig_w - 1, box[0] + box[2]))
                by2 = max(0, min(orig_h - 1, box[1] + box[3]))
                
                cid = int(class_ids[idx])
                label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                conf = float(confidences[idx])
                
                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": [bx1, by1, bx2, by2],
                    "class_id": cid,
                    "inference_ms": round(inference_time_ms, 1)
                })
                
        return detections

    def draw_detections(self, img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate BGR image with bounding boxes and labels."""
        annotated = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            conf = det["confidence"]
            
            # Deterministic color for each class
            color_id = (det["class_id"] * 37) % 255
            color = (int((color_id * 5) % 255), int((color_id * 11) % 255), 255 - int((color_id * 3) % 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label banner
            text = f"{label} {int(conf * 100)}%"
            (txt_w, txt_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - txt_h - 6)), (x1 + txt_w + 6, max(0, y1)), color, -1)
            cv2.putText(annotated, text, (x1 + 3, max(txt_h + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
        return annotated
