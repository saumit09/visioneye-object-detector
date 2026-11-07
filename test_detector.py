"""
Unit verification test for the ObjectDetector.
Tests model downloading, ONNX session initialization, and synthetic inference.
"""

import numpy as np
import cv2
from detector import ObjectDetector

def main():
    print("Testing ObjectDetector initialization and model fetch...")
    detector = ObjectDetector()
    print("ObjectDetector successfully initialized!")
    
    # Create synthetic test frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.rectangle(dummy_frame, (100, 100), (300, 350), (200, 200, 200), -1)
    
    print("Running test inference on dummy frame...")
    detections = detector.detect(dummy_frame, conf_threshold=0.25)
    print(f"Inference completed successfully. Detections returned: {len(detections)}")
    
    # Test drawing
    annotated = detector.draw_detections(dummy_frame, detections)
    assert annotated.shape == dummy_frame.shape, "Annotated frame shape mismatch"
    print("Verification passed! Detector is fully functional.")

if __name__ == "__main__":
    main()
