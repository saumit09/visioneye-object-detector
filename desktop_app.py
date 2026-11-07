"""
VisionEye - Native Desktop Object Recognizer via OpenCV.
v1.1 — Includes temporal tracking for flicker-free detections.
Run: python3 desktop_app.py
"""

import sys
import time
import cv2
from detector import ObjectDetector
from tracker import DetectionTracker

def main():
    print("=" * 60)
    print("VisionEye - Native Desktop Object Recognizer")
    print("=" * 60)
    
    print("Loading YOLO11 neural network...")
    detector = ObjectDetector()
    tracker = DetectionTracker(
        iou_threshold=0.25,
        bbox_smoothing=0.55,
        conf_smoothing=0.45,
        min_hits_to_confirm=3,
        max_misses=8,
    )
    print("Model and tracker initialized successfully.")

    cam_id = 0
    cap = cv2.VideoCapture(cam_id)

    # Test camera access
    if not cap.isOpened():
        print("\n" + "!" * 60)
        print("ERROR: Could not open webcam.")
        print("On macOS, Terminal or your IDE might lack camera permission.")
        print("To fix:")
        print("  1. Open System Settings -> Privacy & Security -> Camera")
        print("  2. Ensure your Terminal application is toggled ON.")
        print("\nAlternatively, run the browser web app which handles permissions automatically:")
        print("  python3 app.py  ->  Open http://localhost:8000")
        print("!" * 60 + "\n")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("\n" + "!" * 60)
        print("ERROR: Camera opened but returned no frames.")
        print("Please check camera permissions in macOS System Settings, or run the web app:")
        print("  python3 app.py  ->  Open http://localhost:8000")
        print("!" * 60 + "\n")
        cap.release()
        sys.exit(1)

    print("\nControls:")
    print("  [q] Quit")
    print("  [s] Save snapshot")
    print("  [+] Increase confidence threshold")
    print("  [-] Decrease confidence threshold")
    print("  [m] Toggle mirror view\n")

    conf_thresh = 0.35
    is_mirrored = True
    prev_time = time.time()
    fps = 0.0

    window_name = "VisionEye - Desktop Object Recognizer (Press 'q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                break

            if is_mirrored:
                frame = cv2.flip(frame, 1)

            t0 = time.time()
            raw_detections = detector.detect(frame, conf_threshold=max(0.15, conf_thresh - 0.10))
            detections = tracker.update(raw_detections)
            annotated = frame.copy()

            # Draw stabilized detections
            for det in detections:
                if det.get("missing"):
                    continue
                x1, y1, x2, y2 = det["bbox"]
                label = det["class"]
                conf = det["confidence"]

                color_id = (det["class_id"] * 37) % 255
                color = (int((color_id * 5) % 255), int((color_id * 11) % 255), 255 - int((color_id * 3) % 255))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {int(conf * 100)}%"
                (txt_w, txt_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - txt_h - 6)), (x1 + txt_w + 6, max(0, y1)), color, -1)
                cv2.putText(annotated, text, (x1 + 3, max(txt_h + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Compute FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # HUD Display
            visible_detections = [d for d in detections if not d.get("missing")]
            hud_text = f"FPS: {int(fps)} | Conf: {int(conf_thresh * 100)}% | Objects: {len(visible_detections)}"
            cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Detected items summary at bottom
            if visible_detections:
                unique_classes = list(set(d["class"] for d in visible_detections))
                summary_text = "Seeing: " + ", ".join(unique_classes)
                cv2.putText(annotated, summary_text, (15, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, summary_text, (15, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Snapshot saved to {filename}")
            elif key == ord('+') or key == ord('='):
                conf_thresh = min(0.90, conf_thresh + 0.05)
                print(f"Confidence threshold set to {int(conf_thresh * 100)}%")
            elif key == ord('-') or key == ord('_'):
                conf_thresh = max(0.15, conf_thresh - 0.05)
                print(f"Confidence threshold set to {int(conf_thresh * 100)}%")
            elif key == ord('m'):
                is_mirrored = not is_mirrored

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application exited cleanly.")

if __name__ == "__main__":
    main()
