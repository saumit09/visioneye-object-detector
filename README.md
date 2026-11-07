# VisionEye — Advanced Real-Time Object Recognition (Laptop & Phone)

[![CI & GitHub Pages Deployment](https://github.com/saumit09/visioneye-object-detector/actions/workflows/deploy.yml/badge.svg)](https://github.com/saumit09/visioneye-object-detector/actions/workflows/deploy.yml)
[![Deep Learning: YOLO11](https://img.shields.io/badge/YOLO11-Deep%20Learning-blue)](https://github.com/ultralytics)
[![Hardware Acceleration: CoreML](https://img.shields.io/badge/Hardware%20Acceleration-CoreML%20Apple%20Neural%20Engine-emerald)](https://developer.apple.com/machine-learning/core-ml/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%2B%20WebSocket-teal)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, cross-platform computer vision application that detects, labels, and tracks real-world objects in real-time through your webcam or smartphone camera. Powered by **YOLO11 (via ONNX Runtime with CoreML Apple Neural Engine acceleration)** and a **client-side WebGL Deep Learning engine (COCO-SSD)** for standalone deployment.

---

## 🌐 Live Accessible Links

| Environment | Accessible URL | Description |
| :--- | :--- | :--- |
| 📱 **Mobile & Global HTTPS** | **[repurpose-usage-trodden.ngrok-free.dev](https://repurpose-usage-trodden.ngrok-free.dev)** | Live secure HTTPS tunnel with full camera access on iPhone & Android |
| 🌐 **GitHub Pages (Serverless)** | **[saumit09.github.io/visioneye-object-detector](https://saumit09.github.io/visioneye-object-detector/)** | Permanent client-side deep learning running directly in your browser |
| 💻 **Local Laptop** | **[http://localhost:8000](http://localhost:8000)** | Local fast development endpoint |

---

## Key Features

- **Cross-Platform (Laptop & Mobile Phone)**:
  - **Phone-Optimized UX**: Responsive touch interface (`100dvh`), high-contrast HUD badges, and distraction-free fullscreen mode.
  - **Front & Rear Camera Switcher**: Switch between selfie and back/environment cameras with auto-mirroring adjustment.
  - **Scan-to-Connect QR Code**: Click **"Phone Link / QR"** on laptop to generate a scannable QR code for instant phone pairing.
- **Advanced Machine Learning & Computer Vision**:
  - **Dual Engine AI**:
    1. **High-Power Mode**: YOLO11 nano ONNX with Apple Silicon CoreML hardware acceleration (< 15-25 ms inference).
    2. **Standalone Mode**: In-browser TensorFlow.js COCO-SSD running over WebGL with zero server dependencies.
  - **Temporal SORT Tracking**: Persistent object IDs and IoU tracking association to eliminate flickering.
  - **60 FPS Client-Side Box Interpolation**: Buttery smooth bounding box transitions via linear interpolation (lerp).
- **Interactive Voice Narration**:
  - Text-to-Speech (TTS) engine that announces newly detected items (e.g. *"I see a laptop and a cell phone"*).
  - Mobile audio unlocked automatically on first tap.
- **Snapshot Studio**:
  - Capture tagged snapshots with bounding boxes and confidence scores.
  - One-click image download and gallery preview.

---

## Quick Start (Laptop & Phone)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/saumit09/visioneye-object-detector.git
cd visioneye-object-detector
pip3 install -r requirements.txt
```

### 2. Launch Server + Public Tunnel (All-in-One)

```bash
python3 run.py
```

This single command:
1. Pre-loads the YOLO11 model with Apple Silicon Neural Engine acceleration.
2. Starts the FastAPI server on port 8000.
3. Launches the secure ngrok HTTPS tunnel.
4. Outputs clickable links and terminal info.

Open `http://localhost:8000` on your laptop, or open `https://repurpose-usage-trodden.ngrok-free.dev` on your smartphone!

---

### 3. Native Desktop App (Optional)

To run as an OpenCV GUI window:

```bash
python3 desktop_app.py
```

**Hotkeys:**
- `q` or `ESC`: Quit
- `s`: Save snapshot image
- `+` / `-`: Increase / decrease confidence threshold
- `m`: Toggle mirror / flip view

---

## Project Structure

```
visioneye-object-detector/
├── .github/
│   └── workflows/
│       └── deploy.yml        # CI test suite & automatic GitHub Pages deployment
├── frontend/
│   ├── index.html            # Responsive UI, QR modal, mobile meta tags
│   ├── style.css             # Glassmorphic dark styling & mobile touch rules
│   └── app.js                # Dual-engine AI, 60fps lerp, camera switcher, TTS
├── models/
│   └── .gitkeep              # Weights auto-downloaded on first run
├── app.py                    # FastAPI server (WebSocket, REST, static endpoints)
├── run.py                    # All-in-one launcher (FastAPI + ngrok HTTPS tunnel)
├── desktop_app.py            # Native OpenCV desktop GUI app
├── detector.py               # YOLO11 ONNX inference & NMS engine
├── tracker.py                # SORT temporal tracker & IoU association
├── requirements.txt          # Python package requirements
├── test_detector.py          # YOLO model unit tests
└── test_api.py               # Automated integration tests
```

---

## Supported Object Classes (COCO 80)

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`.

---

## License

Released under the [MIT License](LICENSE).
