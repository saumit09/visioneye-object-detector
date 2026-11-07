"""
Test FastAPI app using standard requests against a local uvicorn background thread.
"""

import time
import base64
import threading
import requests
import cv2
import numpy as np
import uvicorn
from app import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

def test_api():
    print("Starting background uvicorn server on port 8001...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to spin up
    for _ in range(20):
        try:
            r = requests.get("http://127.0.0.1:8001/", timeout=1)
            if r.status_code == 200:
                print("Server is up and responding!")
                break
        except Exception:
            time.sleep(0.3)
    else:
        raise RuntimeError("Server failed to start within timeout.")

    # 1. Test Root HTML
    res_root = requests.get("http://127.0.0.1:8001/")
    assert res_root.status_code == 200
    assert "VisionEye" in res_root.text
    print("✓ GET / returned 200 and valid HTML")

    # 2. Test static assets at both root and /static
    res_style = requests.get("http://127.0.0.1:8001/style.css")
    assert res_style.status_code == 200
    print("✓ GET /style.css served successfully")

    res_js = requests.get("http://127.0.0.1:8001/app.js")
    assert res_js.status_code == 200
    print("✓ GET /app.js served successfully")

    res_static = requests.get("http://127.0.0.1:8001/static/app.js")
    assert res_static.status_code == 200
    print("✓ GET /static/app.js served successfully")

    # 3. Test /api/info
    res_info = requests.get("http://127.0.0.1:8001/api/info")
    assert res_info.status_code == 200
    info_data = res_info.json()
    assert info_data["status"] == "online"
    assert "tunnel_url" in info_data
    print(f"✓ GET /api/info response: provider={info_data.get('provider')}, tunnel={info_data.get('tunnel_url')}")

    # 4. Test /api/detect with a synthetic frame
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "VisionEye Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buf).decode("utf-8")

    payload = {
        "image": f"data:image/jpeg;base64,{b64_str}",
        "conf_threshold": 0.35
    }

    res_detect = requests.post("http://127.0.0.1:8001/api/detect", json=payload, timeout=10)
    assert res_detect.status_code == 200, f"/api/detect failed: {res_detect.text}"
    data = res_detect.json()
    assert "detections" in data
    assert "inference_ms" in data
    print(f"✓ Inference successful! Latency: {data['inference_ms']}ms")

if __name__ == "__main__":
    test_api()
    print("\nAll automated integration tests PASSED!")
