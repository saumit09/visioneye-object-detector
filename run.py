#!/usr/bin/env python3
"""
VisionEye Unified Launcher (Laptop & Mobile)
Starts FastAPI AI inference server on port 8000 and connects ngrok secure HTTPS tunnel.
Provides scannable mobile link and handles graceful shutdown.
"""

import os
import sys
import time
import signal
import subprocess
import json
import urllib.request

PORT = int(os.environ.get("PORT", 8000))
NGROK_BIN = "/usr/local/bin/ngrok"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

uvicorn_proc = None
ngrok_proc = None


def cleanup(signum=None, frame=None):
    print("\nShutting down VisionEye services...")
    global uvicorn_proc, ngrok_proc
    if ngrok_proc:
        try:
            ngrok_proc.terminate()
            ngrok_proc.wait(timeout=2)
        except Exception:
            ngrok_proc.kill()
    if uvicorn_proc:
        try:
            uvicorn_proc.terminate()
            uvicorn_proc.wait(timeout=2)
        except Exception:
            uvicorn_proc.kill()
    print("All processes stopped. Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def get_ngrok_url():
    """Poll ngrok local inspection API to retrieve public HTTPS tunnel URL."""
    for _ in range(20):
        try:
            req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels")
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read().decode())
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    if t.get("public_url", "").startswith("https://"):
                        return t["public_url"]
        except Exception:
            pass
        time.sleep(0.5)
    return None


def main():
    global uvicorn_proc, ngrok_proc

    print("\n" + "=" * 68)
    print("  🚀 VisionEye — Real-Time Object Recognition (Laptop & Phone)")
    print("=" * 68)

    # 1. Start Uvicorn Server
    print(f"\n[1/2] Starting AI inference server on http://localhost:{PORT}...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    uvicorn_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=PROJECT_DIR,
        env=env
    )

    # 2. Start ngrok Tunnel
    public_url = None
    if os.path.exists(NGROK_BIN):
        print("[2/2] Launching secure HTTPS tunnel via ngrok...")
        ngrok_proc = subprocess.Popen(
            [NGROK_BIN, "http", str(PORT), "--log=stdout"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        public_url = get_ngrok_url()
    else:
        print("[2/2] ngrok not found. Running in local network mode only.")

    print("\n" + "=" * 68)
    print("  ✨ VISIONEYE IS READY!")
    print("=" * 68)
    print(f"  💻 LAPTOP ACCESS:     http://localhost:{PORT}")
    if public_url:
        print(f"  📱 PHONE ACCESS (HTTPS): {public_url}")
        print("\n  👉 Open either link on your Laptop or Smartphone browser!")
        print("  👉 Click the 'Phone Link / QR' button in the app to scan from phone.")
    else:
        print("  📱 PHONE ACCESS:     Connect phone to same Wi-Fi & open laptop IP")
    print("=" * 68)
    print("  Press Ctrl+C to stop the server\n")

    # Keep alive
    try:
        while True:
            if uvicorn_proc.poll() is not None:
                print("Uvicorn server exited unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
