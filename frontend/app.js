/**
 * VisionEye Frontend Application — v2.0 (Cross-Platform Mobile & Desktop)
 *
 * Features:
 *  - Dual Engine AI:
 *      * Primary: Remote YOLO11 ONNX (Apple Silicon CoreML hardware accelerated) over WebSocket
 *      * Fallback: Client-side in-browser Deep Learning (COCO-SSD / WebGL) for standalone/GitHub Pages mode
 *  - Cross-Platform Support:
 *      * Front/Rear camera switching for smartphones (iPhone Safari & Android Chrome)
 *      * Auto-mirroring management (mirrored on selfie camera, unmirrored on rear camera)
 *      * Interactive QR Code modal for instant mobile phone pairing
 *      * Mobile audio unlock for SpeechSynthesis TTS narration
 *      * Fullscreen scanning mode
 *  - Real-Time Smoothing:
 *      * 60 FPS client-side bounding box interpolation (lerp)
 *      * SORT tracking ID persistence
 *      * Debounced voice announcements
 */

// ─── DOM Elements ────────────────────────────────────────────────────────────

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMsg = document.getElementById("loadingMsg");

const connDot = document.getElementById("connDot");
const connStatus = document.getElementById("connStatus");
const providerBadge = document.getElementById("providerBadge");
const providerText = document.getElementById("providerText");
const fpsVal = document.getElementById("fpsVal");
const latencyVal = document.getElementById("latencyVal");
const objCountVal = document.getElementById("objCountVal");

const toggleCamBtn = document.getElementById("toggleCamBtn");
const toggleCamText = document.getElementById("toggleCamText");
const snapshotBtn = document.getElementById("snapshotBtn");
const mirrorBtn = document.getElementById("mirrorBtn");
const mirrorBtnText = document.getElementById("mirrorBtnText");
const flipCamBtn = document.getElementById("flipCamBtn");
const flipCamFloatingBtn = document.getElementById("flipCamFloatingBtn");
const fullscreenFloatingBtn = document.getElementById("fullscreenFloatingBtn");
const sharePhoneBtn = document.getElementById("sharePhoneBtn");

const qrModal = document.getElementById("qrModal");
const closeQrModal = document.getElementById("closeQrModal");
const qrCanvasContainer = document.getElementById("qrCanvasContainer");
const shareUrlInput = document.getElementById("shareUrlInput");
const copyUrlBtn = document.getElementById("copyUrlBtn");

const cameraSelect = document.getElementById("cameraSelect");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

const voiceToggle = document.getElementById("voiceToggle");
const voiceStatus = document.getElementById("voiceStatus");
const voiceStatusText = document.getElementById("voiceStatusText");

const objectsList = document.getElementById("objectsList");
const emptyState = document.getElementById("emptyState");
const activeObjectsBadge = document.getElementById("activeObjectsBadge");
const activityLog = document.getElementById("activityLog");

const snapshotsSection = document.getElementById("snapshotsSection");
const snapshotsGallery = document.getElementById("snapshotsGallery");
const clearSnapshotsBtn = document.getElementById("clearSnapshotsBtn");

// ─── State ───────────────────────────────────────────────────────────────────

let ws = null;
let wsRetries = 0;
let currentStream = null;
let isDetecting = true;
let isMirrored = true;
let currentFacingMode = "user"; // "user" (front) or "environment" (rear)
let confidenceThreshold = 0.35;
let publicTunnelUrl = null;

// FPS tracking
let fpsFrameCount = 0;
let fpsLastTime = performance.now();
let currentFps = 0;

// Voice
let lastAnnouncedKey = "";
let lastAnnounceTime = 0;
const ANNOUNCE_COOLDOWN_MS = 4000;

// Client-Side AI Fallback (COCO-SSD / TensorFlow.js)
let clientModel = null;
let useClientModel = false;
let isClientInferring = false;

// ─── Client-Side Interpolation State ─────────────────────────────────────────

/** @type {Map<number, TrackedBox>} track_id -> interpolation state */
const trackedBoxes = new Map();

/**
 * @typedef {Object} TrackedBox
 * @property {number} track_id
 * @property {string} className
 * @property {number} classId
 * @property {number[]} prevBbox   - [x1, y1, x2, y2]
 * @property {number[]} targetBbox
 * @property {number[]} displayBbox
 * @property {number} confidence
 * @property {boolean} missing
 * @property {number} missingFrames
 */

const LERP_FACTOR = 0.45;
const MAX_MISSING_FRAMES = 12;

// ─── Color Palette for Bounding Boxes ────────────────────────────────────────

const CLASS_COLORS = [
  "#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6",
  "#06b6d4", "#84cc16", "#f97316", "#e11d48", "#6366f1",
  "#14b8a6", "#eab308", "#d946ef", "#3b82f6", "#22c55e"
];

function getColorForClass(className, classId = 0) {
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = (hash << 5) - hash + className.charCodeAt(i);
    hash |= 0;
  }
  const idx = Math.abs(hash + classId) % CLASS_COLORS.length;
  return CLASS_COLORS[idx];
}

// ─── 1. Camera & Viewport Management ─────────────────────────────────────────

async function setupCamera(deviceId = null) {
  loadingOverlay.classList.remove("hidden");
  loadingMsg.textContent = "Connecting to camera...";

  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    audio: false,
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      facingMode: deviceId ? undefined : { ideal: currentFacingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    await new Promise(resolve => {
      video.onloadedmetadata = () => { video.play(); resolve(); };
    });
    updateCanvasDimensions();
    loadingOverlay.classList.add("hidden");
    logActivity(`Camera active (${currentFacingMode === "user" ? "Front" : "Rear"}).`);
    await enumerateCameras();
  } catch (err) {
    console.error("Camera access failed:", err);
    loadingMsg.textContent = `Camera error: ${err.message}. Please allow camera access in browser permissions.`;
    logActivity(`Camera access error: ${err.message}`);
  }
}

function updateCanvasDimensions() {
  if (video.videoWidth && video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}
window.addEventListener("resize", updateCanvasDimensions);

async function enumerateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    cameraSelect.innerHTML = "";
    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });
  } catch (err) {
    console.warn("Could not enumerate devices:", err);
  }
}

cameraSelect.addEventListener("change", (e) => {
  setupCamera(e.target.value);
});

// Flip Camera (Front / Rear)
async function toggleFacingMode() {
  currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
  // Auto-mirror: mirror for front (selfie), don't mirror for rear (environment)
  isMirrored = (currentFacingMode === "user");
  updateMirrorState();
  logActivity(`Switched camera to ${currentFacingMode === "user" ? "Front (Selfie)" : "Rear (Environment)"}.`);
  await setupCamera();
}

flipCamBtn.addEventListener("click", toggleFacingMode);
if (flipCamFloatingBtn) {
  flipCamFloatingBtn.addEventListener("click", toggleFacingMode);
}

// Fullscreen
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    const docEl = document.documentElement;
    if (docEl.requestFullscreen) {
      docEl.requestFullscreen().catch(() => {});
    } else if (docEl.webkitRequestFullscreen) {
      docEl.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    }
  }
}

if (fullscreenFloatingBtn) {
  fullscreenFloatingBtn.addEventListener("click", toggleFullscreen);
}

// Mirror toggle
function updateMirrorState() {
  if (isMirrored) {
    video.classList.add("mirrored");
    canvas.classList.add("mirrored");
    mirrorBtn.classList.add("btn-active");
  } else {
    video.classList.remove("mirrored");
    canvas.classList.remove("mirrored");
    mirrorBtn.classList.remove("btn-active");
  }
}

mirrorBtn.addEventListener("click", () => {
  isMirrored = !isMirrored;
  updateMirrorState();
  logActivity(`Mirror mode ${isMirrored ? "enabled" : "disabled"}.`);
});

// ─── 2. WebSocket & Client-Side Inference ────────────────────────────────────

let sendStartTime = 0;
let awaitingResponse = false;

function connectWebSocket() {
  // If explicitly using client model, do not connect
  if (useClientModel) return;

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/detect`;
  
  try {
    ws = new WebSocket(wsUrl);
  } catch (err) {
    handleWsFailure();
    return;
  }

  ws.onopen = () => {
    wsRetries = 0;
    connDot.classList.add("connected");
    connStatus.innerHTML = `<span class="dot connected"></span> Live`;
    logActivity("Connected to YOLO11 AI inference server.");
    awaitingResponse = false;
    scheduleNextCapture();
  };

  ws.onmessage = (event) => {
    awaitingResponse = false;
    try {
      const data = JSON.parse(event.data);
      if (data.detections !== undefined) {
        handleServerDetections(data);
      }
    } catch (e) {
      console.error("Failed to parse WS response:", e);
    }
    scheduleNextCapture();
  };

  ws.onclose = () => {
    connDot.classList.remove("connected");
    connStatus.innerHTML = `<span class="dot"></span> Disconnected`;
    awaitingResponse = false;
    wsRetries++;
    if (wsRetries > 2 && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
      // Switch to standalone client-side ML
      activateClientSideModel();
    } else {
      setTimeout(connectWebSocket, 2500);
    }
  };

  ws.onerror = () => {
    handleWsFailure();
  };
}

function handleWsFailure() {
  wsRetries++;
  if (wsRetries >= 2 && !useClientModel) {
    activateClientSideModel();
  }
}

async function activateClientSideModel() {
  if (useClientModel) return;
  useClientModel = true;
  providerText.textContent = "COCO-SSD (In-Browser WebGL)";
  connStatus.innerHTML = `<span class="dot connected"></span> Standalone`;
  logActivity("Switching to in-browser Deep Learning (COCO-SSD via WebGL)...");

  try {
    if (typeof cocoSsd !== "undefined") {
      clientModel = await cocoSsd.load();
      logActivity("Client deep learning model loaded successfully (30 FPS WebGL).");
      runClientInferenceLoop();
    } else {
      logActivity("Client-side model scripts unavailable.");
    }
  } catch (err) {
    console.error("Client model loading failed:", err);
    logActivity(`Failed to load client model: ${err.message}`);
  }
}

async function runClientInferenceLoop() {
  if (!useClientModel || !clientModel) return;

  if (isDetecting && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    try {
      const start = performance.now();
      const predictions = await clientModel.detect(video, 10, confidenceThreshold);
      const elapsed = Math.round(performance.now() - start);
      latencyVal.textContent = elapsed;

      // Transform COCO-SSD format to standard VisionEye detections
      const detections = predictions.map((p, idx) => ({
        track_id: idx + 1,
        class_name: p.class,
        class_id: idx,
        confidence: p.score,
        bbox: [p.bbox[0], p.bbox[1], p.bbox[0] + p.bbox[2], p.bbox[1] + p.bbox[3]]
      }));

      handleServerDetections({
        detections: detections,
        count: detections.length,
        inference_ms: elapsed
      });
    } catch (err) {
      console.warn("Client inference frame dropped:", err);
    }
  }

  requestAnimationFrame(runClientInferenceLoop);
}

function scheduleNextCapture() {
  if (useClientModel) return;
  if (isDetecting && !awaitingResponse) {
    requestAnimationFrame(captureAndSendFrame);
  }
}

// Offscreen canvas for frame capture
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

function captureAndSendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !isDetecting || awaitingResponse) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    requestAnimationFrame(captureAndSendFrame);
    return;
  }

  // Scale to 640px wide for optimal real-time latency
  const scale = Math.min(1.0, 640 / video.videoWidth);
  const targetW = Math.round(video.videoWidth * scale);
  const targetH = Math.round(video.videoHeight * scale);

  offscreenCanvas.width = targetW;
  offscreenCanvas.height = targetH;
  offscreenCtx.drawImage(video, 0, 0, targetW, targetH);

  const base64Data = offscreenCanvas.toDataURL("image/jpeg", 0.78);
  sendStartTime = performance.now();
  awaitingResponse = true;

  ws.send(JSON.stringify({
    image: base64Data,
    conf_threshold: confidenceThreshold
  }));
}

// ─── 3. Handle Detections & Interpolation ─────────────────────────────────────

function handleServerDetections(data) {
  if (!useClientModel) {
    const roundtripMs = Math.round(performance.now() - sendStartTime);
    latencyVal.textContent = roundtripMs;
  }
  
  const serverDetections = data.detections || [];
  const seenTrackIds = new Set();

  for (const det of serverDetections) {
    const tid = det.track_id;
    seenTrackIds.add(tid);

    const normBbox = normalizeBbox(det.bbox);

    if (trackedBoxes.has(tid)) {
      const existing = trackedBoxes.get(tid);
      existing.prevBbox = [...existing.displayBbox];
      existing.targetBbox = normBbox;
      existing.confidence = det.confidence;
      existing.className = det.class_name;
      existing.classId = det.class_id;
      existing.missing = false;
      existing.missingFrames = 0;
    } else {
      trackedBoxes.set(tid, {
        track_id: tid,
        className: det.class_name,
        classId: det.class_id,
        prevBbox: [...normBbox],
        targetBbox: [...normBbox],
        displayBbox: [...normBbox],
        confidence: det.confidence,
        missing: false,
        missingFrames: 0
      });
    }
  }

  // Handle missing tracks gracefully
  for (const [tid, box] of trackedBoxes.entries()) {
    if (!seenTrackIds.has(tid)) {
      box.missingFrames = (box.missingFrames || 0) + 1;
      box.missing = true;
      if (box.missingFrames > MAX_MISSING_FRAMES) {
        trackedBoxes.delete(tid);
      }
    }
  }

  // Update badges & narration
  updateActiveObjectsSidebar();
  maybeAnnounceObjects();
}

function normalizeBbox(bbox) {
  const [x1, y1, x2, y2] = bbox;
  return [
    Math.max(0, Math.min(canvas.width, x1)),
    Math.max(0, Math.min(canvas.height, y1)),
    Math.max(0, Math.min(canvas.width, x2)),
    Math.max(0, Math.min(canvas.height, y2))
  ];
}

// ─── 4. 60 FPS Render & Lerp Loop ───────────────────────────────────────────

function renderLoop() {
  calculateFPS();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let activeCount = 0;

  for (const [_, tb] of trackedBoxes.entries()) {
    // Lerp bbox toward target
    for (let i = 0; i < 4; i++) {
      tb.displayBbox[i] += (tb.targetBbox[i] - tb.displayBbox[i]) * LERP_FACTOR;
    }

    const alpha = tb.missing ? Math.max(0, 1 - (tb.missingFrames / MAX_MISSING_FRAMES)) : 1.0;
    if (alpha <= 0.05) continue;

    activeCount++;
    drawBoundingBox(tb, alpha);
  }

  objCountVal.textContent = activeCount;
  requestAnimationFrame(renderLoop);
}

function drawBoundingBox(tb, alpha = 1.0) {
  const [x1, y1, x2, y2] = tb.displayBbox;
  const w = x2 - x1;
  const h = y2 - y1;
  const color = getColorForClass(tb.className, tb.classId);

  ctx.save();
  ctx.globalAlpha = alpha;

  // Box border with glow
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.strokeRect(x1, y1, w, h);

  // Label tag
  ctx.shadowBlur = 0;
  const label = `${tb.className} ${Math.round(tb.confidence * 100)}%`;
  ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
  const textWidth = ctx.measureText(label).width;
  const tagH = 22;
  const tagW = textWidth + 14;
  const tagY = Math.max(0, y1 - tagH);

  ctx.fillStyle = color;
  ctx.fillRect(x1, tagY, tagW, tagH);

  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, x1 + 7, tagY + 15);

  ctx.restore();
}

function calculateFPS() {
  fpsFrameCount++;
  const now = performance.now();
  const delta = now - fpsLastTime;
  if (delta >= 1000) {
    currentFps = Math.round((fpsFrameCount * 1000) / delta);
    fpsVal.textContent = currentFps;
    fpsFrameCount = 0;
    fpsLastTime = now;
  }
}

// ─── 5. Sidebar & Object Counting ───────────────────────────────────────────

function updateActiveObjectsSidebar() {
  const activeList = Array.from(trackedBoxes.values()).filter(b => !b.missing);
  activeObjectsBadge.textContent = `${activeList.length} active`;

  if (activeList.length === 0) {
    emptyState.style.display = "flex";
    return;
  }
  emptyState.style.display = "none";

  // Aggregate counts by class
  const counts = new Map();
  for (const b of activeList) {
    counts.set(b.className, (counts.get(b.className) || 0) + 1);
  }

  // Render object chips
  const existingIds = new Set();
  counts.forEach((count, className) => {
    existingIds.add(className);
    let chip = document.getElementById(`chip-${className}`);
    if (!chip) {
      chip = document.createElement("div");
      chip.id = `chip-${className}`;
      chip.className = "object-chip";
      const color = getColorForClass(className);
      chip.innerHTML = `
        <span class="object-dot" style="background:${color};"></span>
        <span class="object-name">${className}</span>
        <span class="object-badge" id="chip-cnt-${className}">${count}</span>
      `;
      objectsList.appendChild(chip);
    } else {
      document.getElementById(`chip-cnt-${className}`).textContent = count;
    }
  });

  // Remove chips no longer visible
  Array.from(objectsList.children).forEach(child => {
    if (child.id && child.id.startsWith("chip-")) {
      const cls = child.id.replace("chip-", "");
      if (!existingIds.has(cls)) {
        objectsList.removeChild(child);
      }
    }
  });
}

// ─── 6. Speech Narration ────────────────────────────────────────────────────

function maybeAnnounceObjects() {
  if (!voiceToggle.checked) return;
  if (!("speechSynthesis" in window)) return;

  const now = performance.now();
  if (now - lastAnnounceTime < ANNOUNCE_COOLDOWN_MS) return;

  const activeNames = Array.from(
    new Set(Array.from(trackedBoxes.values()).filter(b => !b.missing).map(b => b.className))
  );

  if (activeNames.length === 0) return;

  const currentKey = activeNames.sort().join(",");
  if (currentKey === lastAnnouncedKey) return;

  lastAnnouncedKey = currentKey;
  lastAnnounceTime = now;

  let phrase = "";
  if (activeNames.length === 1) {
    phrase = `I see a ${activeNames[0]}`;
  } else if (activeNames.length === 2) {
    phrase = `I see a ${activeNames[0]} and a ${activeNames[1]}`;
  } else {
    phrase = `I see ${activeNames.slice(0, 3).join(", ")}`;
  }

  voiceStatusText.textContent = `Announcing: "${phrase}"`;
  const utterance = new SpeechSynthesisUtterance(phrase);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// Mobile Audio Unlock: iOS Safari & Chrome require a user interaction
function unlockAudio() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.resume();
  }
}
document.addEventListener("touchstart", unlockAudio, { once: true });
document.addEventListener("click", unlockAudio, { once: true });

// ─── 7. Controls & QR Code Modal ─────────────────────────────────────────────

toggleCamBtn.addEventListener("click", () => {
  isDetecting = !isDetecting;
  toggleCamText.textContent = isDetecting ? "Pause Detection" : "Resume Detection";
  toggleCamBtn.classList.toggle("btn-danger", !isDetecting);
  if (isDetecting) {
    scheduleNextCapture();
    logActivity("Detection resumed.");
  } else {
    logActivity("Detection paused.");
  }
});

confSlider.addEventListener("input", (e) => {
  confidenceThreshold = parseInt(e.target.value) / 100.0;
  confValue.textContent = `${e.target.value}%`;
});

// Snapshots
snapshotBtn.addEventListener("click", () => {
  if (video.videoWidth === 0) return;

  const snapCanvas = document.createElement("canvas");
  snapCanvas.width = video.videoWidth;
  snapCanvas.height = video.videoHeight;
  const sCtx = snapCanvas.getContext("2d");

  if (isMirrored) {
    sCtx.translate(snapCanvas.width, 0);
    sCtx.scale(-1, 1);
  }
  sCtx.drawImage(video, 0, 0);

  if (isMirrored) {
    sCtx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // Draw active boxes on snapshot
  for (const [_, tb] of trackedBoxes.entries()) {
    if (tb.missing) continue;
    let [x1, y1, x2, y2] = tb.displayBbox;
    if (isMirrored) {
      const origX1 = x1;
      x1 = snapCanvas.width - x2;
      x2 = snapCanvas.width - origX1;
    }
    const color = getColorForClass(tb.className, tb.classId);
    sCtx.lineWidth = 3;
    sCtx.strokeStyle = color;
    sCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${tb.className} ${Math.round(tb.confidence * 100)}%`;
    sCtx.font = "bold 16px sans-serif";
    sCtx.fillStyle = color;
    sCtx.fillRect(x1, Math.max(0, y1 - 24), sCtx.measureText(label).width + 12, 24);
    sCtx.fillStyle = "#ffffff";
    sCtx.fillText(label, x1 + 6, Math.max(16, y1 - 6));
  }

  const dataUrl = snapCanvas.toDataURL("image/png");
  addSnapshotToGallery(dataUrl);
  logActivity("Captured snapshot saved.");
});

function addSnapshotToGallery(dataUrl) {
  snapshotsSection.style.display = "block";
  const div = document.createElement("div");
  div.className = "snapshot-item";
  div.innerHTML = `
    <img src="${dataUrl}" alt="Snapshot">
    <a href="${dataUrl}" download="visioneye-snapshot-${Date.now()}.png">Save</a>
  `;
  snapshotsGallery.prepend(div);
}

clearSnapshotsBtn.addEventListener("click", () => {
  snapshotsGallery.innerHTML = "";
  snapshotsSection.style.display = "none";
});

// QR Code Sharing Modal
function openQrModal() {
  qrModal.classList.remove("hidden");
  let targetUrl = window.location.href;

  if (publicTunnelUrl) {
    targetUrl = publicTunnelUrl;
  } else if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    // If running locally, check if user has ngrok
    targetUrl = "https://repurpose-usage-trodden.ngrok-free.dev";
  }

  shareUrlInput.value = targetUrl;
  qrCanvasContainer.innerHTML = "";

  if (typeof QRCode !== "undefined") {
    new QRCode(qrCanvasContainer, {
      text: targetUrl,
      width: 180,
      height: 180,
      colorDark: "#0b0f19",
      colorLight: "#ffffff",
      correctLevel: QRCode.CorrectLevel.M
    });
  }
}

sharePhoneBtn.addEventListener("click", openQrModal);
closeQrModal.addEventListener("click", () => qrModal.classList.add("hidden"));
qrModal.addEventListener("click", (e) => {
  if (e.target === qrModal) qrModal.classList.add("hidden");
});

copyUrlBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(shareUrlInput.value).then(() => {
    const originalText = copyUrlBtn.textContent;
    copyUrlBtn.textContent = "Copied!";
    setTimeout(() => { copyUrlBtn.textContent = originalText; }, 2000);
  });
});

function logActivity(msg) {
  const time = new Date().toLocaleTimeString();
  const li = document.createElement("li");
  li.className = "log-entry";
  li.textContent = `[${time}] ${msg}`;
  activityLog.prepend(li);
  if (activityLog.children.length > 25) {
    activityLog.removeChild(activityLog.lastChild);
  }
}

// ─── 8. Bootstrap & Server Info ──────────────────────────────────────────────

async function fetchServerInfo() {
  try {
    const res = await fetch("/api/info");
    if (res.ok) {
      const data = await res.json();
      if (data.tunnel_url) {
        publicTunnelUrl = data.tunnel_url;
      }
      if (data.provider) {
        providerText.textContent = `YOLO11 / ${data.provider.replace("ExecutionProvider", "")}`;
      }
    }
  } catch (e) {
    // Hosted statically
  }
}

window.addEventListener("DOMContentLoaded", () => {
  updateMirrorState();
  fetchServerInfo();
  setupCamera();
  connectWebSocket();
  renderLoop();
});
