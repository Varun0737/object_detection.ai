/**
 * Real-Time Object Detection - Minimal Camera-Only Version
 * Auto-starts camera and detection on page load
 */

// DOM Elements
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const status = document.getElementById('status');

// State
let stream = null;
let ws = null;
let isRunning = false;
let frameInterval = null;
let offscreenCanvas = null;
let offscreenCtx = null;

// Configuration
const WS_URL = 'ws://localhost:8000/ws';
const FPS = 12;
const MIN_SIMILARITY = 30;  // Lower threshold to detect all objects

function getConfig() {
    return {
        min_area_px: 300,  // Very low to catch small objects
        min_similarity: MIN_SIMILARITY,
        speak_cooldown_ms: 2000,
        max_fps: FPS,
        downscale_width: 640,
        weights: { color: 0.45, shape: 0.45, size: 0.10 },
        single_object_mode: false  // DISABLED - show ALL detections!
    };
}

function updateStatus(message, type = 'default') {
    status.textContent = message;
    status.className = type;
}

async function startCamera() {
    try {
        updateStatus('Starting camera...', 'default');

        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' }
        });

        video.srcObject = stream;
        await new Promise(resolve => video.onloadedmetadata = resolve);

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = video.videoWidth;
        offscreenCanvas.height = video.videoHeight;
        offscreenCtx = offscreenCanvas.getContext('2d');

        updateStatus('Camera ready', 'active');
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        updateStatus('Camera access denied', 'error');
        return false;
    }
}

function connectWebSocket() {
    return new Promise((resolve, reject) => {
        updateStatus('Connecting...', 'default');

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('Connected to server');
            updateStatus('Detecting objects...', 'active');
            resolve();
        };

        ws.onmessage = (event) => handleDetectionResult(JSON.parse(event.data));
        ws.onerror = (error) => updateStatus('Connection error', 'error');
        ws.onclose = () => {
            if (isRunning) {
                updateStatus('Disconnected', 'error');
            }
        };

        setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) reject(new Error('Timeout'));
        }, 5000);
    });
}

function captureAndSendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;

    try {
        offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        const jpegDataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.6);
        const jpegBase64 = jpegDataUrl.split(',')[1];

        ws.send(JSON.stringify({
            jpeg_b64: jpegBase64,
            config: getConfig()
        }));
    } catch (error) {
        console.error('Frame capture error:', error);
    }
}

function handleDetectionResult(data) {
    if (data.error) {
        console.error('Detection error:', data.error);
        return;
    }

    const { frame_w, frame_h, detections } = data;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / frame_w;
    const scaleY = overlay.height / frame_h;

    detections.forEach(detection => {
        drawDetection(detection, scaleX, scaleY);

        // Speak object name
        if (detection.should_speak) {
            speak(detection.spoken_text);
        }
    });
}

function drawDetection(detection, scaleX, scaleY) {
    const [x, y, w, h] = detection.bbox;

    const sx = x * scaleX;
    const sy = y * scaleY;
    const sw = w * scaleX;
    const sh = h * scaleY;

    // Color based on confidence
    let color = detection.similarity >= 80 ? '#10b981' :
        detection.similarity >= 60 ? '#f59e0b' : '#ef4444';

    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(sx, sy, sw, sh);

    // Draw label
    const label = detection.label;
    ctx.font = 'bold 20px Inter, sans-serif';
    const textWidth = ctx.measureText(label).width;
    const padding = 12;

    ctx.fillStyle = color;
    ctx.fillRect(sx, sy - 35, textWidth + padding * 2, 35);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, sx + padding, sy - 10);
}

function speak(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

async function start() {
    if (isRunning) return;

    const cameraStarted = await startCamera();
    if (!cameraStarted) {
        alert('Please allow camera access and reload the page.');
        return;
    }

    try {
        await connectWebSocket();

        const intervalMs = 1000 / FPS;
        frameInterval = setInterval(captureAndSendFrame, intervalMs);

        isRunning = true;
    } catch (error) {
        console.error('Failed to start:', error);
        alert('Failed to connect to server. Make sure backend is running on localhost:8000');
        updateStatus('Connection failed', 'error');
    }
}

// Auto-start when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
} else {
    start();
}
