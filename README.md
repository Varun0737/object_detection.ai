# 🎯 Real-Time Object Detection Web App

Production-ready web application that uses your webcam to detect objects in real-time using **color + shape + size** analysis (not generic AI labels). Fully privacy-friendly - everything runs locally, no data is uploaded.

## ✨ Features

- **Real-time Detection**: Processes 10-15 FPS from your webcam
- **Computer Vision Analysis**:
  - **Color**: Detects red, orange, yellow, green, blue, purple, pink, brown, black, white, gray
  - **Shape**: Classifies as circle, triangle, rectangle, polygon, or unknown
  - **Size**: Categorizes as small, medium, or large (relative to frame)
- **Similarity Scoring**: 0-100% confidence based on weighted features
- **Text-to-Speech**: Announces detections using browser speech synthesis (debounced)
- **Object Tracking**: Centroid-based tracking across frames
- **Live UI**: Bounding boxes, labels, and similarity overlays on video
- **Full Control**: Adjustable weights, thresholds, and FPS

## 🏗️ Project Structure

```
object_detection.ai/
├── backend/
│   ├── main.py          # FastAPI WebSocket server
│   ├── detector.py      # OpenCV detection pipeline
│   ├── tracker.py       # Centroid-based tracker
│   ├── utils.py         # HSV ranges & similarity functions
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── index.html       # UI structure
│   ├── style.css        # Premium dark theme styling
│   └── app.js           # Webcam, WebSocket, canvas, speech
└── README.md
```

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.11+** (recommended: Python 3.11 or 3.12)
- **pip** (package manager)
- **Modern browser** (Chrome/Edge recommended for best compatibility)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended):
   
   **macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   **Windows**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   You should see:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   INFO:     Application startup complete.
   ```

### Frontend Setup

1. **Open the frontend**:
   
   Simply open `frontend/index.html` in your browser:
   - **Chrome**: File → Open File → Select `index.html`
   - **Or use a local server** (optional, recommended for development):
     ```bash
     cd frontend
     python3 -m http.server 8080
     # Then open http://localhost:8080 in browser
     ```

2. **Grant camera permissions** when prompted

3. **Click "Start Camera"** to begin detection

## 🎮 How to Use

### Basic Workflow

1. **Start Backend**: Run `uvicorn main:app --reload` in the backend folder
2. **Open Frontend**: Open `index.html` in Chrome
3. **Start Camera**: Click the "Start Camera" button
4. **Show Objects**: Hold colored objects (paper, toys, etc.) in front of camera
5. **Watch Detection**: See bounding boxes, labels, and similarity scores
6. **Listen**: Hear announcements via text-to-speech

### UI Controls

| Control | Purpose |
|---------|---------|
| **Start Camera** | Begin webcam capture and detection |
| **Stop** | Stop detection and close camera |
| **Min Similarity** | Minimum similarity % to announce (0-100) |
| **Min Area** | Minimum object size in pixels (500-5000) |
| **FPS** | Frame processing rate (1-30) |
| **Color Weight** | Importance of color in similarity (0-1) |
| **Shape Weight** | Importance of shape in similarity (0-1) |
| **Size Weight** | Importance of size in similarity (0-1) |
| **Enable Speech** | Toggle text-to-speech announcements |

### Best Practices

- **Good Lighting**: Ensure room is well-lit for better color detection
- **Solid Colors**: Use objects with solid, vibrant colors for best results
- **Clean Background**: Plain backgrounds improve detection accuracy
- **Distance**: Hold objects 1-3 feet from camera
- **FPS Tuning**: Start with 12 FPS; increase if your machine can handle it

## 🔧 How It Works

### Detection Pipeline

1. **Frame Capture**: Frontend captures webcam frame, encodes as JPEG, sends via WebSocket
2. **Preprocessing**: Backend downscales frame (640px width) and applies Gaussian blur
3. **Color Segmentation**: Converts to HSV, creates masks for each color using thresholds
4. **Morphology**: Applies open/close operations to reduce noise
5. **Contour Detection**: Finds contours, filters by minimum area
6. **Feature Extraction**: For each contour:
   - Bounding box, centroid, area, perimeter
   - Circularity = 4πA / P²
   - Solidity = area / convex_hull_area
   - Polygon approximation (edge count)
7. **Shape Classification**:
   - Circle: circularity ≥ 0.80, solidity ≥ 0.90, edges ≥ 6
   - Triangle: edges == 3, solidity ≥ 0.85
   - Rectangle: edges == 4, solidity ≥ 0.85
   - Polygon: edges ≥ 5, solidity ≥ 0.80
   - Unknown: otherwise
8. **Color Classification**: Median HSV inside contour → mapped to color name
9. **Size Classification**: Box area / frame area → small/medium/large
10. **Similarity Scoring**: Weighted combination of color, shape, size scores
11. **Tracking**: Match detections to previous objects by centroid distance
12. **Speech Decision**: Announce if similarity ≥ threshold AND (new object OR label changed OR cooldown elapsed)

### Similarity Calculation

```
S_color = 1.0 (exact match) | 0.6 (adjacent) | 0.0 (other)
S_shape = 1.0 (strong) | 0.7 (polygon) | 0.4 (unknown)
S_size  = 1.0 - |box_ratio - bucket_center| / 0.20

similarity = 100 * (wC*S_color + wSh*S_shape + wSz*S_size) / (wC+wSh+wSz)
```

## 🎨 Customization

### Adjust HSV Color Ranges

Edit `backend/utils.py` → `DEFAULT_HSV_RANGES`:

```python
"red": [
    (np.array([0, 100, 100]), np.array([10, 255, 255])),
    (np.array([170, 100, 100]), np.array([179, 255, 255]))
],
```

### Adjust Size Buckets

Edit `backend/utils.py` → `get_size_bucket()`:

```python
if box_ratio < 0.03:    # Current: 3%
    return "small"
elif box_ratio < 0.12:  # Current: 12%
    return "medium"
```

### Change Detection Sensitivity

- **More detections**: Decrease Min Area (e.g., 800), lower Min Similarity (e.g., 50)
- **Fewer detections**: Increase Min Area (e.g., 2000), raise Min Similarity (e.g., 80)

## 🐛 Troubleshooting

### Camera Not Working
- **Chrome**: Settings → Privacy and Security → Site Settings → Camera → Allow
- **Safari**: Safari → Settings for This Website → Camera → Allow
- Try restarting browser or computer

### WebSocket Connection Failed
- Ensure backend is running: `uvicorn main:app --reload`
- Check console for errors: F12 → Console tab
- Verify URL is `ws://localhost:8000/ws`
- Firewall: Allow port 8000

### Poor Detection Accuracy
- Improve lighting (add lamps, avoid shadows)
- Use high-contrast colored objects
- Try simpler/cleaner background
- Adjust HSV ranges in `utils.py` for your lighting conditions

### Low FPS / Lag
- Decrease FPS slider (try 8-10)
- Increase Min Area (filters out small objects)
- Close other applications
- Use smaller webcam resolution

### No Speech
- Check browser supports Web Speech API (Chrome/Edge recommended)
- Verify system volume is on
- Toggle "Enable Speech" off and on
- Some browsers require HTTPS (use local file:// for testing)

## 📊 Performance Tips

- **Optimal FPS**: 10-15 FPS balances real-time feel with CPU usage
- **Downscaling**: Detection runs on 640px width (configurable in `app.js`)
- **Min Area**: Setting to 1200-1500 filters noise effectively
- **Weights**: Default `{color: 0.45, shape: 0.45, size: 0.10}` works well

## 🔐 Privacy

- ✅ **100% Local**: All processing happens on your machine
- ✅ **No Uploads**: Frames are never sent to external servers
- ✅ **No Storage**: Frames are processed in memory and discarded
- ✅ **No Tracking**: No analytics or data collection
- ✅ **Open Source**: All code is visible and auditable

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, FastAPI, OpenCV, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Communication**: WebSocket (real-time bidirectional)
- **Computer Vision**: OpenCV contour analysis, HSV color segmentation
- **Speech**: Web Speech API (browser native)

## 📝 API Reference

### WebSocket `/ws`

**Client → Server**:
```json
{
  "jpeg_b64": "base64_encoded_jpeg",
  "config": {
    "min_area_px": 1200,
    "min_similarity": 70,
    "speak_cooldown_ms": 2000,
    "max_fps": 12,
    "downscale_width": 640,
    "weights": {"color": 0.45, "shape": 0.45, "size": 0.10}
  }
}
```

**Server → Client**:
```json
{
  "frame_w": 640,
  "frame_h": 360,
  "detections": [
    {
      "id": 1,
      "bbox": [x, y, w, h],
      "centroid": [cx, cy],
      "color": "red",
      "shape": "circle",
      "size": "medium",
      "label": "Red Circle (Medium)",
      "similarity": 86.4,
      "should_speak": true,
      "spoken_text": "Red circle, medium, eighty six percent"
    }
  ],
  "ts_ms": 1234567890
}
```

## 📄 License

MIT License - Free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 🙏 Acknowledgments

Built with ❤️ using open-source technologies:
- OpenCV for computer vision
- FastAPI for blazing-fast WebSocket server
- Modern web standards (WebRTC, Canvas, Web Speech API)

---

**Happy Detecting! 🎯**

