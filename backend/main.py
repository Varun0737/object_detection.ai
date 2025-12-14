"""
FastAPI WebSocket server for real-time object detection.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import cv2
import numpy as np
import time
from detector import ObjectDetector
from tracker import ObjectTracker

app = FastAPI(title="Object Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and tracker
detector = ObjectDetector()
tracker = ObjectTracker()


def decode_jpeg_base64(jpeg_b64: str) -> np.ndarray:
    """
    Decode base64 JPEG string to OpenCV frame.
    
    Args:
        jpeg_b64: Base64 encoded JPEG string
    
    Returns:
        BGR numpy array
    """
    # Remove data URL prefix if present
    if "," in jpeg_b64:
        jpeg_b64 = jpeg_b64.split(",")[1]
    
    # Decode base64
    jpeg_bytes = base64.b64decode(jpeg_b64)
    
    # Decode JPEG
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return frame


def downscale_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Downscale frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width in pixels
    
    Returns:
        Downscaled frame
    """
    h, w = frame.shape[:2]
    
    if w <= target_width:
        return frame
    
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Object Detection API is running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time object detection.
    
    Protocol:
        Client sends: {"jpeg_b64": "...", "config": {...}}
        Server responds: {"frame_w": int, "frame_h": int, "detections": [...], "ts_ms": int}
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    # Reset tracker for new connection
    tracker.reset()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Extract JPEG and config
            jpeg_b64 = message.get("jpeg_b64", "")
            config = message.get("config", {})
            
            # Set single object mode by default
            if "single_object_mode" not in config:
                config["single_object_mode"] = True
            
            if not jpeg_b64:
                await websocket.send_text(json.dumps({
                    "error": "No jpeg_b64 provided"
                }))
                continue
            
            try:
                # Decode frame
                frame = decode_jpeg_base64(jpeg_b64)
                
                if frame is None:
                    await websocket.send_text(json.dumps({
                        "error": "Failed to decode JPEG"
                    }))
                    continue
                
                # Downscale for performance
                target_width = config.get("downscale_width", 640)
                frame = downscale_frame(frame, target_width)
                
                frame_h, frame_w = frame.shape[:2]
                
                # Detect objects
                detections = detector.detect_objects(frame, config)
                
                # Update tracker and add tracking info
                detections = tracker.update(detections, frame_w, frame_h, config)
                
                # Filter out internal fields for response
                response_detections = []
                for det in detections:
                    response_detections.append({
                        "id": det["id"],
                        "bbox": det["bbox"],
                        "centroid": det["centroid"],
                        "color": det["color"],
                        "shape": det["shape"],
                        "size": det["size"],
                        "label": det["label"],
                        "similarity": det["similarity"],
                        "should_speak": det["should_speak"],
                        "spoken_text": det["spoken_text"]
                    })
                
                # Send response
                response = {
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                    "detections": response_detections,
                    "ts_ms": int(time.time() * 1000)
                }
                
                await websocket.send_text(json.dumps(response))
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"Processing error: {str(e)}"
                }))
    
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
