"""
Object tracker using centroid-based tracking.
Tracks objects across frames and determines when to speak announcements.
"""

import time
import math
from typing import List, Dict, Optional
from utils import format_spoken_text


class ObjectTracker:
    """Simple centroid-based object tracker."""
    
    def __init__(self, max_distance: int = 60):
        """
        Initialize tracker.
        
        Args:
            max_distance: Maximum pixel distance to match objects between frames
        """
        self.max_distance = max_distance
        self.next_id = 1
        self.tracked_objects = {}  # id -> {centroid, last_seen_ms, last_spoken_ms, last_label}
    
    def _calculate_distance(self, p1: List[int], p2: List[int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _find_closest_match(
        self,
        centroid: List[int],
        frame_w: int,
        frame_h: int
    ) -> Optional[int]:
        """
        Find the closest existing object to this centroid.
        
        Args:
            centroid: [cx, cy] of new detection
            frame_w: Frame width (for scaling max distance)
            frame_h: Frame height (for scaling max distance)
        
        Returns:
            Object ID if match found, None otherwise
        """
        # Scale max distance based on frame size
        scaled_max_dist = self.max_distance * (frame_w / 640.0)
        
        closest_id = None
        min_distance = float('inf')
        
        for obj_id, obj_data in self.tracked_objects.items():
            distance = self._calculate_distance(centroid, obj_data["centroid"])
            
            if distance < min_distance and distance < scaled_max_dist:
                min_distance = distance
                closest_id = obj_id
        
        return closest_id
    
    def update(
        self,
        detections: List[Dict],
        frame_w: int,
        frame_h: int,
        config: Dict
    ) -> List[Dict]:
        """
        Update tracked objects with new detections and determine speech.
        
        Args:
            detections: List of detection dictionaries
            frame_w: Frame width
            frame_h: Frame height
            config: Configuration with min_similarity, speak_cooldown_ms
        
        Returns:
            Updated detections with id, should_speak, spoken_text fields
        """
        current_time_ms = int(time.time() * 1000)
        min_similarity = config.get("min_similarity", 70)
        speak_cooldown_ms = config.get("speak_cooldown_ms", 2000)
        
        matched_ids = set()
        updated_detections = []
        
        # Match each detection to existing objects
        for detection in detections:
            centroid = detection["centroid"]
            label = detection["label"]
            similarity = detection["similarity"]
            
            # Try to find a match
            matched_id = self._find_closest_match(centroid, frame_w, frame_h)
            
            if matched_id is not None:
                # Update existing object
                obj_id = matched_id
                matched_ids.add(obj_id)
                
                old_data = self.tracked_objects[obj_id]
                old_label = old_data["last_label"]
                last_spoken_ms = old_data["last_spoken_ms"]
                
                # Determine if we should speak
                should_speak = False
                time_since_spoken = current_time_ms - last_spoken_ms
                
                # Speak if:
                # 1. Similarity >= threshold
                # 2. AND (label changed OR enough time has passed)
                if similarity >= min_similarity:
                    if label != old_label or time_since_spoken >= speak_cooldown_ms:
                        should_speak = True
                
                # Update object data
                self.tracked_objects[obj_id].update({
                    "centroid": centroid,
                    "last_seen_ms": current_time_ms,
                    "last_label": label
                })
                
                if should_speak:
                    self.tracked_objects[obj_id]["last_spoken_ms"] = current_time_ms
            
            else:
                # New object
                obj_id = self.next_id
                self.next_id += 1
                matched_ids.add(obj_id)
                
                # Determine if we should speak for new objects
                should_speak = similarity >= min_similarity
                
                # Add to tracked objects
                self.tracked_objects[obj_id] = {
                    "centroid": centroid,
                    "last_seen_ms": current_time_ms,
                    "last_spoken_ms": current_time_ms if should_speak else 0,
                    "last_label": label
                }
            
            # Add tracking fields to detection
            detection["id"] = obj_id
            detection["should_speak"] = should_speak
            detection["spoken_text"] = format_spoken_text(
                detection["color"],
                detection["shape"],
                detection["size"],
                detection["similarity"]
            )
            
            updated_detections.append(detection)
        
        # Remove objects that weren't seen (timeout after 1 second)
        timeout_ms = 1000
        ids_to_remove = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_id not in matched_ids:
                time_since_seen = current_time_ms - obj_data["last_seen_ms"]
                if time_since_seen > timeout_ms:
                    ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
        
        return updated_detections
    
    def reset(self):
        """Reset tracker state."""
        self.next_id = 1
        self.tracked_objects = {}
