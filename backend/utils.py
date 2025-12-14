"""
Utility functions and constants for object detection.
Includes HSV color ranges and helper methods.
"""

import numpy as np
from typing import Tuple, Dict

# Default HSV color ranges (Hue, Saturation, Value)
# OpenCV HSV: H: 0-179, S: 0-255, V: 0-255
DEFAULT_HSV_RANGES = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Red lower
        (np.array([170, 100, 100]), np.array([179, 255, 255]))    # Red upper (wraps around)
    ],
    "orange": [
        (np.array([11, 100, 100]), np.array([25, 255, 255]))
    ],
    "yellow": [
        (np.array([26, 100, 100]), np.array([35, 255, 255]))
    ],
    "green": [
        (np.array([36, 50, 50]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([86, 50, 50]), np.array([125, 255, 255]))
    ],
    "purple": [
        (np.array([126, 50, 50]), np.array([150, 255, 255]))
    ],
    "pink": [
        (np.array([151, 50, 100]), np.array([169, 255, 255]))
    ],
    "brown": [
        (np.array([10, 100, 20]), np.array([20, 255, 150]))
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([179, 255, 50]))
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([179, 30, 255]))
    ],
    "gray": [
        (np.array([0, 0, 51]), np.array([179, 30, 199]))
    ]
}

# Color adjacency for similarity calculation
COLOR_ADJACENCY = {
    "red": ["orange", "pink"],
    "orange": ["red", "yellow", "brown"],
    "yellow": ["orange", "green"],
    "green": ["yellow", "blue"],
    "blue": ["green", "purple"],
    "purple": ["blue", "pink"],
    "pink": ["purple", "red"],
    "brown": ["orange"],
    "black": ["gray"],
    "white": ["gray"],
    "gray": ["black", "white"]
}

# Size bucket centers for similarity calculation
SIZE_CENTERS = {
    "small": 0.015,
    "medium": 0.075,
    "large": 0.20
}


def hsv_to_color_name(h: int, s: int, v: int) -> str:
    """
    Map HSV values to a color name.
    
    Args:
        h: Hue (0-179)
        s: Saturation (0-255)
        v: Value (0-255)
    
    Returns:
        Color name as string
    """
    # Check black/white/gray first (based on saturation and value)
    if v < 50:
        return "black"
    if s < 30:
        if v > 200:
            return "white"
        return "gray"
    
    # Check hue ranges for colors
    if (h >= 0 and h <= 10) or h >= 170:
        return "red"
    elif h >= 11 and h <= 20:
        # Could be orange or brown based on value
        if v < 150:
            return "brown"
        return "orange"
    elif h >= 21 and h <= 35:
        return "yellow"
    elif h >= 36 and h <= 85:
        return "green"
    elif h >= 86 and h <= 125:
        return "blue"
    elif h >= 126 and h <= 150:
        return "purple"
    elif h >= 151 and h <= 169:
        return "pink"
    
    return "gray"  # fallback


def get_size_bucket(box_ratio: float) -> str:
    """
    Determine size bucket from bounding box ratio.
    
    Args:
        box_ratio: (width * height) / (frame_width * frame_height)
    
    Returns:
        Size bucket: 'small', 'medium', or 'large'
    """
    if box_ratio < 0.03:
        return "small"
    elif box_ratio < 0.12:
        return "medium"
    else:
        return "large"


def color_similarity(color1: str, color2: str) -> float:
    """
    Calculate similarity between two colors.
    
    Args:
        color1: First color name
        color2: Second color name
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if color1 == color2:
        return 1.0
    
    # Check if colors are adjacent
    if color2 in COLOR_ADJACENCY.get(color1, []):
        return 0.6
    
    return 0.0


def shape_similarity(shape: str, circularity: float, solidity: float, edges: int) -> float:
    """
    Calculate how strongly the shape matches its classification.
    
    Args:
        shape: Detected shape name
        circularity: Circularity measure
        solidity: Solidity measure
        edges: Number of polygon edges
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if shape == "circle":
        # Strong match if circularity >= 0.87
        if circularity >= 0.87 and solidity >= 0.90:
            return 1.0
        return 0.8
    
    elif shape == "triangle":
        if edges == 3 and solidity >= 0.90:
            return 1.0
        return 0.8
    
    elif shape == "rectangle":
        if edges == 4 and solidity >= 0.90:
            return 1.0
        return 0.8
    
    elif shape == "polygon":
        if edges >= 6 and solidity >= 0.85:
            return 0.7
        return 0.5
    
    else:  # unknown
        return 0.4


def size_similarity(box_ratio: float, size_bucket: str) -> float:
    """
    Calculate size similarity based on distance from bucket center.
    
    Args:
        box_ratio: Bounding box area ratio
        size_bucket: Size bucket name
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    center = SIZE_CENTERS[size_bucket]
    distance = abs(box_ratio - center)
    similarity = max(0, 1 - (distance / 0.20))
    return similarity


def calculate_similarity(
    color: str,
    shape: str,
    size: str,
    box_ratio: float,
    circularity: float,
    solidity: float,
    edges: int,
    mask_color: str,
    weights: Dict[str, float]
) -> float:
    """
    Calculate overall similarity score (0-100).
    
    Args:
        color: Detected color name
        shape: Detected shape name
        size: Detected size bucket
        box_ratio: Bounding box area ratio
        circularity: Circularity measure
        solidity: Solidity measure
        edges: Number of polygon edges
        mask_color: Color of the mask that detected this object
        weights: Dictionary with keys 'color', 'shape', 'size'
    
    Returns:
        Similarity percentage (0-100)
    """
    # Calculate component similarities
    s_color = color_similarity(mask_color, color)
    s_shape = shape_similarity(shape, circularity, solidity, edges)
    s_size = size_similarity(box_ratio, size)
    
    # Extract weights
    w_color = weights.get("color", 0.45)
    w_shape = weights.get("shape", 0.45)
    w_size = weights.get("size", 0.10)
    
    # Weighted average
    total_weight = w_color + w_shape + w_size
    if total_weight == 0:
        total_weight = 1.0
    
    similarity = 100 * (w_color * s_color + w_shape * s_shape + w_size * s_size) / total_weight
    
    return round(similarity, 1)


def format_spoken_text(color: str, shape: str, size: str, similarity: float) -> str:
    """
    Format the text to be spoken for an object.
    
    Args:
        color: Color name
        shape: Shape name (object class)
        size: Size bucket
        similarity: Similarity percentage
    
    Returns:
        Formatted spoken text - JUST THE OBJECT NAME
    """
    # Return only the object name (shape is the class name)
    return shape.capitalize()
