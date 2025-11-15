"""
Vision Analysis - AI-Powered Visual Recognition
Provides comprehensive scene analysis and object detection
"""

import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import torch
import os

logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============= Data Classes =============

@dataclass
class DetectedObject:
    """Structured object detection result"""
    name: str
    confidence: float
    distance: str
    position: str
    box_width: float
    x_center: float


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2


# ============= Vision Analyzer =============

class VisionAnalyzer:
    """AI-powered vision analysis for accessibility."""
    
    REFERENCE_SIZES = {
        'person': 180, 'chair': 120, 'car': 400, 'cup': 60,
        'bottle': 80, 'book': 100, 'cell phone': 80, 'laptop': 150,
        'keyboard': 120, 'mouse': 60, 'tv': 300, 'clock': 100
    }
    
    POSITION_LEFT = 0.33
    POSITION_RIGHT = 0.66
    
    def __init__(self, model_path: str = "yolov8n.pt", tesseract_path: Optional[str] = None):
        """Initialize vision analyzer."""
        logger.info("🧠 Loading vision AI...")
        self.model = YOLO(model_path)
        self.last_analysis = ""
        self._setup_tesseract(tesseract_path)
    
    @staticmethod
    def _setup_tesseract(tesseract_path: Optional[str]) -> None:
        """Configure Tesseract OCR."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    def _detect_objects(self, frame: np.ndarray, conf: float = 0.4) -> List[DetectedObject]:
        """Run YOLO detection."""
        if frame is None or frame.size == 0:
            return []
        
        try:
            results = self.model(frame, verbose=False, conf=conf)
            frame_width = frame.shape[1]
            detected_objects = []
            
            for r in results:
                for box in r.boxes:
                    cls_idx = int(box.cls.item())
                    name = self.model.names[cls_idx]
                    confidence = float(box.conf.item())
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = BoundingBox(x1, y1, x2, y2)
                    
                    distance = self._estimate_distance(bbox.width, name)
                    position = self._get_position(bbox.x_center, frame_width)
                    
                    detected_objects.append(DetectedObject(
                        name=name,
                        confidence=confidence,
                        distance=distance,
                        position=position,
                        box_width=bbox.width,
                        x_center=bbox.x_center
                    ))
            
            return detected_objects
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _estimate_distance(self, box_width: float, object_type: str) -> str:
        """Estimate distance from bounding box."""
        if box_width == 0:
            return "unknown distance"
        
        ref_size = self.REFERENCE_SIZES.get(object_type, 100)
        estimated_meters = ref_size / box_width
        
        if estimated_meters < 1.0:
            return "very close"
        elif estimated_meters < 2.0:
            return "close"
        elif estimated_meters < 4.0:
            return "a few meters away"
        elif estimated_meters < 8.0:
            return "moderately far"
        else:
            return "far away"
    
    def _get_position(self, x_center: float, frame_width: int) -> str:
        """Get position description."""
        ratio = x_center / frame_width
        
        if ratio < self.POSITION_LEFT:
            return "on your left"
        elif ratio > self.POSITION_RIGHT:
            return "on your right"
        else:
            return "straight ahead"
    
    def _analyze_lighting(self, frame: np.ndarray) -> str:
        """Analyze lighting conditions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        
        if brightness > 180:
            return "very bright"
        elif brightness > 120:
            return "well lit"
        elif brightness > 60:
            return "moderately lit"
        else:
            return "dimly lit"
    
    def analyze_scene_comprehensive(self, frame: np.ndarray) -> str:
        """Comprehensive scene analysis."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            if not detected_objects:
                return "The area appears empty. No recognizable objects."
            
            # Group by type
            by_type = {}
            for obj in detected_objects:
                if obj.name not in by_type:
                    by_type[obj.name] = []
                by_type[obj.name].append(obj)
            
            description_parts = []
            
            for obj_type, objects in sorted(by_type.items(), 
                                           key=lambda x: len(x[1]), 
                                           reverse=True)[:8]:
                if len(objects) == 1:
                    obj = objects[0]
                    description_parts.append(
                        f"one {obj_type} {obj.distance} {obj.position}"
                    )
                else:
                    description_parts.append(
                        f"{len(objects)} {obj_type}s"
                    )
            
            brightness = self._analyze_lighting(frame)
            description_parts.append(f"The room is {brightness}")
            
            full_description = "I can see " + ", ".join(description_parts) + "."
            self.last_analysis = full_description
            return full_description
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def detect_immediate_dangers(self, frame: np.ndarray) -> str:
        """Detect safety hazards."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.5)
            
            dangers = []
            for obj in detected_objects:
                if obj.name in ['car', 'truck', 'bus', 'motorcycle']:
                    dangers.append("moving vehicle")
                elif obj.name in ['knife', 'fire']:
                    dangers.append("danger")
            
            if dangers:
                return f"I see {', '.join(set(dangers))}!"
            
            # Check crowding
            people_count = sum(1 for obj in detected_objects if obj.name == "person")
            if people_count > 5:
                return "The area is crowded. Be careful."
            
            return ""
        except Exception as e:
            logger.error(f"Hazard detection error: {e}")
            return ""
    
    def find_specific_object(self, frame: np.ndarray, object_name: str) -> str:
        """Find specific object."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.3)
            found = [obj for obj in detected_objects if obj.name == object_name]
            
            if not found:
                return f"I don't see any {object_name} nearby."
            
            closest = min(found, key=lambda o: self._distance_to_meters(o.distance))
            return f"I found a {object_name} {closest.distance} {closest.position}."
        except Exception as e:
            logger.error(f"Find object error: {e}")
            return f"Could not search for {object_name}."
    
    def read_text_aloud(self, frame: np.ndarray, max_words: int = 20) -> str:
        """Extract and read text."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            
            clean_text = ' '.join(text.split()[:max_words])
            
            if clean_text.strip():
                return f"I can read: {clean_text}"
            else:
                return "No readable text visible."
        except pytesseract.TesseractNotFoundError:
            return "Tesseract OCR not installed."
        except Exception as e:
            logger.error(f"Text reading error: {e}")
            return "Unable to read text right now."
    
    def provide_navigation_advice(self, frame: np.ndarray) -> str:
        """Navigation guidance."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            advice = []
            
            # Check for exits
            if any(obj.name in ['door'] for obj in detected_objects):
                advice.append("There is a door nearby.")
            
            # Check for seating
            if any(obj.name in ['chair', 'bench'] for obj in detected_objects):
                advice.append("Seating is available.")
            
            # Check crowding
            people_count = sum(1 for obj in detected_objects if obj.name == "person")
            if people_count > 3:
                advice.append("The path is crowded, proceed slowly.")
            elif people_count == 0:
                advice.append("The path appears clear.")
            
            if advice:
                return "Navigation: " + " ".join(advice)
            else:
                return "The area seems navigable. Move forward carefully."
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return "Unable to provide navigation guidance."
    
    def detect_obstacles_in_frame(self, frame: np.ndarray) -> List[str]:
        """Detect obstacles."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            obstacles = []
            
            for obj in detected_objects:
                if obj.name not in ['person', 'chair', 'table']:
                    if 'close' in obj.distance:
                        obstacles.append(obj.name)
            
            return obstacles
        except Exception as e:
            logger.error(f"Obstacle detection error: {e}")
            return []
    
    @staticmethod
    def _distance_to_meters(distance_str: str) -> float:
        """Convert distance to numeric value."""
        distance_map = {
            "very close": 0.5,
            "close": 1.5,
            "a few meters away": 3.0,
            "moderately far": 6.0,
            "far away": 10.0,
            "unknown distance": 999.0
        }
        return distance_map.get(distance_str, 999.0)
    
    def detect_stairs_elevation_changes(self, frame: np.ndarray) -> str:
        """Detect stairs and elevation changes."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            horizontal_lines = np.sum(edges[frame.shape[0]//3:frame.shape[0]//2, :], axis=0)
            
            if np.max(horizontal_lines) > np.mean(horizontal_lines) * 2:
                return "Elevation changes detected. May be stairs ahead."
            
            return "No stairs detected."
        except Exception as e:
            logger.error(f"Stair detection error: {e}")
            return ""
    
    def analyze_lighting_conditions(self, frame: np.ndarray) -> str:
        """Analyze lighting and visibility."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            
            lighting = self._analyze_lighting(frame)
            msg = f"Lighting is {lighting}. "
            
            if contrast > 50:
                msg += "Good contrast. "
            else:
                msg += "Low contrast. "
            
            if brightness < 30:
                msg += "⚠️ Very dark - use additional light."
            elif brightness < 60:
                msg += "Dim - be careful navigating."
            elif brightness > 200:
                msg += "Very bright - may cause glare."
            
            return msg
        except Exception as e:
            logger.error(f"Lighting analysis error: {e}")
            return ""
    
    def describe_room_layout(self, frame: np.ndarray) -> str:
        """Describe room layout."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.35)
            
            if not detected_objects:
                return "The space appears empty or minimally furnished."
            
            # Group by position
            left_objects = [o for o in detected_objects if 'left' in o.position]
            right_objects = [o for o in detected_objects if 'right' in o.position]
            center_objects = [o for o in detected_objects if 'ahead' in o.position]
            
            description = "Room layout: "
            
            if left_objects:
                description += f"Left: {', '.join(set(o.name for o in left_objects))}. "
            if center_objects:
                description += f"Ahead: {', '.join(set(o.name for o in center_objects[:3]))}. "
            if right_objects:
                description += f"Right: {', '.join(set(o.name for o in right_objects))}. "
            
            return description
        except Exception as e:
            logger.error(f"Room layout error: {e}")
            return ""
    
    def detect_color_signals(self, frame: np.ndarray) -> str:
        """Detect traffic lights and color signals."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            signals = [obj for obj in detected_objects 
                      if obj.name in ['traffic light', 'stop sign', 'yield sign', 'warning sign']]
            
            if signals:
                return f"Signals detected: {', '.join(set(obj.name for obj in signals))}."
            
            # Analyze color pixels
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            
            red_pixels = np.count_nonzero(red_mask)
            green_pixels = np.count_nonzero(green_mask)
            
            if red_pixels > frame.size * 0.01:
                return "Red warning color detected."
            if green_pixels > frame.size * 0.01:
                return "Green indicator detected."
            
            return "No significant color signals detected."
        except Exception as e:
            logger.error(f"Color signal detection error: {e}")
            return ""
    
    def detect_water_hazards(self, frame: np.ndarray) -> str:
        """Detect water and wet surfaces."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            bright_ratio = np.count_nonzero(bright_areas) / bright_areas.size
            
            if bright_ratio > 0.1:
                return "⚠️ Wet or glossy surfaces detected - possible slip hazard."
            
            detected_objects = self._detect_objects(frame, conf=0.4)
            water_objects = [o for o in detected_objects 
                            if o.name in ['water', 'puddle', 'pool', 'river']]
            
            if water_objects:
                return f"Water hazard: {', '.join(set(o.name for o in water_objects))}."
            
            return "No water hazards detected."
        except Exception as e:
            logger.error(f"Water hazard detection error: {e}")
            return ""
    
    def get_detailed_spatial_description(self, frame: np.ndarray) -> str:
        """Ultra-detailed spatial description."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.3)
            
            if not detected_objects:
                return "The view is empty or contains no recognizable objects."
            
            # Group by distance
            very_close = [o for o in detected_objects if 'very close' in o.distance]
            close = [o for o in detected_objects if 'close' in o.distance and 'very' not in o.distance]
            far = [o for o in detected_objects if 'far' in o.distance]
            
            description = "Spatial analysis: "
            
            if very_close:
                description += f"IMMEDIATELY: {', '.join(set(o.name for o in very_close))}. "
            if close:
                description += f"Within reach: {', '.join(set(o.name for o in close))}. "
            if far:
                description += f"Distance: {', '.join(set(o.name for o in far))}."
            
            return description
        except Exception as e:
            logger.error(f"Spatial description error: {e}")
            return ""
