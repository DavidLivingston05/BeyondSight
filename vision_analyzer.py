"""
Vision Analyzer - AI-Powered Visual Recognition System
Provides comprehensive scene analysis, object detection, and accessibility features
using state-of-the-art computer vision models.
"""

import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import os
import sys
import logging
import torch

logger = logging.getLogger(__name__)

# ============= GPU OPTIMIZATION =============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    logger.info("🚀 GPU acceleration enabled for vision analysis")
    try:
        cv2.cuda.setDevice(0)
    except:
        logger.debug("OpenCV GPU not available, using CPU")

# ============= Configuration =============
@dataclass
class DistanceCategory:
    """Define distance thresholds"""
    very_close: float = 1.0
    close: float = 2.0
    few_meters: float = 4.0
    moderately_far: float = 8.0


@dataclass
class BrightnessLevel:
    """Define lighting thresholds"""
    very_bright: int = 180
    well_lit: int = 120
    moderately_lit: int = 60


class ObjectCategory(Enum):
    """Categorize objects by type"""
    VEHICLE = ["car", "truck", "bus", "motorcycle"]
    DANGER = ["knife"]
    PERSON = ["person"]
    SEATING = ["chair", "bench"]
    EXIT = ["door"]


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
    """Normalized bounding box coordinates"""
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
    """Enhanced vision analysis for accessibility assistance"""
    
    # Class-level constants
    REFERENCE_SIZES = {
        'person': 180, 'chair': 120, 'car': 400, 'cup': 60,
        'bottle': 80, 'book': 100, 'cell phone': 80, 'laptop': 150,
        'keyboard': 120, 'mouse': 60, 'tv': 300, 'clock': 100
    }
    
    POSITION_LEFT_THRESHOLD = 0.33
    POSITION_RIGHT_THRESHOLD = 0.66
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 tesseract_path: Optional[str] = None):
        """
        Initialize vision analyzer with YOLOv8 model.
        
        Args:
            model_path: Path to YOLO model weights
            tesseract_path: Path to Tesseract OCR binary
        """
        logger.info("🧠 Loading vision AI...")
        self.model = YOLO(model_path)
        self.last_analysis = ""
        self._setup_tesseract(tesseract_path)
        
        # Config objects
        self.distance_config = DistanceCategory()
        self.brightness_config = BrightnessLevel()
    
    @staticmethod
    def _setup_tesseract(tesseract_path: Optional[str]) -> None:
         """Configure Tesseract OCR path with fallback to environment variables and system defaults.
         
         Args:
             tesseract_path: Optional explicit path to tesseract.exe
         """
         if tesseract_path:
             pytesseract.pytesseract.tesseract_cmd = tesseract_path
             logger.debug(f"Tesseract configured with explicit path: {tesseract_path}")
         else:
             # Check environment variable first
             env_path = os.getenv('TESSERACT_CMD')
             if env_path and os.path.exists(env_path):
                 pytesseract.pytesseract.tesseract_cmd = env_path
                 logger.debug(f"Tesseract configured from environment: {env_path}")
             else:
                 # Check common system paths based on platform
                 common_paths = []
                 if sys.platform == 'win32':
                     common_paths = [
                         r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                         r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                     ]
                 elif sys.platform == 'darwin':
                     common_paths = ['/usr/local/bin/tesseract']
                 else:  # Linux
                     common_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract']
                 
                 for path in common_paths:
                     if os.path.exists(path):
                         pytesseract.pytesseract.tesseract_cmd = path
                         logger.debug(f"Tesseract found at: {path}")
                         return
                 
                 logger.warning("Tesseract OCR not found. Please install or set TESSERACT_CMD environment variable")
    
    # ============= Core Detection =============
    def _detect_objects(self, frame: np.ndarray, conf: float = 0.4) -> List[DetectedObject]:
        """
        Run YOLO detection once and return structured results.
        
        Args:
            frame: Input image frame
            conf: Confidence threshold
            
        Returns:
            List of DetectedObject instances
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        results = self.model(frame, verbose=False, conf=conf)
        frame_width = frame.shape[1]
        detected_objects = []
        
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                name = self.model.names[cls_idx]
                confidence = float(box.conf.item())
                
                # Extract and normalize box coordinates
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
    
    # ============= Distance & Position =============
    def _estimate_distance(self, box_width: float, object_type: str) -> str:
        """
        Estimate distance based on bounding box width.
        
        Args:
            box_width: Width of detected object bounding box
            object_type: Type of object (affects reference size)
            
        Returns:
            Human-readable distance description
        """
        if box_width == 0:
            return "unknown distance"
        
        ref_size = self.REFERENCE_SIZES.get(object_type, 100)
        estimated_meters = ref_size / box_width
        
        if estimated_meters < self.distance_config.very_close:
            return "very close"
        elif estimated_meters < self.distance_config.close:
            return "close"
        elif estimated_meters < self.distance_config.few_meters:
            return "a few meters away"
        elif estimated_meters < self.distance_config.moderately_far:
            return "moderately far"
        else:
            return "far away"
    
    def _get_position(self, x_center: float, frame_width: int) -> str:
        """
        Determine horizontal position relative to frame center.
        
        Args:
            x_center: Horizontal center of object
            frame_width: Width of frame
            
        Returns:
            Position description (left/right/ahead)
        """
        ratio = x_center / frame_width
        
        if ratio < self.POSITION_LEFT_THRESHOLD:
            return "on your left"
        elif ratio > self.POSITION_RIGHT_THRESHOLD:
            return "on your right"
        else:
            return "straight ahead"
    
    # ============= Lighting Analysis =============
    def _analyze_lighting(self, frame: np.ndarray) -> str:
        """
        Analyze frame brightness.
        
        Args:
            frame: Input image frame
            
        Returns:
            Lighting condition description
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        
        if brightness > self.brightness_config.very_bright:
            return "very bright"
        elif brightness > self.brightness_config.well_lit:
            return "well lit"
        elif brightness > self.brightness_config.moderately_lit:
            return "moderately lit"
        else:
            return "dimly lit"
    
    # ============= Scene Analysis =============
    def analyze_scene_comprehensive(self, frame: np.ndarray) -> str:
        """
        Provide comprehensive scene description.
        
        Args:
            frame: Input image frame
            
        Returns:
            Human-readable scene description
        """
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            if not detected_objects:
                return "The area appears empty. I don't see any recognizable objects."
            
            # Group by object type and create descriptions
            by_type = self._group_by_type(detected_objects)
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
                    closest = min(objects, key=lambda o: self._distance_to_meters(o.distance))
                    description_parts.append(
                        f"{len(objects)} {obj_type}s, the closest is {closest.distance} {closest.position}"
                    )
            
            brightness = self._analyze_lighting(frame)
            description_parts.append(f"The room is {brightness}")
            
            full_description = "I can see " + ", ".join(description_parts) + "."
            self.last_analysis = full_description
            return full_description
            
        except Exception as e:
            return f"Analysis incomplete: {str(e)}"
    
    @staticmethod
    def _group_by_type(objects: List[DetectedObject]) -> Dict[str, List[DetectedObject]]:
        """Group detected objects by type"""
        grouped = {}
        for obj in objects:
            if obj.name not in grouped:
                grouped[obj.name] = []
            grouped[obj.name].append(obj)
        return grouped
    
    @staticmethod
    def _distance_to_meters(distance_str: str) -> float:
        """Convert distance description to numeric value for sorting"""
        distance_map = {
            "very close": 0.5,
            "close": 1.5,
            "a few meters away": 3.0,
            "moderately far": 6.0,
            "far away": 10.0,
            "unknown distance": 999.0
        }
        return distance_map.get(distance_str, 999.0)
    
    # ============= Safety Detection =============
    def detect_immediate_dangers(self, frame: np.ndarray) -> str:
        """
        Detect potential safety hazards.
        
        Args:
            frame: Input image frame
            
        Returns:
            Warning message if dangers detected, empty string otherwise
        """
        try:
            detected_objects = self._detect_objects(frame, conf=0.5)
            
            # Check for dangerous objects
            dangers = []
            for obj in detected_objects:
                if obj.name in ObjectCategory.VEHICLE.value:
                    dangers.append("moving vehicle")
                elif obj.name in ObjectCategory.DANGER.value:
                    dangers.append("sharp object")
                elif obj.name == "fire":
                    dangers.append("fire hazard")
            
            if dangers:
                return f"⚠️ SAFETY WARNING: I see {', '.join(set(dangers))}!"
            
            # Check for crowding
            people_count = sum(1 for obj in detected_objects if obj.name == "person")
            if people_count > 5:
                return "⚠️ The area is crowded. Please be careful."
            
            return ""
            
        except Exception as e:
             logger.error(f"Error detecting dangers: {e}", exc_info=True)
             return ""
    
    # ============= Object Search =============
    def find_specific_object(self, frame: np.ndarray, object_name: str) -> str:
        """
        Locate a specific object type.
        
        Args:
            frame: Input image frame
            object_name: Name of object to find
            
        Returns:
            Location description or "not found" message
        """
        try:
            detected_objects = self._detect_objects(frame, conf=0.3)
            found = [obj for obj in detected_objects if obj.name == object_name]
            
            if not found:
                return f"I don't see any {object_name} nearby."
            
            closest = min(found, key=lambda o: self._distance_to_meters(o.distance))
            return f"I found a {object_name} {closest.distance} {closest.position}."
            
        except Exception as e:
            return f"Could not search for {object_name}: {str(e)}"
    
    # ============= Text Recognition =============
    def read_text_aloud(self, frame: np.ndarray, max_words: int = 20) -> str:
        """
        Extract and read text from frame.
        
        Args:
            frame: Input image frame
            max_words: Maximum words to return
            
        Returns:
            Extracted text or "no text found" message
        """
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
            return "Tesseract OCR not installed. Please install from: https://github.com/UB-Mannheim/tesseract/wiki"
        except Exception as e:
            return f"Unable to read text: {str(e)}"
    
    # ============= Navigation =============
    def provide_navigation_advice(self, frame: np.ndarray) -> str:
        """
        Provide navigation guidance based on scene analysis.
        
        Args:
            frame: Input image frame
            
        Returns:
            Navigation advice message
        """
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            advice = []
            
            # Check for exits
            if any(obj.name in ObjectCategory.EXIT.value for obj in detected_objects):
                advice.append("There is a door nearby that you can use.")
            
            # Check for seating
            if any(obj.name in ObjectCategory.SEATING.value for obj in detected_objects):
                advice.append("Seating is available if you need to rest.")
            
            # Check for crowding
            people_count = sum(1 for obj in detected_objects if obj.name == "person")
            if people_count > 3:
                advice.append("The path ahead is crowded, proceed slowly.")
            elif people_count == 0:
                advice.append("The path appears clear ahead.")
            
            if advice:
                return "Navigation advice: " + " ".join(advice)
            else:
                return "The area seems navigable. Move forward carefully."
                
        except Exception as e:
             logger.error(f"Error providing navigation advice: {e}", exc_info=True)
             return "Unable to provide navigation guidance."
    
    # ============= NEW ACCESSIBILITY FEATURES =============
    
    def detect_stairs_elevation_changes(self, frame: np.ndarray) -> str:
        """Detect stairs, ramps, steps, and elevation changes."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            # Look for common stair-related objects
            stair_indicators = ['stair', 'stairs', 'step', 'steps', 'ramp', 'escalator', 'slope']
            stairs = [obj for obj in detected_objects if any(ind in obj.name.lower() for ind in stair_indicators)]
            
            if stairs:
                closest = min(stairs, key=lambda o: self._distance_to_meters(o.distance))
                return f"I detected stairs or a ramp {closest.distance} {closest.position}. Please be careful."
            
            # Analyze image edges for potential step detection
            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
            horizontal_lines = np.sum(edges[frame.shape[0]//3:frame.shape[0]//2, :], axis=0)
            
            if np.max(horizontal_lines) > np.mean(horizontal_lines) * 2:
                return "Elevation changes detected. There may be stairs or a slope ahead."
            
            return "No stairs or major elevation changes detected in the current view."
        except Exception as e:
            return f"Unable to detect stairs: {str(e)}"
    
    def detect_intersections_crosswalks(self, frame: np.ndarray) -> str:
        """Detect street intersections and crosswalks."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect white lines (typical of crosswalks)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            large_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            if len(large_contours) > 5:
                return "I detect a crosswalk or intersection marking nearby. Use caution when crossing."
            
            # Look for traffic-related objects
            detected_objects = self._detect_objects(frame, conf=0.4)
            traffic_objects = [obj for obj in detected_objects if obj.name in ['traffic light', 'stop sign', 'yield sign']]
            
            if traffic_objects:
                return f"I see traffic signals ahead. {', '.join(set(obj.name for obj in traffic_objects))}."
            
            return "No intersection or crosswalk markings detected."
        except Exception as e:
            return f"Unable to detect intersections: {str(e)}"
    
    def analyze_lighting_conditions(self, frame: np.ndarray) -> str:
        """Analyze and describe current lighting and visibility conditions."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            contrast = float(gray.std())
            
            lighting = self._analyze_lighting(frame)
            
            msg = f"Lighting is {lighting}. Visibility: "
            if contrast > 50:
                msg += "Good contrast. "
            else:
                msg += "Low contrast, objects may be hard to distinguish. "
            
            if brightness < 30:
                msg += "⚠️ Very dark conditions - consider using additional light."
            elif brightness < 60:
                msg += "Dim lighting may affect navigation safety."
            elif brightness > 200:
                msg += "Very bright - may cause glare."
            
            return msg
        except Exception as e:
            return f"Unable to analyze lighting: {str(e)}"
    
    def describe_room_layout(self, frame: np.ndarray) -> str:
        """Provide detailed spatial description of room or space layout."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.35)
            
            if not detected_objects:
                return "The space appears empty or minimally furnished."
            
            # Separate objects by position and distance
            left_objects = [o for o in detected_objects if 'left' in o.position]
            right_objects = [o for o in detected_objects if 'right' in o.position]
            center_objects = [o for o in detected_objects if 'ahead' in o.position]
            
            description = "Room layout: "
            
            if left_objects:
                description += f"To your left: {', '.join(set(o.name for o in left_objects))}. "
            if center_objects:
                description += f"Ahead of you: {', '.join(set(o.name for o in center_objects[:3]))}. "
            if right_objects:
                description += f"To your right: {', '.join(set(o.name for o in right_objects))}. "
            
            return description
        except Exception as e:
            return f"Unable to describe room layout: {str(e)}"
    
    def detect_color_signals(self, frame: np.ndarray) -> str:
        """Detect traffic lights, warning signs, and other color-coded signals."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            # Look for color-significant objects
            signal_objects = [obj for obj in detected_objects if obj.name in ['traffic light', 'stop sign', 'yield sign', 'warning sign']]
            
            if signal_objects:
                signals_info = []
                for obj in signal_objects:
                    signals_info.append(f"a {obj.name} {obj.distance} {obj.position}")
                return f"Color signals detected: {', '.join(signals_info)}."
            
            # Analyze color distribution
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            
            red_pixels = np.count_nonzero(red_mask)
            green_pixels = np.count_nonzero(green_mask)
            
            if red_pixels > frame.size * 0.01:
                return "Red warning color detected in the scene."
            if green_pixels > frame.size * 0.01:
                return "Green go/pass indicator detected."
            
            return "No significant color signals detected."
        except Exception as e:
            return f"Unable to detect color signals: {str(e)}"
    
    def read_text_with_position(self, frame: np.ndarray, focus_area: str = 'center') -> str:
        """Read text with position information (top, bottom, left, right, center)."""
        try:
            h, w = frame.shape[:2]
            
            # Define focus area
            if focus_area == 'top':
                roi = frame[0:h//3, :]
            elif focus_area == 'bottom':
                roi = frame[2*h//3:, :]
            elif focus_area == 'left':
                roi = frame[:, 0:w//3]
            elif focus_area == 'right':
                roi = frame[:, 2*w//3:]
            else:  # center
                roi = frame[h//3:2*h//3, w//3:2*w//3]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            
            if text.strip():
                return f"Text in {focus_area}: {text.strip()}"
            else:
                return f"No readable text in the {focus_area} area."
        except Exception as e:
            return f"Unable to read text: {str(e)}"
    
    def get_detailed_spatial_description(self, frame: np.ndarray, focus: str = 'foreground') -> str:
        """Get ultra-detailed spatial and positional description."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.3)
            
            if not detected_objects:
                return "The view is empty or contains no recognizable objects."
            
            description = f"Detailed spatial analysis ({focus}): "
            
            # Group by distance
            very_close = [o for o in detected_objects if 'very close' in o.distance]
            close = [o for o in detected_objects if 'close' in o.distance and 'very' not in o.distance]
            far = [o for o in detected_objects if 'far' in o.distance or 'moderately' in o.distance]
            
            if very_close:
                description += f"IMMEDIATELY: {', '.join(set(o.name for o in very_close))}. "
            if close:
                description += f"Within arm's reach: {', '.join(set(o.name for o in close))}. "
            if far:
                description += f"In the distance: {', '.join(set(o.name for o in far))}."
            
            return description
        except Exception as e:
            return f"Unable to provide detailed description: {str(e)}"
    
    def detect_people_and_activities(self, frame: np.ndarray) -> str:
        """Detect people and infer their activities."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            people = [obj for obj in detected_objects if obj.name == 'person']
            
            if not people:
                return "No people detected in the current view."
            
            description = f"Found {len(people)} person" + ("s" if len(people) != 1 else "") + ": "
            
            for person in people[:3]:
                description += f"One {person.distance} {person.position}. "
            
            # Analyze for activity indicators
            activity_objects = [o for o in detected_objects if o.name in ['bicycle', 'skateboard', 'sports ball', 'ski']]
            if activity_objects:
                description += f"Activity detected: {', '.join(set(o.name for o in activity_objects))}."
            
            return description
        except Exception as e:
            return f"Unable to detect people: {str(e)}"
    
    def analyze_time_of_day(self, frame: np.ndarray) -> str:
        """Estimate time of day from lighting and scene analysis."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            
            if brightness < 40:
                time_est = "Night time. Visibility is very low."
            elif brightness < 80:
                time_est = "Early morning or late evening. Lighting is dim."
            elif brightness < 150:
                time_est = "Daytime with moderate lighting."
            else:
                time_est = "Bright daytime or strong sunlight."
            
            # Check for shadows (midday indicator)
            edges = cv2.Canny(gray, 50, 150)
            if np.mean(edges) > 10:
                time_est += " Strong shadows suggest midday sun."
            
            return time_est
        except Exception as e:
            return f"Unable to analyze time of day: {str(e)}"
    
    def suggest_audio_cues(self, frame: np.ndarray) -> str:
        """Identify sounds and audio cues user should be aware of."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            audio_cues = []
            
            for obj in detected_objects:
                if obj.name in ['car', 'truck', 'bus', 'motorcycle']:
                    audio_cues.append(f"Watch for {obj.name} sounds ({obj.name} noise)")
                elif obj.name == 'person':
                    audio_cues.append("Listen for people talking or moving")
                elif obj.name == 'dog':
                    audio_cues.append("Dog present - be aware of barking or movement")
            
            if audio_cues:
                return "Audio cues to be aware of: " + ", ".join(set(audio_cues))
            else:
                return "No significant audio cues detected based on visual analysis."
        except Exception as e:
            return f"Unable to suggest audio cues: {str(e)}"
    
    def get_distance_estimates(self, frame: np.ndarray, direction: str = 'all') -> List[Dict]:
        """Get distance estimates to multiple objects."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            if direction != 'all':
                detected_objects = [o for o in detected_objects if direction.lower() in o.position.lower()]
            
            estimates = []
            for obj in detected_objects[:5]:
                estimates.append({
                    'object': obj.name,
                    'distance': obj.distance,
                    'position': obj.position,
                    'confidence': f"{obj.confidence:.1%}"
                })
            
            return estimates
        except Exception as e:
            return [{'error': str(e)}]
    
    def analyze_social_context(self, frame: np.ndarray) -> str:
        """Analyze social context and gestures in the scene."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            people = [obj for obj in detected_objects if obj.name == 'person']
            
            if not people:
                return "No social context - area appears empty of people."
            
            context = f"Social context: {len(people)} person" + ("s" if len(people) != 1 else "") + " present. "
            
            # Check for interaction-related objects
            interaction_objects = [o for o in detected_objects if o.name in ['chair', 'table', 'couch']]
            if interaction_objects:
                context += "Seating/gathering areas available."
            
            return context
        except Exception as e:
            return f"Unable to analyze social context: {str(e)}"
    
    def get_safe_path_guidance(self, frame: np.ndarray, destination_direction: str = 'forward') -> str:
        """Provide detailed guidance for safe navigation path."""
        try:
            detected_objects = self._detect_objects(frame, conf=0.4)
            
            # Filter for obstacles in the intended direction
            obstacles = [o for o in detected_objects if o.name not in ['chair', 'table', 'person']]
            
            guidance = f"Path analysis ({destination_direction}): "
            
            close_obstacles = [o for o in obstacles if 'close' in o.distance or 'very close' in o.distance]
            if close_obstacles:
                guidance += f"⚠️ Obstacles near: {', '.join(set(o.name for o in close_obstacles[:2]))}. Proceed carefully. "
            else:
                guidance += "Path appears clear ahead. "
            
            # Check if seating is available for rest
            seats = [o for o in detected_objects if o.name in ['chair', 'bench']]
            if seats:
                guidance += f"Seating available {seats[0].distance} {seats[0].position}."
            
            return guidance
        except Exception as e:
            return f"Unable to provide path guidance: {str(e)}"
    
    def detect_water_hazards(self, frame: np.ndarray) -> str:
        """Detect water, puddles, wet surfaces - slip hazards."""
        try:
            # Analyze for water/wet indicators (glossy, reflective areas)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Water often has low saturation and moderate brightness
            # Check for very reflective/shiny areas
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            bright_pixel_ratio = np.count_nonzero(bright_areas) / bright_areas.size
            
            if bright_pixel_ratio > 0.1:
                return "⚠️ Wet or glossy surfaces detected - possible slip hazard. Use caution."
            
            # Look for water-related objects
            detected_objects = self._detect_objects(frame, conf=0.4)
            water_objects = [o for o in detected_objects if o.name in ['water', 'puddle', 'pool', 'river']]
            
            if water_objects:
                return f"Water hazard detected: {', '.join(set(o.name for o in water_objects))}."
            
            return "No water hazards detected."
        except Exception as e:
            return f"Unable to detect water hazards: {str(e)}"
    
    def analyze_ambient_conditions(self, frame: np.ndarray) -> str:
        """Analyze ambient environmental conditions."""
        try:
            lighting = self._analyze_lighting(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Analyze weather indicators
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            ambient_desc = f"Ambient conditions: {lighting}. "
            
            # Sky color analysis (rough weather detection)
            sky_region = frame[0:frame.shape[0]//4, :]
            sky_hsv = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
            sky_brightness = np.mean(sky_hsv[:,:,2])
            
            if sky_brightness < 100:
                ambient_desc += "Sky is overcast or dark - expect possible rain."
            elif sky_brightness > 200:
                ambient_desc += "Clear, bright sky visible."
            
            # Temperature estimate (rough, based on color tone)
            b, g, r = cv2.split(frame)
            if np.mean(r) > np.mean(b) + 20:
                ambient_desc += " Warm color tones."
            elif np.mean(b) > np.mean(r) + 20:
                ambient_desc += " Cool color tones."
            
            return ambient_desc
        except Exception as e:
            return f"Unable to analyze ambient conditions: {str(e)}"
