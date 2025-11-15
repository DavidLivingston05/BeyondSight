"""New accessibility features for Beyond Sight"""
import cv2
import numpy as np
from typing import List, Dict


def detect_stairs_elevation_changes(self, frame: np.ndarray) -> str:
    """Detect stairs, ramps, steps, and elevation changes."""
    try:
        detected_objects = self._detect_objects(frame, conf=0.4)
        
        stair_indicators = ['stair', 'stairs', 'step', 'steps', 'ramp', 'escalator', 'slope']
        stairs = [obj for obj in detected_objects if any(ind in obj.name.lower() for ind in stair_indicators)]
        
        if stairs:
            closest = min(stairs, key=lambda o: self._distance_to_meters(o.distance))
            return f"I detected stairs or a ramp {closest.distance} {closest.position}. Please be careful."
        
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
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if len(large_contours) > 5:
            return "I detect a crosswalk or intersection marking nearby. Use caution when crossing."
        
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
        
        signal_objects = [obj for obj in detected_objects if obj.name in ['traffic light', 'stop sign', 'yield sign', 'warning sign']]
        
        if signal_objects:
            signals_info = []
            for obj in signal_objects:
                signals_info.append(f"a {obj.name} {obj.distance} {obj.position}")
            return f"Color signals detected: {', '.join(signals_info)}."
        
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
        import pytesseract
        h, w = frame.shape[:2]
        
        if focus_area == 'top':
            roi = frame[0:h//3, :]
        elif focus_area == 'bottom':
            roi = frame[2*h//3:, :]
        elif focus_area == 'left':
            roi = frame[:, 0:w//3]
        elif focus_area == 'right':
            roi = frame[:, 2*w//3:]
        else:
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
