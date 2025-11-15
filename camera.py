"""
Camera Processor - Real-time Video Capture
Handles camera initialization and face/hand detection
"""

import cv2
import time
import mediapipe as mp
import numpy as np
import logging
import base64

logger = logging.getLogger(__name__)

class CameraProcessor:
    """Processes camera frames and detects faces/hands."""
    
    def __init__(self):
        """Initialize camera processor."""
        self.cap = None
        self.camera_started = False
        
        # MediaPipe initialization
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def start(self) -> bool:
        """Start camera."""
        if not self.camera_started:
            logger.info("📷 Starting camera...")
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if not self.cap.isOpened():
                    logger.error("Failed to open camera")
                    return False
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                
                # Warm up camera
                for i in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info("✅ Camera activated")
                        self.camera_started = True
                        return True
                    time.sleep(0.1)
                
                logger.error("Camera not responding")
                return False
            except Exception as e:
                logger.error(f"Camera error: {e}")
                return False
        return True
    
    def get_frame(self) -> np.ndarray:
        """Get current frame."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect
                return frame
        return None
    
    def detect_face_and_hands(self, frame: np.ndarray):
        """Detect faces and hands in frame."""
        if frame is None:
            return frame, {'faces': [], 'hands': []}
        
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = {'faces': [], 'hands': []}
        
        # Face detection
        try:
            face_results = self.face_detector.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    x_min = int(bbox.xmin * w)
                    y_min = int(bbox.ymin * h)
                    x_max = int((bbox.xmin + bbox.width) * w)
                    y_max = int((bbox.ymin + bbox.height) * h)
                    
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (100, 255, 100), 1)
                    
                    # Draw confidence
                    confidence = detection.score[0]
                    cv2.putText(frame, f"Face: {confidence:.2f}", 
                               (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    detections['faces'].append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'confidence': float(confidence)
                    })
        except Exception as e:
            logger.error(f"Face detection error: {e}")
        
        # Hand detection
        try:
            hand_results = self.hands_detector.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                ):
                    # Get bounding box
                    h_positions = [lm.x for lm in hand_landmarks.landmark]
                    v_positions = [lm.y for lm in hand_landmarks.landmark]
                    
                    x_min = int(min(h_positions) * w)
                    y_min = int(min(v_positions) * h)
                    x_max = int(max(h_positions) * w)
                    y_max = int(max(v_positions) * h)
                    
                    # Add padding
                    padding = 20
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)
                    
                    # Draw box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (100, 100, 255), 1)
                    
                    # Draw label
                    hand_label = handedness.classification[0].label
                    cv2.putText(frame, f"{hand_label} Hand", 
                               (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Draw keypoints
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        x_pos = int(lm.x * w)
                        y_pos = int(lm.y * h)
                        cv2.circle(frame, (x_pos, y_pos), 4, (0, 255, 255), -1)
                    
                    detections['hands'].append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'label': hand_label,
                        'landmarks': [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    })
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
        
        return frame, detections
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.face_detector:
                self.face_detector.close()
            if self.hands_detector:
                self.hands_detector.close()
            if self.cap:
                self.cap.release()
            logger.info("Camera cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def decode_mobile_frame(frame_data: str) -> np.ndarray:
    """Decode base64 frame from mobile device."""
    try:
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None
