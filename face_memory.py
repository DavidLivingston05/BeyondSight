"""
Face Memory Module - Face Recognition and Memory for Beyond Sight

Provides:
- Face detection and recognition
- Face encoding storage and retrieval
- Named person tracking
- Face-to-person association
"""

import logging
import pickle
import json
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import face_recognition library
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition library not installed. Face recognition disabled.")

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class RecognizedPerson:
    """Data class for recognized person."""
    name: str
    confidence: float
    location: Tuple[int, int, int, int]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class FaceMemory:
    """
    Face recognition and memory system for Beyond Sight.
    
    Maintains:
    - Known face encodings
    - Person name mappings
    - Recognition confidence
    - Face detection history
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Face Memory system.
        
        Args:
            data_dir: Directory for storing face data
        """
        logger.info("Initializing Face Memory...")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.face_encodings_file = self.data_dir / "face_encodings.pkl"
        self.face_names_file = self.data_dir / "face_names.json"
        
        # Face data storage
        self.known_face_encodings: List = []
        self.known_face_names: List[str] = []
        self.face_confidence: Dict[str, float] = {}
        
        # Recognition history
        self.recognition_history: List[Dict] = []
        self.last_recognized_faces: List[RecognizedPerson] = []
        
        # Threading
        self.data_lock = threading.Lock()
        
        # Load existing face data
        self._load_face_data()
        
        logger.info(f"✅ Face Memory initialized with {len(self.known_face_names)} known faces")
    
    def _load_face_data(self) -> None:
        """Load face encodings and names from disk."""
        try:
            if self.face_encodings_file.exists():
                with open(self.face_encodings_file, 'rb') as f:
                    self.known_face_encodings = pickle.load(f)
                logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
            
            if self.face_names_file.exists():
                with open(self.face_names_file, 'r') as f:
                    data = json.load(f)
                    self.known_face_names = data.get('names', [])
                    self.face_confidence = data.get('confidence', {})
                logger.info(f"Loaded {len(self.known_face_names)} face names")
        except Exception as e:
            logger.error(f"Error loading face data: {e}")
    
    def _save_face_data(self) -> None:
        """Save face encodings and names to disk."""
        try:
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(self.known_face_encodings, f)
            
            data = {
                'names': self.known_face_names,
                'confidence': self.face_confidence,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.face_names_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Face data saved")
        except Exception as e:
            logger.error(f"Error saving face data: {e}")
    
    def add_face(self, face_encoding, person_name: str) -> bool:
        """
        Add a new face encoding for a person.
        
        Args:
            face_encoding: Face encoding from face_recognition
            person_name: Name of the person
            
        Returns:
            bool: Success status
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("Face recognition not available")
            return False
        
        try:
            with self.data_lock:
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(person_name)
                self.face_confidence[person_name] = 1.0
            
            self._save_face_data()
            logger.info(f"Face added for {person_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
    
    def remember_face(self, frame, person_name: str, notes: str = '') -> bool:
        """
        Remember a face from a frame.
        
        Args:
            frame: OpenCV frame
            person_name: Name to remember the person as
            notes: Optional notes about the person
            
        Returns:
            bool: Success status
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("Face recognition not available")
            return False
        
        try:
            face_locations = face_recognition.face_locations(frame, model='hog')
            if not face_locations:
                logger.warning("No face detected in frame")
                return False
            
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            if not face_encodings:
                logger.warning("Could not encode face")
                return False
            
            self.add_face(face_encodings[0], person_name)
            logger.info(f"Remembered face for {person_name}")
            return True
        except Exception as e:
            logger.error(f"Error remembering face: {e}")
            return False
    
    def recognize_people(self, frame) -> Tuple[List[RecognizedPerson], int]:
        """
        Recognize people in a frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Tuple of (recognized_people, unknown_count)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("Face recognition not available")
            return [], 0
        
        try:
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(frame, model='hog')
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            recognized = []
            unknown_count = 0
            
            with self.data_lock:
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=0.6
                    )
                    name = "Unknown"
                    confidence = 0.0
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                    
                    if name == "Unknown":
                        unknown_count += 1
                    
                    person = RecognizedPerson(
                        name=name,
                        confidence=float(confidence),
                        location=face_location,
                        timestamp=datetime.now().isoformat()
                    )
                    recognized.append(person)
            
            self.last_recognized_faces = recognized
            self._add_to_history(recognized, unknown_count)
            
            return recognized, unknown_count
        except Exception as e:
            logger.error(f"Error recognizing people: {e}")
            return [], 0
    
    def _add_to_history(self, recognized: List[RecognizedPerson], unknown_count: int) -> None:
        """Add recognition to history."""
        try:
            with self.data_lock:
                self.recognition_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'recognized_count': len(recognized),
                    'unknown_count': unknown_count,
                    'people': [p.to_dict() for p in recognized]
                })
                
                # Keep only last 100 entries
                if len(self.recognition_history) > 100:
                    self.recognition_history = self.recognition_history[-100:]
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
    
    def get_known_faces(self) -> Dict[str, any]:
        """Get list of known faces."""
        try:
            with self.data_lock:
                return {
                    'count': len(self.known_face_names),
                    'faces': list(set(self.known_face_names))
                }
        except Exception as e:
            logger.error(f"Error getting known faces: {e}")
            return {'count': 0, 'faces': []}
    
    def get_recognition_history(self, limit: int = 10) -> List[Dict]:
        """Get recent recognition history."""
        try:
            with self.data_lock:
                return self.recognition_history[-limit:]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def remove_face(self, person_name: str) -> bool:
        """Remove all encodings for a person."""
        try:
            with self.data_lock:
                indices = [i for i, name in enumerate(self.known_face_names) if name == person_name]
                for i in reversed(indices):
                    del self.known_face_encodings[i]
                    del self.known_face_names[i]
                
                if person_name in self.face_confidence:
                    del self.face_confidence[person_name]
            
            self._save_face_data()
            logger.info(f"Removed faces for {person_name}")
            return True
        except Exception as e:
            logger.error(f"Error removing face: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all face data."""
        try:
            with self.data_lock:
                self.known_face_encodings = []
                self.known_face_names = []
                self.face_confidence = {}
                self.recognition_history = []
            
            self._save_face_data()
            logger.info("All face data cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing face data: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get face memory statistics."""
        try:
            with self.data_lock:
                return {
                    'known_faces': len(set(self.known_face_names)),
                    'total_encodings': len(self.known_face_encodings),
                    'recognition_history_count': len(self.recognition_history),
                    'last_recognized_faces': len(self.last_recognized_faces),
                    'face_recognition_available': FACE_RECOGNITION_AVAILABLE
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
