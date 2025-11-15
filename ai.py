"""
AI Integration Module
Includes Face Recognition and DeepSeek API integration
"""

import json
import hashlib
import logging
import os
import requests
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Face recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available")

# ============= Face Recognition =============

@dataclass
class FacePerson:
    """Stored face data."""
    person_id: str
    name: str
    face_encoding: List[float]
    first_seen: str
    last_seen: str
    encounter_count: int
    notes: str = ""


class FaceMemory:
    """Face recognition and memory management."""
    
    DB_PATH = "face_database.json"
    SIMILARITY_THRESHOLD = 0.6
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize face memory."""
        self.logger = logger
        self.db_path = db_path
        self.people: Dict[str, FacePerson] = {}
        
        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("Face recognition not available")
        else:
            self._load_database()
    
    def _load_database(self) -> None:
        """Load face database from file."""
        try:
            if Path(self.db_path).exists():
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for person_data in data.get('people', []):
                        person = FacePerson(**person_data)
                        self.people[person.person_id] = person
                self.logger.info(f"✅ Loaded {len(self.people)} people")
        except Exception as e:
            self.logger.error(f"Load database error: {e}")
    
    def _save_database(self) -> None:
        """Save face database to file."""
        try:
            data = {
                'people': [asdict(person) for person in self.people.values()],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Database saved")
        except Exception as e:
            self.logger.error(f"Save database error: {e}")
    
    def recognize_people(self, frame) -> Tuple[List[Dict], int]:
        """Recognize people in frame."""
        if not FACE_RECOGNITION_AVAILABLE or not self.people:
            return [], 0
        
        try:
            import face_recognition as fr
            import cv2
            import numpy as np
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = fr.face_locations(rgb_frame, model="hog")
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            
            recognized = []
            unknown_count = 0
            
            for face_encoding in face_encodings:
                # Compare with known faces
                found = False
                for person_id, person in self.people.items():
                    known_encoding = face_encoding.tolist()
                    stored_encoding = person.face_encoding
                    
                    # Simple similarity check
                    distance = sum(abs(a - b) for a, b in zip(known_encoding, stored_encoding)) / len(known_encoding)
                    
                    if distance < self.SIMILARITY_THRESHOLD:
                        person.last_seen = datetime.now().isoformat()
                        person.encounter_count += 1
                        recognized.append({
                            'person_id': person_id,
                            'name': person.name,
                            'confidence': 1.0 - (distance / 2.0)
                        })
                        found = True
                        break
                
                if not found:
                    unknown_count += 1
            
            if recognized or unknown_count > 0:
                self._save_database()
            
            return recognized, unknown_count
        except Exception as e:
            self.logger.error(f"Recognition error: {e}")
            return [], 0
    
    def remember_face(self, frame, name: str, notes: str = '') -> str:
        """Remember a face."""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        
        try:
            import face_recognition as fr
            import cv2
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_frame, model="hog")
            face_encodings = fr.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                return None
            
            # Use first face encoding
            person_id = hashlib.md5(f"{name}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            
            person = FacePerson(
                person_id=person_id,
                name=name,
                face_encoding=face_encodings[0].tolist(),
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                encounter_count=1,
                notes=notes
            )
            
            self.people[person_id] = person
            self._save_database()
            self.logger.info(f"Remembered: {name}")
            
            return person_id
        except Exception as e:
            self.logger.error(f"Remember error: {e}")
            return None
    
    def forget_person(self, person_id: str) -> bool:
        """Forget a person."""
        try:
            if person_id in self.people:
                name = self.people[person_id].name
                del self.people[person_id]
                self._save_database()
                self.logger.info(f"Forgot: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Forget error: {e}")
            return False


# ============= DeepSeek Integration =============

def set_api_key(key: str) -> None:
    """Set DeepSeek API key."""
    os.environ['DEEPSEEK_API_KEY'] = key


class DeepSeekClient:
    """DeepSeek AI API client."""
    
    def __init__(self):
        """Initialize DeepSeek client."""
        self.api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.enabled = bool(self.api_key)
        self.model = "deepseek-chat"
        self.request_count = 0
        self.error_count = 0
        
        if self.enabled:
            logger.info("✅ DeepSeek API enabled")
        else:
            logger.info("ℹ️  DeepSeek API not configured")
    
    def set_api_key(self, key: str) -> None:
        """Set or update API key."""
        self.api_key = key
        self.enabled = bool(key)
        if self.enabled:
            logger.info("✅ DeepSeek API enabled")
    
    def health_check(self) -> bool:
        """Test API connectivity."""
        if not self.enabled:
            return False
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": "ping"}]},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def _call_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> Optional[str]:
        """Call DeepSeek API."""
        if not self.enabled:
            return None
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"API call error: {e}")
            self.error_count += 1
        
        return None
    
    def enhance_scene_description(self, basic_description: str, detected_objects: List[str]) -> Optional[str]:
        """Enhance scene analysis."""
        if not self.enabled:
            return None
        
        prompt = f"""You are a vision assistant. Scene detected:
{', '.join(detected_objects[:10])}
Basic: {basic_description}

Enhance with important navigation details (2-3 sentences max):"""
        
        result = self._call_api(prompt)
        if result:
            self.request_count += 1
        return result
    
    def analyze_hazards(self, hazard_description: str) -> Optional[str]:
        """Analyze hazards."""
        if not self.enabled:
            return None
        
        prompt = f"""Hazard detected: {hazard_description}

Brief assessment (2 sentences max):"""
        
        result = self._call_api(prompt)
        if result:
            self.request_count += 1
        return result
    
    def provide_navigation_guidance(self, scene_description: str, obstacles: List[str]) -> Optional[str]:
        """Navigation guidance."""
        if not self.enabled:
            return None
        
        obstacle_text = f"Obstacles: {', '.join(obstacles)}" if obstacles else "No major obstacles"
        
        prompt = f"""Navigation for visually impaired user.
Scene: {scene_description}
{obstacle_text}

Guidance (2-3 sentences max):"""
        
        result = self._call_api(prompt)
        if result:
            self.request_count += 1
        return result
    
    def analyze_context(self, scene_description: str) -> Optional[str]:
        """Analyze social context."""
        if not self.enabled:
            return None
        
        prompt = f"""Scene analysis for accessibility.
Description: {scene_description}

Context and recommendations (brief):"""
        
        result = self._call_api(prompt)
        if result:
            self.request_count += 1
        return result
    
    def get_statistics(self) -> Dict:
        """Get API statistics."""
        return {
            'enabled': self.enabled,
            'requests': self.request_count,
            'errors': self.error_count
        }


# ============= Factory Functions =============

def get_deepseek_client() -> Optional[DeepSeekClient]:
    """Get DeepSeek client instance."""
    try:
        return DeepSeekClient()
    except Exception as e:
        logger.error(f"DeepSeek client error: {e}")
        return None
