"""
Core BeyondSight Assistant Logic
Main application class managing vision, speech, and navigation
"""

import cv2
import threading
import time
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, Set
from enum import Enum

from camera import CameraProcessor
from vision import VisionAnalyzer
from speech import SpeechEnginePro, VoiceRate, VoiceVolume, SpeechPriority
from ai import FaceMemory, get_deepseek_client, FACE_RECOGNITION_AVAILABLE
from navigation import get_navigator, get_place_memory

logger = logging.getLogger(__name__)

# GPU acceleration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

# Constants
FRAME_RATE, FRAME_HEIGHT, FRAME_WIDTH = 30, 720, 1280
CONTINUOUS_ANALYSIS_INTERVAL = 5

class WebBeyondSightAssistant:
    """Main Beyond Sight assistant managing all components."""
    
    def __init__(self):
        """Initialize assistant with all components."""
        logger.info("🚀 Initializing Beyond Sight Assistant...")
        
        # Core components
        self.camera = CameraProcessor()
        self.vision = VisionAnalyzer()
        self.speech = SpeechEnginePro()
        self.face_memory = FaceMemory()
        self.navigator = get_navigator()
        self.place_memory = get_place_memory()
        self.deepseek = get_deepseek_client()
        
        # State
        self.current_frame = None
        self.is_running = True
        self.camera_started = False
        self.welcome_given = False
        
        # Camera thread
        self.camera_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0
        self.fps_update_time = time.time()
        
        # Continuous analysis
        self.continuous_analysis_enabled = False
        self.continuous_analysis_thread: Optional[threading.Thread] = None
        self.last_analysis_time = 0
        self.last_analysis_result = ""
        
        # WebSocket
        self.connected_clients: Set[str] = set()
        self.client_lock = threading.Lock()
        
        # Cache
        self.analysis_cache = {
            'scene': {'result': '', 'timestamp': 0},
            'objects': {'result': '', 'timestamp': 0},
            'text': {'result': '', 'timestamp': 0}
        }
        
        logger.info("✅ Assistant initialized")
        self._welcome_user()
    
    def _welcome_user(self) -> None:
        """Play welcome message."""
        if self.welcome_given:
            return
        self.welcome_given = True
        messages = [
            "Welcome to Beyond Sight.",
            "Your personal AI vision assistant.",
            "I'm here to help you see the world.",
            "Preparing your camera now."
        ]
        for msg in messages:
            self.speech.speak(msg, priority=SpeechPriority.NORMAL)
            time.sleep(0.3)
    
    def start_camera(self) -> bool:
        """Start camera."""
        if self.camera_started:
            return True
        
        try:
            logger.info("Starting camera...")
            if not self.camera.start():
                logger.error("Camera start failed")
                self.speech.speak("Failed to start camera", priority=SpeechPriority.PRIORITY)
                return False
            
            self.camera_started = True
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=False)
            self.camera_thread.start()
            
            if self.continuous_analysis_enabled:
                self._start_continuous_analysis()
            
            logger.info("✅ Camera started")
            for msg in ["Camera activated", "Ready to assist"]:
                self.speech.speak(msg, priority=SpeechPriority.NORMAL)
                time.sleep(0.2)
            return True
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return False
    
    def stop_camera(self) -> bool:
        """Stop camera."""
        try:
            logger.info("Stopping camera...")
            self.camera_started = False
            self.continuous_analysis_enabled = False
            
            if self.camera.cap:
                self.camera.cap.release()
            
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2)
            
            if self.continuous_analysis_thread and self.continuous_analysis_thread.is_alive():
                self.continuous_analysis_thread.join(timeout=2)
            
            logger.info("✅ Camera stopped")
            self.speech.speak("Camera deactivated", priority=SpeechPriority.NORMAL)
            return True
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            return False
    
    def _camera_loop(self) -> None:
        """Continuous frame capture loop."""
        logger.info("Camera loop started")
        
        while self.is_running and self.camera_started:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    frame_with_tracking, _ = self.camera.detect_face_and_hands(frame)
                    with self.frame_lock:
                        self.current_frame = frame_with_tracking
                        self.frame_count += 1
                        
                        current_time = time.time()
                        if current_time - self.fps_update_time >= 1.0:
                            self.fps = self.frame_count
                            self.frame_count = 0
                            self.fps_update_time = current_time
                
                time.sleep(1 / FRAME_RATE)
            except Exception as e:
                logger.error(f"Camera loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Camera loop ended")
    
    def get_frame_bytes(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes."""
        try:
            with self.frame_lock:
                if self.current_frame is not None and self.camera_started:
                    _, buffer = cv2.imencode('.jpg', self.current_frame)
                    return buffer.tobytes()
            
            black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', black_frame)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Frame encode error: {e}")
            return None
    
    def get_camera_status(self) -> Dict:
        """Get camera status."""
        return {
            'camera_active': self.camera_started,
            'fps': self.fps,
            'has_frame': self.current_frame is not None,
            'continuous_analysis': self.continuous_analysis_enabled,
            'speech_queue': self.speech.queue_size() if hasattr(self.speech, 'queue_size') else 0,
            'is_speaking': self.speech.is_speaking(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_camera_ready(self) -> Tuple[bool, str]:
        """Check if camera is ready."""
        if not self.camera_started:
            return False, "Camera not active"
        if self.current_frame is None:
            return False, "No camera feed available"
        return True, ""
    
    def _perform_analysis(self, analysis_func, error_prefix: str = "Analysis") -> Dict:
        """Helper for analysis operations."""
        ready, error = self._check_camera_ready()
        if not ready:
            self.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return {'result': error, 'status': 'error', 'error_code': 'camera_not_ready'}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            result = analysis_func(frame)
            self.speech.speak(result, priority=SpeechPriority.PRIORITY)
            logger.info(f"{error_prefix} completed")
            return {
                'result': result,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            error_msg = f"{error_prefix} failed: {str(e)}"
            self.speech.speak(error_msg, priority=SpeechPriority.PRIORITY)
            return {'result': error_msg, 'status': 'error', 'error_code': 'analysis_failed'}
    
    # ============= ANALYSIS METHODS =============
    
    def analyze_scene(self) -> Dict:
        """Analyze scene comprehensively."""
        return self._perform_analysis(
            self.vision.analyze_scene_comprehensive,
            "Scene analysis"
        )
    
    def find_object(self, object_name: str) -> Dict:
        """Find specific object."""
        return self._perform_analysis(
            lambda frame: self.vision.find_specific_object(frame, object_name),
            f"Finding {object_name}"
        )
    
    def read_text(self) -> Dict:
        """Read text in scene."""
        return self._perform_analysis(
            self.vision.read_text_aloud,
            "Text reading"
        )
    
    def get_navigation_advice(self) -> Dict:
        """Get navigation guidance."""
        return self._perform_analysis(
            self.vision.provide_navigation_advice,
            "Navigation advice"
        )
    
    def detect_hazards(self) -> Dict:
        """Detect immediate hazards."""
        ready, error = self._check_camera_ready()
        if not ready:
            return {'result': error, 'status': 'error'}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            result = self.vision.detect_immediate_dangers(frame)
            if result:
                self.speech.speak_critical(f"WARNING: {result}", interrupt=True)
                return {'result': result, 'status': 'warning'}
            else:
                self.speech.speak("Area appears safe", priority=SpeechPriority.PRIORITY)
                return {'result': "No hazards detected", 'status': 'success'}
        except Exception as e:
            logger.error(f"Hazard detection error: {e}")
            return {'result': f"Error: {str(e)}", 'status': 'error'}
    
    def toggle_continuous_analysis(self, enabled: bool) -> bool:
        """Enable/disable continuous analysis."""
        try:
            self.continuous_analysis_enabled = enabled
            if enabled and self.camera_started:
                if not self.continuous_analysis_thread or not self.continuous_analysis_thread.is_alive():
                    self._start_continuous_analysis()
            
            msg = "Continuous analysis " + ("enabled" if enabled else "disabled")
            self.speech.speak(msg, priority=SpeechPriority.NORMAL)
            return True
        except Exception as e:
            logger.error(f"Toggle error: {e}")
            return False
    
    def _start_continuous_analysis(self) -> None:
        """Start continuous analysis thread."""
        if self.continuous_analysis_thread and self.continuous_analysis_thread.is_alive():
            return
        
        self.continuous_analysis_thread = threading.Thread(
            target=self._continuous_analysis_loop,
            daemon=False
        )
        self.continuous_analysis_thread.start()
        logger.info("Continuous analysis started")
    
    def _continuous_analysis_loop(self) -> None:
        """Background continuous analysis."""
        logger.info("Continuous analysis loop started")
        
        while self.continuous_analysis_enabled and self.camera_started:
            try:
                current_time = time.time()
                if current_time - self.last_analysis_time >= CONTINUOUS_ANALYSIS_INTERVAL:
                    with self.frame_lock:
                        if self.current_frame is not None:
                            result = self.vision.analyze_scene_comprehensive(self.current_frame)
                            if result and result != self.last_analysis_result:
                                self.last_analysis_result = result
                                self.last_analysis_time = current_time
                                self.speech.speak(result, priority=SpeechPriority.BACKGROUND)
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Continuous analysis error: {e}")
                time.sleep(1)
        
        logger.info("Continuous analysis ended")
    
    def set_voice_settings(self, rate: str = 'NORMAL', volume: str = 'LOUD') -> bool:
        """Set voice parameters."""
        try:
            rate_enum = VoiceRate[rate.upper()] if hasattr(VoiceRate, rate.upper()) else VoiceRate.NORMAL
            volume_enum = VoiceVolume[volume.upper()] if hasattr(VoiceVolume, volume.upper()) else VoiceVolume.LOUD
            self.speech.set_rate(rate_enum)
            self.speech.set_volume(volume_enum)
            logger.info(f"Voice settings: rate={rate}, volume={volume}")
            return True
        except Exception as e:
            logger.error(f"Voice settings error: {e}")
            return False
    
    def recognize_people(self) -> Dict:
        """Recognize people in scene."""
        if not FACE_RECOGNITION_AVAILABLE:
            msg = "Face recognition not available"
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'error', 'message': msg}
        
        ready, error = self._check_camera_ready()
        if not ready:
            return {'status': 'error', 'message': error}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            recognized, unknown_count = self.face_memory.recognize_people(frame)
            msg = f"Found {len(recognized)} people"
            if unknown_count > 0:
                msg += f" and {unknown_count} unknown"
            
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'success', 'recognized': recognized, 'unknown_count': unknown_count}
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def remember_current_person(self, name: str, notes: str = '') -> Dict:
        """Remember current person."""
        if not FACE_RECOGNITION_AVAILABLE:
            return {'status': 'error', 'message': 'Face recognition not available'}
        
        ready, error = self._check_camera_ready()
        if not ready:
            return {'status': 'error', 'message': error}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            person_id = self.face_memory.remember_face(frame, name, notes)
            msg = f"Remembered {name}"
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'success', 'person_id': person_id, 'name': name}
        except Exception as e:
            logger.error(f"Remember error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def forget_person(self, person_id: str) -> Dict:
        """Forget a person."""
        try:
            if self.face_memory.forget_person(person_id):
                self.speech.speak("Person forgotten", priority=SpeechPriority.NORMAL)
                return {'status': 'success'}
            return {'status': 'error', 'message': 'Person not found'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _process_command(self, command: str) -> Dict:
        """Process voice command."""
        if 'help' in command:
            help_text = "Available: Analyze scene, Find people, Find doors, Read text, Check hazards, Get navigation"
            self.speech.speak(help_text, priority=SpeechPriority.PRIORITY)
            return {'status': 'success', 'result': help_text}
        elif any(kw in command for kw in ['analyze', 'scene']):
            return self.analyze_scene()
        elif any(kw in command for kw in ['people', 'person']):
            return self.find_object('person')
        elif any(kw in command for kw in ['door', 'doors']):
            return self.find_object('door')
        elif any(kw in command for kw in ['text', 'read']):
            return self.read_text()
        elif any(kw in command for kw in ['hazard', 'danger', 'safe']):
            return self.detect_hazards()
        elif any(kw in command for kw in ['navigate', 'guidance']):
            return self.get_navigation_advice()
        else:
            return {'status': 'error', 'result': 'Command not recognized. Say help.'}
