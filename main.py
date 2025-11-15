"""
Beyond Sight - AI-Powered Vision Assistant for the Visually Impaired

A professional web-based application that provides real-time audio descriptions
of the user's surroundings using computer vision and AI analysis. Acts as a
personal AI assistant (similar to Alexa) but focused on visual accessibility.

Features:
- Real-time scene analysis with voice feedback
- Object detection and distance estimation
- Text recognition (OCR)
- Obstacle detection and safety warnings
- Navigation guidance
- Context-aware descriptions
"""

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import threading
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, List, Set, Tuple, Callable, Any
import json
import torch
import asyncio
from pathlib import Path

from camera_processor import CameraProcessor
from vision_analyzer import VisionAnalyzer
from speech_engine_pro import SpeechEnginePro, VoiceRate, VoiceVolume, SpeechPriority
from face_memory import FaceMemory, FACE_RECOGNITION_AVAILABLE
from gps_navigator import get_navigator, GPSNavigator
from place_memory import get_place_memory
from deepseek_integration import get_deepseek_client, AnalysisType, set_api_key
from utils import setup_logging, ensure_directories

# ============= GPU ACCELERATION SETUP =============
# Force GPU usage for YOLO
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger_gpu = logging.getLogger('GPU_CONFIG')

if DEVICE == 'cuda':
    logger_gpu.info("🚀 GPU CUDA detected - Accelerating with GPU")
    try:
        cv2.cuda.setDevice(0)
        logger_gpu.info("✅ OpenCV GPU acceleration enabled")
    except (RuntimeError, cv2.error) as e:
        logger_gpu.warning(f"⚠️ OpenCV GPU not available: {e}, using CPU fallback")
else:
    logger_gpu.info("💻 GPU not available - Using CPU (install CUDA toolkit for GPU acceleration)")

# OpenCV DNN GPU acceleration (if available)
def setup_opencv_dnn_gpu():
    """Configure OpenCV DNN for GPU acceleration.
    
    Returns:
        cv2.dnn network object if successful, None otherwise
    """
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net = cv2.dnn.readNetFromONNX('yolov8n.onnx')
            if net:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger_gpu.info("✅ OpenCV DNN GPU acceleration enabled")
                return net
    except FileNotFoundError:
        logger_gpu.debug("ONNX file not found, skipping DNN GPU setup")
    except (RuntimeError, cv2.error) as e:
        logger_gpu.debug(f"DNN GPU setup failed: {e}")
    return None

# ============= CONFIGURATION =============
app = FastAPI(
    title="Beyond Sight",
    description="AI-Powered Vision Assistant for the Visually Impaired",
    version="2.0"
)

# Add CORS middleware with restricted origins
from config import Config
allowed_origins = Config.CORS_ALLOW_ORIGINS if not Config.CORS_DEBUG_MODE else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods instead of wildcard
    allow_headers=["Content-Type", "Authorization"],  # Explicit headers instead of wildcard
    max_age=3600  # Cache preflight requests for 1 hour
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('beyond_sight.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Serve static files from dedicated directory
try:
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files directory issue: {e}")

# Import constants from config instead of hardcoding
FRAME_RATE = Config.CAMERA.frame_rate
FRAME_HEIGHT = Config.CAMERA.frame_height
FRAME_WIDTH = Config.CAMERA.frame_width
CONTINUOUS_ANALYSIS_INTERVAL = Config.ANALYSIS.interval
ANALYSIS_TIMEOUT = Config.ANALYSIS.timeout


# ============= WEB ASSISTANT CLASS =============
class WebBeyondSightAssistant:
    """Main application class that manages the vision assistance experience."""
    
    def __init__(self):
        """Initialize the assistant with all required components."""
        logger.info("Initializing Beyond Sight Assistant...")
        
        self.camera = CameraProcessor()
        self.vision = VisionAnalyzer()
        self.speech = SpeechEnginePro()
        self.face_memory = FaceMemory()
        self.navigator = get_navigator()
        self.place_memory = get_place_memory()
        self.deepseek = get_deepseek_client()
        
        # State management
        self.current_frame = None
        self.is_running = True
        self.camera_started = False
        self.camera_thread: Optional[threading.Thread] = None
        self.welcome_given = False
        
        # Continuous analysis
        self.continuous_analysis_enabled = False
        self.continuous_analysis_thread: Optional[threading.Thread] = None
        self.last_analysis_time = 0
        self.last_analysis_result = ""
        
        # Face recognition
        self.face_recognition_enabled = False
        self.last_recognized_faces = []
        
        # Frame processing
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.fps = 0
        self.fps_update_time = time.time()
        
        # Analysis cache
        self.analysis_cache = {
            'scene': {'result': '', 'timestamp': 0},
            'objects': {'result': '', 'timestamp': 0},
            'text': {'result': '', 'timestamp': 0}
        }
        
        # WebSocket state
        self.connected_clients: Set[str] = set()
        self.client_lock = threading.Lock()
        self.status_update_interval = 0.5
        self.last_status_update = 0
        
        logger.info("✅ Beyond Sight Assistant initialized successfully")
        self._welcome_user()
    
    def _welcome_user(self) -> None:
        """Play welcoming introduction."""
        if self.welcome_given:
            return
        self.welcome_given = True
        for msg in ["Welcome to Beyond Sight.", 
                    "Your personal AI vision assistant, powered by advanced artificial intelligence.",
                    "I'm here to help you see the world around you like never before.",
                    "Let's unlock a new experience together.",
                    "Preparing your camera now."]:
            self.speech.speak(msg, priority=SpeechPriority.NORMAL)
            time.sleep(0.5)
    
    def start_camera(self) -> bool:
        """Start the camera and begin capturing frames."""
        if self.camera_started:
            logger.warning("Camera already started")
            return True
        
        try:
            logger.info("Starting camera...")
            success = self.camera.start()
            
            if success:
                self.camera_started = True
                self.camera_thread = threading.Thread(target=self._camera_loop, name="CameraWorker", daemon=False)
                self.camera_thread.start()
                
                if self.continuous_analysis_enabled:
                    self._start_continuous_analysis()
                
                logger.info("✅ Camera started successfully")
                for msg in ["Camera activated successfully.", 
                           "I can now see what you see.",
                           "Ready to assist you with real-time vision analysis."]:
                    self.speech.speak(msg, priority=SpeechPriority.NORMAL)
                    time.sleep(0.3)
                return True
            else:
                logger.error("Failed to initialize camera")
                self.speech.speak("Failed to start camera. Please check your device and try again.", priority=SpeechPriority.PRIORITY)
                return False
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            self.speech.speak(f"Error starting camera: {str(e)}", priority=SpeechPriority.PRIORITY)
            return False
    
    def stop_camera(self) -> bool:
        """Stop the camera and clean up resources."""
        try:
            logger.info("Stopping camera...")
            self.camera_started = False
            self.continuous_analysis_enabled = False
            
            if hasattr(self.camera, 'cleanup'):
                self.camera.cleanup()
            elif self.camera.cap:
                self.camera.cap.release()
            
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2)
            
            if self.continuous_analysis_thread and self.continuous_analysis_thread.is_alive():
                self.continuous_analysis_thread.join(timeout=2)
            
            logger.info("✅ Camera stopped")
            self.speech.speak("Camera deactivated.", priority=SpeechPriority.NORMAL)
            return True
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            return False
    
    def _camera_loop(self) -> None:
        """Background thread that continuously captures frames."""
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
                logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)
        
        logger.info("Camera loop ended")
    
    def get_frame_bytes(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes for streaming."""
        try:
            with self.frame_lock:
                if self.current_frame is not None and self.camera_started:
                    _, buffer = cv2.imencode('.jpg', self.current_frame)
                    return buffer.tobytes()
            
            black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', black_frame)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None
    
    def get_camera_status(self) -> Dict:
        """Get current camera and system status."""
        status = {
            'camera_active': self.camera_started,
            'fps': self.fps,
            'has_frame': self.current_frame is not None,
            'continuous_analysis': self.continuous_analysis_enabled,
            'speech_queue': self.speech.queue_size(),
            'is_speaking': self.speech.is_speaking(),
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def toggle_continuous_analysis(self, enabled: bool) -> bool:
        """Enable or disable continuous scene descriptions."""
        try:
            self.continuous_analysis_enabled = enabled
            if enabled and self.camera_started:
                if not self.continuous_analysis_thread or not self.continuous_analysis_thread.is_alive():
                    self._start_continuous_analysis()
                    self.speech.speak("Continuous description enabled.", priority=SpeechPriority.NORMAL)
                    return True
            else:
                self.speech.speak("Continuous description disabled.", priority=SpeechPriority.NORMAL)
                return True
        except Exception as e:
            logger.error(f"Error toggling continuous analysis: {e}")
            return False
    
    def _start_continuous_analysis(self) -> None:
        """Start background thread for continuous analysis."""
        if self.continuous_analysis_thread and self.continuous_analysis_thread.is_alive():
            return
        
        self.continuous_analysis_thread = threading.Thread(
            target=self._continuous_analysis_loop, name="ContinuousAnalysis", daemon=False
        )
        self.continuous_analysis_thread.start()
        logger.info("Continuous analysis thread started")
    
    def _continuous_analysis_loop(self) -> None:
        """Background thread that periodically describes the scene."""
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
                logger.error(f"Error in continuous analysis: {e}")
                time.sleep(1)
        
        logger.info("Continuous analysis loop ended")
    
    def _check_camera_ready(self) -> Tuple[bool, str]:
        """Check if camera is ready for analysis.
        
        Returns:
            Tuple of (is_ready, error_message)
        """
        if not self.camera_started:
            return False, "Camera is not active. Please start the camera first."
        if self.current_frame is None:
            return False, "No camera feed available yet. Please wait a moment."
        return True, ""
    
    def _perform_analysis(self, analysis_func, error_prefix: str = "Analysis") -> Dict[str, str]:
        """Helper method to reduce code repetition for analysis operations.
        
        Args:
            analysis_func: Callable that takes a frame and returns analysis result
            error_prefix: Text to use in logging and error messages
            
        Returns:
            Dict with 'result', 'status', and optional 'error_code'
        """
        ready, error = self._check_camera_ready()
        if not ready:
            self.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return {
                'result': error,
                'status': 'error',
                'error_code': 'camera_not_ready'
            }
        
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
            logger.error(f"Error in {error_prefix}: {e}", exc_info=True)
            error_msg = f"{error_prefix} failed: {str(e)}"
            self.speech.speak(error_msg, priority=SpeechPriority.PRIORITY)
            return {
                'result': error_msg,
                'status': 'error',
                'error_code': 'analysis_failed'
            }
    
    def analyze_scene(self) -> Dict[str, str]:
        """Perform comprehensive scene analysis with voice output.
        
        Analyzes the current camera frame to provide detailed information about:
        - Objects and their locations
        - People and their activities
        - Text visible in the scene
        - Environmental hazards
        - Spatial relationships
        
        Returns:
            Dict with keys:
                - 'result': Description of the scene
                - 'status': 'success' or 'error'
                - 'timestamp': When analysis was performed (on success)
        """
        result = self._perform_analysis(self.vision.analyze_scene_comprehensive, "Scene analysis")
        if result['status'] == 'success':
            self.last_analysis_result = result['result']
            self.last_analysis_time = time.time()
            result['timestamp'] = datetime.now().isoformat()
        return result
    
    def find_object(self, object_name: str) -> Dict[str, str]:
        """Find specific objects in the scene."""
        return self._perform_analysis(
            lambda frame: self.vision.find_specific_object(frame, object_name),
            f"Finding {object_name}"
        )
    
    def read_text(self) -> Dict[str, str]:
        """Read text visible in the scene."""
        return self._perform_analysis(self.vision.read_text_aloud, "Text reading")
    
    def get_navigation_advice(self) -> Dict[str, str]:
        """Get navigation guidance for safe movement."""
        return self._perform_analysis(self.vision.provide_navigation_advice, "Navigation advice")
    
    def detect_hazards(self) -> Dict[str, str]:
        """Check for immediate dangers and obstacles."""
        ready, error = self._check_camera_ready()
        if not ready:
            self.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return {'result': error, 'status': 'error'}
        
        try:
            logger.info("Checking for hazards...")
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            result = self.vision.detect_immediate_dangers(frame)
            if result:
                warning_msg = f"⚠️ WARNING: {result}"
                self.speech.speak_critical(warning_msg, interrupt=True)
                logger.warning(f"Hazard detected: {result}")
                return {'result': result, 'status': 'warning'}
            else:
                safe_msg = "Area appears safe. No immediate hazards detected."
                self.speech.speak(safe_msg, priority=SpeechPriority.PRIORITY)
                return {'result': safe_msg, 'status': 'success'}
        except Exception as e:
            logger.error(f"Error checking hazards: {e}")
            error_msg = f"Failed to check hazards: {str(e)}"
            self.speech.speak(error_msg, priority=SpeechPriority.PRIORITY)
            return {'result': error_msg, 'status': 'error'}
    
    def set_voice_settings(self, rate: str = 'NORMAL', volume: str = 'LOUD') -> bool:
        """Configure voice settings."""
        try:
            rate_enum = VoiceRate[rate.upper()] if hasattr(VoiceRate, rate.upper()) else VoiceRate.NORMAL
            volume_enum = VoiceVolume[volume.upper()] if hasattr(VoiceVolume, volume.upper()) else VoiceVolume.LOUD
            self.speech.set_rate(rate_enum)
            self.speech.set_volume(volume_enum)
            logger.info(f"Voice settings updated: rate={rate}, volume={volume}")
            return True
        except Exception as e:
            logger.error(f"Error setting voice settings: {e}")
            return False
    
    def recognize_people(self) -> Dict[str, Any]:
        """Recognize people in the current frame."""
        if not FACE_RECOGNITION_AVAILABLE:
            msg = "Face recognition is not available on this system."
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'error', 'message': msg}
        
        ready, error = self._check_camera_ready()
        if not ready:
            self.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return {'status': 'error', 'message': error}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            recognized, unknown_count = self.face_memory.recognize_people(frame)
            self.last_recognized_faces = recognized
            
            if not recognized and unknown_count == 0:
                msg = "No people detected in the scene."
            else:
                msg = f"Recognized {len(recognized)} people" + (f" and {unknown_count} unknown" if unknown_count > 0 else "")
                for person in recognized:
                    msg += f". Found {person['name']}"
            
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'success', 'recognized': recognized, 'unknown_count': unknown_count}
        except Exception as e:
            logger.error(f"Error recognizing people: {e}")
            self.speech.speak(f"Error recognizing people: {str(e)}", priority=SpeechPriority.PRIORITY)
            return {'status': 'error', 'message': str(e)}
    
    def remember_current_person(self, name: str, notes: str = '') -> Dict[str, Any]:
        """Remember a person's face from the current frame."""
        if not FACE_RECOGNITION_AVAILABLE:
            return {'status': 'error', 'message': 'Face recognition not available'}
        
        ready, error = self._check_camera_ready()
        if not ready:
            return {'status': 'error', 'message': error}
        
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            success = self.face_memory.remember_face(frame, name, notes)
            if success:
                msg = f"Remembered {name}."
                self.speech.speak(msg, priority=SpeechPriority.NORMAL)
                return {'status': 'success', 'message': f'Remembered {name}'}
            else:
                msg = f"Could not find a face to remember."
                self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
                return {'status': 'error', 'message': msg}
        except Exception as e:
            logger.error(f"Error remembering person: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_navigation_guidance(self) -> Dict[str, str]:
        """Get navigation guidance with GPS data."""
        try:
            guidance = self.navigator.get_navigation_guidance()
            self.speech.speak(guidance, priority=SpeechPriority.PRIORITY)
            return {'result': guidance, 'status': 'success'}
        except Exception as e:
            logger.error(f"Error getting navigation guidance: {e}")
            return {'result': str(e), 'status': 'error'}
    
    def add_waypoint(self, latitude: float, longitude: float, name: str = '') -> Dict:
        """Add a waypoint to navigation route."""
        try:
            self.navigator.add_waypoint(latitude, longitude, name)
            msg = f"Waypoint '{name}' added."
            self.speech.speak(msg, priority=SpeechPriority.NORMAL)
            return {'status': 'success', 'message': msg}
        except Exception as e:
            logger.error(f"Error adding waypoint: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_navigation_stats(self) -> Dict:
        """Get navigation statistics."""
        try:
            stats = self.navigator.get_statistics()
            return {'status': 'success', 'stats': stats}
        except Exception as e:
            logger.error(f"Error getting navigation stats: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def detect_stairs(self) -> Dict:
        """Detect stairs and elevation changes."""
        return self._perform_analysis(
            self.vision.detect_stairs_elevation_changes,
            "Stair detection"
        )
    
    def detect_intersections(self) -> Dict:
        """Detect street intersections."""
        return self._perform_analysis(
            self.vision.detect_intersections_crosswalks,
            "Intersection detection"
        )
    
    def analyze_lighting_conditions(self) -> Dict:
        """Analyze lighting conditions."""
        return self._perform_analysis(
            self.vision.analyze_lighting_conditions,
            "Lighting analysis"
        )
    
    def describe_room_layout(self) -> Dict:
        """Describe room layout and spatial layout."""
        return self._perform_analysis(
            self.vision.describe_room_layout,
            "Room description"
        )
    
    def detect_color_signals(self) -> Dict:
        """Detect color signals (traffic lights, warnings)."""
        return self._perform_analysis(
            self.vision.detect_color_signals,
            "Color signal detection"
        )
    
    def read_printed_text(self, focus_area: str = 'center') -> Dict:
        """Read printed text with position information."""
        try:
            result = self.vision.read_text_with_position(focus_area)
            self.speech.speak(result, priority=SpeechPriority.PRIORITY)
            return {'result': result, 'status': 'success', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error reading text with position: {e}")
            return {'result': str(e), 'status': 'error'}
    
    def get_nearby_locations(self, radius: float = 1.0) -> Dict:
        """Get nearby places of interest."""
        try:
            # This would use place_memory instead for location-based queries
            context = self.navigator.get_location_context()
            msg = f"Current location context: {context}"
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'status': 'success', 'context': context}
        except Exception as e:
            logger.error(f"Error getting nearby locations: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def emergency_alert(self, alert_type: str = 'general', message: str = '') -> Dict:
        """Send emergency alert."""
        try:
            alert_msg = f"EMERGENCY ALERT: {message}" if message else f"EMERGENCY: {alert_type} alert activated"
            self.speech.speak_critical(alert_msg, interrupt=True)
            logger.critical(f"Emergency alert: {alert_msg}")
            return {'status': 'success', 'message': 'Alert sent'}
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_detailed_description(self, focus: str = 'foreground') -> Dict:
        """Get detailed spatial description."""
        try:
            desc = self.vision.get_detailed_spatial_description(focus)
            self.speech.speak(desc, priority=SpeechPriority.PRIORITY)
            return {'result': desc, 'status': 'success', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error getting detailed description: {e}")
            return {'result': str(e), 'status': 'error'}
    
    def detect_people_and_activity(self) -> Dict:
        """Detect people and their activities."""
        return self._perform_analysis(
            self.vision.detect_people_and_activities,
            "People detection"
        )
    
    def get_time_of_day_info(self) -> Dict:
        """Analyze time of day from scene."""
        return self._perform_analysis(
            self.vision.analyze_time_of_day,
            "Time of day analysis"
        )
    
    def detect_audio_cues_needed(self) -> Dict:
        """Identify sounds user should be aware of."""
        return self._perform_analysis(
            self.vision.suggest_audio_cues,
            "Audio cue detection"
        )
    
    def get_distance_estimates(self, direction: str = 'all') -> Dict:
        """Get distance estimates to objects."""
        try:
            distances = self.vision.get_distance_estimates(self.current_frame, direction)
            msg = f"Distance estimates: {distances}"
            self.speech.speak(msg, priority=SpeechPriority.PRIORITY)
            return {'result': distances, 'status': 'success', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error getting distances: {e}")
            return {'result': str(e), 'status': 'error'}
    
    def explain_gesture_context(self) -> Dict:
        """Explain gesture context and social cues."""
        return self._perform_analysis(
            self.vision.analyze_social_context,
            "Gesture analysis"
        )
    
    def get_safe_path_guidance(self, direction: str = 'forward') -> Dict:
        """Get safe path guidance."""
        try:
            guidance = self.vision.get_safe_path_guidance(self.current_frame, direction)
            self.speech.speak(guidance, priority=SpeechPriority.PRIORITY)
            return {'result': guidance, 'status': 'success', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error getting safe path: {e}")
            return {'result': str(e), 'status': 'error'}
    
    def detect_water_and_wet_surfaces(self) -> Dict:
        """Detect water and wet surfaces."""
        return self._perform_analysis(
            self.vision.detect_water_hazards,
            "Water detection"
        )
    
    def get_ambient_analysis(self) -> Dict:
        """Get ambient conditions analysis."""
        return self._perform_analysis(
            self.vision.analyze_ambient_conditions,
            "Ambient analysis"
        )


# Initialize global assistant
assistant = WebBeyondSightAssistant()


# ============= API ROUTES - CORE FUNCTIONALITY =============

@app.get("/")
async def root():
    """Serve the main application page."""
    try:
        index_path = Path("templates/index.html")
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        return JSONResponse({"status": "error", "message": "Index file not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/assets/{file_path:path}")
async def serve_asset(file_path: str):
    """Serve static assets from static directory."""
    try:
        asset_path = Path("static") / file_path
        if asset_path.exists() and asset_path.is_file():
            return FileResponse(asset_path)
        return JSONResponse({"status": "error", "message": "Asset not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving asset {file_path}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/video_feed")
async def video_feed():
    """Stream video feed as MJPEG."""
    async def generate():
        while assistant.camera_started:
            frame_bytes = assistant.get_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
            await asyncio.sleep(1 / FRAME_RATE)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/frame")
async def get_frame():
    """Get single current frame as JPEG."""
    frame_bytes = assistant.get_frame_bytes()
    if frame_bytes:
        return StreamingResponse(iter([frame_bytes]), media_type="image/jpeg")
    return JSONResponse({"status": "error", "message": "No frame available"}, status_code=500)


@app.get("/api/status")
async def get_status():
    """Get application status."""
    return assistant.get_camera_status()


@app.post("/api/camera/start")
async def start_camera():
    """Start the camera."""
    try:
        success = assistant.start_camera()
        return {
            "status": "success" if success else "error",
            "camera_started": success,
            "message": "Camera started" if success else "Failed to start camera"
        }
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/camera/stop")
async def stop_camera():
    """Stop the camera."""
    try:
        success = assistant.stop_camera()
        return {
            "status": "success" if success else "error",
            "camera_started": not success,
            "message": "Camera stopped" if success else "Failed to stop camera"
        }
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= ANALYSIS ENDPOINTS =============

@app.post("/api/analyze/scene")
async def analyze_scene_route():
    """Analyze the current scene."""
    try:
        return assistant.analyze_scene()
    except Exception as e:
        logger.error(f"Error analyzing scene: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/find")
async def find_object_route(object_name: str = Query(...)):
    """Find specific object in scene."""
    try:
        return assistant.find_object(object_name)
    except Exception as e:
        logger.error(f"Error finding object: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/find/people")
async def find_people_route(body: Dict = Body(None)):
    """Find people in scene."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.find_specific_object(frame, 'person')
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.find_object('person')
    except Exception as e:
        logger.error(f"Error finding people: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/find/doors")
async def find_doors_route(body: Dict = Body(None)):
    """Find doors in scene."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.find_specific_object(frame, 'door')
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.find_object('door')
    except Exception as e:
        logger.error(f"Error finding doors: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/find/chairs")
async def find_chairs_route(body: Dict = Body(None)):
    """Find chairs in scene."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.find_specific_object(frame, 'chair')
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.find_object('chair')
    except Exception as e:
        logger.error(f"Error finding chairs: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/hazards")
async def detect_hazards_route(body: Dict = Body(None)):
    """Detect hazards in the scene."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.detect_immediate_dangers(frame)
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.detect_hazards()
    except Exception as e:
        logger.error(f"Error detecting hazards: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/text")
async def read_text_route(body: Dict = Body(None)):
    """Read text from the scene."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.read_text_aloud(frame)
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.read_text()
    except Exception as e:
        logger.error(f"Error reading text: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/navigation")
async def get_navigation_route(body: Dict = Body(None)):
    """Get navigation advice."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.provide_navigation_advice(frame)
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        return assistant.get_navigation_advice()
    except Exception as e:
        logger.error(f"Error getting navigation: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/continuous")
async def toggle_continuous_route(enabled: bool = Query(...)):
    """Toggle continuous analysis."""
    try:
        success = assistant.toggle_continuous_analysis(enabled)
        return {"status": "success" if success else "error", "continuous": enabled}
    except Exception as e:
        logger.error(f"Error toggling continuous: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= FACE RECOGNITION ROUTES =============

@app.post("/api/faces/recognize")
async def recognize_people_route():
    """Recognize people in the scene."""
    try:
        return assistant.recognize_people()
    except Exception as e:
        logger.error(f"Error recognizing people: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/faces/remember")
async def remember_person_route(name: str = Query(...), notes: str = Query(default="")):
    """Remember a person."""
    try:
        return assistant.remember_current_person(name, notes)
    except Exception as e:
        logger.error(f"Error remembering person: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= VOICE SETTINGS ROUTES =============

@app.post("/api/voice/settings")
async def set_voice_route(rate: str = Query("NORMAL"), volume: str = Query("LOUD")):
    """Set voice settings."""
    try:
        success = assistant.set_voice_settings(rate, volume)
        return {"status": "success" if success else "error"}
    except Exception as e:
        logger.error(f"Error setting voice: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= NAVIGATION ROUTES =============

@app.get("/api/navigation/guidance")
async def get_guidance():
    """Get navigation guidance to destination."""
    try:
        return assistant.get_navigation_guidance()
    except Exception as e:
        logger.error(f"Error getting guidance: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/navigation/waypoint/add")
async def add_waypoint(latitude: float = Query(...), longitude: float = Query(...), name: str = Query(default="")):
    """Add waypoint to route."""
    try:
        return assistant.add_waypoint(latitude, longitude, name)
    except Exception as e:
        logger.error(f"Error adding waypoint: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/navigation/stats")
async def get_nav_stats():
    """Get navigation statistics."""
    try:
        return assistant.get_navigation_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= ACCESSIBILITY API ROUTES =============

@app.post("/api/detect/stairs")
async def detect_stairs_route():
    """Detect stairs and elevation changes."""
    try:
        return assistant.detect_stairs()
    except Exception as e:
        logger.error(f"Error detecting stairs: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/intersections")
async def detect_intersections_route():
    """Detect street intersections and crosswalks."""
    try:
        return assistant.detect_intersections()
    except Exception as e:
        logger.error(f"Error detecting intersections: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/lighting")
async def analyze_lighting_route():
    """Analyze lighting conditions."""
    try:
        return assistant.analyze_lighting_conditions()
    except Exception as e:
        logger.error(f"Error analyzing lighting: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/describe/room-layout")
async def describe_room_route():
    """Describe room layout."""
    try:
        return assistant.describe_room_layout()
    except Exception as e:
        logger.error(f"Error describing room: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/color-signals")
async def detect_signals_route():
    """Detect color signals (traffic lights, warnings)."""
    try:
        return assistant.detect_color_signals()
    except Exception as e:
        logger.error(f"Error detecting signals: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/read/text-position")
async def read_text_position_route(focus_area: str = Query(default="center")):
    """Read text with position information."""
    try:
        return assistant.read_printed_text(focus_area)
    except Exception as e:
        logger.error(f"Error reading text with position: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.get("/api/location/nearby")
async def get_nearby_locations_route(radius: float = Query(default=1.0)):
    """Find nearby places of interest."""
    try:
        return assistant.get_nearby_locations(radius)
    except Exception as e:
        logger.error(f"Error getting nearby locations: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/emergency/alert")
async def emergency_alert_route(alert_type: str = Query(default="general"), message: str = Query(default="")):
    """Send emergency alert."""
    try:
        return assistant.emergency_alert(alert_type, message)
    except Exception as e:
        logger.error(f"Error sending emergency alert: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/describe/detailed")
async def get_detailed_description_route(focus: str = Query(default="foreground")):
    """Get detailed spatial description."""
    try:
        return assistant.get_detailed_description(focus)
    except Exception as e:
        logger.error(f"Error getting detailed description: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/people-activity")
async def detect_people_activity_route():
    """Detect people and their activities."""
    try:
        return assistant.detect_people_and_activity()
    except Exception as e:
        logger.error(f"Error detecting people activity: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/time-of-day")
async def get_time_of_day_route():
    """Analyze time of day from scene."""
    try:
        return assistant.get_time_of_day_info()
    except Exception as e:
        logger.error(f"Error getting time of day: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/audio-cues")
async def detect_audio_cues_route():
    """Identify sounds user should be aware of."""
    try:
        return assistant.detect_audio_cues_needed()
    except Exception as e:
        logger.error(f"Error detecting audio cues: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/measure/distances")
async def get_distances_route(direction: str = Query(default="all")):
    """Get distance estimates to objects."""
    try:
        return assistant.get_distance_estimates(direction)
    except Exception as e:
        logger.error(f"Error getting distances: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/social-context")
async def analyze_social_context_route():
    """Analyze social context and gestures."""
    try:
        return assistant.explain_gesture_context()
    except Exception as e:
        logger.error(f"Error analyzing social context: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/guidance/safe-path")
async def get_safe_path_route(direction: str = Query(default="forward")):
    """Get safe path guidance."""
    try:
        return assistant.get_safe_path_guidance(direction)
    except Exception as e:
        logger.error(f"Error getting safe path: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/detect/water-hazards")
async def detect_water_hazards_route():
    """Detect water and wet surfaces."""
    try:
        return assistant.detect_water_and_wet_surfaces()
    except Exception as e:
        logger.error(f"Error detecting water hazards: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


@app.post("/api/analyze/ambient")
async def get_ambient_analysis_route():
    """Get ambient conditions analysis."""
    try:
        return assistant.get_ambient_analysis()
    except Exception as e:
        logger.error(f"Error analyzing ambient: {e}")
        return JSONResponse({"status": "error", "result": str(e)}, status_code=500)


# ============= NAMED PLACES / PLACE MEMORY ROUTES =============

@app.post("/api/places/save")
async def save_place(name: str = Query(...), latitude: float = Query(...), 
                     longitude: float = Query(...), description: str = Query(default=""),
                     tags: str = Query(default="")):
    """Save a named place."""
    try:
        tag_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
        success = assistant.place_memory.save_place(name, latitude, longitude, description, tag_list)
        
        if success:
            msg = f"Saved place: {name}"
            assistant.speech.speak(msg, priority=SpeechPriority.NORMAL)
            return {
                "status": "success",
                "message": msg,
                "place": {
                    "name": name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "description": description,
                    "tags": tag_list
                }
            }
        else:
            return JSONResponse({"status": "error", "message": "Failed to save place"}, status_code=500)
    except Exception as e:
        logger.error(f"Error saving place: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/get")
async def get_place(name: str = Query(...)):
    """Get a saved place by name."""
    try:
        place = assistant.place_memory.get_place(name)
        if place:
            return {"status": "success", "place": place}
        else:
            return JSONResponse({"status": "error", "message": f"Place '{name}' not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error getting place: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/all")
async def get_all_places():
    """Get all saved places."""
    try:
        places = assistant.place_memory.get_all_places()
        return {"status": "success", "total": len(places), "places": places}
    except Exception as e:
        logger.error(f"Error getting all places: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.delete("/api/places/delete")
async def delete_place(name: str = Query(...)):
    """Delete a saved place."""
    try:
        success = assistant.place_memory.delete_place(name)
        if success:
            msg = f"Deleted place: {name}"
            assistant.speech.speak(msg, priority=SpeechPriority.NORMAL)
            return {"status": "success", "message": msg}
        else:
            return JSONResponse({"status": "error", "message": f"Place '{name}' not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error deleting place: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/search")
async def search_places(query: str = Query(...)):
    """Search places by name or description."""
    try:
        results = assistant.place_memory.search_places(query)
        return {"status": "success", "query": query, "found": len(results), "places": results}
    except Exception as e:
        logger.error(f"Error searching places: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/nearby")
async def get_nearby_places(latitude: float = Query(...), longitude: float = Query(...),
                            radius: float = Query(default=1.0)):
    """Find places within a radius."""
    try:
        places = assistant.place_memory.get_places_within_radius(latitude, longitude, radius)
        
        if places:
            # Speak the nearest place
            nearest = places[0]
            msg = f"Nearest place: {nearest['name']}, {nearest['distance_km']} kilometers away"
            assistant.speech.speak(msg, priority=SpeechPriority.NORMAL)
        
        return {
            "status": "success",
            "current_location": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius,
            "found": len(places),
            "places": places
        }
    except Exception as e:
        logger.error(f"Error finding nearby places: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/nearest")
async def get_nearest_place(latitude: float = Query(...), longitude: float = Query(...)):
    """Find the nearest saved place."""
    try:
        nearest = assistant.place_memory.get_nearest_place(latitude, longitude)
        
        if nearest:
            msg = f"Nearest place: {nearest['name']}, {nearest['distance_km']} kilometers away"
            assistant.speech.speak(msg, priority=SpeechPriority.NORMAL)
            return {"status": "success", "nearest_place": nearest}
        else:
            return {"status": "success", "message": "No saved places found"}
    except Exception as e:
        logger.error(f"Error finding nearest place: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/distance")
async def get_distance_to_place(place_name: str = Query(...),
                                latitude: float = Query(...),
                                longitude: float = Query(...)):
    """Get distance to a saved place."""
    try:
        distance = assistant.place_memory.get_distance_to_place(place_name, latitude, longitude)
        
        if distance is not None:
            # Format distance nicely
            if distance < 1:
                distance_text = f"{distance * 1000:.0f} meters"
            else:
                distance_text = f"{distance:.1f} kilometers"
            
            msg = f"{place_name} is {distance_text} away"
            assistant.speech.speak(msg, priority=SpeechPriority.NORMAL)
            
            return {
                "status": "success",
                "place": place_name,
                "distance_km": round(distance, 2),
                "distance_text": distance_text
            }
        else:
            return JSONResponse({"status": "error", "message": f"Place '{place_name}' not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/places/stats")
async def get_place_stats():
    """Get place memory statistics."""
    try:
        stats = assistant.place_memory.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        logger.error(f"Error getting place stats: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= DEEPSEEK AI INTEGRATION ROUTES =============

@app.post("/api/deepseek/setkey")
async def set_deepseek_key(api_key: str = Query(...)):
    """Set Deepseek API key."""
    try:
        if not api_key or not api_key.strip():
            return JSONResponse({"status": "error", "message": "API key cannot be empty"}, status_code=400)
        
        success = set_api_key(api_key.strip())
        return {
            "status": "success" if success else "error",
            "api_key_set": success
        }
    except Exception as e:
        logger.error(f"Error setting API key: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/deepseek/status")
async def get_deepseek_status():
    """Get Deepseek API status and statistics."""
    try:
        stats = assistant.deepseek.get_statistics()
        health = assistant.deepseek.health_check()
        return {
            "enabled": assistant.deepseek.enabled,
            "api_healthy": health,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting Deepseek status: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/analyze/scene-enhanced")
async def analyze_scene_enhanced():
    """Perform enhanced scene analysis using Deepseek AI."""
    try:
        ready, error = assistant._check_camera_ready()
        if not ready:
            assistant.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return JSONResponse({"status": "error", "result": error}, status_code=400)
        
        with assistant.frame_lock:
            frame = assistant.current_frame.copy()
        
        basic_result = assistant.vision.analyze_scene_comprehensive(frame)
        
        if assistant.deepseek.enabled:
            objects = [obj for obj in basic_result.split(',') if len(obj) > 3][:10]
            enhanced = assistant.deepseek.enhance_scene_description(basic_result, objects)
            result = enhanced if enhanced else basic_result
        else:
            result = basic_result
        
        assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
        logger.info("Enhanced scene analysis completed")
        
        return {
            "result": result,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "enhanced": assistant.deepseek.enabled
        }
    except Exception as e:
        logger.error(f"Error in enhanced scene analysis: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/analyze/hazards-enhanced")
async def analyze_hazards_enhanced():
    """Detect and analyze hazards using Deepseek AI."""
    try:
        ready, error = assistant._check_camera_ready()
        if not ready:
            assistant.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return JSONResponse({"status": "error", "result": error}, status_code=400)
        
        with assistant.frame_lock:
            frame = assistant.current_frame.copy()
        
        hazard_description = assistant.vision.detect_immediate_dangers(frame)
        
        if not hazard_description:
            safe_msg = "Area appears safe. No immediate hazards detected."
            assistant.speech.speak(safe_msg, priority=SpeechPriority.PRIORITY)
            return {
                "result": safe_msg,
                "status": "success",
                "hazard_detected": False
            }
        
        if assistant.deepseek.enabled:
            enhanced = assistant.deepseek.analyze_hazards(hazard_description)
            result = enhanced if enhanced else hazard_description
        else:
            result = hazard_description
        
        warning_msg = f"⚠️ WARNING: {result}"
        assistant.speech.speak_critical(warning_msg, interrupt=True)
        logger.warning(f"Hazard detected and analyzed: {result}")
        
        return {
            "result": result,
            "status": "warning",
            "hazard_detected": True,
            "timestamp": datetime.now().isoformat(),
            "enhanced": assistant.deepseek.enabled
        }
    except Exception as e:
        logger.error(f"Error in enhanced hazard analysis: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/analyze/navigation-enhanced")
async def analyze_navigation_enhanced():
    """Get enhanced navigation guidance using Deepseek AI."""
    try:
        ready, error = assistant._check_camera_ready()
        if not ready:
            assistant.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return JSONResponse({"status": "error", "result": error}, status_code=400)
        
        with assistant.frame_lock:
            frame = assistant.current_frame.copy()
        
        basic_advice = assistant.vision.provide_navigation_advice(frame)
        scene_analysis = assistant.vision.analyze_scene_comprehensive(frame)
        obstacles = assistant.vision.detect_immediate_dangers(frame)
        
        if assistant.deepseek.enabled:
            enhanced = assistant.deepseek.provide_navigation_guidance(
                scene_analysis,
                obstacles if obstacles else []
            )
            result = enhanced if enhanced else basic_advice
        else:
            result = basic_advice
        
        assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
        logger.info("Enhanced navigation guidance provided")
        
        return {
            "result": result,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "enhanced": assistant.deepseek.enabled
        }
    except Exception as e:
        logger.error(f"Error in enhanced navigation: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/analyze/context-enhanced")
async def analyze_context_enhanced():
    """Analyze social context using Deepseek AI."""
    try:
        ready, error = assistant._check_camera_ready()
        if not ready:
            assistant.speech.speak(error, priority=SpeechPriority.PRIORITY)
            return JSONResponse({"status": "error", "result": error}, status_code=400)
        
        with assistant.frame_lock:
            frame = assistant.current_frame.copy()
        
        scene_desc = assistant.vision.analyze_scene_comprehensive(frame)
        
        if assistant.deepseek.enabled:
            context = assistant.deepseek.analyze_context(scene_desc)
            result = context if context else scene_desc
        else:
            result = scene_desc
        
        assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
        logger.info("Context analysis completed")
        
        return {
            "result": result,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "enhanced": assistant.deepseek.enabled
        }
    except Exception as e:
        logger.error(f"Error in context analysis: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ============= WEBSOCKET ROUTES =============

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates."""
    await websocket.accept()
    logger.info(f"🔗 WebSocket client connected")
    
    with assistant.client_lock:
        assistant.connected_clients.add(id(websocket))
    
    try:
        while True:
            # Send status updates periodically
            status = assistant.get_camera_status()
            await websocket.send_json({"type": "status_update", "data": status})
            await asyncio.sleep(assistant.status_update_interval)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        with assistant.client_lock:
            assistant.connected_clients.discard(id(websocket))
        logger.info(f"🔌 WebSocket client disconnected")


@app.websocket("/ws/commands")
async def websocket_commands(websocket: WebSocket):
    """WebSocket endpoint for voice commands."""
    await websocket.accept()
    logger.info(f"🔗 Command WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get('command', '').strip().lower()
            
            if not command:
                await websocket.send_json({"type": "error", "error": "No command provided"})
                continue
            
            logger.info(f"Voice command received: {command}")
            await websocket.send_json({"type": "command_received", "command": command, "status": "processing"})
            
            # Command handlers mapping
            result = None
            
            try:
                if any(kw in command for kw in ['help']):
                    help_text = "Available commands: Analyze scene, Find people, Find doors, Read text, Check hazards, Get navigation guidance, Toggle continuous analysis, Set voice settings."
                    assistant.speech.speak(help_text, priority=SpeechPriority.PRIORITY)
                    result = {'status': 'success', 'result': help_text}
                elif any(kw in command for kw in ['analyze', 'scene']):
                    result = assistant.analyze_scene()
                elif any(kw in command for kw in ['people', 'person']):
                    result = assistant.find_object('person')
                elif any(kw in command for kw in ['door', 'doors']):
                    result = assistant.find_object('door')
                elif any(kw in command for kw in ['text', 'read']):
                    result = assistant.read_text()
                elif any(kw in command for kw in ['hazard', 'danger', 'safe']):
                    result = assistant.detect_hazards()
                elif any(kw in command for kw in ['navigate', 'guidance']):
                    result = assistant.get_navigation_advice()
                elif any(kw in command for kw in ['continuous', 'describe']):
                    enabled = not assistant.continuous_analysis_enabled
                    result = {'status': 'success' if assistant.toggle_continuous_analysis(enabled) else 'error', 
                             'continuous_analysis': assistant.continuous_analysis_enabled}
                else:
                    result = {'status': 'error', 'result': 'Command not recognized. Say "help" for available commands.'}
                
                # Send result back to client
                await websocket.send_json({"type": "command_result", "command": command, "data": result})
                logger.info(f"Command result sent: {result.get('status', 'unknown')}")
            
            except Exception as cmd_error:
                logger.error(f"Error executing command '{command}': {cmd_error}")
                await websocket.send_json({
                    "type": "command_error", 
                    "command": command, 
                    "error": str(cmd_error)
                })
    
    except Exception as e:
        logger.error(f"Command WebSocket error: {e}")
    finally:
        logger.info(f"🔌 Command WebSocket disconnected")


# ============= HEALTH CHECK =============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "camera": assistant.camera_started,
        "timestamp": datetime.now().isoformat()
    }


# ============= ERROR HANDLERS =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        {"status": "error", "message": exc.detail},
        status_code=exc.status_code
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        {"status": "error", "message": "Internal server error"},
        status_code=500
    )


def get_local_ip():
    """Get the local IP address for network connectivity."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# ============= MOBILE HTTP ROUTES =============

@app.get("/mobile")
async def mobile_interface():
    """Serve mobile interface."""
    try:
        mobile_path = Path("templates/mobile.html")
        if mobile_path.exists():
            return FileResponse(mobile_path, media_type="text/html")
        return JSONResponse({"status": "error", "message": "Mobile interface not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving mobile interface: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/voice-test")
async def voice_test():
    """Serve voice recognition diagnostic tool."""
    try:
        voice_path = Path("templates/voice_test.html")
        if voice_path.exists():
            return FileResponse(voice_path, media_type="text/html")
        return JSONResponse({"status": "error", "message": "Voice test not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error serving voice test: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/mobile/authorize")
async def mobile_authorize(device_id: str = Query(...), device_name: str = Query(default="")):
    """Authorize a mobile device for full access."""
    try:
        auth_token = f"{device_id}_{datetime.now().timestamp()}"
        logger.info(f"✅ Mobile device authorized: {device_name or device_id}")
        return {
            "status": "authorized",
            "token": auth_token,
            "device_id": device_id,
            "server": "Beyond Sight",
            "version": "2.0"
        }
    except Exception as e:
        logger.error(f"Authorization error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/mobile/status")
async def mobile_status():
    """Get full system status for mobile."""
    try:
        status = {
            "camera_active": assistant.camera_started,
            "fps": assistant.fps,
            "continuous_analysis": assistant.continuous_analysis_enabled,
            "speech_queue": assistant.speech.queue_size(),
            "is_speaking": assistant.speech.is_speaking(),
            "device": DEVICE.upper(),
            "timestamp": datetime.now().isoformat(),
            "system_healthy": True
        }
        return status
    except Exception as e:
        logger.error(f"Status error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/mobile/start-camera")
async def mobile_start_camera():
    """Start camera from mobile with full authorization."""
    try:
        success = assistant.start_camera()
        if success:
            logger.info("📱 Camera started from mobile device")
            return {
                "status": "success",
                "message": "Camera activated",
                "camera_active": True
            }
        else:
            return JSONResponse(
                {"status": "error", "message": "Failed to start camera"},
                status_code=400
            )
    except Exception as e:
        logger.error(f"Mobile camera start error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/mobile/stop-camera")
async def mobile_stop_camera():
    """Stop camera from mobile."""
    try:
        success = assistant.stop_camera()
        logger.info("📱 Camera stopped from mobile device")
        return {
            "status": "success",
            "message": "Camera deactivated",
            "camera_active": False
        }
    except Exception as e:
        logger.error(f"Mobile camera stop error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Helper to decode mobile frame
def decode_mobile_frame(frame_data: str):
    """Decode base64 frame from mobile device."""
    try:
        import base64
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except:
        return None

@app.post("/api/mobile/analyze")
async def mobile_analyze(body: Dict = Body(None)):
    """Scene analysis from mobile device camera."""
    try:
        if body and 'frame' in body:
            frame = decode_mobile_frame(body['frame'])
            if frame is not None:
                result = assistant.vision.analyze_scene_comprehensive(frame)
                assistant.speech.speak(result, priority=SpeechPriority.PRIORITY)
                return {'result': result, 'status': 'success'}
        
        # Fallback to server camera
        result = assistant.analyze_scene()
        return result
    except Exception as e:
        logger.error(f"Mobile analyze error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/mobile/video-stream")
async def mobile_video_stream():
    """Stream video feed to mobile devices."""
    async def frame_generator():
        while True:
            try:
                frame_bytes = assistant.get_frame_bytes()
                if frame_bytes:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                await asyncio.sleep(0.03)
            except Exception as e:
                logger.error(f"Stream error: {e}")
                break
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/mobile/continuous-toggle")
async def mobile_continuous_toggle(enabled: bool = Query(...)):
    """Toggle continuous analysis from mobile."""
    try:
        result = assistant.toggle_continuous_analysis(enabled)
        return {
            "status": "success" if result else "error",
            "continuous_analysis": assistant.continuous_analysis_enabled,
            "message": "Continuous analysis " + ("enabled" if enabled else "disabled")
        }
    except Exception as e:
        logger.error(f"Toggle continuous error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/mobile/voice-settings")
async def mobile_voice_settings(rate: str = Query(default="NORMAL"), volume: str = Query(default="LOUD")):
    """Update voice settings from mobile."""
    try:
        success = assistant.set_voice_settings(rate, volume)
        return {
            "status": "success" if success else "error",
            "rate": rate,
            "volume": volume
        }
    except Exception as e:
        logger.error(f"Voice settings error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == '__main__':
    try:
        import uvicorn
        
        local_ip = get_local_ip()
        
        logger.info("🚀 Starting Beyond Sight Web Server...")
        logger.info("=" * 80)
        logger.info("📍 NETWORK ADDRESSES:")
        logger.info(f"   🖥️  Desktop:  http://localhost:5000")
        logger.info(f"   📱 Mobile:   http://{local_ip}:5000/mobile")
        logger.info(f"   🌐 Network:  http://{local_ip}:5000")
        logger.info("=" * 80)
        logger.info(f"📊 Application logs saved to: beyond_sight.log")
        logger.info(f"🖥️  Device: {DEVICE.upper()}")
        logger.info(f"📡 CORS: Enabled for all origins (full authorization)")
        logger.info("-" * 80)
        logger.info("✅ Ready for Desktop & Mobile connections!")
        logger.info("=" * 80)
        
        # Run with uvicorn - production-ready ASGI server
        # Host 0.0.0.0 allows connections from any device on the network
        uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info", workers=1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Shutting down...")
        assistant.stop_camera()
        assistant.is_running = False
        logger.info("Goodbye!")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise SystemExit(1)
