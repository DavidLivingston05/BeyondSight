"""
Configuration Module - Centralized settings for Beyond Sight application
Manages all constants, thresholds, and configuration parameters.
"""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# ============= Application Paths =============
APP_ROOT = Path(__file__).parent
LOG_DIR = APP_ROOT / "logs"
MODEL_DIR = APP_ROOT / "models"
DATA_DIR = APP_ROOT / "data"

# Create directories if they don't exist
for directory in [LOG_DIR, MODEL_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)


# ============= Vision Configuration =============
@dataclass
class VisionConfig:
    """Vision analysis settings"""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    frame_width: int = 1280
    frame_height: int = 720
    frame_rate: int = 30


# ============= Camera Configuration =============
@dataclass
class CameraConfig:
    """Camera capture settings"""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    frame_rate: int = 30
    buffer_size: int = 1
    auto_exposure: int = 1


# ============= Speech Configuration =============
@dataclass
class SpeechConfig:
    """Speech engine settings"""
    rate: int = 140  # Words per minute
    volume: float = 1.0
    engine: str = "sapi5"  # Windows, espeak (Linux), nsss (macOS)
    worker_threads: int = 1
    max_queue_size: int = 200


# ============= Server Configuration =============
@dataclass
class ServerConfig:
    """Flask/WebSocket server settings"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "beyond-sight-secret-key-2025"
    cors_enabled: bool = True
    websocket_timeout: int = 30


# ============= Analysis Configuration =============
@dataclass
class AnalysisConfig:
    """Continuous analysis settings"""
    interval: int = 5  # Seconds between analyses
    timeout: int = 30  # Analysis timeout in seconds
    cache_enabled: bool = True
    cache_ttl: int = 60  # Cache time-to-live in seconds


# ============= Feature Flags =============
@dataclass
class FeatureFlags:
    """Enable/disable features"""
    face_recognition: bool = True
    hand_tracking: bool = True
    gps_navigation: bool = True
    continuous_analysis: bool = True
    audio_cues: bool = True
    emergency_alerts: bool = True
    color_detection: bool = True
    water_hazard_detection: bool = True
    stair_detection: bool = True
    gesture_recognition: bool = True
    deepseek_ai_enhancement: bool = True


# ============= Deepseek Configuration =============
@dataclass
class DeepseekConfig:
    """Deepseek AI API settings"""
    enabled: bool = True
    api_key: str = ""  # Set via environment variable or API
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: int = 30
    enable_scene_enhancement: bool = True
    enable_hazard_analysis: bool = True
    enable_navigation_guidance: bool = True
    enable_context_awareness: bool = True


# ============= Global Configuration Instance =============
class Config:
    """Master configuration class"""
    
    # Create directories
    LOG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Core configurations
    VISION = VisionConfig()
    CAMERA = CameraConfig()
    SPEECH = SpeechConfig()
    SERVER = ServerConfig()
    ANALYSIS = AnalysisConfig()
    FEATURES = FeatureFlags()
    DEEPSEEK = DeepseekConfig()
    
    # Logging
    LOG_FILE = LOG_DIR / "beyond_sight.log"
    LOG_LEVEL = "INFO"
    
    # Paths
    YOLO_MODEL = MODEL_DIR / "yolov8n.pt"
    FACE_ENCODINGS_DB = DATA_DIR / "face_encodings.pkl"
    FACE_NAMES_DB = DATA_DIR / "face_names.json"
    
    # Tesseract OCR path (configurable via env var, defaults to common paths)
    TESSERACT_PATH = os.getenv('TESSERACT_CMD', '')
    
    # Default voice settings
    VOICE_RATE_DEFAULT = 140
    VOICE_VOLUME_DEFAULT = 1.0
    
    # Frame streaming constants
    FRAME_STREAM_INTERVAL = 0.033  # ~30 FPS
    FRAME_STREAM_MAX_ERRORS = 3
    FRAME_STREAM_ERROR_BACKOFF = 0.5
    
    # Analysis timeout constants
    ANALYSIS_TIMEOUT_SECONDS = 30
    ANALYSIS_CACHE_TTL = 300  # 5 minutes
    
    # Input validation constraints
    OBJECT_NAME_MAX_LENGTH = 50
    PROMPT_MAX_LENGTH = 4000
    DEEPSEEK_MAX_TOKENS = 2000
    DEEPSEEK_MIN_TOKENS = 50
    
    # CORS and networking
    CORS_ALLOW_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5000,http://localhost:3000').split(',')
    CORS_DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Distance estimation
    REFERENCE_DISTANCES = {
        'person': 180,
        'chair': 120,
        'car': 400,
        'cup': 60,
        'bottle': 80,
        'book': 100,
        'cell phone': 80,
        'laptop': 150,
        'keyboard': 120,
        'mouse': 60,
        'tv': 300,
        'clock': 100
    }
    
    # Safety thresholds
    DANGER_DISTANCE_THRESHOLD = 2.0  # meters
    CROWDING_THRESHOLD = 5  # people
    MIN_BRIGHTNESS_WARNING = 30  # brightness units
    MAX_BRIGHTNESS_WARNING = 200  # brightness units
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Get all configuration as dictionary"""
        return {
            'vision': cls.VISION.__dict__,
            'camera': cls.CAMERA.__dict__,
            'speech': cls.SPEECH.__dict__,
            'server': cls.SERVER.__dict__,
            'analysis': cls.ANALYSIS.__dict__,
            'features': cls.FEATURES.__dict__,
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        try:
            # Check required files
            if not cls.TESSERACT_PATH and os.name == 'nt':
                import logging
                logging.warning(f"Tesseract not found at {cls.TESSERACT_PATH}")
            
            # Validate ranges
            assert 0 < cls.SPEECH.rate < 500, "Invalid speech rate"
            assert 0.0 <= cls.SPEECH.volume <= 1.0, "Invalid speech volume"
            assert cls.CAMERA.frame_width > 0, "Invalid frame width"
            assert cls.CAMERA.frame_height > 0, "Invalid frame height"
            
            return True
        except AssertionError as e:
            import logging
            logging.error(f"Configuration validation failed: {e}")
            return False


# Initialize and validate
if __name__ == "__main__":
    Config.validate()
    print("✅ Configuration loaded successfully")
    print(Config.get_config_dict())
