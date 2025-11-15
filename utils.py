"""
Utility Functions and Helpers
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============= Path Utilities =============

def get_app_root() -> Path:
    """Get application root directory."""
    return Path(__file__).parent


def get_log_dir() -> Path:
    """Get logs directory."""
    log_dir = get_app_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_data_dir() -> Path:
    """Get data directory."""
    data_dir = get_app_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """Get models directory."""
    models_dir = get_app_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def ensure_directories() -> bool:
    """Ensure all required directories exist."""
    try:
        get_log_dir()
        get_data_dir()
        get_models_dir()
        logger.info("✅ All required directories ensured")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure directories: {e}")
        return False


# ============= Configuration Utilities =============

def get_env_or_default(key: str, default: str) -> str:
    """Get environment variable or default."""
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float."""
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


# ============= Logging Utilities =============

def setup_logging(name: str = "beyond_sight", level: int = logging.INFO) -> logging.Logger:
    """Setup application logging."""
    log_dir = get_log_dir()
    log_file = log_dir / f"{name}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# ============= File Utilities =============

def ensure_file_exists(file_path: Path, create_if_missing: bool = True) -> bool:
    """Check if file exists, create if needed."""
    if file_path.exists():
        return True
    
    if create_if_missing:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            logger.info(f"Created file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create file: {e}")
            return False
    
    return False


# ============= Network Utilities =============

def get_local_ip() -> str:
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Could not get IP: {e}")
        return "127.0.0.1"


# ============= Device Utilities =============

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Get compute device."""
    return "cuda" if is_cuda_available() else "cpu"


# ============= Format Utilities =============

def format_distance(distance_km: float) -> str:
    """Format distance for display."""
    if distance_km < 0.1:
        return f"{distance_km * 1000:.0f}m"
    elif distance_km < 1:
        return f"{distance_km * 1000:.0f}m"
    else:
        return f"{distance_km:.1f}km"


def format_time_elapsed(seconds: int) -> str:
    """Format elapsed time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


# ============= Validation Utilities =============

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate GPS coordinates."""
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def validate_image_path(path: str) -> bool:
    """Validate image file path."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    return Path(path).suffix.lower() in valid_extensions
