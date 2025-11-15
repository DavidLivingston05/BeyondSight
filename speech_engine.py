"""
Speech Engine Module

Provides text-to-speech (TTS) functionality with threading support for
non-blocking audio output. Designed for accessibility applications.
"""

import pyttsx3
import threading
import logging
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty
import time


# ============= Configuration =============
logger = logging.getLogger(__name__)


class VoiceRate(Enum):
    """Voice speed settings"""
    SLOW = 120
    NORMAL = 160
    FAST = 200
    VERY_FAST = 250


class VoiceVolume(Enum):
    """Volume levels"""
    SILENT = 0.0
    LOW = 0.5
    MEDIUM = 0.75
    LOUD = 1.0


@dataclass
class SpeechConfig:
    """Speech engine configuration"""
    rate: int = VoiceRate.NORMAL.value
    volume: float = VoiceVolume.LOUD.value
    engine: str = "sapi5"  # Windows default; use "espeak" on Linux, "nsss" on macOS


# ============= Speech Engine =============
class SpeechEngine:
    """
    Thread-safe text-to-speech engine for accessibility applications.
    
    Features:
    - Non-blocking speech synthesis
    - Queue-based processing for thread safety
    - Configurable voice rate and volume
    - Error handling and recovery
    - Grace shutdown support
    
    Example:
        >>> engine = SpeechEngine()
        >>> engine.speak("Hello world")
        >>> engine.shutdown()
    """
    
    # Maximum queue size to prevent memory bloat
    _MAX_QUEUE_SIZE = 100
    
    def __init__(self, config: Optional[SpeechConfig] = None):
        """
        Initialize speech engine.
        
        Args:
            config: SpeechConfig instance. Defaults to standard config.
            
        Raises:
            RuntimeError: If TTS engine fails to initialize
        """
        self.config = config or SpeechConfig()
        self._speech_queue: Queue[str] = Queue(maxsize=self._MAX_QUEUE_SIZE)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._tts: Optional[pyttsx3.engine.Engine] = None
        
        self._initialize_engine()
        self._start_worker()
        logger.info("🔊 Speech engine initialized")
    
    def _initialize_engine(self) -> None:
        """
        Initialize the TTS engine with configured properties.
        
        Raises:
            RuntimeError: If engine fails to initialize
        """
        try:
            self._tts = pyttsx3.init()
            self._tts.setProperty('rate', self.config.rate)
            self._tts.setProperty('volume', self.config.volume)
            logger.debug(f"TTS engine initialized: rate={self.config.rate}, volume={self.config.volume}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise RuntimeError(f"TTS initialization failed: {str(e)}")
    
    # ============= Speech Control =============
    def speak(self, text: str, priority: bool = False) -> bool:
        """
        Queue text for speech synthesis.
        
        Args:
            text: Text to speak
            priority: If True, prepend to queue (not yet implemented)
            
        Returns:
            True if queued successfully, False if queue is full
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided to speak()")
            return False
        
        text = text.strip()
        if not text:
            return False
        
        try:
            # Log output for debugging/monitoring
            logger.info(f"📢 Queued: {text[:100]}...")
            
            # Try to add to queue without blocking
            self._speech_queue.put(text, block=False)
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue speech: {e}")
            return False
    
    def set_rate(self, rate: VoiceRate) -> None:
        """
        Change voice speed.
        
        Args:
            rate: VoiceRate enum value
        """
        if self._tts:
            self._tts.setProperty('rate', rate.value)
            self.config.rate = rate.value
            logger.info(f"Voice rate changed to {rate.name}")
    
    def set_volume(self, volume: VoiceVolume) -> None:
        """
        Change volume level.
        
        Args:
            volume: VoiceVolume enum value
        """
        if self._tts and 0.0 <= volume.value <= 1.0:
            self._tts.setProperty('volume', volume.value)
            self.config.volume = volume.value
            logger.info(f"Volume changed to {volume.name}")
    
    def set_voice(self, voice_id: int = 0) -> None:
        """
        Set available voice.
        
        Args:
            voice_id: Voice index (0=default, check available voices)
        """
        if not self._tts:
            logger.warning("TTS engine not initialized")
            return
        
        try:
            voices = self._tts.getProperty('voices')
            if voice_id < len(voices):
                self._tts.setProperty('voice', voices[voice_id].id)
                logger.info(f"Voice changed to: {voices[voice_id].name}")
            else:
                logger.warning(f"Voice ID {voice_id} not available")
        except Exception as e:
            logger.error(f"Failed to set voice: {e}")
    
    def is_speaking(self) -> bool:
        """Check if engine is currently speaking."""
        return not self._speech_queue.empty() or (
            self._tts and self._tts.isBusy()
        ) if self._tts else False
    
    # ============= Worker Thread =============
    def _start_worker(self) -> None:
        """Start background speech processing thread."""
        if self._is_running:
            logger.warning("Worker thread already running")
            return
        
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._speech_worker,
            daemon=False,  # Changed to False for graceful shutdown
            name="SpeechWorker"
        )
        self._worker_thread.start()
        logger.debug("Speech worker thread started")
    
    def _speech_worker(self) -> None:
        """
        Background worker that processes speech queue.
        Runs until shutdown() is called.
        """
        while self._is_running:
            try:
                # Wait for text with timeout to allow graceful shutdown
                text = self._speech_queue.get(timeout=0.5)
                
                if text is None:  # Sentinel value for shutdown
                    break
                
                self._synthesize_speech(text)
                self._speech_queue.task_done()
                
            except Empty:
                # Timeout - just continue waiting
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
                continue
    
    def _synthesize_speech(self, text: str) -> None:
        """
        Synthesize and play audio.
        
        Args:
            text: Text to convert to speech
        """
        if not self._tts:
            logger.error("TTS engine not available")
            return
        
        try:
            self._tts.say(text)
            self._tts.runAndWait()
            logger.debug(f"✓ Speech output complete")
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
    
    # ============= Lifecycle =============
    def shutdown(self, timeout: float = 5.0) -> bool:
        """
        Gracefully shutdown speech engine.
        
        Waits for queued speech to complete before exiting.
        
        Args:
            timeout: Maximum seconds to wait for queue drain
            
        Returns:
            True if shutdown successful, False if timeout
        """
        logger.info("🔇 Shutting down speech engine...")
        
        if not self._is_running:
            logger.warning("Engine not running")
            return True
        
        try:
            # Wait for queue to empty
            self._speech_queue.join()
            
            # Signal worker to stop
            self._is_running = False
            self._speech_queue.put(None)  # Sentinel value
            
            # Wait for worker thread
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                
                if self._worker_thread.is_alive():
                    logger.warning("Worker thread did not shut down gracefully")
                    return False
            
            logger.info("✓ Speech engine shut down")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.shutdown()
    
    def __repr__(self) -> str:
        return (
            f"SpeechEngine(rate={self.config.rate}, "
            f"volume={self.config.volume})"
        )


# ============= Helper Functions =============
def get_available_voices() -> list:
    """
    List available system voices.
    
    Returns:
        List of voice names available on system
    """
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return [voice.name for voice in voices]
    except Exception as e:
        logger.error(f"Failed to get available voices: {e}")
        return []


# ============= Example Usage =============
if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example 1: Basic usage
    print("\n=== Example 1: Basic Usage ===")
    engine = SpeechEngine()
    engine.speak("Hello, this is a test of the speech engine.")
    time.sleep(3)
    engine.shutdown()
    
    # Example 2: With context manager
    print("\n=== Example 2: Context Manager ===")
    with SpeechEngine() as engine:
        engine.speak("Speaking with context manager.")
        engine.set_rate(VoiceRate.SLOW)
        engine.speak("Now speaking slower.")
        time.sleep(5)
    
    # Example 3: List available voices
    print("\n=== Example 3: Available Voices ===")
    voices = get_available_voices()
    for i, voice in enumerate(voices):
        print(f"  {i}: {voice}")
