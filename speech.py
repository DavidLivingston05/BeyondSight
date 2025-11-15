"""
Professional Speech Engine - Text-to-Speech with Priority Queue
Supports both pyttsx3 and Edge-TTS with priority-based queuing
"""

import threading
import logging
from enum import Enum
from typing import Optional
from queue import Queue, Empty
import time
import pyttsx3

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============= Enums =============

class VoiceRate(Enum):
    """Speech rate options"""
    SLOW = 0.5
    NORMAL = 1.0
    FAST = 1.5
    VERY_FAST = 2.0


class VoiceVolume(Enum):
    """Volume levels"""
    QUIET = 0.5
    NORMAL = 1.0
    LOUD = 1.5


class SpeechPriority(Enum):
    """Speech priority levels"""
    BACKGROUND = 0
    NORMAL = 1
    PRIORITY = 2


# ============= Speech Engine =============

class SpeechEnginePro:
    """Professional speech engine with queuing and priority support."""
    
    def __init__(self):
        """Initialize speech engine."""
        self.use_edge_tts = EDGE_TTS_AVAILABLE
        
        # Initialize pyttsx3 as fallback
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}")
            self.engine = None
        
        # Speech queue
        self.speech_queue = Queue(maxsize=50)
        self.speaking = False
        self.current_priority = SpeechPriority.NORMAL
        
        # Worker thread
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._speech_worker,
            daemon=False,
            name="SpeechWorker"
        )
        self.worker_thread.start()
        
        # Voice settings
        self.rate = VoiceRate.NORMAL
        self.volume = VoiceVolume.LOUD
        
        if self.use_edge_tts:
            logger.info("✅ Edge-TTS enabled")
        else:
            logger.info("ℹ️  Using pyttsx3")
    
    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.NORMAL) -> None:
        """Queue text for speech output."""
        if not text or len(text.strip()) == 0:
            return
        
        try:
            self.speech_queue.put_nowait((text, priority, time.time()))
        except Exception as e:
            logger.warning(f"Queue full: {e}")
    
    def speak_critical(self, text: str, interrupt: bool = True) -> None:
        """Speak critical message immediately."""
        if interrupt:
            # Clear lower priority items
            temp_queue = []
            try:
                while True:
                    item = self.speech_queue.get_nowait()
                    if item[1].value >= SpeechPriority.PRIORITY.value:
                        temp_queue.append(item)
            except Empty:
                pass
            
            # Restore higher priority items
            for item in temp_queue:
                try:
                    self.speech_queue.put_nowait(item)
                except:
                    pass
        
        # Queue critical message
        self.speak(text, SpeechPriority.PRIORITY)
    
    def _speech_worker(self) -> None:
        """Background worker processing speech queue."""
        while self.is_running:
            try:
                text, priority, timestamp = self.speech_queue.get(timeout=1)
                
                if text is None:  # Shutdown signal
                    break
                
                self.speaking = True
                self.current_priority = priority
                
                try:
                    if self.use_edge_tts:
                        self._speak_with_edge_tts(text)
                    else:
                        self._speak_with_pyttsx3(text)
                except Exception as e:
                    logger.error(f"Speech error: {e}")
                finally:
                    self.speaking = False
                
                self.speech_queue.task_done()
            
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _speak_with_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3."""
        if not self.engine:
            logger.error("pyttsx3 not available")
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            logger.debug(f"✓ Speech complete: {text[:50]}")
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
    
    def _speak_with_edge_tts(self, text: str) -> None:
        """Speak using Edge-TTS."""
        try:
            import asyncio
            
            async def async_speak():
                try:
                    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
                    await communicate.save("temp_speech.mp3")
                    
                    # Play the audio
                    import os
                    os.system("start temp_speech.mp3")
                    time.sleep(len(text.split()) * 0.3)  # Rough estimate
                except Exception as e:
                    logger.error(f"Edge-TTS error: {e}")
                    # Fallback to pyttsx3
                    self._speak_with_pyttsx3(text)
            
            try:
                asyncio.run(async_speak())
            except RuntimeError:
                # Already in event loop
                self._speak_with_pyttsx3(text)
        
        except Exception as e:
            logger.error(f"Edge-TTS error: {e}")
            self._speak_with_pyttsx3(text)
    
    def set_rate(self, rate: VoiceRate) -> None:
        """Set speech rate."""
        if self.engine:
            self.engine.setProperty('rate', int(rate.value * 150))
        self.rate = rate
        logger.info(f"Rate set to {rate.name}")
    
    def set_volume(self, volume: VoiceVolume) -> None:
        """Set volume level."""
        if self.engine and 0.0 <= volume.value <= 1.5:
            self.engine.setProperty('volume', min(1.0, volume.value))
        self.volume = volume
        logger.info(f"Volume set to {volume.name}")
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.speaking or not self.speech_queue.empty()
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return self.speech_queue.qsize()
    
    def shutdown(self, timeout: float = 5.0) -> bool:
        """Gracefully shutdown speech engine."""
        logger.info("🔇 Shutting down speech engine...")
        
        try:
            # Wait for queue to empty
            self.speech_queue.join()
            
            # Signal worker to stop
            self.is_running = False
            self.speech_queue.put(None)
            
            # Wait for worker thread
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=timeout)
            
            if self.engine:
                self.engine.stop()
            
            logger.info("✓ Speech engine shut down")
            return True
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
