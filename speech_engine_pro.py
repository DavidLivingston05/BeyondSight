"""
Professional Speech Engine with Edge-TTS Support
Faster, higher-quality text-to-speech with multiple voice options
"""

import asyncio
import threading
import logging
from enum import Enum
from typing import Optional
from queue import Queue, Empty
import time

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.warning("edge-tts not installed, falling back to pyttsx3")

import pyttsx3

logger = logging.getLogger(__name__)

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

class SpeechEnginePro:
    """Professional speech engine with Edge-TTS and fallback"""
    
    def __init__(self):
        self.use_edge_tts = EDGE_TTS_AVAILABLE
        
        # Always init pyttsx3 as fallback
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}")
            self.engine = None
        
        # Speech queue management
        self.speech_queue = Queue(maxsize=20)
        self.speaking = False
        self.current_priority = SpeechPriority.NORMAL
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
        
        # Voice settings
        self.rate = VoiceRate.NORMAL
        self.volume = VoiceVolume.LOUD
        
        if self.engine:
            try:
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)
            except Exception as e:
                logger.warning(f"pyttsx3 property setting failed: {e}")
        
        if self.use_edge_tts:
            logger.info("✅ Edge-TTS enabled (Microsoft Natural Voices)")
        else:
            logger.info("ℹ️  Using pyttsx3 (install edge-tts for better quality)")
    
    def speak(self, text: str, priority: SpeechPriority = SpeechPriority.NORMAL) -> None:
        """Queue text for speech output"""
        if not text or len(text.strip()) == 0:
            return
        
        try:
            self.speech_queue.put_nowait((text, priority, time.time()))
        except:
            logger.warning("Speech queue full, dropping message")
    
    def speak_critical(self, text: str, interrupt: bool = True) -> None:
        """Speak critical message immediately"""
        # Clear lower priority items if interrupt is True
        if interrupt:
            temp_queue = []
            try:
                while True:
                    item = self.speech_queue.get_nowait()
                    if item[1].value >= SpeechPriority.PRIORITY.value:
                        temp_queue.append(item)
            except Empty:
                pass
            
            for item in temp_queue:
                self.speech_queue.put(item)
        
        self.speak(text, priority=SpeechPriority.PRIORITY)
    
    def set_rate(self, rate: VoiceRate) -> None:
        """Set speech rate"""
        self.rate = rate
        if self.engine and not self.use_edge_tts:
            self.engine.setProperty('rate', int(150 * rate.value))
    
    def set_volume(self, volume: VoiceVolume) -> None:
        """Set volume level"""
        self.volume = volume
        if self.engine and not self.use_edge_tts:
            self.engine.setProperty('volume', volume.value)
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speaking
    
    def queue_size(self) -> int:
        """Get number of items in speech queue"""
        return self.speech_queue.qsize()
    
    def _speech_worker(self) -> None:
        """Background worker that processes speech queue"""
        while True:
            try:
                text, priority, timestamp = self.speech_queue.get(timeout=1)
                
                # Skip if message is too old (>30 seconds)
                if time.time() - timestamp > 30:
                    logger.debug("Skipping old queued message")
                    continue
                
                self.speaking = True
                self.current_priority = priority
                
                if self.use_edge_tts:
                    self._speak_edge_tts(text)
                else:
                    self._speak_pyttsx3(text)
                
                self.speaking = False
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in speech worker: {e}")
                self.speaking = False
    
    def _speak_edge_tts(self, text: str) -> None:
        """Speak using Edge TTS (much faster & better quality)"""
        try:
            # Run async function in thread-safe way
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_edge_tts(text))
            loop.close()
        except Exception as e:
            logger.error(f"Edge TTS error: {e}, falling back to pyttsx3")
            self._speak_pyttsx3(text)
    
    async def _async_edge_tts(self, text: str) -> None:
        """Async Edge TTS implementation"""
        try:
            # Use a professional voice
            voice = "en-US-AriaNeural"  # Clear, professional voice
            
            communicate = edge_tts.Communicate(text, voice, rate=self._get_rate_str())
            await communicate.save("temp_speech.mp3")
            
            # Play audio file
            try:
                import subprocess
                import os
                # Use winsound for direct audio playback on Windows
                if os.name == 'nt':
                    import winsound
                    # Use SND_FILENAME | SND_NODEFAULT (blocking by default is correct)
                    winsound.PlaySound('temp_speech.mp3', winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                else:
                    # Fallback for non-Windows
                    subprocess.run(["ffplay", "-nodisp", "-autoexit", "temp_speech.mp3"], 
                                 capture_output=True, check=False, timeout=30)
            except Exception as play_error:
                logger.debug(f"Audio playback skipped (may be running headless): {play_error}")
        except Exception as e:
            logger.error(f"Async Edge TTS failed: {e}")
    
    def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 (fallback)"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
    
    def _get_rate_str(self) -> str:
        """Convert rate to Edge TTS format"""
        rates = {
            VoiceRate.SLOW: "-50%",
            VoiceRate.NORMAL: "+0%",
            VoiceRate.FAST: "+50%",
            VoiceRate.VERY_FAST: "+100%"
        }
        return rates.get(self.rate, "+0%")
    
    def shutdown(self) -> None:
        """Clean shutdown"""
        try:
            if self.engine:
                self.engine.endLoop()
        except:
            pass
