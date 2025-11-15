"""
DeepSeek AI Integration for Enhanced Vision Analysis
Uses DeepSeek's advanced LLM for context-aware scene analysis
"""

import requests
import logging
import os
from typing import Optional, List, Dict
from enum import Enum

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis that can use DeepSeek enhancement"""
    SCENE_DESCRIPTION = "scene"
    HAZARD_ANALYSIS = "hazard"
    NAVIGATION = "navigation"
    CONTEXT = "context"

class DeepSeekClient:
    """Client for DeepSeek API integration"""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.enabled = bool(self.api_key)
        self.model = "deepseek-chat"
        self.request_count = 0
        self.error_count = 0
        
        if self.enabled:
            logger.info("✅ DeepSeek API enabled")
        else:
            logger.info("ℹ️  DeepSeek API not configured (set DEEPSEEK_API_KEY)")
    
    def health_check(self) -> bool:
         """Test API connectivity with timeout and validation"""
         if not self.enabled or not self.api_key or len(self.api_key.strip()) < 10:
             logger.warning("API health check failed: invalid API key")
             return False
         try:
             response = requests.post(
                 self.api_url,
                 headers={
                     "Authorization": f"Bearer {self.api_key.strip()}",
                     "Content-Type": "application/json"
                 },
                 json={"model": self.model, "messages": [{"role": "user", "content": "ping"}]},
                 timeout=5
             )
             return response.status_code == 200
         except requests.exceptions.Timeout:
             logger.error("API health check timeout")
             return False
         except requests.exceptions.RequestException as e:
             logger.error(f"API health check failed: {e}")
             return False
    
    def enhance_scene_description(self, basic_description: str, detected_objects: List[str]) -> Optional[str]:
        """Enhance basic scene analysis with DeepSeek context awareness"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""You are a vision assistant for visually impaired users. 
            
The camera detected: {', '.join(detected_objects[:10])}
Basic description: {basic_description}

Provide a natural, helpful enhancement that:
1. Highlights the most important elements for navigation
2. Suggests caution areas
3. Is concise (2-3 sentences max)
            
Enhanced description:"""
            
            response = self._call_api(prompt)
            self.request_count += 1
            return response
        except Exception as e:
            logger.error(f"Error enhancing scene: {e}")
            self.error_count += 1
            return None
    
    def analyze_hazards(self, hazard_description: str) -> Optional[str]:
        """Analyze detected hazards for severity and recommendations"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""You are a safety assistant. A hazard was detected:
{hazard_description}

Provide:
1. Brief severity assessment
2. Immediate safety recommendation
(Max 2 sentences)

Response:"""
            
            response = self._call_api(prompt)
            self.request_count += 1
            return response
        except Exception as e:
            logger.error(f"Error analyzing hazards: {e}")
            self.error_count += 1
            return None
    
    def provide_navigation_guidance(self, scene_analysis: str, obstacles: List[str]) -> Optional[str]:
        """Generate context-aware navigation guidance"""
        if not self.enabled:
            return None
        
        try:
            obstacles_str = ', '.join(obstacles) if obstacles else "none detected"
            prompt = f"""You are a navigation assistant for visually impaired users.

Scene: {scene_analysis}
Obstacles: {obstacles_str}

Provide clear, actionable navigation advice (2-3 sentences):

Guidance:"""
            
            response = self._call_api(prompt)
            self.request_count += 1
            return response
        except Exception as e:
            logger.error(f"Error providing navigation: {e}")
            self.error_count += 1
            return None
    
    def analyze_context(self, scene_description: str) -> Optional[str]:
        """Analyze social and environmental context"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""You are a context analyzer. Given this scene:
{scene_description}

Analyze:
1. Social context (if people present)
2. Environmental conditions
3. Time-of-day hints

Response (2-3 sentences):

Analysis:"""
            
            response = self._call_api(prompt)
            self.request_count += 1
            return response
        except Exception as e:
            logger.error(f"Error analyzing context: {e}")
            self.error_count += 1
            return None
    
    def _call_api(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
         """Call DeepSeek API with validation and error handling.
         
         Args:
             prompt: The prompt to send to the API
             max_tokens: Maximum tokens in response (50-2000)
             
         Returns:
             API response text or None if failed
         """
         if not self.enabled or not self.api_key:
             return None
         
         # Validate API key
         if len(self.api_key.strip()) < 10:
             logger.error("Invalid API key configuration")
             self.error_count += 1
             return None
         
         # Validate and sanitize inputs
         if not prompt or len(prompt.strip()) == 0:
             logger.error("Empty prompt provided")
             return None
         
         if len(prompt) > 4000:
             logger.warning("Prompt exceeds 4000 chars, truncating")
             prompt = prompt[:4000]
         
         # Clamp max_tokens between 50-2000
         max_tokens = max(50, min(max_tokens, 2000))
         
         try:
             response = requests.post(
                 self.api_url,
                 headers={
                     "Authorization": f"Bearer {self.api_key.strip()}",
                     "Content-Type": "application/json"
                 },
                 json={
                     "model": self.model,
                     "messages": [{"role": "user", "content": prompt.strip()}],
                     "max_tokens": max_tokens,
                     "temperature": 0.7,
                     "top_p": 0.9
                 },
                 timeout=10
             )
             
             if response.status_code == 200:
                 data = response.json()
                 if 'choices' in data and len(data['choices']) > 0:
                     return data['choices'][0]['message']['content'].strip()
             else:
                 logger.warning(f"API returned {response.status_code}: {response.text[:100]}")
                 self.error_count += 1
         except requests.exceptions.Timeout:
             logger.error("API call timeout after 10 seconds")
             self.error_count += 1
         except requests.exceptions.RequestException as e:
             logger.error(f"API call failed: {e}")
             self.error_count += 1
         except (KeyError, ValueError) as e:
             logger.error(f"API response parsing error: {e}")
             self.error_count += 1
         
         return None
    
    def get_statistics(self) -> Dict:
        """Get API usage statistics"""
        return {
            'enabled': self.enabled,
            'requests': self.request_count,
            'errors': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(1, self.request_count)
        }

# Global client instance
_deepseek_client: Optional[DeepSeekClient] = None

def get_deepseek_client() -> DeepSeekClient:
    """Get or create DeepSeek client"""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient()
    return _deepseek_client

def set_api_key(api_key: str) -> bool:
    """Set DeepSeek API key at runtime"""
    try:
        client = get_deepseek_client()
        client.api_key = api_key.strip()
        client.enabled = bool(client.api_key)
        
        if client.enabled:
            logger.info("✅ DeepSeek API key updated")
            return client.health_check()
        else:
            logger.warning("API key cleared")
            return False
    except Exception as e:
        logger.error(f"Error setting API key: {e}")
        return False
