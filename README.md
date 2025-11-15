# 🌍 Beyond Sight - AI-Powered Vision Assistant

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()

**Beyond Sight** is a professional AI-powered web application that provides real-time audio descriptions of the user's surroundings using advanced computer vision and artificial intelligence. It acts as a personal AI assistant specifically designed for accessibility, helping visually impaired users navigate and interact with their environment.

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage & Operations](#-usage--operations)
- [API Reference](#-api-reference)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### Core Vision Analysis
- 🎥 **Real-time Scene Analysis**: AI-powered scene understanding with object detection and spatial relationships
- 📍 **Object Detection**: Identify and locate specific objects with distance estimation
- 🔤 **Text Recognition (OCR)**: Read text from signs, documents, and displays
- ⚠️ **Hazard Detection**: Identify obstacles, stairs, water hazards, and dangerous conditions
- 🎯 **Distance Estimation**: Calculate approximate distances to detected objects

### Navigation & Mobility
- 🗺️ **GPS Navigation**: Real-time location tracking and navigation guidance
- 👣 **Safe Path Guidance**: Obstacle-aware route recommendations
- 🚦 **Intersection Detection**: Identify crosswalks and intersections
- 📌 **Place Memory**: Remember locations and landmarks

### Accessibility Features
- 🔊 **Text-to-Speech**: Natural audio descriptions using Edge-TTS
- 🎤 **Voice Commands**: Control the application with natural voice commands
- 🔔 **Priority Alerts**: Critical warnings interrupt normal speech
- 🎨 **Color Detection**: Identify and describe colors in the scene
- 👤 **Face Recognition**: Recognize and remember familiar people

### User Interface
- 💻 **Web Dashboard**: Desktop interface for configuration and monitoring
- 📱 **Mobile Interface**: Optimized mobile application for on-the-go use
- 🔴 **Live Video Feed**: Real-time camera stream with visual feedback
- 📊 **System Status**: Monitor performance metrics and system health
- ⚙️ **Voice Settings**: Customize speech rate and volume

### Advanced Features
- 🚀 **GPU Acceleration**: CUDA support for faster processing
- 🤖 **Deepseek AI Enhancement**: Optional AI enhancement for richer descriptions
- 🔄 **Continuous Analysis**: Periodic automatic scene descriptions
- 📡 **WebSocket Support**: Real-time bidirectional communication
- 🧠 **Hand Gesture Recognition**: Detect and interpret hand gestures

---

## 💻 System Requirements

### Hardware
- **CPU**: Intel Core i5 / AMD Ryzen 5 or better
- **RAM**: Minimum 8GB (16GB recommended for GPU)
- **GPU** (Optional): NVIDIA CUDA-capable GPU with 4GB+ VRAM for acceleration
- **Camera**: USB webcam or built-in camera (1080p+ recommended)
- **Microphone**: For voice command input
- **Speakers**: For audio output

### Software
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.9 or higher
- **Git**: For cloning and version control

### Optional Dependencies
- **NVIDIA CUDA Toolkit**: For GPU acceleration
- **Tesseract OCR**: For enhanced text recognition
- **Deepseek API Key**: For AI-powered descriptions

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DavidLivingston05/BeyondSight.git
cd BeyondSight
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install CUDA Support

For GPU acceleration, install NVIDIA CUDA Toolkit:
```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Then run the installer and follow instructions

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. (Optional) Install Tesseract OCR

**Windows:**
- Download installer: https://github.com/UB-Mannheim/tesseract/wiki
- Run installer (default: `C:\Program Files\Tesseract-OCR`)

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 6. Verify Installation

```bash
python verify_setup.py
```

---

## 🚀 Quick Start

### 1. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys (optional):
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_ENABLED=true
FLASK_ENV=production
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
```

### 2. Start the Server

```bash
python main.py
```

You should see output like:
```
🚀 Starting Beyond Sight Web Server...
================================================================================
📍 NETWORK ADDRESSES:
   🖥️  Desktop:  http://localhost:5000
   📱 Mobile:   http://192.168.x.x:5000/mobile
   🌐 Network:  http://192.168.x.x:5000
================================================================================
✅ Ready for Desktop & Mobile connections!
```

### 3. Access the Application

**Desktop:**
- Open browser: `http://localhost:5000`

**Mobile:**
- On same network: `http://YOUR_COMPUTER_IP:5000/mobile`
- Replace `YOUR_COMPUTER_IP` with your computer's IP address

### 4. Enable Camera

1. Click "Start Camera" button
2. Grant camera permissions when prompted
3. Wait for camera to initialize
4. Ready for analysis!

---

## ⚙️ Configuration

### Main Configuration File: `config.py`

Edit `config.py` to customize:

```python
# Camera settings
FRAME_WIDTH = 1280      # Resolution width
FRAME_HEIGHT = 720      # Resolution height
FRAME_RATE = 30         # Frames per second

# Speech settings
SPEECH_RATE = 140       # Words per minute (100-300)
SPEECH_VOLUME = 1.0     # Volume (0.0-1.0)

# Analysis settings
CONTINUOUS_ANALYSIS_INTERVAL = 5  # Seconds between analyses
ANALYSIS_TIMEOUT = 30              # Max analysis time

# Feature flags
FACE_RECOGNITION = True
HAND_TRACKING = True
CONTINUOUS_ANALYSIS = True
DEEPSEEK_AI_ENHANCEMENT = True
```

### Environment Variables

```bash
# Deepseek AI (optional)
DEEPSEEK_API_KEY=sk_xxx_yyy_zzz
DEEPSEEK_ENABLED=true

# Server
FLASK_ENV=production
DEBUG_MODE=false

# Camera
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# Speech
SPEECH_RATE=140
SPEECH_VOLUME=1.0
```

---

## 🎮 Usage & Operations

### Desktop Interface

#### Main Dashboard
1. **Camera Controls**
   - Click "Start Camera" to begin
   - Click "Stop Camera" to disable
   - Monitor FPS and frame count

2. **Analysis Tools**
   - **Analyze Scene**: Get comprehensive description of surroundings
   - **Find Object**: Search for specific items ("Find door", "Find person")
   - **Read Text**: Extract and read text visible on screen
   - **Check Hazards**: Identify dangerous obstacles and conditions
   - **Navigation**: Get safe movement guidance

3. **Settings**
   - Adjust speech rate (Slow/Normal/Fast)
   - Control volume (Quiet/Normal/Loud)
   - Toggle continuous analysis
   - Enable/disable features

#### Continuous Analysis
- Automatically describes surroundings at intervals
- Use "Toggle Continuous" to enable/disable
- Useful for awareness during mobility

### Mobile Interface

Access via: `http://YOUR_IP:5000/mobile`

Features:
- Full camera control
- Scene analysis
- Voice commands
- System status
- Video streaming
- Settings control

### Voice Commands

Activate voice input and say:

**Navigation:**
- "Analyze scene" → Describe current surroundings
- "Check hazards" → Identify dangers
- "Navigation guidance" → Get movement advice
- "Find [object]" → Locate specific items

**Text:**
- "Read text" → Extract visible text

**People:**
- "Find people" → Detect people in scene
- "Find doors" → Locate doorways

**System:**
- "Toggle continuous" → Enable/disable auto-descriptions
- "Set volume [level]" → Adjust audio volume
- "Help" → List available commands

### Video Stream

View live camera feed:
```
http://localhost:5000/api/video-stream
```

Use in HTML:
```html
<img src="http://localhost:5000/api/video-stream" />
```

---

## 📡 API Reference

### REST Endpoints

#### Camera Control

**Start Camera**
```http
POST /api/camera/start
Response: { status: "success", message: "Camera started" }
```

**Stop Camera**
```http
POST /api/camera/stop
Response: { status: "success", message: "Camera stopped" }
```

**Get Status**
```http
GET /api/camera/status
Response: { 
  camera_active: true,
  fps: 30,
  has_frame: true,
  continuous_analysis: false
}
```

#### Analysis Endpoints

**Scene Analysis**
```http
POST /api/analyze/scene
Response: {
  result: "Description of scene",
  status: "success",
  timestamp: "2025-11-16T10:30:00Z"
}
```

**Find Object**
```http
POST /api/analyze/find-object?object=door
Response: {
  result: "Door is 2 meters ahead on the right",
  status: "success"
}
```

**Read Text**
```http
POST /api/analyze/read-text
Response: {
  result: "Text from image",
  status: "success"
}
```

**Hazard Detection**
```http
POST /api/analyze/hazards
Response: {
  result: "Obstacle detected",
  status: "warning",
  hazard_detected: true
}
```

**Navigation**
```http
POST /api/analyze/navigation
Response: {
  result: "Safe path guidance",
  status: "success"
}
```

#### Video Streaming

**MJPEG Stream**
```http
GET /api/video-stream
Content-Type: multipart/x-mixed-replace; boundary=frame
```

**Single Frame**
```http
GET /api/frame
Content-Type: image/jpeg
```

#### Mobile Endpoints

**Authorize Device**
```http
POST /api/mobile/authorize?device_id=xxx&device_name=iPhone
Response: {
  status: "authorized",
  token: "token_string",
  version: "2.0"
}
```

**Mobile Status**
```http
GET /api/mobile/status
Response: {
  camera_active: true,
  fps: 30,
  speech_queue: 0,
  device: "CUDA"
}
```

**Toggle Continuous Analysis**
```http
POST /api/mobile/continuous-toggle?enabled=true
Response: {
  status: "success",
  continuous_analysis: true
}
```

### WebSocket Endpoints

**Status Updates**
```
ws://localhost:5000/ws/status
Subscribes to real-time status updates
```

**Voice Commands**
```
ws://localhost:5000/ws/commands
Send: { command: "analyze scene" }
Receive: { type: "command_result", data: {...} }
```

---

## 🏗️ Architecture

### Component Overview

```
Beyond Sight Application
├── main.py (FastAPI Server)
├── camera.py (Video Capture)
├── vision_analyzer.py (AI Analysis)
├── speech_engine_pro.py (Audio Output)
├── face_memory.py (Face Recognition)
├── navigation.py (GPS & Routes)
├── place_memory.py (Location Memory)
├── deepseek_integration.py (AI Enhancement)
└── config.py (Configuration)
```

### Data Flow

```
Camera Input
    ↓
Frame Capture (camera.py)
    ↓
Vision Analysis (vision_analyzer.py)
    ├── Object Detection
    ├── Text Recognition
    ├── Hazard Analysis
    └── Distance Estimation
    ↓
AI Enhancement (deepseek_integration.py) [Optional]
    ↓
Speech Generation (speech_engine_pro.py)
    ↓
Audio Output + WebSocket Update
```

### Key Classes

**WebBeyondSightAssistant**: Main application controller
- Manages camera, vision, speech, and navigation
- Handles API requests and WebSocket connections
- Coordinates all system components

**CameraProcessor**: Real-time video capture
- Frame acquisition and buffering
- Face/hand detection
- Frame encoding for streaming

**VisionAnalyzer**: AI-powered visual understanding
- YOLO object detection
- OCR text reading
- Hazard identification
- Scene comprehension

**SpeechEnginePro**: Priority-based audio output
- Text-to-speech using Edge-TTS
- Fallback to pyttsx3
- Queue management
- Critical alert interruption

**FaceMemory**: Person recognition system
- Face encoding and storage
- Person identification
- Memory persistence

---

## 🔧 Troubleshooting

### Camera Issues

**Problem**: Camera won't start
```
Solution:
1. Check camera permissions (Windows: Settings > Privacy > Camera)
2. Verify no other app is using camera
3. Try different camera index in config.py: CAMERA_INDEX = 0,1,2...
4. Restart application
```

**Problem**: Low FPS / Slow performance
```
Solution:
1. Reduce frame size: FRAME_WIDTH = 640, FRAME_HEIGHT = 480
2. Lower FPS: FRAME_RATE = 15
3. Enable GPU acceleration: Install CUDA
4. Close other applications using GPU
```

### Audio Issues

**Problem**: No audio output
```
Solution:
1. Check system volume is not muted
2. Verify speakers are connected and enabled
3. Check in config.py: SPEECH_ENGINE = 'sapi5' (Windows)
4. Test with: python -c "from speech_engine_pro import SpeechEnginePro; SpeechEnginePro().speak('test')"
```

**Problem**: Speech is too fast/slow
```
Solution:
Edit config.py:
SPEECH_RATE = 140  # Range: 80-300 WPM
```

### Network/Web Interface

**Problem**: Can't access from mobile
```
Solution:
1. Find your computer IP: ipconfig (Windows) or ifconfig (Linux)
2. Use format: http://COMPUTER_IP:5000/mobile
3. Ensure both devices on same network
4. Check firewall: Allow port 5000
```

**Problem**: CORS errors
```
Solution:
1. Check .env CORS_ORIGINS
2. Disable debug mode if testing: CORS_DEBUG_MODE=true
3. Ensure correct protocol (http vs https)
```

### API Errors

**Error**: "Camera not active" or "Camera is not ready"
```
Solution: Click "Start Camera" in web interface first
```

**Error**: Analysis timeout
```
Solution: Increase ANALYSIS_TIMEOUT in config.py or reduce frame size
```

**Error**: Deepseek API errors
```
Solution:
1. Verify API key in .env
2. Check Deepseek service status
3. Disable AI enhancement: DEEPSEEK_ENABLED=false
4. Check rate limits on API
```

### Performance Optimization

**For slow systems:**
```python
# Reduce resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Reduce FPS
FRAME_RATE = 15

# Disable continuous analysis
CONTINUOUS_ANALYSIS = False

# Reduce analysis interval
CONTINUOUS_ANALYSIS_INTERVAL = 10
```

**For GPU systems:**
```
Install CUDA Toolkit
Set DEVICE = 'cuda' in main.py (automatic detection)
```

---

## 🐛 Debugging

### Enable Debug Logging

**In .env:**
```env
FLASK_ENV=development
FLASK_DEBUG=true
DEBUG_MODE=true
```

### View Logs

**Real-time logs:**
```bash
tail -f beyond_sight.log  # Linux/macOS
Get-Content -Tail 50 -Wait beyond_sight.log  # Windows
```

**Log file location:** `./beyond_sight.log`

### Test Individual Components

```python
# Test camera
python -c "from camera_processor import CameraProcessor; c = CameraProcessor(); c.start()"

# Test speech
python -c "from speech_engine_pro import SpeechEnginePro; SpeechEnginePro().speak('Hello')"

# Test vision
python verify_setup.py
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test
4. Commit: `git commit -m "Add feature: description"`
5. Push: `git push origin feature/your-feature`
6. Submit Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Test before submitting

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLOv8**: Object detection (Ultralytics)
- **MediaPipe**: Hand and face detection
- **Edge-TTS**: Natural text-to-speech
- **Deepseek AI**: Optional AI enhancement
- **FastAPI**: Modern web framework
- **OpenCV**: Computer vision processing

---

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/DavidLivingston05/BeyondSight/issues)
- **Documentation**: Check [STATIC_CONFIG.md](STATIC_CONFIG.md) for advanced configuration
- **Troubleshooting**: See [#Troubleshooting](#-troubleshooting) section above

---

## 🚀 Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Advanced gesture recognition
- [ ] Real-time object tracking
- [ ] Multi-language support
- [ ] Custom AI model training
- [ ] Enhanced face recognition
- [ ] Offline mode support
- [ ] Database integration

---

**Last Updated**: November 16, 2025  
**Version**: 2.0  
**Status**: Production Ready ✅

🌍 **Empowering accessibility through AI-powered vision assistance** 🌍
