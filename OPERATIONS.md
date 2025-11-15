# 🎮 Beyond Sight Operations Guide

## Complete Instructions for Running & Operating Beyond Sight

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#-pre-deployment-checklist)
2. [Starting the Server](#-starting-the-server)
3. [Web Interface Navigation](#-web-interface-navigation)
4. [Mobile App Usage](#-mobile-app-usage)
5. [Voice Commands Reference](#-voice-commands-reference)
6. [Advanced Operations](#-advanced-operations)
7. [Maintenance](#-maintenance)
8. [Emergency Procedures](#-emergency-procedures)

---

## ✅ Pre-Deployment Checklist

Before starting Beyond Sight, verify:

### Hardware Setup
- [ ] Webcam connected and functioning
- [ ] Microphone connected and set as default input
- [ ] Speakers connected and not muted
- [ ] Adequate disk space (minimum 5GB)
- [ ] RAM available (monitor with `top` or Task Manager)

### Software Setup
- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip list | grep fastapi`
- [ ] Environment file configured (`.env` present)
- [ ] Network connectivity verified
- [ ] Firewall allows port 5000

### Optional Enhancements
- [ ] NVIDIA CUDA installed (for GPU support)
- [ ] Tesseract OCR installed (for better text recognition)
- [ ] Deepseek API key obtained
- [ ] GPS device connected (for navigation)

### Pre-Flight Verification
```bash
# Run setup verification
python verify_setup.py

# Test camera
python -c "from camera_processor import CameraProcessor; print('Camera OK')"

# Test speech
python -c "from speech_engine_pro import SpeechEnginePro; SpeechEnginePro().speak('Test')"

# Test vision
python -c "from vision_analyzer import VisionAnalyzer; print('Vision OK')"
```

---

## 🚀 Starting the Server

### Step 1: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 2: Verify Current Directory

```bash
cd /path/to/BeyondSight
pwd  # Linux/macOS or cd # Windows
```

### Step 3: Start the Application

```bash
python main.py
```

### Step 4: Wait for Initialization

You'll see output:
```
🚀 Starting Beyond Sight Web Server...
================================================================================
📍 NETWORK ADDRESSES:
   🖥️  Desktop:  http://localhost:5000
   📱 Mobile:   http://192.168.1.100:5000/mobile
   🌐 Network:  http://192.168.1.100:5000
================================================================================
✅ Ready for Desktop & Mobile connections!
```

**Note your IP address** for mobile access.

### Step 5: Access the Web Interface

Open your browser:
```
http://localhost:5000
```

---

## 🖥️ Web Interface Navigation

### Main Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  BEYOND SIGHT - AI Vision Assistant           Status: ✅ │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  📹 [START CAMERA]  [STOP CAMERA]     FPS: 30  ⚠️ Ready │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                  Video Feed                         │ │
│  │                   (Live Stream)                      │ │
│  │                                                      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  QUICK ACTIONS:                                          │
│  [🔍 Analyze Scene]  [🔎 Find Object]  [📖 Read Text]   │
│  [⚠️ Check Hazards]  [🗺️ Navigation]                     │
│                                                           │
│  SETTINGS:                                              │
│  Speech Rate:  [Slow] [Normal] [Fast]                  │
│  Volume:       [Quiet] [Normal] [Loud]                 │
│  ☑️ Continuous Analysis    ☐ Face Recognition         │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Camera Control Section

**Start Camera**
1. Click blue "START CAMERA" button
2. Wait 2-3 seconds for camera to initialize
3. You should hear "Camera activated successfully"
4. Video feed appears in main area
5. FPS counter shows (e.g., FPS: 30)

**Stop Camera**
1. Click red "STOP CAMERA" button
2. Feed stops and shows black screen
3. You hear "Camera deactivated"

**Monitor Status**
- ✅ Green: Camera running and healthy
- 🟡 Yellow: Camera running but slow
- ❌ Red: Camera offline or error

### Quick Action Buttons

**1. Analyze Scene**
- Gives comprehensive description
- Takes 2-5 seconds
- Audio output describes: objects, people, environment
- Best for: General awareness, room assessment

**2. Find Object**
- Search for specific items
- Enter item name: "door", "person", "chair"
- Returns location: "2 meters ahead on your right"
- Best for: Navigation, locating items

**3. Read Text**
- Extracts visible text from scene
- Reads: signs, documents, screens
- Audio output reads the text
- Best for: Reading information

**4. Check Hazards**
- Safety analysis
- Detects: obstacles, stairs, edges
- Priority alert if danger found
- Audio: "WARNING: Staircase ahead"
- Best for: Safety checks, unfamiliar spaces

**5. Get Navigation**
- Movement guidance
- Based on scene analysis
- Recommends safe paths
- Considers obstacles
- Best for: Planning movement

### Settings Panel

**Speech Rate Control**
- **Slow**: 100 WPM (clear, deliberate)
- **Normal**: 140 WPM (standard pace)
- **Fast**: 180 WPM (quick, energetic)
- Default: Normal

**Volume Control**
- **Quiet**: 50% system volume
- **Normal**: 80% system volume
- **Loud**: 100% system volume
- Default: Loud

**Feature Toggles**
- ☑️ **Continuous Analysis**: Auto-describe scene every 5 seconds
- ☑️ **Face Recognition**: Remember and identify people
- ☑️ **Hazard Alerts**: Warn about dangers
- ☑️ **Audio Cues**: Sound effects for alerts

### Status Panel

Monitor real-time information:
- **Camera Status**: Active/Inactive with indicator
- **FPS**: Frames per second (30 is optimal)
- **Frame Count**: Number of frames processed
- **Speech Queue**: Pending audio messages
- **System Health**: Overall system status

---

## 📱 Mobile App Usage

### Access Mobile Interface

**From your mobile device on same network:**
```
http://YOUR_COMPUTER_IP:5000/mobile
```

**Find Your Computer IP:**
- Windows: Open Command Prompt, type `ipconfig`, look for IPv4 Address
- Linux/macOS: Open Terminal, type `ifconfig`, look for inet address

**Example:**
```
http://192.168.1.100:5000/mobile
```

### Mobile Interface Features

**Main Controls**
```
┌──────────────────────────────────┐
│ 🔄 Status: Connected    ⚡ Health │
├──────────────────────────────────┤
│                                  │
│  📹 [START]  [STOP]  [REFRESH]   │
│                                  │
│  📡 Server: Online               │
│  🎥 Camera: Ready                │
│  🔊 Audio: Connected             │
│                                  │
│  ┌────────────────────────────┐  │
│  │   Video Stream             │  │
│  │   (Real-time feed)         │  │
│  └────────────────────────────┘  │
│                                  │
│  🎤 [VOICE COMMAND]              │
│                                  │
│  [Analyze] [Find] [Read] [Safety]│
│                                  │
└──────────────────────────────────┘
```

### Mobile Voice Commands

**Tap the microphone icon** and say commands naturally:
- "Analyze the scene"
- "What's in front of me?"
- "Find the door"
- "Read the text"
- "Are there hazards?"
- "Give me directions"

### Mobile Settings

**Swipe settings icon** to access:
- Voice rate adjustment
- Volume control
- Continuous analysis toggle
- Face recognition toggle
- API key configuration

---

## 🎤 Voice Commands Reference

### Voice Command Activation

**How to Give Commands:**
1. **Web Interface**: Click microphone icon in nav bar
2. **Mobile**: Tap voice icon
3. **WebSocket**: Send JSON via connection

**Command Format:**
- Natural speech (e.g., "What do you see?")
- Simple commands (e.g., "Analyze scene")
- Clear pronunciation for accuracy

### Available Commands

#### Vision Analysis Commands

| Command | Examples | Output |
|---------|----------|--------|
| **Analyze** | "Analyze scene", "What do you see?" | Full scene description with audio |
| **Find Object** | "Find door", "Where's a person?" | Object location with distance |
| **Read Text** | "Read text", "What does it say?" | Text extraction and audio reading |
| **Check Hazards** | "Are there hazards?", "Safety check" | Warning if dangers detected |
| **Navigation** | "Give me directions", "Which way?" | Safe path guidance |

#### Settings Commands

| Command | Examples | Result |
|---------|----------|--------|
| **Volume** | "Louder", "Volume max" | Adjusts audio volume |
| **Speed** | "Slower", "Talk faster" | Adjusts speech rate |
| **Continuous** | "Toggle description", "Keep talking" | Enables/disables auto-analysis |
| **Recognition** | "Remember faces", "Identify people" | Toggles face recognition |

#### System Commands

| Command | Examples | Result |
|---------|----------|--------|
| **Help** | "What can you do?", "Help" | Lists available commands |
| **Status** | "System status", "How are you?" | Returns system health info |
| **Stop** | "Stop", "Cancel" | Halts current operation |

### Voice Command Examples

**Scene Understanding:**
```
User: "Analyze scene"
System: "Processing... I see a living room with a couch, 
         television on the wall 3 meters away, and a coffee 
         table 1 meter in front of you. There are windows 
         to your left with natural light coming through."
```

**Object Finding:**
```
User: "Find door"
System: "Door detected. It's located 4 meters ahead on 
         your right side. Appears to be an open doorway."
```

**Text Reading:**
```
User: "Read text"
System: "I can see the following text: 'Welcome to Beyond Sight'
         in blue letters at the top of the sign."
```

**Safety Check:**
```
User: "Check hazards"
System: "Area appears safe. No immediate dangers detected.
         You are clear to move forward."
```

---

## ⚙️ Advanced Operations

### Continuous Analysis Mode

**Enable Automatic Descriptions:**
1. Toggle switch or say "Enable continuous description"
2. System describes scene every 5 seconds
3. Useful during mobility for situational awareness

**Customize Interval:**
Edit `config.py`:
```python
CONTINUOUS_ANALYSIS_INTERVAL = 5  # seconds
```

### Face Recognition

**Remember a Person:**
1. Person appears on camera
2. System captures face
3. Say "Remember this face as John"
4. Future detections: "John is here"

**View Remembered People:**
```bash
python -c "from face_memory import FaceMemory; f = FaceMemory(); print(f.get_known_people())"
```

### Custom Analysis

**Create Custom Analysis Function:**

```python
# In main.py, add to WebBeyondSightAssistant class:
def custom_analysis(self, query: str) -> Dict[str, str]:
    """Custom AI-powered analysis using Deepseek."""
    try:
        with self.frame_lock:
            frame = self.current_frame.copy()
        
        result = self.vision.analyze_scene_comprehensive(frame)
        
        if self.deepseek.enabled:
            enhanced = self.deepseek.analyze_custom(result, query)
            return {'result': enhanced, 'status': 'success'}
        return {'result': result, 'status': 'success'}
    except Exception as e:
        return {'result': str(e), 'status': 'error'}
```

### GPS Navigation

**Enable GPS Mode:**
1. Ensure GPS device connected (USB)
2. In config: `GPS_NAVIGATION = True`
3. Commands: "Get navigation", "Where am I?"

**Set Destination:**
```python
navigator = get_navigator()
navigator.set_destination(latitude, longitude)
guidance = navigator.get_guidance()
```

### API Integration

**Call API Directly:**

```bash
# Start camera
curl -X POST http://localhost:5000/api/camera/start

# Analyze scene
curl -X POST http://localhost:5000/api/analyze/scene

# Get status
curl http://localhost:5000/api/camera/status
```

**Python Integration:**

```python
import requests

# Analyze scene
response = requests.post('http://localhost:5000/api/analyze/scene')
print(response.json()['result'])

# Get video stream
response = requests.get('http://localhost:5000/api/video-stream', stream=True)
```

### WebSocket Connection

**Connect to Status Updates:**

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/status');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('FPS:', data.data.fps);
};
```

**Send Voice Commands:**

```javascript
const cmd_ws = new WebSocket('ws://localhost:5000/ws/commands');
cmd_ws.send(JSON.stringify({
    command: 'analyze scene'
}));
```

---

## 🔧 Maintenance

### Daily Maintenance

**Check System Health:**
```bash
# View last 20 log lines
tail -n 20 beyond_sight.log

# Check for errors
grep ERROR beyond_sight.log

# Monitor memory usage
# Windows: Task Manager
# Linux: top -p $(pgrep -f "python main.py")
```

**Clean Temporary Files:**
```bash
# Remove temporary speech files
rm -f temp_speech.mp3

# Clear log if too large (>100MB)
if [ $(wc -c < beyond_sight.log) -gt 104857600 ]; then
    > beyond_sight.log
fi
```

### Weekly Maintenance

**Update Dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Backup Configuration:**
```bash
cp config.py backups/config_backup_$(date +%Y%m%d).py
cp .env backups/env_backup_$(date +%Y%m%d)
```

**Review Face Recognition Database:**
```bash
# Check stored faces
ls -la data/

# Verify database integrity
python -c "from face_memory import FaceMemory; f = FaceMemory(); print(f'Stored faces: {len(f.known_encodings)}')"
```

### Monthly Maintenance

**Database Cleanup:**
```bash
# Remove old logs (older than 30 days)
find logs -name "*.log" -mtime +30 -delete

# Rebuild face database
rm -f data/face_encodings.pkl data/face_names.json
echo "Database cleaned. Re-train faces on next use."
```

**Performance Optimization:**
```bash
# Clear cache
rm -rf __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +

# Reindex models
python verify_setup.py
```

**Security Review:**
```bash
# Check for exposed credentials
grep -r "sk_live" .
grep -r "sk_test" .

# Verify .gitignore
cat .gitignore
```

---

## 🆘 Emergency Procedures

### Application Crashes

**If Beyond Sight stops responding:**

1. **Force Stop (Ctrl+C in Terminal)**
   ```bash
   # Terminal shows: ^C
   # Wait 2 seconds for graceful shutdown
   ```

2. **Force Kill (if not responding)**
   
   **Windows:**
   ```bash
   # Find process
   tasklist | findstr python
   # Kill process
   taskkill /PID <PID> /F
   ```
   
   **Linux/macOS:**
   ```bash
   # Find and kill
   pkill -f "python main.py"
   ```

3. **Check Logs**
   ```bash
   tail -n 50 beyond_sight.log
   # Look for ERROR or CRITICAL lines
   ```

4. **Restart**
   ```bash
   python main.py
   ```

### Camera Freezes

**If camera feed is frozen:**

1. Click "STOP CAMERA"
2. Wait 3 seconds
3. Click "START CAMERA"
4. If still frozen, restart application

**If camera won't start:**

```bash
# Check camera device
# Windows: Device Manager > Cameras
# Linux: ls -la /dev/video*

# Try different camera index
# Edit config.py: CAMERA_INDEX = 1
```

### Audio Issues

**If no audio output:**

1. Check system volume (not muted)
2. Test with: `python -c "from speech_engine_pro import SpeechEnginePro; SpeechEnginePro().speak('test')"`
3. Verify speakers connected
4. Restart audio service

**If audio is garbled:**

```bash
# Reset speech engine
rm -f temp_speech.mp3
```

### Network Issues

**If can't access from mobile:**

1. **Verify Server Running**
   ```bash
   curl http://localhost:5000/health
   # Should return: {"status": "healthy", ...}
   ```

2. **Check Firewall**
   ```bash
   # Windows: Windows Defender Firewall > Advanced > Inbound Rules
   # Add rule for port 5000
   ```

3. **Find Correct IP**
   ```bash
   # Windows
   ipconfig | findstr "IPv4"
   
   # Linux/macOS
   ifconfig | grep "inet "
   ```

4. **Test Connection**
   ```bash
   # From mobile: ping YOUR_COMPUTER_IP
   # Should get responses (not timeout)
   ```

### High Memory Usage

**If application uses >2GB RAM:**

1. Check what's running:
   ```bash
   # See memory per process
   ps aux | grep python
   ```

2. Reduce load:
   - Disable continuous analysis
   - Lower frame resolution
   - Disable face recognition
   - Stop video stream

3. Restart application:
   ```bash
   # Forces memory cleanup
   python main.py
   ```

### API Rate Limiting

**If getting 429 errors from Deepseek:**

1. Check API quota
2. Increase interval: `CONTINUOUS_ANALYSIS_INTERVAL = 10`
3. Disable AI enhancement: `DEEPSEEK_ENABLED = False`
4. Wait before retrying (5 minute cooldown)

---

## 📊 Performance Tuning

### For Slow Systems

```python
# config.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 15
CONTINUOUS_ANALYSIS_INTERVAL = 10
VOICE_RATE = 180  # Faster to feel responsive
```

### For High-Performance Systems

```python
# config.py
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60
CONTINUOUS_ANALYSIS_INTERVAL = 2
DEEPSEEK_ENABLED = True
```

### GPU-Optimized Settings

```python
# config.py
# Automatically set if CUDA available
DEVICE = 'cuda'  # Auto-detected
# For manual override:
# In main.py: DEVICE = 'cuda'
```

---

## 📝 Logging & Monitoring

### View Logs in Real-Time

```bash
# Linux/macOS
tail -f beyond_sight.log

# Windows PowerShell
Get-Content -Tail 50 -Wait beyond_sight.log
```

### Filter Specific Messages

```bash
# Show only errors
grep ERROR beyond_sight.log

# Show only warnings
grep WARNING beyond_sight.log

# Show last 100 lines
tail -n 100 beyond_sight.log

# Show specific time window
grep "2025-11-16 10:" beyond_sight.log
```

### Create Log Snapshot

```bash
# Backup current log
cp beyond_sight.log beyond_sight_backup_$(date +%Y%m%d_%H%M%S).log

# Clear log for fresh start
> beyond_sight.log
```

---

## ✅ Shutdown Procedure

**Proper Application Shutdown:**

1. In web interface, click "STOP CAMERA"
2. In terminal, press Ctrl+C
3. Wait for: "Goodbye!" message
4. Terminal prompt returns

**Files to Check Before Shutdown:**
- No unsaved analysis results
- Camera stopped
- Audio playing completed

**Post-Shutdown Cleanup:**
```bash
# Optional
rm -f temp_speech.mp3
# Backup logs
cp beyond_sight.log backups/logs_$(date +%Y%m%d).log
```

---

## 📚 Additional Resources

- **README.md**: Feature overview and installation
- **STATIC_CONFIG.md**: Advanced configuration options
- **config.py**: All configuration constants
- **Logs**: `beyond_sight.log` for debugging
- **API Docs**: http://localhost:5000/docs (FastAPI OpenAPI)

---

**Last Updated**: November 16, 2025  
**Version**: 2.0  
**Status**: Production Ready ✅

🚀 **Ready to assist you with every seeing experience!** 🚀
