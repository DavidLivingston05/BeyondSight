#!/usr/bin/env python
"""
Verification script for Beyond Sight setup.
Checks all critical components and configurations.
"""

import sys
from pathlib import Path
import importlib.util

def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    p = Path(path)
    status = "✅" if p.exists() else "❌"
    print(f"{status} {description}: {path}")
    return p.exists()

def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    p = Path(path)
    status = "✅" if p.is_dir() else "❌"
    print(f"{status} {description}: {path}")
    return p.is_dir()

def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 80)
    print("🔍 BEYOND SIGHT SETUP VERIFICATION")
    print("=" * 80)
    
    all_ok = True
    
    # Check directories
    print("\n📁 Checking Directory Structure...")
    all_ok &= check_directory_exists("templates", "Templates directory")
    all_ok &= check_directory_exists("static", "Static assets directory")
    all_ok &= check_directory_exists("data", "Data directory")
    all_ok &= check_directory_exists("logs", "Logs directory")
    all_ok &= check_directory_exists("models", "Models directory")
    
    # Check critical files
    print("\n📄 Checking Critical Files...")
    all_ok &= check_file_exists("main.py", "Main application")
    all_ok &= check_file_exists("config.py", "Configuration")
    all_ok &= check_file_exists("face_memory.py", "Face memory module")
    all_ok &= check_file_exists("gps_navigator.py", "GPS navigator")
    all_ok &= check_file_exists("vision_analyzer.py", "Vision analyzer")
    all_ok &= check_file_exists("speech_engine_pro.py", "Speech engine")
    all_ok &= check_file_exists("camera_processor.py", "Camera processor")
    
    # Check templates
    print("\n🌐 Checking Templates...")
    all_ok &= check_file_exists("templates/index.html", "Main template")
    all_ok &= check_file_exists("templates/mobile.html", "Mobile template")
    all_ok &= check_file_exists("templates/voice_test.html", "Voice test template")
    
    # Check static assets
    print("\n🎨 Checking Static Assets...")
    all_ok &= check_file_exists("static/magicstudio-art-clean.png", "Application logo")
    
    # Check imports
    print("\n📦 Checking Python Imports...")
    all_ok &= check_import("fastapi", "FastAPI")
    all_ok &= check_import("cv2", "OpenCV")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("pyttsx3", "pyttsx3")
    all_ok &= check_import("pytesseract", "Tesseract OCR")
    all_ok &= check_import("ultralytics", "YOLO")
    
    # Check local modules
    print("\n🔧 Checking Local Modules...")
    all_ok &= check_import("config", "Config module")
    all_ok &= check_import("camera_processor", "Camera processor")
    all_ok &= check_import("vision_analyzer", "Vision analyzer")
    all_ok &= check_import("speech_engine_pro", "Speech engine")
    all_ok &= check_import("face_memory", "Face memory")
    all_ok &= check_import("gps_navigator", "GPS navigator")
    all_ok &= check_import("place_memory", "Place memory")
    all_ok &= check_import("deepseek_integration", "Deepseek integration")
    all_ok &= check_import("utils", "Utilities")
    
    # Final check
    print("\n🚀 Testing Main Application Load...")
    try:
        import main
        print("✅ Main application loads successfully")
        
        # Check assistant initialization
        if hasattr(main, 'assistant'):
            print("✅ Assistant object created")
            
            # Check key components
            components = [
                ('camera', 'Camera processor'),
                ('vision', 'Vision analyzer'),
                ('speech', 'Speech engine'),
                ('face_memory', 'Face memory'),
                ('navigator', 'GPS navigator'),
                ('place_memory', 'Place memory'),
                ('deepseek', 'Deepseek integration'),
            ]
            
            for attr, name in components:
                if hasattr(main.assistant, attr):
                    print(f"✅ {name} initialized")
                else:
                    print(f"❌ {name} NOT initialized")
                    all_ok = False
        else:
            print("❌ Assistant object not found")
            all_ok = False
            
    except Exception as e:
        print(f"❌ Failed to load main application: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ ALL CHECKS PASSED - System is ready!")
        print("\nStart the application with:")
        print("  python main.py")
        print("\nAccess the application at:")
        print("  Desktop:  http://localhost:5000")
        print("  Mobile:   http://<your-ip>:5000/mobile")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
