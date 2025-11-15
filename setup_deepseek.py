#!/usr/bin/env python3
"""
Setup wizard for Deepseek API integration.
Helps users configure their API key and verify the setup.
"""

import os
import sys
import json
from pathlib import Path


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text: str):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text: str):
    """Print info message"""
    print(f"ℹ️  {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"⚠️  {text}")


def check_env_file():
    """Check if .env file exists"""
    env_path = Path('.env')
    if env_path.exists():
        print_success(".env file exists")
        return True
    return False


def read_env_file():
    """Read environment variables from .env"""
    env_path = Path('.env')
    if not env_path.exists():
        return {}
    
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def write_env_file(api_key: str):
    """Write API key to .env file"""
    env_path = Path('.env')
    
    # Read existing content
    content = ""
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
    
    # Update or add API key
    if 'DEEPSEEK_API_KEY' in content:
        # Replace existing key
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith('DEEPSEEK_API_KEY='):
                new_lines.append(f'DEEPSEEK_API_KEY={api_key}')
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    else:
        # Add new key
        if content and not content.endswith('\n'):
            content += '\n'
        content += f'DEEPSEEK_API_KEY={api_key}\n'
    
    # Write back
    with open(env_path, 'w') as f:
        f.write(content)
    
    print_success(f"API key saved to .env")
    return True


def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key:
        return False
    
    # Check format (Deepseek keys typically start with sk_live_ or sk_test_)
    if api_key.startswith(('sk_live_', 'sk_test_')):
        return True
    
    # Allow other formats too (just not empty)
    return len(api_key) > 10


def test_api_key(api_key: str) -> bool:
    """Test if API key works"""
    try:
        import requests
        
        print_info("Testing API key connectivity...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print_success("API key is valid and working!")
            return True
        elif response.status_code == 401:
            print_error("API key is invalid or expired")
            return False
        else:
            print_warning(f"API returned status {response.status_code}")
            print_info("This might be a temporary issue. Try again later.")
            return False
    
    except requests.exceptions.ConnectionError:
        print_warning("Could not connect to Deepseek API")
        print_info("Check your internet connection")
        return False
    except Exception as e:
        print_warning(f"Error testing API: {e}")
        return False


def main():
    """Main setup wizard"""
    print_header("🚀 Beyond Sight - Deepseek AI Setup Wizard")
    
    print("""
This wizard will help you set up Deepseek API integration for
enhanced scene descriptions, hazard analysis, and navigation.

Time: ~2 minutes
""")
    
    # Step 1: Check if setup is needed
    env_vars = read_env_file()
    existing_key = env_vars.get('DEEPSEEK_API_KEY', '')
    
    if existing_key:
        print_success(f"Found existing API key: {existing_key[:20]}...")
        
        response = input("\n1. Do you want to use the existing key? (y/n): ").lower().strip()
        if response == 'y':
            print_success("Using existing API key")
            return show_summary(existing_key, verified=False)
    
    # Step 2: Get API key from user
    print("\n2. Getting API Key")
    print("""
To get a Deepseek API key:
1. Visit: https://platform.deepseek.com/api_keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with sk_live_)
""")
    
    while True:
        api_key = input("Enter your Deepseek API key: ").strip()
        
        if not api_key:
            print_error("API key cannot be empty")
            continue
        
        if not validate_api_key(api_key):
            print_warning("API key format looks unusual")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != 'y':
                continue
        
        break
    
    # Step 3: Save to .env
    print("\n3. Saving Configuration")
    try:
        write_env_file(api_key)
    except Exception as e:
        print_error(f"Failed to save .env: {e}")
        print_info("You can manually add to .env: DEEPSEEK_API_KEY=" + api_key)
        return False
    
    # Step 4: Test API key
    print("\n4. Verifying API Key")
    response = input("Test the API key now? (y/n): ").lower().strip()
    
    verified = False
    if response == 'y':
        verified = test_api_key(api_key)
    else:
        print_info("Skipping verification. You can test later.")
    
    # Step 5: Summary
    return show_summary(api_key, verified)


def show_summary(api_key: str, verified: bool = False) -> bool:
    """Show setup summary"""
    print_header("✨ Setup Summary")
    
    print(f"API Key: {api_key[:20]}...{api_key[-5:]}")
    print(f"Configuration: Saved to .env")
    print(f"Status: {'✅ Verified' if verified else '⚠️  Not verified'}")
    
    print(f"""
Next Steps:
1. Start the application:
   python main.py

2. Open in browser:
   http://localhost:5000

3. Test with voice commands:
   "Describe the scene"
   "Check for hazards"
   "Guide my navigation"

Documentation:
- Quick Start: DEEPSEEK_QUICKSTART.md
- Full Setup: DEEPSEEK_SETUP.md  
- Full Docs: DEEPSEEK_INTEGRATION.md

Troubleshooting:
Run: curl http://localhost:5000/api/deepseek/status
Expected: {{"enabled": true, "api_healthy": true, ...}}
""")
    
    print_success("Setup complete! Ready to use Deepseek AI enhancement!")
    return True


def show_status():
    """Show current Deepseek setup status"""
    print_header("📊 Deepseek Setup Status")
    
    env_vars = read_env_file()
    api_key = env_vars.get('DEEPSEEK_API_KEY', '')
    
    if api_key:
        print_success(f"API Key configured: {api_key[:20]}...{api_key[-5:]}")
        print_info("To test: python main.py (then curl http://localhost:5000/api/deepseek/status)")
    else:
        print_warning("No API key configured")
        print_info("Run this script without arguments to set up: python setup_deepseek.py")
    
    # Check imports
    try:
        from deepseek_integration import get_deepseek_client
        print_success("Deepseek integration module found")
    except ImportError:
        print_error("Deepseek integration module not found")
    
    # Check if running
    try:
        import requests
        response = requests.get('http://localhost:5000/api/deepseek/status', timeout=2)
        if response.status_code == 200:
            data = response.json()
            print_success("Application is running")
            print(f"  Deepseek enabled: {data.get('enabled', False)}")
            print(f"  API healthy: {data.get('api_healthy', False)}")
        else:
            print_warning(f"Application returned status {response.status_code}")
    except:
        print_info("Application is not running (start with: python main.py)")


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--status':
            show_status()
        else:
            success = main()
            sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
