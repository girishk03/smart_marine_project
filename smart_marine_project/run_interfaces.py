#!/usr/bin/env python3
"""
Smart Marine Project - Interface Launcher
=========================================

Easy launcher for all Smart Marine Project interfaces.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'web': ['flask', 'werkzeug'],
        'desktop': [],  # tkinter is built-in
        'streamlit': ['streamlit'],
        'api': ['fastapi', 'uvicorn']
    }
    
    missing_packages = []
    
    for interface, packages in required_packages.items():
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} (for {interface} interface)")
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements_interfaces.txt")
        return False
    
    return True

def run_web_interface():
    """Run the Flask web interface"""
    print("🌐 Starting Flask Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "web_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")
    except Exception as e:
        print(f"❌ Error running web interface: {e}")

def run_desktop_app():
    """Run the desktop GUI application"""
    print("🖥️ Starting Desktop GUI Application...")
    print("📱 GUI window will open shortly...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "desktop_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Desktop app stopped")
    except Exception as e:
        print(f"❌ Error running desktop app: {e}")

def run_streamlit_app():
    """Run the Streamlit web app"""
    print("🌊 Starting Streamlit Web App...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

def run_api_server():
    """Run the REST API server"""
    print("🔌 Starting REST API Server...")
    print("📱 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "api_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error running API server: {e}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Smart Marine Project Interface Launcher")
    parser.add_argument(
        "interface",
        choices=["web", "desktop", "streamlit", "api", "all"],
        help="Interface to run"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements, don't run interface"
    )
    
    args = parser.parse_args()
    
    print("🌊 Smart Marine Project - Interface Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        if not args.check_only:
            sys.exit(1)
        else:
            print("✅ Requirements check complete")
            return
    
    if args.check_only:
        print("✅ All requirements satisfied!")
        return
    
    # Run selected interface
    if args.interface == "web":
        run_web_interface()
    elif args.interface == "desktop":
        run_desktop_app()
    elif args.interface == "streamlit":
        run_streamlit_app()
    elif args.interface == "api":
        run_api_server()
    elif args.interface == "all":
        print("🚀 Starting all interfaces...")
        print("⚠️  Note: You'll need to run each interface in a separate terminal")
        print("\n1. Web Interface: python run_interfaces.py web")
        print("2. Desktop App: python run_interfaces.py desktop")
        print("3. Streamlit App: python run_interfaces.py streamlit")
        print("4. API Server: python run_interfaces.py api")

if __name__ == "__main__":
    main()
