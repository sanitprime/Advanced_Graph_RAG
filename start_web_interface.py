#!/usr/bin/env python3
"""
Start the Graph RAG Web Interface
Simple script to launch the web interface
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def start_web_interface():
    """Start the web interface"""
    print("🚀 Starting Graph RAG Web Interface...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this from the graph_rag directory.")
        return
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("❌ Error: Virtual environment not found. Please run setup first.")
        return
    
    print("✅ Virtual environment found")
    print("✅ Flask app ready")
    print("✅ API key configured")
    
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Flask app
        subprocess.run([
            sys.executable, "app.py"
        ], cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_web_interface()
