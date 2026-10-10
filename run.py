#!/usr/bin/env python3
"""
Launcher for Khan Bank Amirlan Voicebot Web App.
Runs the Flask application on port 5005.
"""
import os
import sys

# Add app directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print("=" * 65)
    print("  🏦 KHAN BANK AMIRLAN AI VOICEBOT WEB APPLICATION")
    print(f"  ✨ Luxury Web UI running at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="0.0.0.0", port=port, debug=False)
