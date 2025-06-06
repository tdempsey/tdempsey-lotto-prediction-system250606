#!/usr/bin/env python3
"""Run the Lotto Prediction System web GUI."""

import os
import sys

# Add the current directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lotto_prediction_system import web_gui

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Lotto Prediction System web GUI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Print startup information
    print(f"Starting Lotto Prediction System web GUI on http://{args.host}:{args.port}")
    print("Use Ctrl+C to stop the server")
    
    # Run the app
    web_gui.run_app(host=args.host, port=args.port, debug=args.debug)