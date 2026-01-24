"""
Vercel Serverless Function Entry Point
Healthcare Fraud Detection API

This file serves as the entry point for Vercel's Python serverless runtime.
It imports the FastAPI app from the backend module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from backend.main import app

# Vercel expects the app to be available as 'app' or 'handler'
handler = app
