import os
import sys

# Add project root to python path for Vercel serverless functions
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.app import app
