import sys
import os

# Add the parent directory to the Python path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export the Flask app for Vercel
# This is the standard pattern for Vercel Python functions
app = app
