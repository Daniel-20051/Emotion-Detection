# Vercel serverless function entry point
from app import app

# This is required for Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)
