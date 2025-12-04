#!/usr/bin/env python3
"""
HuggingFace Spaces compatible FastAPI app for MusicGen backend
Deploy to: https://huggingface.co/spaces
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Import existing main.py components
from main import app as main_app

# HuggingFace Spaces requires the app to be named 'app'
app = main_app

# Add CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    # HF Spaces sets PORT environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
