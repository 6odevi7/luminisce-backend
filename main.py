import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io

# Import our custom modules
from ai_generator import AIGenerator
from audio_engine import AudioEngine
from audio_analyzer import AudioAnalyzer

app = FastAPI(title="Luminisce Backend")

# Configuration
SAMPLE_DIRS = [
    r"D:\freekits",
    r"D:\reform original\Reboot\Chosen-1"
]
# VST Path - Set to None to disable VST processing
# Note: Pedalboard works best with simple effect plugins (reverb, delay, EQ)
# Complex synth plugins like Spawn may not be supported
VST_PATH = None  # Disabled - Spawn is not compatible with Pedalboard

# Initialize AI, Audio Engine, and Analyzer
ai_generator = AIGenerator()
audio_engine = AudioEngine()
# Try to load VST, but don't fail if it doesn't work
if VST_PATH:
    audio_engine.load_vst(VST_PATH)
else:
    print("VST processing disabled - audio will be generated without effects")
audio_analyzer = AudioAnalyzer()

# Auto-index samples on startup (DISABLED - use /index endpoint instead)
# import threading
# def index_samples_background():
#     for directory in SAMPLE_DIRS:
#         if os.path.exists(directory):
#             print(f"Indexing samples in {directory}...")
#             count = audio_analyzer.quick_index_directory(directory)
#             print(f"Indexed {count} new samples from {directory}")

# threading.Thread(target=index_samples_background, daemon=True).start()

# Configure CORS - allow all origins for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 5
    creativity: float = 0.5

class RemixRequest(BaseModel):
    genre: str
    mood: str
    tempo: int = 100
    pitch: int = 0
    intensity: int = 50
    prompt: Optional[str] = None
    audioData: Optional[str] = None  # Base64 encoded audio
    audioFile: Optional[str] = None  # File path fallback

@app.get("/api/status")
async def api_status():
    print("\n[STATUS] Backend status check")
    return {"message": "Luminisce Backend is running", "version": "1.0.0"}

@app.post("/test-remix")
async def test_remix():
    """Test endpoint to verify backend is receiving remix requests"""
    print("\n" + "="*80)
    print("[TEST] ✓ TEST REMIX ENDPOINT REACHED")
    print("="*80)
    return {"success": True, "message": "Backend is receiving requests!"}

import asyncio
from fastapi.concurrency import run_in_threadpool

# Global lock to ensure only one generation happens at a time on the GPU
# This prevents OOM errors while allowing other requests (status, static files) to proceed
generation_lock = asyncio.Lock()

@app.post("/generate")
async def generate_sound(request: GenerateRequest):
    """Generate audio using MusicGen with optional prompt enhancement."""
    try:
        import sys
        print("\n" + "="*80)
        print("[GENERATE] ✓✓✓ GENERATE REQUEST RECEIVED ✓✓✓")
        print(f"[GENERATE] Prompt: '{request.prompt}'")
        print(f"[GENERATE] Duration: {request.duration}s, Creativity: {request.creativity}")
        print("="*80)
        sys.stderr.write(f"\n[GENERATE] Starting generation for prompt: '{request.prompt}'\n")
        sys.stderr.flush()
        
        # Enhance prompt using audio analyzer
        enhanced_prompt = audio_analyzer.get_prompt_enhancement(request.prompt)
        sys.stderr.write(f"[GENERATE] Original: {request.prompt}\n")
        sys.stderr.write(f"[GENERATE] Enhanced: {enhanced_prompt}\n")
        sys.stderr.write(f"[GENERATE] Duration: {request.duration}s\n")
        sys.stderr.flush()
        
        # Acquire lock to ensure GPU safety (serialize generations)
        async with generation_lock:
            # Run generation in a separate thread to avoid blocking the main event loop
            # This keeps the server responsive for other users/requests
            audio_bytes = await run_in_threadpool(ai_generator.generate, enhanced_prompt, request.duration)
        
        sys.stderr.write(f"[GENERATE] Audio generated, size: {len(audio_bytes.getvalue())} bytes\n")
        sys.stderr.flush()
        
        # Ensure we're at the start of the stream
        audio_bytes.seek(0)
        
        # Return as direct response
        return StreamingResponse(
            audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated.wav",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    except Exception as e:
        error_msg = str(e) if isinstance(e, (ValueError, FileNotFoundError, IOError)) else repr(e)
        sys.stderr.write(f"[GENERATE] ERROR: {error_msg}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail="Generation processing failed. Check server logs for details.")

@app.post("/remix")
async def remix_audio(request: RemixRequest):
    """Remix audio using MusicGen - same backend as generate endpoint."""
    try:
        import sys
        import base64
        import librosa
        import numpy as np
        
        print("\n" + "="*80)
        print("[REMIX] ✓✓✓ REMIX REQUEST RECEIVED ✓✓✓")
        print(f"[REMIX] Genre: {request.genre}, Mood: {request.mood}, Tempo: {request.tempo}%, Pitch: {request.pitch}, Intensity: {request.intensity}%")
        print(f"[REMIX] Prompt: {request.prompt}")
        print(f"[REMIX] Has audio data: {bool(request.audioData)}")
        print(f"[REMIX] Audio file: {request.audioFile}")
        print(f"[REMIX] Audio data length: {len(request.audioData) if request.audioData else 0}")
        print("="*80)
        sys.stderr.write(f"\n[REMIX] Starting remix request\n")
        sys.stderr.write(f"[REMIX] Genre: {request.genre}, Mood: {request.mood}, Tempo: {request.tempo}%, Pitch: {request.pitch}, Intensity: {request.intensity}%\n")
        sys.stderr.write(f"[REMIX] Audio data length: {len(request.audioData) if request.audioData else 0}\n")
        sys.stderr.flush()
        
        # Load audio from base64 or file
        audio_data = None
        sr = None
        
        if request.audioData:
            sys.stderr.write(f"[REMIX] Loading audio from base64...\n")
            sys.stderr.flush()
            try:
                # Extract base64 from data URL if needed
                if ',' in request.audioData:
                    base64_string = request.audioData.split(',')[1]
                else:
                    base64_string = request.audioData
                
                audio_bytes = base64.b64decode(base64_string)
                audio_io = io.BytesIO(audio_bytes)
                audio_data, sr = librosa.load(audio_io, sr=None)
                sys.stderr.write(f"[REMIX] Audio loaded from base64: {len(audio_data)} samples at {sr}Hz\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[REMIX] Failed to load from base64: {str(e)}\n")
                sys.stderr.flush()
        
        if audio_data is None and request.audioFile:
            sys.stderr.write(f"[REMIX] Loading audio from file: {request.audioFile}\n")
            sys.stderr.flush()
            try:
                if request.audioFile.startswith('blob:'):
                    raise ValueError('Blob URLs not supported on server')
                
                # Try to load from local file system
                if os.path.exists(request.audioFile):
                    audio_data, sr = librosa.load(request.audioFile, sr=None)
                else:
                    # Try in uploads folder
                    upload_path = os.path.join(os.path.dirname(__file__), '..', 'Remixer', 'uploads', os.path.basename(request.audioFile))
                    if os.path.exists(upload_path):
                        audio_data, sr = librosa.load(upload_path, sr=None)
                    else:
                        raise FileNotFoundError(f'Audio file not found: {request.audioFile}')
                
                sys.stderr.write(f"[REMIX] Audio loaded from file: {len(audio_data)} samples at {sr}Hz\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[REMIX] Failed to load from file: {str(e)}\n")
                sys.stderr.flush()
                raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")
        
        if audio_data is None:
            raise HTTPException(status_code=400, detail="No audio data provided (audioData or audioFile required)")
        
        sys.stderr.write(f"[REMIX] Audio loaded: {len(audio_data)} samples at {sr}Hz\n")
        sys.stderr.flush()
        
        # Acquire lock to serialize with generate requests
        async with generation_lock:
            # Run remix in a separate thread to avoid blocking
            remixed_bytes = await run_in_threadpool(
                ai_generator.remix,
                audio_data,
                sr,
                request.genre,
                request.mood,
                request.tempo,
                request.pitch,
                request.intensity,
                request.prompt
            )
        
        if remixed_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to remix audio")
        
        remixed_bytes.seek(0)
        
        sys.stderr.write(f"[REMIX] Remix complete, returning {remixed_bytes.getbuffer().nbytes} bytes\n")
        sys.stderr.flush()
        
        return StreamingResponse(
            remixed_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=remixed.wav",
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) if isinstance(e, (ValueError, FileNotFoundError, IOError)) else repr(e)
        sys.stderr.write(f"[REMIX] ERROR: {error_msg}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail="Remix processing failed. Check server logs for details.")

@app.get("/analyze-status")
async def analyze_status():
    """Get status of sample analysis."""
    analyzed_count = sum(1 for meta in audio_analyzer.sample_index.values() if meta.get("analyzed", False))
    return {
        "samples_indexed": len(audio_analyzer.sample_index),
        "samples_analyzed": analyzed_count,
        "directories": SAMPLE_DIRS
    }

@app.post("/index-samples")
async def index_samples_manually():
    """Manually trigger sample indexing."""
    import threading
    def index_samples():
        total = 0
        for directory in SAMPLE_DIRS:
            if os.path.exists(directory):
                print(f"Indexing samples in {directory}...")
                count = audio_analyzer.quick_index_directory(directory)
                print(f"Indexed {count} new samples from {directory}")
                total += count
        print(f"Total indexed: {total}")
    
    threading.Thread(target=index_samples, daemon=True).start()
    return {"message": "Indexing started in background", "directories": SAMPLE_DIRS}

@app.post("/match-prompt")
async def match_prompt(request: GenerateRequest):
    """Find samples matching the prompt."""
    matches = audio_analyzer.match_prompt_to_samples(request.prompt, top_k=10)
    return {
        "prompt": request.prompt,
        "matches": [{"path": path, "features": features} for path, features in matches]
    }

@app.get("/scale/{root}/{scale_type}")
async def get_scale(root: int, scale_type: str):
    """Get notes for a specific scale (Music Theory)."""
    return {"notes": audio_engine.get_scale_notes(root, scale_type)}

@app.get("/samples")
async def list_samples():
    """List available samples from the configured directories."""
    samples = []
    for directory in SAMPLE_DIRS:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        samples.append({
                            "name": file,
                            "path": os.path.join(root, file),
                            "source": directory
                        })
    return {"count": len(samples), "samples": samples[:100]} # Limit to 100 for now

# Mount static files LAST so API routes take precedence
import pathlib
static_dir = pathlib.Path(__file__).parent.parent / "dist"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    # Allow configuring workers via env var (default to 1)
    # Note: reload only works with workers=1
    workers = int(os.getenv("WORKERS", "1"))
    reload = workers == 1
    
    # Long timeout for generation requests (up to 15 minutes)
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=reload, 
        workers=workers,
        timeout_keep_alive=900,  # 15 minutes for keepalive
    )
