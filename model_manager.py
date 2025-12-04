"""
Shared MusicGen Model Manager
Used by both main.py and remixer_service.py to load and share the same model instance
Prevents duplicate model loads and GPU memory conflicts
"""

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from datetime import datetime
import sys
import traceback
import threading

# Global model instance (shared across services)
MODEL = None
PROCESSOR = None
MODEL_LOCK = threading.Lock()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[MODEL_MANAGER] Initializing on device: {DEVICE}")


def load_model():
    """Load MusicGen model (thread-safe, loads only once)"""
    global MODEL, PROCESSOR
    
    with MODEL_LOCK:
        if MODEL is not None:
            print(f"[MODEL_MANAGER] Model already loaded, returning cached instance")
            return True
        
        print(f"\n[MODEL_MANAGER] Loading MusicGen model (this may take a minute)...")
        try:
            # Try loading large model first (requires GPU or sufficient RAM)
            model_loaded = False
            
            # First try: musicgen-large with local files only (cached)
            try:
                print(f"[{datetime.now()}] Attempting to load cached facebook/musicgen-large...")
                sys.stdout.flush()
                PROCESSOR = AutoProcessor.from_pretrained("facebook/musicgen-large", local_files_only=True)
                print(f"[{datetime.now()}] Processor loaded from cache")
                sys.stdout.flush()
                
                print(f"[{datetime.now()}] Loading model weights (facebook/musicgen-large)...")
                sys.stdout.flush()
                
                load_kwargs = {
                    "local_files_only": True,
                    "attn_implementation": "eager",
                }
                
                if DEVICE == "cpu":
                    load_kwargs["torch_dtype"] = torch.float32
                    load_kwargs["device_map"] = "cpu"
                    load_kwargs["low_cpu_mem_usage"] = True
                else:
                    load_kwargs["torch_dtype"] = torch.float16
                
                MODEL = MusicgenForConditionalGeneration.from_pretrained(
                    "facebook/musicgen-large",
                    **load_kwargs
                )
                model_loaded = True
                print(f"[{datetime.now()}] Successfully loaded musicgen-large from cache")
            except Exception as e:
                print(f"[{datetime.now()}] ⚠️  Could not load cached musicgen-large: {str(e)[:100]}")
                print(f"[{datetime.now()}] Attempting to download musicgen-small (this may take a while)...")
                sys.stdout.flush()
            
            # Second try: musicgen-small with download if needed
            if not model_loaded:
                try:
                    PROCESSOR = AutoProcessor.from_pretrained("facebook/musicgen-small")
                    print(f"[{datetime.now()}] Processor loaded")
                    sys.stdout.flush()
                    
                    load_kwargs = {
                        "attn_implementation": "eager",
                        "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32
                    }
                    
                    if DEVICE == "cpu":
                        load_kwargs["device_map"] = "cpu"
                        load_kwargs["low_cpu_mem_usage"] = True
                    
                    MODEL = MusicgenForConditionalGeneration.from_pretrained(
                        "facebook/musicgen-small",
                        **load_kwargs
                    )
                    model_loaded = True
                    print(f"[{datetime.now()}] Successfully loaded musicgen-small")
                except Exception as e:
                    print(f"[{datetime.now()}] ❌ ERROR: Could not load any MusicGen model: {e}")
                    sys.stdout.flush()
                    raise
            
            print(f"[{datetime.now()}] Model weights loaded")
            sys.stdout.flush()
            
            if DEVICE == "cuda":
                print(f"[{datetime.now()}] Moving to CUDA...")
                sys.stdout.flush()
                MODEL = MODEL.to("cuda")
                print("MusicGen model loaded on CUDA (float16)")
            else:
                print("MusicGen model loaded on CPU")
            
            # Set to eval mode for inference
            MODEL.eval()
            
            print(f"[{datetime.now()}] Model fully initialized and ready to use")
            sys.stdout.flush()
            
            return True
        except Exception as e:
            print(f"[MODEL_MANAGER] Failed to load MusicGen: {e}")
            traceback.print_exc()
            return False


def get_model():
    """Get the shared MusicGen model (loads if not already loaded)"""
    global MODEL
    if MODEL is None:
        load_model()
    return MODEL


def get_processor():
    """Get the shared MusicGen processor (loads if not already loaded)"""
    global PROCESSOR
    if PROCESSOR is None:
        load_model()
    return PROCESSOR


def get_device():
    """Get the device (cuda or cpu)"""
    return DEVICE


def is_loaded():
    """Check if model is loaded"""
    return MODEL is not None
