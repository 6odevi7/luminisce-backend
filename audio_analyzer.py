import librosa
import numpy as np
import os
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib

class AudioAnalyzer:
    def __init__(self, cache_file="sample_cache.json"):
        self.cache_file = cache_file
        self.sample_index = {}  # {file_path: basic_metadata}
        self.feature_cache = {}  # Lazy-loaded features
        self.load_cache()
        
        self.keyword_map = {
            # Tempo-based
            "slow": {"tempo_range": (0, 90)},
            "medium": {"tempo_range": (90, 130)},
            "fast": {"tempo_range": (130, 200)},
            "energetic": {"tempo_range": (130, 200), "energy": "high"},
            
            # Mood-based
            "dark": {"spectral_centroid": "low"},
            "bright": {"spectral_centroid": "high"},
            "aggressive": {"energy": "high"},
            "calm": {"energy": "low"},
            "ambient": {"energy": "low", "spectral_centroid": "low"},
            
            # Genre hints
            "drum": {"percussion": True},
            "bass": {"spectral_centroid": "low"},
            "synth": {"spectral_centroid": "high"},
        }

    def load_cache(self):
        """Load cached sample index from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.sample_index = json.load(f)
                print(f"Loaded {len(self.sample_index)} samples from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def save_cache(self):
        """Save sample index to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.sample_index, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def quick_index_directory(self, directory: str) -> int:
        """Quickly index files without analyzing them."""
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    path = os.path.join(root, file)
                    if path not in self.sample_index:
                        # Store basic metadata only
                        self.sample_index[path] = {
                            "name": file,
                            "category": self._categorize_by_path(path),
                            "analyzed": False
                        }
                        count += 1
        
        self.save_cache()
        return count

    def _categorize_by_path(self, path: str) -> str:
        """Simple categorization based on file path."""
        path_lower = path.lower()
        if "drum" in path_lower or "kick" in path_lower or "snare" in path_lower:
            return "drums"
        elif "bass" in path_lower:
            return "bass"
        elif "synth" in path_lower or "lead" in path_lower:
            return "synth"
        elif "pad" in path_lower or "ambient" in path_lower:
            return "ambient"
        elif "fx" in path_lower or "effect" in path_lower:
            return "fx"
        else:
            return "other"

    def extract_features_lazy(self, audio_path: str) -> Dict:
        """Extract features only when needed (lazy loading)."""
        if audio_path in self.feature_cache:
            return self.feature_cache[audio_path]
        
        try:
            y, sr = librosa.load(audio_path, duration=10)  # Only load 10 seconds for speed
            
            if len(y) < 2048:
                return None
            
            n_fft = min(2048, len(y))
            hop_length = n_fft // 4
            
            # Quick feature extraction
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length))
            rms = np.mean(librosa.feature.rms(y=y, hop_length=hop_length))
            
            features = {
                "tempo": float(tempo),
                "spectral_centroid": float(spectral_centroid),
                "energy": float(rms),
            }
            
            # Cache it
            self.feature_cache[audio_path] = features
            self.sample_index[audio_path]["analyzed"] = True
            
            return features
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")
            return None

    def match_prompt_to_samples(self, prompt: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        """Match prompt to samples using category + lazy feature extraction."""
        prompt_lower = prompt.lower()
        
        # First pass: category filtering
        category_matches = []
        for path, metadata in self.sample_index.items():
            category = metadata.get("category", "other")
            
            # Check if prompt mentions this category
            if category in prompt_lower or any(keyword in prompt_lower for keyword in ["drum", "bass", "synth", "ambient", "fx"]):
                if category in prompt_lower or category == "other":
                    category_matches.append(path)
        
        # If no category matches, use all samples
        if not category_matches:
            category_matches = list(self.sample_index.keys())
        
        # Limit to avoid over-analyzing
        category_matches = category_matches[:20]
        
        # Second pass: lazy feature extraction for matched samples
        scored_samples = []
        for path in category_matches:
            features = self.extract_features_lazy(path)
            if features:
                score = self._score_sample(prompt_lower, features)
                scored_samples.append((path, score, features))
        
        # Sort and return top matches
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        return [(path, features) for path, _, features in scored_samples[:top_k]]

    def _score_sample(self, prompt: str, features: Dict) -> float:
        """Score a sample based on prompt keywords."""
        score = 0
        
        # Tempo matching
        if "slow" in prompt and features["tempo"] < 90:
            score += 2
        elif "fast" in prompt and features["tempo"] > 130:
            score += 2
        elif "medium" in prompt and 90 <= features["tempo"] <= 130:
            score += 2
        
        # Energy matching
        if "energetic" in prompt or "aggressive" in prompt:
            if features["energy"] > 0.1:
                score += 1
        elif "calm" in prompt or "ambient" in prompt:
            if features["energy"] < 0.05:
                score += 1
        
        # Spectral matching
        if "bright" in prompt and features["spectral_centroid"] > 2000:
            score += 1
        elif "dark" in prompt and features["spectral_centroid"] < 1000:
            score += 1
        
        return score

    def get_prompt_enhancement(self, prompt: str) -> str:
        """Enhance prompt based on matched samples."""
        matches = self.match_prompt_to_samples(prompt, top_k=3)
        
        if not matches:
            return prompt
        
        # Aggregate features
        avg_tempo = np.mean([f["tempo"] for _, f in matches])
        avg_energy = np.mean([f["energy"] for _, f in matches])
        
        enhancements = []
        if avg_tempo < 90:
            enhancements.append("slow tempo")
        elif avg_tempo > 130:
            enhancements.append("fast tempo")
        
        if avg_energy > 0.1:
            enhancements.append("high energy")
        elif avg_energy < 0.05:
            enhancements.append("calm")
        
        enhanced = f"{prompt}, {', '.join(enhancements)}" if enhancements else prompt
        return enhanced
