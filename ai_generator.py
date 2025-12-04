import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import numpy as np
import io
import random
import librosa
import soundfile as sf

class AIGenerator:
    def __init__(self):
        print("Loading MusicGen model...")
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            attn_implementation="eager"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Model loaded on CUDA (float32)")
        else:
            print("Model loaded on CPU")

    def generate(self, prompt: str, duration: int = 5):
        # Check if prompt already specifies a BPM
        import re
        bpm_match = re.search(r'(\d{1,3})\s*BPM', prompt, re.IGNORECASE)
        
        tempo_descriptors = {
            1: "ultra-slow, ambient, minimal",
            10: "extremely slow, experimental",
            20: "very slow, drone-like",
            30: "very slow, sparse",
            40: "slow, relaxed",
            50: "slow, contemplative",
            60: "slow, calm tempo",
            70: "very slow, downtempo",
            80: "slow, relaxed tempo",
            90: "slow, laid-back tempo",
            100: "moderate, steady tempo",
            110: "moderate, upbeat tempo",
            120: "energetic, driving tempo",
            130: "fast, energetic tempo",
            140: "fast, intense tempo",
            150: "very fast, high-energy tempo",
            160: "extremely fast, intense high-energy",
            170: "extremely fast, hardcore tempo",
            180: "ultra-fast, breakcore-like tempo",
            190: "ultra-fast, extreme tempo",
            200: "maximum speed, extreme high-energy"
        }
        
        if bpm_match:
            # User specified a BPM - respect their choice
            bpm = int(bpm_match.group(1))
            tempo_phrase = tempo_descriptors.get(bpm, "steady tempo")
            enhanced_prompt = prompt
            print(f"[GENERATE] Original prompt: {prompt}")
            print(f"[GENERATE] User-specified BPM: {bpm}")
            print(f"[GENERATE] Using prompt as-is: {enhanced_prompt}")
        else:
            # No BPM in prompt - add random one for variety
            bpm = random.choice([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
            tempo_phrase = tempo_descriptors.get(bpm, "steady tempo")
            enhanced_prompt = f"{prompt} at {bpm} BPM with {tempo_phrase}"
            print(f"[GENERATE] Original prompt: {prompt}")
            print(f"[GENERATE] Random BPM selected: {bpm}")
            print(f"[GENERATE] Enhanced prompt: {enhanced_prompt}")
        
        inputs = self.processor(
            text=[enhanced_prompt],
            padding=True,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate audio with optimized parameters
        # max_new_tokens: ~50 tokens per second of audio
        # do_sample=True: Faster than beam search and more creative
        # guidance_scale=3.0: Good balance of adherence to prompt and speed
        with torch.inference_mode(): # Disable gradient calculation for speed
            audio_values = self.model.generate(
                **inputs, 
                max_new_tokens=int(duration * 50),
                do_sample=True,
                guidance_scale=3.0,
                temperature=1.0
            )
        
        # Convert to numpy
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        # Convert back to float32 for audio processing if needed
        audio_data = audio_values[0, 0].cpu().float().numpy()
        
        print(f"[GENERATE] Audio data shape: {audio_data.shape}")
        print(f"[GENERATE] Audio data type: {audio_data.dtype}")
        print(f"[GENERATE] Audio data min/max before normalization: {np.min(audio_data)}/{np.max(audio_data)}")
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            print(f"[GENERATE] Audio normalized by max value: {max_val}")
        else:
            print(f"[GENERATE] WARNING: Audio data is all zeros or invalid!")
            # Create minimal valid audio to prevent empty files
            audio_data = np.zeros(sampling_rate * duration, dtype=np.float32) + 0.001
        
        # Convert to int16 for WAV file (more compatible than float32)
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        # Save to bytes
        bytes_io = io.BytesIO()
        scipy.io.wavfile.write(bytes_io, rate=sampling_rate, data=audio_int16)
        bytes_io.seek(0)
        
        final_size = bytes_io.getbuffer().nbytes
        print(f"[GENERATE] Final WAV file size: {final_size} bytes")
        
        if final_size < 100:
            print(f"[GENERATE] WARNING: Generated file is suspiciously small ({final_size} bytes)")
        
        bytes_io.seek(0)
        return bytes_io
    
    def remix(self, audio_data: np.ndarray, sr: int, genre: str, mood: str, tempo: int = 100, pitch: int = 0, intensity: int = 50, custom_prompt: str = None, duration: int = None):
        """
        Remix audio by:
        1. Separating vocals from instrumental
        2. Generating new beat with MusicGen
        3. Adjusting tempo/pitch
        4. Mixing vocals + new beat
        """
        try:
            print(f"[REMIX] Starting remix: genre={genre}, mood={mood}, tempo={tempo}%, pitch={pitch}, intensity={intensity}%")
            print(f"[REMIX] Input audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
            # Use original duration if not specified
            if duration is None:
                duration = len(audio_data) / sr
            
            # Step 1: Use original audio as vocals (skip HPSS for speed)
            # HPSS is slow; just use the input audio directly as "vocals"
            print(f"[REMIX] Using input audio as vocals (skipping HPSS for speed)")
            print(f"[REMIX] Input audio: {len(audio_data)} samples at {sr}Hz, max: {np.max(np.abs(audio_data)):.4f}")
            vocals = audio_data
            
            # Step 2: Generate new beat
            print(f"[REMIX] Generating new beat with MusicGen...")
            beat = self._generate_beat(genre, mood, duration, sr, intensity, custom_prompt)
            
            if beat is None or len(beat) == 0 or np.max(np.abs(beat)) < 0.001:
                print(f"[REMIX] ERROR: Beat generation failed or returned silent audio")
                print(f"[REMIX] Beat generation must produce actual audio for remix to work")
                raise Exception("Beat generation failed - cannot remix without generated beat. Check MusicGen model and GPU/CPU resources.")
            
            # Step 3: Adjust tempo and pitch of beat
            if tempo != 100:
                print(f"[REMIX] Adjusting tempo to {tempo}%...")
                rate = tempo / 100.0
                beat = librosa.effects.time_stretch(beat, rate=rate)
            
            if pitch != 0:
                print(f"[REMIX] Adjusting pitch by {pitch} semitones...")
                beat = librosa.effects.pitch_shift(beat, sr=sr, n_steps=pitch)
            
            # Step 4: Synchronize lengths and mix
            print(f"[REMIX] Synchronizing and mixing...")
            
            # Ensure same length
            beat = np.asarray(beat, dtype=np.float32)
            vocals = np.asarray(vocals, dtype=np.float32)
            
            vocal_length = len(vocals)
            beat_length = len(beat)
            
            if beat_length > vocal_length:
                beat = beat[:vocal_length]
            elif beat_length < vocal_length:
                beat = np.pad(beat, (0, vocal_length - beat_length), mode='constant')
            
            # Dynamic mixing based on intensity (0-100)
            # Intensity 0: 100% Original, 0% Beat
            # Intensity 50: 50% Original, 50% Beat
            # Intensity 100: 0% Original, 100% Beat
            
            # Map intensity 0-100 to mix ratio 0.0-1.0
            # We clamp it slightly to ensure we never lose either component completely unless extreme
            mix_ratio = intensity / 100.0
            
            # Apply non-linear curve for better feel (optional, but linear is predictable)
            # mix_ratio = mix_ratio  # Linear for now
            
            original_weight = 1.0 - mix_ratio
            beat_weight = mix_ratio
            
            # Boost beat slightly as it tends to be quieter than mastered tracks
            beat_weight = min(1.0, beat_weight * 1.2)
            
            print(f"[REMIX] Mixing - Intensity: {intensity}%")
            print(f"[REMIX] Weights - Original: {original_weight:.2f}, Beat: {beat_weight:.2f}")
            
            # Normalize inputs before mixing to ensure consistent ratios
            beat_max = np.max(np.abs(beat))
            vocals_max = np.max(np.abs(vocals))
            
            print(f"[REMIX] DEBUG: Beat max amplitude: {beat_max:.4f}")
            print(f"[REMIX] DEBUG: Vocals max amplitude: {vocals_max:.4f}")
            
            if beat_max > 0:
                beat = beat / beat_max
            if vocals_max > 0:
                vocals = vocals / vocals_max
                
            # Apply weights
            beat_mixed = beat * beat_weight
            vocals_mixed = vocals * original_weight
            
            print(f"[REMIX] DEBUG: Beat weight: {beat_weight:.2f}, Vocals weight: {original_weight:.2f}")
            print(f"[REMIX] DEBUG: Beat mixed max: {np.max(np.abs(beat_mixed)):.4f}")
            print(f"[REMIX] DEBUG: Vocals mixed max: {np.max(np.abs(vocals_mixed)):.4f}")
            
            remixed = beat_mixed + vocals_mixed
            
            print(f"[REMIX] After mixing - Remixed max: {np.max(np.abs(remixed)):.4f}")
            
            # Normalize final mix
            max_val = np.max(np.abs(remixed))
            if max_val > 1.0:
                remixed = remixed / max_val * 0.95
            
            print(f"[REMIX] After final normalization - Remixed max: {np.max(np.abs(remixed)):.4f}")
            
            # VALIDATION: Verify remix is actually different from original
            if np.max(np.abs(remixed)) < 0.001:
                raise Exception("Remix resulted in silent or near-silent audio - beat generation likely failed")
            
            # Compare with original to ensure remix happened
            original_max = np.max(np.abs(audio_data))
            remixed_max = np.max(np.abs(remixed))
            similarity = min(original_max, remixed_max) / max(original_max, remixed_max) if max(original_max, remixed_max) > 0 else 1.0
            
            if similarity > 0.95:
                print(f"[REMIX] WARNING: Remixed audio is very similar to original (similarity: {similarity:.2%})")
                print(f"[REMIX] Original max: {original_max:.4f}, Remixed max: {remixed_max:.4f}")
            
            print(f"[REMIX] Remix complete: 95% beat + 5% vocals")
            print(f"[REMIX] Original peak: {original_max:.4f}, Remixed peak: {remixed_max:.4f}")
            
            # Convert to int16 and save
            audio_int16 = np.clip(remixed * 32767, -32768, 32767).astype(np.int16)
            
            print(f"[REMIX] Final audio to save: {len(audio_int16)} samples, max: {np.max(np.abs(audio_int16)):.0f}")
            
            bytes_io = io.BytesIO()
            scipy.io.wavfile.write(bytes_io, rate=sr, data=audio_int16)
            bytes_io.seek(0)
            
            final_size = bytes_io.getbuffer().nbytes
            print(f"[REMIX] Saved WAV file: {final_size} bytes")
            return bytes_io
            
        except Exception as e:
            print(f"[REMIX] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_beat(self, genre: str, mood: str, duration: float, sr: int, intensity: int = 50, custom_prompt: str = None):
        """Generate a beat using MusicGen for remix"""
        try:
            # Use custom prompt if provided
            if custom_prompt and custom_prompt.strip():
                enhanced_prompt = custom_prompt.strip()
            else:
                bpm = random.choice([90, 100, 110, 120])
                enhanced_prompt = f"{genre.lower()} {mood.lower()} beat at {bpm} BPM, drum pattern, bass line"
            
            print(f"[REMIX] Beat prompt: {enhanced_prompt}")
            
            # Cap duration for speed - generate only 20 seconds, then loop it
            safe_duration = 20
            
            inputs = self.processor(
                text=[enhanced_prompt],
                padding=True,
                return_tensors="pt",
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.inference_mode():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=int(safe_duration * 50),
                    do_sample=True,
                    guidance_scale=9.0,
                    temperature=0.9,
                    use_cache=True
                )
            
            # Convert to numpy
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            audio_data = audio_values[0, 0].cpu().float().numpy()
            
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            else:
                audio_data = np.zeros(sampling_rate * safe_duration, dtype=np.float32) + 0.001
            
            print(f"[REMIX] Beat generated - shape: {audio_data.shape}, max: {np.max(np.abs(audio_data)):.4f}")
            
            # Resample if needed
            if sr != sampling_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=sr)
            
            # Ensure correct length
            target_samples = int(safe_duration * sr)
            if len(audio_data) > target_samples:
                audio_data = audio_data[:target_samples]
            else:
                audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
            
            # Loop to extend if duration is different from generated
            target_samples = int(duration * sr)
            if len(audio_data) > target_samples:
                audio_data = audio_data[:target_samples]
            elif len(audio_data) < target_samples:
                print(f"[REMIX] Looping beat to fill duration (Target: {duration}s)")
                num_repeats = int(np.ceil(target_samples / len(audio_data)))
                audio_data = np.tile(audio_data, num_repeats)
                audio_data = audio_data[:target_samples]
            
            # Apply intensity
            intensity_factor = 0.5 + (intensity / 100)
            audio_data = audio_data * intensity_factor
            
            print(f"[REMIX] Beat final: {len(audio_data)} samples, max: {np.max(np.abs(audio_data)):.4f}")
            return audio_data
            
        except Exception as e:
            print(f"[REMIX] Beat generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
