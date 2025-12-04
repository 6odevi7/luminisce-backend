from pedalboard import load_plugin, Compressor, Reverb
import numpy as np
from typing import Literal

class FXBus:
    """Individual FX bus for mastering chain"""
    def __init__(self, name: str, channel_type: Literal["vocal", "beat"]):
        self.name = name
        self.channel_type = channel_type
        self.vst = None
        self.effects_chain = []
        self.enabled = True
        self.gain = 0.0  # dB
        self.mute = False
        
    def add_effect(self, effect):
        """Add effect to chain"""
        self.effects_chain.append(effect)
        print(f"[{self.name}] Added effect: {type(effect).__name__}")
        
    def remove_effect(self, index: int):
        """Remove effect from chain"""
        if 0 <= index < len(self.effects_chain):
            removed = self.effects_chain.pop(index)
            print(f"[{self.name}] Removed effect: {type(removed).__name__}")
            
    def load_vst(self, path: str):
        """Load VST plugin for this bus"""
        try:
            self.vst = load_plugin(path)
            print(f"[{self.name}] Loaded VST: {self.vst.name}")
        except Exception as e:
            print(f"[{self.name}] Failed to load VST: {e}")
            
    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio through effects chain"""
        if self.mute or not self.enabled:
            return audio_data
            
        output = audio_data.copy()
        
        # Apply effects chain
        for effect in self.effects_chain:
            try:
                output = effect(output, sample_rate)
            except Exception as e:
                print(f"[{self.name}] Error processing effect: {e}")
                
        # Apply VST if loaded
        if self.vst:
            try:
                output = self.vst(output, sample_rate)
            except Exception as e:
                print(f"[{self.name}] Error processing VST: {e}")
                
        # Apply gain
        if self.gain != 0:
            gain_linear = 10 ** (self.gain / 20.0)
            output = output * gain_linear
            
        return output
        
    def set_gain(self, gain_db: float):
        """Set output gain in dB"""
        self.gain = np.clip(gain_db, -48, 24)
        
    def toggle(self, enabled: bool = None):
        """Toggle bus on/off"""
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = not self.enabled
        return self.enabled


class AudioEngine:
    def __init__(self):
        self.vst = None
        self.vocal_bus = FXBus("Vocal Bus", "vocal")
        self.beat_bus = FXBus("Beat Bus", "beat")
        self.master_bus = FXBus("Master Bus", "master")
        
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10]
        }
        
        # Initialize default mastering chain for master bus
        self._init_default_mastering()

    def _init_default_mastering(self):
        """Initialize default mastering effects"""
        # Master bus gets subtle EQ and compression
        try:
            self.master_bus.add_effect(Compressor(threshold_db=-20, ratio=4))
        except:
            pass

    def load_vst(self, path: str):
        try:
            self.vst = load_plugin(path)
            print(f"Loaded VST: {self.vst.name}")
        except Exception as e:
            print(f"Failed to load VST at {path}: {e}")

    def get_scale_notes(self, root_note: int, scale_type: str = "minor"):
        """
        Returns MIDI note numbers for a given scale.
        root_note: MIDI note number (e.g., 60 for Middle C)
        """
        intervals = self.scales.get(scale_type, self.scales["minor"])
        return [root_note + interval for interval in intervals]

    def process_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Apply VST effects to audio data.
        """
        if self.vst:
            return self.vst(audio_data, sample_rate)
        return audio_data
        
    def process_track(self, audio_data: np.ndarray, sample_rate: int, track_type: str = "beat") -> np.ndarray:
        """Process audio through appropriate FX bus"""
        if track_type == "vocal":
            output = self.vocal_bus.process_audio(audio_data, sample_rate)
        elif track_type == "beat":
            output = self.beat_bus.process_audio(audio_data, sample_rate)
        else:
            output = audio_data.copy()
            
        # Route through master bus
        output = self.master_bus.process_audio(output, sample_rate)
        return output
        
    def get_fx_bus(self, bus_type: str) -> FXBus:
        """Get FX bus by type"""
        if bus_type == "vocal":
            return self.vocal_bus
        elif bus_type == "beat":
            return self.beat_bus
        elif bus_type == "master":
            return self.master_bus
        return None
