"""
FX Bus API - Manage vocal and beat mastering channels
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from audio_engine import AudioEngine

router = APIRouter()
audio_engine = AudioEngine()


class EffectConfig(BaseModel):
    """Effect configuration"""
    type: str  # compressor, reverb, eq, etc.
    parameters: dict


class FXBusConfig(BaseModel):
    """FX Bus configuration"""
    enabled: bool = True
    gain_db: float = 0.0
    mute: bool = False
    effects: List[EffectConfig] = []


class FXBusStatus(BaseModel):
    """FX Bus status response"""
    name: str
    channel_type: str
    enabled: bool
    gain_db: float
    mute: bool
    effect_count: int


class TrackProcessRequest(BaseModel):
    """Request to process track through FX bus"""
    audio_data: List[List[float]]  # Interleaved audio samples
    sample_rate: int
    track_type: Literal["vocal", "beat", "other"] = "beat"


@router.get("/fx-buses")
async def get_fx_buses() -> dict:
    """Get all FX buses status"""
    return {
        "vocal": FXBusStatus(
            name=audio_engine.vocal_bus.name,
            channel_type=audio_engine.vocal_bus.channel_type,
            enabled=audio_engine.vocal_bus.enabled,
            gain_db=audio_engine.vocal_bus.gain,
            mute=audio_engine.vocal_bus.mute,
            effect_count=len(audio_engine.vocal_bus.effects_chain)
        ),
        "beat": FXBusStatus(
            name=audio_engine.beat_bus.name,
            channel_type=audio_engine.beat_bus.channel_type,
            enabled=audio_engine.beat_bus.enabled,
            gain_db=audio_engine.beat_bus.gain,
            mute=audio_engine.beat_bus.mute,
            effect_count=len(audio_engine.beat_bus.effects_chain)
        ),
        "master": FXBusStatus(
            name=audio_engine.master_bus.name,
            channel_type=audio_engine.master_bus.channel_type,
            enabled=audio_engine.master_bus.enabled,
            gain_db=audio_engine.master_bus.gain,
            mute=audio_engine.master_bus.mute,
            effect_count=len(audio_engine.master_bus.effects_chain)
        )
    }


@router.get("/fx-buses/{bus_type}")
async def get_fx_bus(bus_type: Literal["vocal", "beat", "master"]) -> FXBusStatus:
    """Get specific FX bus status"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    return FXBusStatus(
        name=bus.name,
        channel_type=bus.channel_type,
        enabled=bus.enabled,
        gain_db=bus.gain,
        mute=bus.mute,
        effect_count=len(bus.effects_chain)
    )


@router.post("/fx-buses/{bus_type}/toggle")
async def toggle_fx_bus(bus_type: Literal["vocal", "beat", "master"]) -> dict:
    """Toggle FX bus on/off"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    new_state = bus.toggle()
    return {"bus": bus_type, "enabled": new_state}


@router.post("/fx-buses/{bus_type}/mute")
async def mute_fx_bus(bus_type: Literal["vocal", "beat", "master"], mute: bool) -> dict:
    """Mute/unmute FX bus"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    bus.mute = mute
    return {"bus": bus_type, "mute": mute}


@router.post("/fx-buses/{bus_type}/gain")
async def set_fx_bus_gain(bus_type: Literal["vocal", "beat", "master"], gain_db: float) -> dict:
    """Set FX bus output gain"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    bus.set_gain(gain_db)
    return {"bus": bus_type, "gain_db": bus.gain}


@router.post("/fx-buses/{bus_type}/config")
async def configure_fx_bus(bus_type: Literal["vocal", "beat", "master"], config: FXBusConfig) -> dict:
    """Configure FX bus"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    bus.enabled = config.enabled
    bus.mute = config.mute
    bus.set_gain(config.gain_db)
    
    # Clear existing effects and add new ones
    bus.effects_chain = []
    for effect_config in config.effects:
        try:
            if effect_config.type == "compressor":
                from pedalboard import Compressor
                params = effect_config.parameters
                compressor = Compressor(
                    threshold_db=params.get("threshold_db", -20),
                    ratio=params.get("ratio", 4),
                    attack_ms=params.get("attack_ms", 5),
                    release_ms=params.get("release_ms", 50)
                )
                bus.add_effect(compressor)
            elif effect_config.type == "reverb":
                from pedalboard import Reverb
                params = effect_config.parameters
                reverb = Reverb(
                    room_scale=params.get("room_scale", 0.5),
                    damping=params.get("damping", 0.5),
                    wet_level=params.get("wet_level", 0.2),
                    dry_level=params.get("dry_level", 0.8)
                )
                bus.add_effect(reverb)
            elif effect_config.type == "eq":
                from pedalboard import Equalizer
                params = effect_config.parameters
                eq = Equalizer(
                    low_gain_db=params.get("low_gain_db", 0),
                    mid_gain_db=params.get("mid_gain_db", 0),
                    high_gain_db=params.get("high_gain_db", 0)
                )
                bus.add_effect(eq)
        except Exception as e:
            print(f"Error adding {effect_config.type}: {e}")
    
    return {"bus": bus_type, "configured": True, "effect_count": len(bus.effects_chain)}


@router.post("/fx-buses/{bus_type}/add-effect")
async def add_effect_to_bus(bus_type: Literal["vocal", "beat", "master"], effect: EffectConfig) -> dict:
    """Add effect to FX bus"""
    bus = audio_engine.get_fx_bus(bus_type)
    if not bus:
        raise HTTPException(status_code=404, detail=f"FX Bus '{bus_type}' not found")
    
    try:
        if effect.type == "compressor":
            from pedalboard import Compressor
            compressor = Compressor(**effect.parameters)
            bus.add_effect(compressor)
        elif effect.type == "reverb":
            from pedalboard import Reverb
            reverb = Reverb(**effect.parameters)
            bus.add_effect(reverb)
        elif effect.type == "eq":
            from pedalboard import Equalizer
            eq = Equalizer(**effect.parameters)
            bus.add_effect(eq)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add effect: {str(e)}")
    
    return {"bus": bus_type, "effect_added": effect.type, "effect_count": len(bus.effects_chain)}
