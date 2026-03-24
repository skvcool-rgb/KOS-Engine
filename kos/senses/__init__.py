"""KOS Senses — Eyes, Ears, Mouth, and Perception Loop."""

try:
    from .ears import Ears
except ImportError:
    Ears = None

try:
    from .eyes import Eyes
except ImportError:
    Eyes = None

try:
    from .mouth import Mouth
except ImportError:
    Mouth = None

from .perception import PerceptionLoop, SensoryPrediction, MultimodalBinder, EmotionGrounding

__all__ = [
    "Ears", "Eyes", "Mouth",
    "PerceptionLoop", "SensoryPrediction", "MultimodalBinder", "EmotionGrounding",
]
