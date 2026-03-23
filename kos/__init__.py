"""
KOS (Knowledge Operating System) — Neurosymbolic Knowledge Engine.

A spreading activation engine with biological neuron physics that
confines LLMs to thin I/O while routing all reasoning through a
deterministic graph. Zero hallucination, deterministic evidence
scoring, 6-layer typo recovery.
"""

from .node import ConceptNode
from .graph import KOSKernel
from .lexicon import KASMLexicon
from .weaver import AlgorithmicWeaver
from .router import KOSShell
from .daemon import KOSDaemon

# Lazy driver imports — avoid hard crashes if optional deps are missing
try:
    from .drivers.text import TextDriver, KOSResolver
except ImportError:
    TextDriver = KOSResolver = None
try:
    from .drivers.math import MathDriver
except ImportError:
    MathDriver = None
try:
    from .drivers.ast import ASTDriver
except ImportError:
    ASTDriver = None
try:
    from .drivers.vision import VisionDriver
except ImportError:
    VisionDriver = None

__version__ = "4.1.0"
__all__ = [
    "ConceptNode", "KOSKernel", "KASMLexicon", "AlgorithmicWeaver",
    "KOSShell", "KOSDaemon", "TextDriver", "KOSResolver",
    "MathDriver", "ASTDriver", "VisionDriver",
]
