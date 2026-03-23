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
from .drivers.text import TextDriver, KOSResolver
from .drivers.math import MathDriver
from .drivers.ast import ASTDriver
from .drivers.vision import VisionDriver

__version__ = "4.1.0"
__all__ = [
    "ConceptNode", "KOSKernel", "KASMLexicon", "AlgorithmicWeaver",
    "KOSShell", "KOSDaemon", "TextDriver", "KOSResolver",
    "MathDriver", "ASTDriver", "VisionDriver",
]
