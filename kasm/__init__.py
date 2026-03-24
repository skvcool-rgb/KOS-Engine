"""
KASM — Knowledge Assembly: A Domain-Specific Language for Vector Symbolic Architectures.

The SQL of Topological Algebra.

Usage:
    from kasm import KASMEngine, KASMInterpreter

    # Direct API
    engine = KASMEngine(dimensions=10000, seed=42)
    a = engine.node("sun")
    b = engine.node("center")
    c = engine.bind(a, b)

    # DSL Interpreter
    interpreter = KASMInterpreter()
    interpreter.run_file("example.kasm")
"""

from .vsa import KASMEngine
from .interpreter import KASMInterpreter

__all__ = ["KASMEngine", "KASMInterpreter"]
