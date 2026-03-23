# Lazy imports to avoid crashes when optional dependencies are missing
try:
    from .text import TextDriver, KOSResolver
except ImportError:
    TextDriver = None
    KOSResolver = None

try:
    from .vision import VisionDriver
except ImportError:
    VisionDriver = None

try:
    from .ast import ASTDriver
except ImportError:
    ASTDriver = None

try:
    from .math import MathDriver
except ImportError:
    MathDriver = None

__all__ = ['TextDriver', 'KOSResolver', 'VisionDriver', 'ASTDriver',
           'MathDriver']
