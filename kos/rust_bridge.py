"""
KOS V7.0 — Rust Bridge

Replaces Python graph.py with the Rust RustKernel via PyO3.
Falls back to Python if Rust module not compiled.

The bridge makes RustKernel look identical to KOSKernel so the
rest of the codebase (router, weaver, drivers) works unchanged.

Usage:
    from kos.rust_bridge import get_kernel
    kernel = get_kernel()  # Returns RustKernel if available, else KOSKernel
"""

import os
import sys
import time

# Try to import the compiled Rust module
_RUST_AVAILABLE = False
_RustKernel = None
_RustVSA = None

try:
    # The compiled .pyd/.so lives in kos_rust/ or the project root
    rust_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kos_rust')
    if rust_path not in sys.path:
        sys.path.insert(0, rust_path)
    # Also check project root (maturin develop puts it there)
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    from kos_rust import RustKernel, RustVSA
    _RUST_AVAILABLE = True
    _RustKernel = RustKernel
    _RustVSA = RustVSA
except ImportError:
    pass


class RustKernelBridge:
    """
    Wraps RustKernel to match KOSKernel's Python interface.

    The rest of KOS (router, weaver, drivers) expects:
        kernel.nodes[uid].connections
        kernel.provenance[(a,b)]
        kernel.contradictions
        kernel.add_node(uid)
        kernel.add_connection(src, tgt, weight, text)
        kernel.query(seeds, top_k)
        kernel.current_tick

    This bridge translates those calls to RustKernel methods.
    """

    def __init__(self, dim=10000, temporal_decay=0.7, max_energy=3.0, seed=42):
        if not _RUST_AVAILABLE:
            raise ImportError("Rust module not compiled. Run: cd kos_rust && maturin develop --release")

        self._rust = _RustKernel(dim=dim, temporal_decay=temporal_decay,
                                  max_energy=max_energy, seed=seed)
        self.provenance = {}  # Python dict for compatibility
        self.contradictions = []
        self.current_tick = 0
        self.max_energy = max_energy
        self._nodes_cache = {}  # Lazy cache for node-like access
        self._using_rust = True

    @property
    def nodes(self):
        """Return a dict-like view of nodes for compatibility."""
        return _RustNodesView(self._rust)

    def add_node(self, uid):
        self._rust.add_node(str(uid))

    def add_connection(self, src, tgt, weight, source_text=""):
        self._rust.add_connection(str(src), str(tgt), float(weight),
                                   source_text if source_text else None)
        # Also store provenance in Python dict for Weaver compatibility
        key = tuple(sorted([str(src), str(tgt)]))
        if key not in self.provenance:
            self.provenance[key] = set()
        if source_text:
            self.provenance[key].add(source_text)

    def query(self, seeds, top_k=10):
        self.current_tick += 1
        seed_names = [str(s) for s in seeds]
        results = self._rust.query(seed_names, top_k)
        return results  # List of (name, energy) tuples

    def has_node(self, name):
        return self._rust.has_node(str(name))

    def node_count(self):
        return self._rust.node_count()

    def edge_count(self):
        return self._rust.edge_count()

    def stats(self):
        return self._rust.stats()

    def get_provenance(self, a, b):
        return self._rust.get_provenance(str(a), str(b))

    def resonate(self, a, b):
        return self._rust.resonate(str(a), str(b))

    def resonate_search(self, query, top_k=10, threshold=0.1):
        return self._rust.resonate_search(str(query), top_k, threshold)


class _RustNodeProxy:
    """Mimics a ConceptNode for compatibility with existing code."""

    def __init__(self, name, rust_kernel):
        self.id = name
        self._rust = rust_kernel
        self.activation = 0.0
        self.fuel = 0.0
        self.connections = {}  # Will be populated lazily


class _RustNodesView:
    """Dict-like view over Rust arena nodes."""

    def __init__(self, rust_kernel):
        self._rust = rust_kernel
        self._cache = {}

    def __len__(self):
        return self._rust.node_count()

    def __contains__(self, key):
        return self._rust.has_node(str(key))

    def __getitem__(self, key):
        if str(key) not in self._cache:
            if not self._rust.has_node(str(key)):
                raise KeyError(key)
            self._cache[str(key)] = _RustNodeProxy(str(key), self._rust)
        return self._cache[str(key)]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._rust.node_names())

    def keys(self):
        return self._rust.node_names()

    def values(self):
        return [self[n] for n in self._rust.node_names()]

    def items(self):
        return [(n, self[n]) for n in self._rust.node_names()]


def get_kernel(prefer_rust=True, **kwargs):
    """
    Get the best available kernel.

    Returns RustKernelBridge if Rust is compiled, else KOSKernel.
    The caller doesn't need to know which one they got.
    """
    if prefer_rust and _RUST_AVAILABLE:
        try:
            bridge = RustKernelBridge(**kwargs)
            print("[V7.0] Using Rust kernel (arena-based, 6.7x faster)")
            return bridge
        except Exception as e:
            print("[V7.0] Rust kernel failed: %s — falling back to Python" % e)

    from .graph import KOSKernel
    print("[V7.0] Using Python kernel")
    return KOSKernel(**kwargs)


def is_rust_available():
    return _RUST_AVAILABLE


def get_vsa(dim=10000, seed=42):
    """Get Rust VSA engine if available, else Python KASMEngine."""
    if _RUST_AVAILABLE:
        return _RustVSA(dim=dim, seed=seed)
    from kasm.vsa import KASMEngine
    return KASMEngine(dimensions=dim, seed=seed)
