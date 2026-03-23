"""
KASM VSA Engine — Phase 1: NumPy Virtual Machine

Implements the four fundamental operations of Vector Symbolic Architecture
(Multiply-Add-Permute / MAP encoding) using 10,000-dimensional bipolar vectors.

Mathematical foundation: Kanerva (2009), "Hyperdimensional Computing"
Architecture: MAP (Multiply-Add-Permute) bipolar vectors {-1, +1}^D

Operations:
    NODE      — Generate random pseudo-orthogonal bipolar vector
    BIND (*)  — Element-wise multiplication (XOR analog for bipolar)
    SUPERPOSE (+) — Element-wise addition + threshold (bundling)
    PERMUTE (ρ)  — Circular shift (sequence encoding)
    RESONATE (<=>)  — Cosine similarity (O(1) approximate match)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class KASMEngine:
    """
    The KASM Virtual Machine.

    All concepts exist as D-dimensional bipolar vectors in {-1, +1}^D.
    Default D=10000 guarantees pseudo-orthogonality between random vectors
    (expected cosine similarity ≈ 0.0 ± 0.01).
    """

    __slots__ = ['D', 'symbols', '_rng']

    def __init__(self, dimensions: int = 10_000, seed: Optional[int] = None):
        self.D = dimensions
        self.symbols: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    # ── NODE: Instantiation ──────────────────────────────────────────

    def node(self, name: str) -> np.ndarray:
        """
        Spawn a new atomic concept as a random bipolar vector.

        Math: v ∈ {-1, +1}^D, each element iid Rademacher(0.5)
        Property: E[cos(v_i, v_j)] = 0 for i ≠ j (pseudo-orthogonal)
        """
        vec = self._rng.choice([-1, 1], size=self.D).astype(np.int8)
        self.symbols[name] = vec
        return vec

    def node_batch(self, *names: str) -> Dict[str, np.ndarray]:
        """Spawn multiple nodes at once."""
        return {name: self.node(name) for name in names}

    # ── BIND (*): Association ────────────────────────────────────────

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Element-wise multiplication of two bipolar vectors.

        Math: c_i = a_i * b_i  (equivalent to XOR for binary encoding)

        Properties:
            - bind(a, b) is orthogonal to both a and b
            - bind(a, bind(a, b)) = b  (self-inverse: a is its own unbinding key)
            - Commutative: bind(a, b) = bind(b, a)
            - Distributes over superpose
        """
        return (a * b).astype(np.int8)

    # ── SUPERPOSE (+): Bundling ──────────────────────────────────────

    @staticmethod
    def superpose(*vectors: np.ndarray) -> np.ndarray:
        """
        Element-wise addition + majority threshold back to {-1, +1}.

        Math: s_i = sign(Σ v_k_i), with ties broken randomly

        Properties:
            - superpose(a, b, c) is similar to each of a, b, c
            - Capacity: ~D/2 vectors before noise floor overwhelms signal
            - This is how KASM creates "contexts" or "sets"
        """
        summed = np.sum(vectors, axis=0).astype(np.float32)
        # Break ties (zeros) randomly to maintain bipolar
        ties = summed == 0
        result = np.sign(summed).astype(np.int8)
        if np.any(ties):
            result[ties] = np.random.choice([-1, 1], size=int(ties.sum())).astype(np.int8)
        return result

    # ── PERMUTE (ρ): Sequence ────────────────────────────────────────

    @staticmethod
    def permute(v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """
        Circular shift of vector elements.

        Math: ρ(v)_i = v_{(i-shifts) mod D}

        Properties:
            - permute(v) is orthogonal to v (encodes position/order)
            - "dog bites man" ≠ "man bites dog" because shifted vectors differ
            - Invertible: permute(v, -shifts) recovers original
        """
        return np.roll(v, shifts)

    # ── RESONATE (<=>): Similarity Query ─────────────────────────────

    @staticmethod
    def resonate(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two bipolar vectors.

        Math: cos(a, b) = (a · b) / (||a|| * ||b||)

        For bipolar vectors of same dimension:
            cos(a, b) = (a · b) / D

        Complexity: O(D) — but D is fixed, so effectively O(1) per query.

        Interpretation:
            ≈ 0.00: Unrelated (orthogonal)
            > 0.15: Weak similarity
            > 0.30: Strong structural match
            > 0.50: Very high overlap
            = 1.00: Identical
        """
        dot = int(np.dot(a.astype(np.int32), b.astype(np.int32)))
        return dot / len(a)

    # ── UNBIND: Inverse query ────────────────────────────────────────

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Query a composite vector with a key to recover the bound partner.

        Math: unbind(bind(a, b), a) = b  (because a * a = 1 for bipolar)

        This is the mechanism behind analogical reasoning:
            mapping = bind(system_A, system_B)
            answer  = unbind(mapping, query_concept)
        """
        return self.bind(composite, key)

    # ── CLEANUP: Find nearest known symbol ───────────────────────────

    def cleanup(self, noisy_vector: np.ndarray, threshold: float = 0.10) -> List[Tuple[str, float]]:
        """
        Find the closest known symbols to a noisy/composite vector.

        This is the "associative memory" — given a degraded signal,
        recover the original concept(s) it most resembles.
        """
        scores = []
        for name, vec in self.symbols.items():
            sim = self.resonate(noisy_vector, vec)
            if abs(sim) >= threshold:
                scores.append((name, sim))
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores

    # ── Convenience: named operations ────────────────────────────────

    def store(self, name: str, vec: np.ndarray) -> np.ndarray:
        """Store a computed vector under a name."""
        self.symbols[name] = vec
        return vec

    def get(self, name: str) -> np.ndarray:
        """Retrieve a named vector."""
        return self.symbols[name]

    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "dimensions": self.D,
            "symbols": len(self.symbols),
            "memory_bytes": len(self.symbols) * self.D,  # int8 = 1 byte each
            "memory_mb": round(len(self.symbols) * self.D / (1024 * 1024), 2),
        }
