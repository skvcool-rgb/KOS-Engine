"""
KASM Layer 3: Analogical Abstraction Engine

The system automatically discovers structural metaphors between concepts
that share NO surface-level words, purely through VSA state vector similarity.

How it works:
    1. Every node accumulates a state vector via BIND + SUPERPOSE on each edge
    2. Two nodes that connect to structurally similar neighbors develop
       similar state vectors — even if they share zero edges
    3. The abstraction daemon sweeps the state matrix and discovers these
       hidden isomorphisms

Example:
    After ingesting:
        "The heart pumps blood through arteries"
        "A water pump pushes water through pipes"

    The daemon discovers:
        heart <=> water_pump (structural similarity: 0.15+)
        blood <=> water     (structural similarity: 0.12+)
        arteries <=> pipes  (structural similarity: 0.10+)

    No human told it these are related. The algebra found the isomorphism.

Architecture:
    AbsLayer plugs into the KOSDaemon as a 4th maintenance protocol.
    It runs during idle cycles alongside prune/merge/triadic.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from .vsa import KASMEngine


class AnalogicalAbstraction:
    """
    Layer 3 Engine: discovers structural metaphors in the knowledge graph.

    Attached to a VSABackplane, it sweeps the state vector matrix
    and surfaces concept pairs that are structurally isomorphic
    despite having no explicit edge between them.
    """

    def __init__(self, backplane, lexicon=None):
        """
        Args:
            backplane: VSABackplane instance (from kasm.bridge)
            lexicon: KASMLexicon for human-readable names (optional)
        """
        self.bp = backplane
        self.lexicon = lexicon
        # Discovered analogies: (node_a, node_b) -> similarity score
        self.discoveries: Dict[Tuple[str, str], float] = {}
        # Domains: group nodes by their neighborhood clusters
        self._domain_cache: Dict[str, str] = {}

    def sweep(self, threshold: float = 0.08,
              max_comparisons: int = 50_000,
              exclude_connected: bool = True,
              graph_nodes: Optional[dict] = None) -> List[Tuple[str, str, float]]:
        """
        Full analogical sweep: compare all node state vectors and
        discover hidden structural similarities.

        Args:
            threshold: minimum cosine similarity to report (default 0.08)
            max_comparisons: safety cap to prevent O(N^2) explosion
            exclude_connected: skip pairs that already share a direct edge
            graph_nodes: KOSKernel.nodes dict (for edge exclusion)

        Returns:
            List of (node_a, node_b, similarity) sorted by strength
        """
        node_ids = list(self.bp.state_vectors.keys())
        n = len(node_ids)

        if n < 2:
            return []

        # Build connected-pairs set for exclusion
        connected: Set[Tuple[str, str]] = set()
        if exclude_connected and graph_nodes:
            for nid, node in graph_nodes.items():
                for tgt in node.connections:
                    pair = tuple(sorted([nid, tgt]))
                    connected.add(pair)

        discoveries = []
        comparisons = 0

        for i in range(n):
            for j in range(i + 1, n):
                if comparisons >= max_comparisons:
                    break

                nid_a = node_ids[i]
                nid_b = node_ids[j]

                # Skip directly connected pairs
                pair = tuple(sorted([nid_a, nid_b]))
                if exclude_connected and pair in connected:
                    continue

                # Skip nodes with zero bindings (they're just random noise)
                if self.bp._binding_count.get(nid_a, 0) < 2:
                    continue
                if self.bp._binding_count.get(nid_b, 0) < 2:
                    continue

                score = self.bp.engine.resonate(
                    self.bp.state_vectors[nid_a],
                    self.bp.state_vectors[nid_b]
                )
                comparisons += 1

                if abs(score) >= threshold:
                    discoveries.append((nid_a, nid_b, score))
                    self.discoveries[pair] = score

            if comparisons >= max_comparisons:
                break

        # Sort by strength
        discoveries.sort(key=lambda x: abs(x[2]), reverse=True)
        return discoveries

    def discover_role_mappings(self, node_a: str, node_b: str,
                                graph_nodes: dict) -> List[Tuple[str, str, float]]:
        """
        Given two structurally similar nodes, discover which of their
        neighbors map to each other (role-filler correspondence).

        This is the "unbind" operation at the graph level:
            If heart <=> water_pump, then:
                heart's neighbor "blood" <=> pump's neighbor "water"
                heart's neighbor "arteries" <=> pump's neighbor "pipes"

        Uses VSA base vectors to find the mapping.
        """
        if node_a not in graph_nodes or node_b not in graph_nodes:
            return []

        neighbors_a = list(graph_nodes[node_a].connections.keys())
        neighbors_b = list(graph_nodes[node_b].connections.keys())

        if not neighbors_a or not neighbors_b:
            return []

        # Compare all neighbor pairs via state vector similarity
        mappings = []
        for na in neighbors_a:
            if na not in self.bp.state_vectors:
                continue
            best_match = None
            best_score = -1.0

            for nb in neighbors_b:
                if nb not in self.bp.state_vectors:
                    continue
                # Use base vectors for role comparison
                # (state vectors would be influenced by shared edges)
                score = self.bp.engine.resonate(
                    self.bp.base_vectors.get(na, self.bp.state_vectors[na]),
                    self.bp.base_vectors.get(nb, self.bp.state_vectors[nb])
                )
                # Also check state similarity
                state_score = self.bp.engine.resonate(
                    self.bp.state_vectors[na],
                    self.bp.state_vectors[nb]
                )
                # Combined: state similarity matters more
                combined = state_score * 0.7 + score * 0.3

                if combined > best_score:
                    best_score = combined
                    best_match = nb

            if best_match and best_score > 0.02:
                mappings.append((na, best_match, best_score))

        mappings.sort(key=lambda x: abs(x[2]), reverse=True)
        return mappings

    def find_analogies_for(self, node_id: str,
                           top_k: int = 5,
                           threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Find concepts structurally similar to a given concept.

        Unlike RESONATE search (which finds similar vectors),
        this specifically looks for cross-domain structural matches
        by filtering out directly connected nodes.
        """
        if node_id not in self.bp.state_vectors:
            return []

        query_vec = self.bp.state_vectors[node_id]
        results = []

        for nid, state_vec in self.bp.state_vectors.items():
            if nid == node_id:
                continue
            # Skip nodes with too few bindings
            if self.bp._binding_count.get(nid, 0) < 2:
                continue

            score = self.bp.engine.resonate(query_vec, state_vec)
            if abs(score) >= threshold:
                results.append((nid, score))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results[:top_k]

    def format_discoveries(self, discoveries: list,
                           lexicon=None) -> str:
        """Pretty-print discovered analogies."""
        lex = lexicon or self.lexicon
        lines = []
        for a, b, score in discoveries:
            name_a = lex.get_word(a) if lex else a
            name_b = lex.get_word(b) if lex else b
            bar_len = int(abs(score) * 100)
            bar = "#" * min(bar_len, 40)
            lines.append(f"  {name_a:20s} <=> {name_b:20s}  {score:+.4f}  [{bar}]")
        return "\n".join(lines)


class Layer3Daemon:
    """
    Daemon protocol for Layer 3: runs during idle cycles.

    Plugs into KOSDaemon.run_maintenance_cycle() as a 4th protocol
    alongside prune_orphans, merge_isomorphs, and triadic_closure.
    """

    def __init__(self, kernel, lexicon=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.engine = None
        self._last_node_count = 0

    def _ensure_engine(self):
        """Initialize the abstraction engine if VSA is available."""
        if self.engine is None and self.kernel.vsa is not None:
            self.engine = AnalogicalAbstraction(
                self.kernel.vsa, self.lexicon
            )

    def run(self, threshold: float = 0.08) -> dict:
        """
        Execute one Layer 3 sweep.

        Returns:
            dict with discoveries count, top analogies, and timing
        """
        self._ensure_engine()

        if self.engine is None:
            return {"analogies_found": 0, "time_ms": 0, "top": []}

        t0 = time.perf_counter()

        # Only re-sweep if graph has changed
        current_count = len(self.kernel.nodes)
        if current_count == self._last_node_count and self.engine.discoveries:
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "analogies_found": len(self.engine.discoveries),
                "time_ms": elapsed,
                "top": [],
                "cached": True,
            }

        discoveries = self.engine.sweep(
            threshold=threshold,
            exclude_connected=True,
            graph_nodes=self.kernel.nodes
        )

        self._last_node_count = current_count
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "analogies_found": len(discoveries),
            "time_ms": elapsed,
            "top": discoveries[:10],
            "cached": False,
        }
