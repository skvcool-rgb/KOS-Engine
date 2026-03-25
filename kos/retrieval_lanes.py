"""
KOS V8.0 -- Specialized Retrieval Lanes

Three retrieval lanes beyond the default beam search:

1. CausalLane: Follow only CAUSES/TEMPORAL edges to build causal chains.
   Use case: "Why does X happen?" / "What causes Y?"

2. TemporalLane: Follow TEMPORAL_BEFORE/AFTER edges to build timelines.
   Use case: "What happened before X?" / "Timeline of events"

3. AnalogicalLane: Use VSA (hyperdimensional computing) to find
   structurally similar concepts via BIND/SUPERPOSE/CLEANUP.
   Use case: "What is X like?" / "Find something similar to X"
"""

from .edge_types import CAUSES, TEMPORAL_BEFORE, TEMPORAL_AFTER


class CausalLane:
    """Extract causal chains from the knowledge graph."""

    # Edge types that represent causation
    CAUSAL_TYPES = [CAUSES, TEMPORAL_BEFORE, TEMPORAL_AFTER]

    def trace(self, kernel, seed_ids: list, max_depth: int = 5,
              top_k: int = 10) -> list:
        """
        Trace causal chains from seed nodes.

        Returns:
            list of (node_id, score) following only causal edges
        """
        if hasattr(kernel, '_rust') and kernel._rust is not None:
            return kernel._rust.query_causal(seed_ids, top_k)

        # Python fallback: filter edges by type during traversal
        if hasattr(kernel, 'query_beam'):
            return kernel.query_beam(
                seed_ids, top_k=top_k,
                beam_width=24, max_depth=max_depth,
                allowed_edge_types=self.CAUSAL_TYPES)

        # Last resort: regular query
        return kernel.query(seed_ids, top_k)

    def build_chain(self, kernel, seed_id: str, max_depth: int = 5) -> list:
        """
        Build an ordered causal chain from a seed node.

        Returns:
            list of (node_id, score) in causal order
        """
        chain = []
        visited = {seed_id}
        current = [seed_id]

        for _ in range(max_depth):
            next_level = []
            for nid in current:
                if hasattr(kernel, '_rust') and kernel._rust is not None:
                    neighbors = kernel._rust.get_neighbors(nid)
                    for tgt, w, m, et in neighbors:
                        if et in self.CAUSAL_TYPES and tgt not in visited:
                            chain.append((tgt, float(w)))
                            visited.add(tgt)
                            next_level.append(tgt)
                elif nid in kernel.nodes:
                    for tgt, w in kernel.nodes[nid].connections.items():
                        if tgt not in visited:
                            chain.append((tgt, w))
                            visited.add(tgt)
                            next_level.append(tgt)
            current = next_level
            if not current:
                break

        return chain


class TemporalLane:
    """Extract temporal sequences from the knowledge graph."""

    TEMPORAL_TYPES = [TEMPORAL_BEFORE, TEMPORAL_AFTER]

    def trace(self, kernel, seed_ids: list, direction: str = "both",
              max_depth: int = 5, top_k: int = 10) -> list:
        """
        Trace temporal sequences.

        Args:
            direction: "before", "after", or "both"
        """
        if direction == "before":
            types = [TEMPORAL_BEFORE]
        elif direction == "after":
            types = [TEMPORAL_AFTER]
        else:
            types = self.TEMPORAL_TYPES

        if hasattr(kernel, 'query_beam'):
            return kernel.query_beam(
                seed_ids, top_k=top_k,
                beam_width=24, max_depth=max_depth,
                allowed_edge_types=types)

        return kernel.query(seed_ids, top_k)

    def build_timeline(self, kernel, seed_id: str,
                       max_depth: int = 5) -> dict:
        """
        Build a timeline around a seed node.

        Returns:
            {"before": [...], "after": [...]} with ordered events
        """
        before = []
        after = []
        visited = {seed_id}

        # Trace backwards
        current = [seed_id]
        for _ in range(max_depth):
            next_level = []
            for nid in current:
                if hasattr(kernel, '_rust') and kernel._rust is not None:
                    neighbors = kernel._rust.get_neighbors(nid)
                    for tgt, w, m, et in neighbors:
                        if et == TEMPORAL_BEFORE and tgt not in visited:
                            before.append((tgt, float(w)))
                            visited.add(tgt)
                            next_level.append(tgt)
                elif nid in kernel.nodes:
                    for tgt, w in kernel.nodes[nid].connections.items():
                        if tgt not in visited:
                            before.append((tgt, w))
                            visited.add(tgt)
                            next_level.append(tgt)
            current = next_level
            if not current:
                break

        # Trace forwards
        current = [seed_id]
        for _ in range(max_depth):
            next_level = []
            for nid in current:
                if hasattr(kernel, '_rust') and kernel._rust is not None:
                    neighbors = kernel._rust.get_neighbors(nid)
                    for tgt, w, m, et in neighbors:
                        if et == TEMPORAL_AFTER and tgt not in visited:
                            after.append((tgt, float(w)))
                            visited.add(tgt)
                            next_level.append(tgt)
                elif nid in kernel.nodes:
                    for tgt, w in kernel.nodes[nid].connections.items():
                        if tgt not in visited:
                            after.append((tgt, w))
                            visited.add(tgt)
                            next_level.append(tgt)
            current = next_level
            if not current:
                break

        return {"before": before, "after": after}


class AnalogicalLane:
    """Find structural analogies using VSA (hyperdimensional computing)."""

    def find_analogies(self, kernel, source_id: str, target_domain: list = None,
                       top_k: int = 5, threshold: float = 0.05) -> list:
        """
        Find structural analogies to source_id.

        Uses VSA RESONATE to find nodes with similar relational structure.

        Args:
            source_id: the concept to find analogies for
            target_domain: optional list of node IDs to search within
            top_k: number of results
            threshold: minimum similarity

        Returns:
            list of (node_id, similarity_score)
        """
        # Try Rust VSA resonate
        if hasattr(kernel, '_rust') and kernel._rust is not None:
            try:
                results = kernel._rust.resonate_search(source_id, top_k * 2, threshold)
                # Filter to target domain if specified
                if target_domain:
                    domain_set = set(target_domain)
                    results = [(n, s) for n, s in results if n in domain_set]
                return results[:top_k]
            except Exception:
                pass

        # Fallback: use Python VSA backplane
        if hasattr(kernel, 'resonate_match'):
            results = kernel.resonate_match(source_id, threshold)
            if target_domain:
                domain_set = set(target_domain)
                results = [(n, s) for n, s in results if n in domain_set]
            return results[:top_k]

        return []

    def structural_map(self, kernel, source_id: str, target_id: str) -> dict:
        """
        Build a structural mapping between two concepts.

        Compares their neighbor patterns to find role correspondences.

        Returns:
            {"similarity": float, "mappings": [(src_neighbor, tgt_neighbor, sim), ...]}
        """
        # Get neighbors of both
        src_neighbors = []
        tgt_neighbors = []

        if hasattr(kernel, '_rust') and kernel._rust is not None:
            try:
                src_neighbors = [(n, w, et) for n, w, _, et in
                                 kernel._rust.get_neighbors(source_id)]
                tgt_neighbors = [(n, w, et) for n, w, _, et in
                                 kernel._rust.get_neighbors(target_id)]
            except Exception:
                pass

        if not src_neighbors and source_id in kernel.nodes:
            src_neighbors = [(n, w, 0) for n, w in
                             kernel.nodes[source_id].connections.items()]
        if not tgt_neighbors and target_id in kernel.nodes:
            tgt_neighbors = [(n, w, 0) for n, w in
                             kernel.nodes[target_id].connections.items()]

        # Overall similarity via resonate
        overall_sim = 0.0
        if hasattr(kernel, '_rust') and kernel._rust is not None:
            try:
                overall_sim = kernel._rust.resonate(source_id, target_id)
            except Exception:
                pass

        # Map neighbors by edge type similarity
        mappings = []
        used_tgt = set()
        for sn, sw, set_ in src_neighbors:
            best_match = None
            best_sim = -1
            for tn, tw, tet in tgt_neighbors:
                if tn in used_tgt:
                    continue
                # Same edge type = structural match
                type_sim = 1.0 if set_ == tet else 0.3
                weight_sim = 1.0 - abs(sw - tw)
                sim = type_sim * 0.7 + weight_sim * 0.3
                if sim > best_sim:
                    best_sim = sim
                    best_match = tn
            if best_match:
                mappings.append((sn, best_match, best_sim))
                used_tgt.add(best_match)

        return {
            "similarity": overall_sim,
            "mappings": mappings,
        }
