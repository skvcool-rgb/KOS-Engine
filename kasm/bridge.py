"""
KASM-KOS Bridge — Fuses Hyperdimensional Computing into the Knowledge Graph.

This module connects two systems:
    KOS (Spreading Activation Graph) — scalar edges, provenance, text ingestion
    KASM (Vector Symbolic Architecture) — 10,000-D bipolar vectors, algebra

The bridge provides:
    1. VSA-backed ConceptNodes (each node carries a state hypervector)
    2. Auto BIND/SUPERPOSE on edge creation (silent background VSA ops)
    3. RESONATE-based semantic matching (replaces difflib fuzzy match)
    4. Thought Transfer protocol (export/import graph as .npy vectors)
    5. Cross-graph analogical queries

Architecture:
    ConceptNode.vsa_state = SUPERPOSE of all (self * neighbor) bindings
    When add_connection(A, B) fires:
        A.vsa_state = SUPERPOSE(A.vsa_state, BIND(A.vsa_base, B.vsa_base))
        B.vsa_state = SUPERPOSE(B.vsa_state, BIND(B.vsa_base, A.vsa_base))
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .vsa import KASMEngine


class VSABackplane:
    """
    The hyperdimensional substrate shared across all KOS nodes.

    Every concept gets:
        - vsa_base:  immutable random identity vector (its "DNA")
        - vsa_state: mutable accumulated state (SUPERPOSE of all bindings)

    The backplane is a singleton attached to the KOSKernel.
    """

    def __init__(self, dimensions: int = 10_000, seed: int = 42):
        self.engine = KASMEngine(dimensions=dimensions, seed=seed)
        self.base_vectors: Dict[str, np.ndarray] = {}   # node_id -> immutable base
        self.state_vectors: Dict[str, np.ndarray] = {}   # node_id -> mutable state
        self._binding_count: Dict[str, int] = {}          # node_id -> num bindings

    @property
    def D(self) -> int:
        return self.engine.D

    # ── Node Lifecycle ───────────────────────────────────────────────

    def register_node(self, node_id: str):
        """
        Create a base vector for a new concept.
        Called automatically when KOSKernel.add_node() fires.
        """
        if node_id not in self.base_vectors:
            vec = self.engine.node(f"base_{node_id}")
            self.base_vectors[node_id] = vec
            self.state_vectors[node_id] = vec.copy()  # state starts as identity
            self._binding_count[node_id] = 0

    def remove_node(self, node_id: str):
        """Remove a concept's vectors (called on pruning)."""
        self.base_vectors.pop(node_id, None)
        self.state_vectors.pop(node_id, None)
        self._binding_count.pop(node_id, None)
        self.engine.symbols.pop(f"base_{node_id}", None)

    # ── Edge Events (Auto BIND + SUPERPOSE) ──────────────────────────

    def on_edge_created(self, source_id: str, target_id: str, weight: float):
        """
        Called when KOSKernel.add_connection() creates an edge.

        BIDIRECTIONAL binding with ROLE VECTORS for structural fingerprinting.

        The key insight for Layer 3 analogical abstraction:
        Instead of binding source_base * target_base (which are orthogonal
        random vectors and can never match across domains), we bind each node
        with STRUCTURAL ROLE VECTORS derived from the edge's position:

            - The source gets bound with a "connects_to" role vector
            - The target gets bound with a "connected_from" role vector

        This way, two nodes that occupy the same structural position
        (e.g., heart and pump both "connect_to" two things and are
        "connected_from" zero things) develop similar state vectors.

        Additionally, we propagate the target's CURRENT STATE into the source,
        so nodes that connect to structurally similar neighborhoods converge.
        """
        self.register_node(source_id)
        self.register_node(target_id)

        # 1. Identity binding (preserves which specific nodes are connected)
        identity_binding = self.engine.bind(
            self.base_vectors[source_id],
            self.base_vectors[target_id]
        )

        # 2. Structural binding: source absorbs target's STATE
        # This propagates structural information across edges.
        # If target has similar structure to another node's target,
        # the source will develop similar state.
        structural_binding = self.engine.bind(
            self.base_vectors[source_id],
            self.state_vectors[target_id]
        )

        # 3. Target absorbs source's STATE (bidirectional structural propagation)
        reverse_binding = self.engine.bind(
            self.base_vectors[target_id],
            self.state_vectors[source_id]
        )

        # Accumulate into source: identity + structural
        self._binding_count[source_id] = self._binding_count.get(source_id, 0) + 1
        self.state_vectors[source_id] = self.engine.superpose(
            self.state_vectors[source_id],
            identity_binding,
            structural_binding
        )

        # Accumulate into target: identity + reverse structural
        self._binding_count[target_id] = self._binding_count.get(target_id, 0) + 1
        self.state_vectors[target_id] = self.engine.superpose(
            self.state_vectors[target_id],
            identity_binding,
            reverse_binding
        )

    # ── Structural Convergence (Layer 3 prerequisite) ──────────────

    def converge_structure(self, graph_nodes: dict, iterations: int = 3):
        """
        Run structural convergence: build ROLE-BASED fingerprints.

        The key insight: random base vectors are orthogonal, so BIND(heart, blood)
        can never equal BIND(pump, water). But we can create SHARED ROLE VECTORS
        based on graph topology:

            role_out_1 = a fixed vector for "first outbound edge"
            role_out_2 = a fixed vector for "second outbound edge"
            role_in_1  = a fixed vector for "first inbound edge"

        Then heart's state = SUPERPOSE(BIND(base, role_out_1), BIND(base, role_out_2), ...)
        And  pump's state  = SUPERPOSE(BIND(base, role_out_1), BIND(base, role_out_2), ...)

        Because they share the SAME role vectors, their states become similar
        when they have the same structural topology.
        """
        import numpy as np

        # Create deterministic role vectors based on structural position
        rng = np.random.default_rng(seed=12345)
        max_degree = 20
        role_vectors = {}
        for i in range(max_degree):
            role_vectors[f"out_{i}"] = rng.choice([-1, 1], size=self.engine.D).astype(np.int8)
            role_vectors[f"in_{i}"] = rng.choice([-1, 1], size=self.engine.D).astype(np.int8)

        # Build inbound index
        inbound = {}
        for nid, node in graph_nodes.items():
            for tgt_id in node.connections:
                inbound.setdefault(tgt_id, []).append(nid)

        # For each node, create structural fingerprint
        for nid in graph_nodes:
            if nid not in self.base_vectors:
                continue

            bindings = []

            # Outbound role bindings
            out_neighbors = list(graph_nodes[nid].connections.keys())
            for i, _ in enumerate(out_neighbors[:max_degree]):
                role_vec = role_vectors[f"out_{i}"]
                bindings.append(self.engine.bind(self.base_vectors[nid], role_vec))

            # Inbound role bindings
            in_neighbors = inbound.get(nid, [])
            for i, _ in enumerate(in_neighbors[:max_degree]):
                role_vec = role_vectors[f"in_{i}"]
                bindings.append(self.engine.bind(self.base_vectors[nid], role_vec))

            if bindings:
                # Structural fingerprint: ONLY role vectors, no base vector
                # This makes two nodes with same topology produce identical fingerprints
                structural = self.engine.superpose(*bindings)

                # Also create a PURE topology vector (no identity binding at all)
                # Just superpose the raw role vectors for each edge
                topo_parts = []
                for i, _ in enumerate(out_neighbors[:max_degree]):
                    topo_parts.append(role_vectors[f"out_{i}"])
                for i, _ in enumerate(in_neighbors[:max_degree]):
                    topo_parts.append(role_vectors[f"in_{i}"])

                if topo_parts:
                    topology = self.engine.superpose(*topo_parts)
                    # The topology vector is the same for all nodes with same degree
                    # Superpose it into state to create structural similarity
                    self.state_vectors[nid] = self.engine.superpose(
                        self.state_vectors[nid],
                        topology,
                        structural
                    )

    # ── RESONATE: Semantic Matching ──────────────────────────────────

    def resonate_query(self, query_id: str, candidates: List[str],
                       threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Find the closest concepts to a query using cosine similarity
        on state vectors.

        This replaces difflib fuzzy matching — it works on meaning,
        not character overlap.
        """
        if query_id not in self.state_vectors:
            return []

        query_vec = self.state_vectors[query_id]
        results = []

        for cid in candidates:
            if cid in self.state_vectors and cid != query_id:
                score = self.engine.resonate(query_vec, self.state_vectors[cid])
                if abs(score) >= threshold:
                    results.append((cid, score))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results

    def resonate_vector(self, query_vec: np.ndarray,
                        threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Find closest concepts to an arbitrary vector.
        Used for cross-graph queries and thought transfer.
        """
        results = []
        for nid, state_vec in self.state_vectors.items():
            score = self.engine.resonate(query_vec, state_vec)
            if abs(score) >= threshold:
                results.append((nid, score))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results

    # ── Unbind: Analogical Query ─────────────────────────────────────

    def analogical_query(self, system_a_ids: List[str],
                         system_b_ids: List[str],
                         query_id: str) -> List[Tuple[str, float]]:
        """
        "What is the X of system B?"

        Given two systems (e.g., solar system concepts, atom concepts)
        and a query concept from system A, find its analog in system B.

        Math:
            system_a_vec = SUPERPOSE(all state vectors in system A)
            system_b_vec = SUPERPOSE(all state vectors in system B)
            mapping = BIND(system_a_vec, system_b_vec)
            answer = UNBIND(mapping, query_base)
            cleanup(answer) -> nearest known concept
        """
        # Build composite vectors for each system
        a_vecs = [self.state_vectors[nid] for nid in system_a_ids
                  if nid in self.state_vectors]
        b_vecs = [self.state_vectors[nid] for nid in system_b_ids
                  if nid in self.state_vectors]

        if not a_vecs or not b_vecs:
            return []

        system_a = self.engine.superpose(*a_vecs)
        system_b = self.engine.superpose(*b_vecs)

        # Create mapping and query
        mapping = self.engine.bind(system_a, system_b)
        query_base = self.base_vectors.get(query_id)
        if query_base is None:
            return []

        answer_vec = self.engine.unbind(mapping, query_base)

        # Find nearest concepts (excluding the query itself)
        results = []
        for nid, base_vec in self.base_vectors.items():
            if nid != query_id:
                score = self.engine.resonate(answer_vec, base_vec)
                if abs(score) >= 0.05:
                    results.append((nid, score))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results

    # ── Thought Transfer Protocol ────────────────────────────────────

    def export_vectors(self, filepath: str):
        """
        Export all state vectors as a compressed .npz file.

        This is the "thought transfer" — another KOS instance can
        import this file and instantly gain the structural knowledge
        without re-ingesting documents.

        File format:
            base_<node_id>: base identity vector
            state_<node_id>: accumulated state vector
            _meta_ids: node ID list (for reconstruction)
            _meta_dims: dimensionality
        """
        arrays = {}
        node_ids = list(self.base_vectors.keys())

        for nid in node_ids:
            arrays[f"base_{nid}"] = self.base_vectors[nid]
            arrays[f"state_{nid}"] = self.state_vectors[nid]

        arrays['_meta_ids'] = np.array(node_ids, dtype=object)
        arrays['_meta_dims'] = np.array([self.D])
        arrays['_meta_counts'] = np.array(
            [self._binding_count.get(nid, 0) for nid in node_ids]
        )

        np.savez_compressed(filepath, **arrays)
        return len(node_ids)

    def import_vectors(self, filepath: str, merge: bool = True) -> int:
        """
        Import vectors from a .npz file.

        If merge=True, new vectors are superposed with existing ones
        (additive knowledge). If False, they replace existing vectors.

        Returns: number of concepts imported
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        data = np.load(filepath, allow_pickle=True)
        node_ids = list(data['_meta_ids'])
        imported_dims = int(data['_meta_dims'][0])
        counts = data['_meta_counts']

        if imported_dims != self.D:
            raise ValueError(
                f"Dimension mismatch: this backplane is {self.D}-D, "
                f"imported file is {imported_dims}-D"
            )

        for i, nid in enumerate(node_ids):
            base_key = f"base_{nid}"
            state_key = f"state_{nid}"

            if base_key in data and state_key in data:
                imported_base = data[base_key]
                imported_state = data[state_key]

                if merge and nid in self.state_vectors:
                    # Merge: superpose existing and imported states
                    self.state_vectors[nid] = self.engine.superpose(
                        self.state_vectors[nid], imported_state
                    )
                    self._binding_count[nid] = (
                        self._binding_count.get(nid, 0) + int(counts[i])
                    )
                else:
                    # Replace or new
                    self.base_vectors[nid] = imported_base
                    self.state_vectors[nid] = imported_state
                    self._binding_count[nid] = int(counts[i])

        return len(node_ids)

    # ── Diagnostics ──────────────────────────────────────────────────

    def stats(self) -> dict:
        n = len(self.base_vectors)
        return {
            "nodes": n,
            "dimensions": self.D,
            "memory_mb": round(n * 2 * self.D / (1024 * 1024), 2),
            "total_bindings": sum(self._binding_count.values()),
        }
