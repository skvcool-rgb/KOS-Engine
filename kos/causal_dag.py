"""
KOS V8.0 -- Causal DAG Engine (Separate from Association Graph)

CRITICAL: Causation != Association. The KOS association graph stores
"X is related to Y." The Causal DAG stores "X causes Y" with direction,
strength, and confidence.

Mixing them in one graph is a category error. This module maintains
a SEPARATE directed acyclic graph for causal relationships.

Features:
    - Directed causal edges with confidence
    - Topological ordering (causes before effects)
    - Causal path tracing (A -> B -> C -> D)
    - Intervention queries: "What happens if we remove X?"
    - Fault tree analysis: "What could cause Y?"
"""

from collections import defaultdict, deque


class CausalEdge:
    """A directed causal link with metadata."""
    __slots__ = ['source', 'target', 'strength', 'confidence',
                 'mechanism', 'provenance']

    def __init__(self, source: str, target: str, strength: float = 1.0,
                 confidence: float = 0.5, mechanism: str = "",
                 provenance: str = ""):
        self.source = source
        self.target = target
        self.strength = strength      # How strong is the causal effect
        self.confidence = confidence  # How confident are we this is causal
        self.mechanism = mechanism    # HOW does it cause (optional)
        self.provenance = provenance  # Evidence source


class CausalDAG:
    """
    Directed Acyclic Graph for causal relationships.
    SEPARATE from the KOS association graph.
    """

    def __init__(self):
        self.nodes = set()
        self.edges = {}          # (src, tgt) -> CausalEdge
        self.forward = defaultdict(list)   # src -> [tgt, ...]
        self.backward = defaultdict(list)  # tgt -> [src, ...]

    def add_cause(self, source: str, target: str, strength: float = 1.0,
                  confidence: float = 0.5, mechanism: str = "",
                  provenance: str = "") -> bool:
        """
        Add a causal link: source CAUSES target.
        Returns False if it would create a cycle (violating DAG property).
        """
        # Check for cycle
        if self._would_create_cycle(source, target):
            return False

        self.nodes.add(source)
        self.nodes.add(target)

        edge = CausalEdge(source, target, strength, confidence,
                          mechanism, provenance)
        self.edges[(source, target)] = edge
        if target not in self.forward[source]:
            self.forward[source].append(target)
        if source not in self.backward[target]:
            self.backward[target].append(source)

        return True

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding source->target would create a cycle."""
        if source == target:
            return True
        # BFS from target: can we reach source?
        visited = set()
        queue = deque([target])
        while queue:
            node = queue.popleft()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self.forward.get(node, []))
        return False

    def get_effects(self, cause: str, depth: int = 1) -> list:
        """What does this cause? Returns list of (effect, strength, depth)."""
        results = []
        visited = set()
        queue = deque([(cause, 0)])

        while queue:
            node, d = queue.popleft()
            if d > depth or node in visited:
                continue
            visited.add(node)

            for target in self.forward.get(node, []):
                edge = self.edges.get((node, target))
                if edge:
                    results.append((target, edge.strength, d + 1))
                    queue.append((target, d + 1))

        return results

    def get_causes(self, effect: str, depth: int = 1) -> list:
        """What causes this? Returns list of (cause, strength, depth)."""
        results = []
        visited = set()
        queue = deque([(effect, 0)])

        while queue:
            node, d = queue.popleft()
            if d > depth or node in visited:
                continue
            visited.add(node)

            for source in self.backward.get(node, []):
                edge = self.edges.get((source, node))
                if edge:
                    results.append((source, edge.strength, d + 1))
                    queue.append((source, d + 1))

        return results

    def causal_path(self, source: str, target: str) -> list:
        """Find the causal path from source to target. Returns list of nodes."""
        if source == target:
            return [source]

        visited = set()
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            if node in visited:
                continue
            visited.add(node)

            for tgt in self.forward.get(node, []):
                if tgt not in visited:
                    queue.append((tgt, path + [tgt]))

        return []  # No path

    def intervene(self, removed_node: str) -> dict:
        """
        Intervention: What changes if we remove a node?
        Returns nodes that lose their causal support.
        """
        affected = set()
        # Find all downstream effects
        queue = deque(self.forward.get(removed_node, []))
        while queue:
            node = queue.popleft()
            if node in affected:
                continue
            # Check if ALL causes of this node are removed/affected
            causes = self.backward.get(node, [])
            active_causes = [c for c in causes
                             if c != removed_node and c not in affected]
            if not active_causes:
                affected.add(node)
                queue.extend(self.forward.get(node, []))

        return {
            "removed": removed_node,
            "affected_nodes": list(affected),
            "cascade_depth": len(affected),
        }

    def fault_tree(self, failure: str, max_depth: int = 5) -> dict:
        """
        Build a fault tree: what combination of causes could lead to failure?
        Returns tree structure.
        """
        def _build_tree(node, depth):
            if depth <= 0:
                return {"node": node, "causes": []}
            causes = self.backward.get(node, [])
            return {
                "node": node,
                "causes": [_build_tree(c, depth - 1) for c in causes],
            }

        return _build_tree(failure, max_depth)

    def topological_order(self) -> list:
        """Return nodes in causal order (causes before effects)."""
        in_degree = defaultdict(int)
        for node in self.nodes:
            if node not in in_degree:
                in_degree[node] = 0
        for (src, tgt) in self.edges:
            in_degree[tgt] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for tgt in self.forward.get(node, []):
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    queue.append(tgt)

        return order

    def sync_from_kos(self, kernel):
        """
        Import causal edges from KOS graph (edge_type == CAUSES).
        Only imports edges typed as causal.
        """
        if not hasattr(kernel, '_rust') or kernel._rust is None:
            return 0

        count = 0
        for nid in kernel.nodes:
            try:
                neighbors = kernel._rust.get_neighbors(nid)
                for tgt, w, m, et in neighbors:
                    if et == 2:  # CAUSES
                        added = self.add_cause(nid, tgt, float(w), 0.7)
                        if added:
                            count += 1
            except Exception:
                continue
        return count

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "root_causes": sum(1 for n in self.nodes
                               if not self.backward.get(n)),
            "leaf_effects": sum(1 for n in self.nodes
                                if not self.forward.get(n)),
        }
