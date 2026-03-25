"""
KOS V8.0 -- Contradiction-Aware Hypothesis Forking

When the graph contains contradictions (A supports X, B contradicts X),
the retrieval pipeline forks into competing hypotheses rather than
averaging conflicting evidence.

Each hypothesis is a branch of evidence that is internally consistent.
The system returns ranked hypotheses rather than a single blended answer.
"""


class Hypothesis:
    """A single branch of internally consistent evidence."""

    def __init__(self, label: str):
        self.label = label
        self.evidence = []      # list of (node_id, score, provenance_text)
        self.supporting = []    # edges that support this hypothesis
        self.contradicting = [] # edges that contradict this hypothesis
        self.confidence = 0.0

    def add_evidence(self, node_id: str, score: float, text: str = ""):
        self.evidence.append((node_id, score, text))

    def compute_confidence(self):
        """Confidence = (sum of support scores) / (support + contradiction)."""
        sup = sum(s for _, s, _ in self.evidence) + len(self.supporting) * 0.1
        con = len(self.contradicting) * 0.3
        self.confidence = sup / max(sup + con, 0.001)
        return self.confidence

    def __repr__(self):
        return f"Hypothesis({self.label}, evidence={len(self.evidence)}, conf={self.confidence:.2f})"


class HypothesisForker:
    """Fork retrieval results into contradiction-aware hypotheses."""

    def fork(self, results, kernel) -> list:
        """
        Given retrieval results, detect contradictions and fork into hypotheses.

        Args:
            results: list of (node_id, score)
            kernel: KOSKernel with .contradictions list

        Returns:
            list of Hypothesis objects, sorted by confidence descending
        """
        contradictions = getattr(kernel, 'contradictions', [])

        if not contradictions:
            # No contradictions -- single hypothesis with all evidence
            h = Hypothesis("primary")
            for node_id, score in results:
                h.add_evidence(node_id, score)
            h.compute_confidence()
            return [h]

        # Build contradiction map: node_id -> set of contradicting node_ids
        contra_map = {}
        for c in contradictions:
            a = c.get('existing_target', '')
            b = c.get('new_target', '')
            if a and b:
                contra_map.setdefault(a, set()).add(b)
                contra_map.setdefault(b, set()).add(a)

        # Greedy partitioning: assign results to hypotheses
        # such that no hypothesis contains contradicting nodes
        result_nodes = set(r[0] for r in results)
        hypotheses = []
        assigned = set()

        for node_id, score in results:
            if node_id in assigned:
                continue

            # Find or create a compatible hypothesis
            placed = False
            for h in hypotheses:
                h_nodes = set(e[0] for e in h.evidence)
                # Check if node_id contradicts anything in this hypothesis
                node_contras = contra_map.get(node_id, set())
                if not (h_nodes & node_contras):
                    h.add_evidence(node_id, score)
                    assigned.add(node_id)
                    placed = True
                    break

            if not placed:
                h = Hypothesis(f"branch_{len(hypotheses)+1}")
                h.add_evidence(node_id, score)
                assigned.add(node_id)
                hypotheses.append(h)

        # Mark supporting/contradicting edges
        for h in hypotheses:
            h_nodes = set(e[0] for e in h.evidence)
            for node_id in h_nodes:
                contras = contra_map.get(node_id, set())
                for other_h in hypotheses:
                    if other_h is h:
                        continue
                    other_nodes = set(e[0] for e in other_h.evidence)
                    overlap = contras & other_nodes
                    if overlap:
                        h.contradicting.extend(list(overlap))

        # Compute confidence and sort
        for h in hypotheses:
            h.compute_confidence()

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses
