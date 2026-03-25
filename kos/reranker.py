"""
KOS V8.0 -- Multi-Signal Reranker

Reranks retrieval results using multiple scoring signals:
    1. query_match      (0.30) - How well does the node match the query?
    2. path_coherence   (0.20) - Is this reachable via coherent edges?
    3. edge_trust       (0.15) - What type of edges led here?
    4. source_trust     (0.15) - How reliable is the provenance?
    5. recency          (0.10) - How recently was this node accessed?
    6. wm_bias          (0.05) - Is this in working memory?
    7. multi_path       (0.05) - Is this reachable via multiple paths?

Penalties:
    - contradiction_penalty (0.15) - Does this contradict known facts?
    - hub_penalty           (0.10) - Is this a generic hub node?
"""


class MultiSignalReranker:
    """Rerank retrieval results using weighted multi-signal scoring."""

    # Signal weights (sum to 1.0 before penalties)
    W_QUERY_MATCH = 0.30
    W_PATH_COHERENCE = 0.20
    W_EDGE_TRUST = 0.15
    W_SOURCE_TRUST = 0.15
    W_RECENCY = 0.10
    W_WM_BIAS = 0.05
    W_MULTI_PATH = 0.05

    # Penalty weights
    W_CONTRADICTION = 0.15
    W_HUB = 0.10

    def rerank(self, results, kernel, query_words, working_memory=None):
        """
        Rerank a list of (node_id, activation_score) tuples.

        Args:
            results: list of (node_id, score) from beam search
            kernel: KOSKernel instance
            query_words: list of content words from query
            working_memory: list of recent query seeds

        Returns:
            list of (node_id, reranked_score) sorted descending
        """
        if not results:
            return results

        working_memory = working_memory or []
        wm_set = set(working_memory)
        query_set = set(w.lower() for w in query_words)

        scored = []
        max_activation = max(abs(s) for _, s in results) if results else 1.0

        for node_id, activation in results:
            signals = {}

            # 1. Query match: does node ID contain query words?
            node_lower = node_id.lower().replace('.', ' ').replace('_', ' ')
            node_words = set(node_lower.split())
            overlap = len(query_set & node_words)
            signals["query_match"] = min(overlap / max(len(query_set), 1), 1.0)

            # 2. Path coherence: normalized activation (proxy for path quality)
            signals["path_coherence"] = abs(activation) / max(max_activation, 0.001)

            # 3. Edge trust: average trust of edges leading to this node
            edge_trust_val = self._compute_edge_trust(node_id, kernel)
            signals["edge_trust"] = edge_trust_val

            # 4. Source trust: provenance quality (presence + length)
            source_trust_val = self._compute_source_trust(node_id, kernel)
            signals["source_trust"] = source_trust_val

            # 5. Recency: tier bias (hot=1.0, warm=0.66, cold=0.33)
            recency_val = self._compute_recency(node_id, kernel)
            signals["recency"] = recency_val

            # 6. Working memory bias
            signals["wm_bias"] = 1.0 if node_id in wm_set else 0.0

            # 7. Multi-path support: degree as proxy
            signals["multi_path"] = self._compute_multi_path(node_id, kernel)

            # Penalties
            contradiction = 1.0 if self._has_contradiction(node_id, kernel) else 0.0
            hub_pen = self._compute_hub_penalty(node_id, kernel)

            # Weighted score
            score = (
                self.W_QUERY_MATCH * signals["query_match"]
                + self.W_PATH_COHERENCE * signals["path_coherence"]
                + self.W_EDGE_TRUST * signals["edge_trust"]
                + self.W_SOURCE_TRUST * signals["source_trust"]
                + self.W_RECENCY * signals["recency"]
                + self.W_WM_BIAS * signals["wm_bias"]
                + self.W_MULTI_PATH * signals["multi_path"]
                - self.W_CONTRADICTION * contradiction
                - self.W_HUB * hub_pen
            )

            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_edge_trust(self, node_id, kernel):
        """Average edge trust for edges pointing to this node."""
        try:
            from .edge_types import EDGE_CONFIG
            if hasattr(kernel, '_rust') and kernel._rust is not None:
                neighbors = kernel._rust.get_neighbors(node_id)
                if not neighbors:
                    return 0.5
                trusts = [EDGE_CONFIG.get(et, {}).get("trust", 0.5)
                          for _, _, _, et in neighbors]
                return sum(trusts) / len(trusts)
        except (ImportError, Exception):
            pass
        return 0.5

    def _compute_source_trust(self, node_id, kernel):
        """Score based on provenance availability and quality."""
        prov = getattr(kernel, 'provenance', {})
        total_text = 0
        for pair, texts in prov.items():
            if node_id in pair:
                total_text += len(texts)
        # More provenance = higher trust (capped at 1.0)
        return min(total_text / 5.0, 1.0)

    def _compute_recency(self, node_id, kernel):
        """Tier-based recency score."""
        from .tiers import classify
        tick = getattr(kernel, 'current_tick', 0)
        node = kernel.nodes.get(node_id)
        if node:
            last_active = getattr(node, 'last_tick', 0)
            tier = classify(tick - last_active)
            return {"hot": 1.0, "warm": 0.66, "cold": 0.33}.get(tier, 0.33)
        return 0.33

    def _compute_multi_path(self, node_id, kernel):
        """Multi-path support based on in-degree."""
        if hasattr(kernel, '_rust') and kernel._rust is not None:
            neighbors = kernel._rust.get_neighbors(node_id)
            count = len(neighbors) if neighbors else 0
        elif node_id in kernel.nodes:
            count = len(kernel.nodes[node_id].connections)
        else:
            count = 0
        # Normalize: 3+ paths = full score
        return min(count / 3.0, 1.0)

    def _has_contradiction(self, node_id, kernel):
        """Check if this node appears in any known contradictions."""
        for c in getattr(kernel, 'contradictions', []):
            if node_id in (c.get('source'), c.get('existing_target'),
                           c.get('new_target')):
                return True
        return False

    def _compute_hub_penalty(self, node_id, kernel):
        """Hub penalty: high-degree nodes get penalized."""
        import math
        if hasattr(kernel, '_rust') and kernel._rust is not None:
            neighbors = kernel._rust.get_neighbors(node_id)
            degree = len(neighbors) if neighbors else 0
        elif node_id in kernel.nodes:
            degree = len(kernel.nodes[node_id].connections)
        else:
            degree = 0
        if degree <= 5:
            return 0.0
        # 1 - 1/(1+ln(degree)) -- inverted so high degree = high penalty
        return 1.0 - 1.0 / (1.0 + math.log(degree))
