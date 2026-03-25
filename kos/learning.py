"""
KOS V7.0 — Learning Coordinator (The Brain)

Orchestrates all learning subsystems after each query.
This is the missing link: the predict -> compare -> update loop
that makes every query a learning opportunity.

Learning behaviors:
1. Predictive coding update (every query)
2. Hebbian reinforcement (strengthen successful paths)
3. Anti-Hebbian weakening (weaken irrelevant paths)
4. Self-growing graph (create nodes from unknown query words)
5. Abstraction formation (detect repeated patterns -> new concepts)
6. Sleep consolidation (periodic pruning + strengthening)
"""

import time
import re
from collections import defaultdict


class LearningCoordinator:
    """Wire all learning subsystems into a continuous loop."""

    def __init__(self, kernel, lexicon, pce=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.pce = pce

        # Tracking
        self._query_count = 0
        self._last_consolidation = time.time()
        self._activation_history = []  # Last N query activation patterns
        self._max_history = 100
        self._last_seeds = []
        self._last_results = {}

        # Learning rates
        self._hebbian_rate = 2       # Myelin increment per reinforced edge
        self._antihebbian_rate = 0.02  # Weight decay per weakened edge
        self._growth_min_len = 3     # Min word length to create new nodes
        self._abstraction_interval = 10  # Check every N queries
        self._consolidation_interval = 300  # seconds (5 min)

        # Stats
        self.stats = {
            'queries_learned': 0,
            'edges_strengthened': 0,
            'edges_weakened': 0,
            'nodes_grown': 0,
            'abstractions_formed': 0,
            'consolidations': 0,
        }

    def after_query(self, seed_ids, results, user_prompt, answer_text):
        """Called after every chat() — the core learning loop.

        Args:
            seed_ids: list of node IDs that were used as query seeds
            results: dict {node_id: activation} from spreading activation
            user_prompt: the original user query string
            answer_text: the answer that was returned
        """
        self._query_count += 1
        self._last_seeds = seed_ids
        self._last_results = results

        # 1. Predictive coding: predict -> compare -> update weights
        self._learn_from_prediction(seed_ids, results)

        # 2. Hebbian: strengthen paths that were activated
        self._hebbian_reinforce(seed_ids, results)

        # 3. Self-growing: create nodes from unknown words in the query
        self._grow_from_query(user_prompt)

        # 4. Store activation pattern for abstraction detection
        if results:
            top_nodes = sorted(results.keys(), key=lambda k: results[k], reverse=True)[:10]
            self._activation_history.append(set(top_nodes))
            if len(self._activation_history) > self._max_history:
                self._activation_history.pop(0)

        # 5. Abstraction formation (every N queries)
        if self._query_count % self._abstraction_interval == 0:
            self._detect_abstractions()

        # 6. Anti-Hebbian on re-asks
        self._detect_reask_and_weaken(user_prompt, seed_ids)

        # 7. Sleep consolidation (periodic)
        now = time.time()
        if now - self._last_consolidation > self._consolidation_interval:
            self._consolidate()
            self._last_consolidation = now

        self.stats['queries_learned'] = self._query_count

    def _learn_from_prediction(self, seed_ids, actual_results):
        """Run predictive coding: predict what should activate, compare, adjust."""
        if not self.pce or not seed_ids:
            return
        try:
            report = self.pce.query_with_prediction(
                seed_ids, top_k=10, verbose=False)
            # The PCE internally adjusts weights based on prediction error.
            # We just need to call it — the learning happens inside.
        except Exception:
            pass

    def _hebbian_reinforce(self, seed_ids, results):
        """Strengthen edges along successful activation paths.

        Hebbian rule: "neurons that fire together wire together."
        If seed A activated node B which activated node C,
        the A->B and B->C edges get myelin increments.
        """
        if not self.kernel._rust or not results:
            return

        rust = self.kernel._rust
        strengthened = 0

        # For each activated node, check if its parent (seed or
        # intermediate) also activated — that's a successful path
        activated_set = set(results.keys())
        for node_id in activated_set:
            try:
                neighbors = rust.get_neighbors(node_id)
            except Exception:
                continue
            for tgt_name, weight, myelin in neighbors:
                tgt_id = self.lexicon.word_to_uuid.get(tgt_name)
                if tgt_id and tgt_id in activated_set:
                    # Both ends activated — reinforce this edge
                    try:
                        rust.myelinate(node_id, tgt_id, self._hebbian_rate)
                        strengthened += 1
                    except Exception:
                        pass

        self.stats['edges_strengthened'] += strengthened

    def _detect_reask_and_weaken(self, user_prompt, seed_ids):
        """If the user re-asks a similar query, weaken the old paths.

        Re-ask = user wasn't satisfied with the answer.
        Weaken edges that were activated in the previous answer.
        """
        if not self._last_seeds or not self.kernel._rust:
            return

        # Simple overlap check: if >50% of seeds overlap with last query
        if not seed_ids:
            return
        overlap = set(seed_ids) & set(self._last_seeds)
        if len(overlap) < len(seed_ids) * 0.5:
            return  # Not a re-ask

        # It's a re-ask — weaken the old results
        rust = self.kernel._rust
        weakened = 0
        for node_id in list(self._last_results.keys())[:20]:
            for seed in self._last_seeds:
                try:
                    if rust.get_edge(seed, node_id) is not None:
                        rust.adjust_weight(seed, node_id, -self._antihebbian_rate)
                        weakened += 1
                except Exception:
                    pass

        self.stats['edges_weakened'] += weakened

    def _grow_from_query(self, user_prompt):
        """Create nodes from unknown words in the user's query.

        If the user asks about "decoherence" and it's not in the graph,
        create a node for it and weakly connect it to other query words.
        """
        if not user_prompt:
            return

        words = re.findall(r'[a-zA-Z]{3,}', user_prompt.lower())
        known_ids = []
        unknown_words = []

        for w in words:
            uid = self.lexicon.word_to_uuid.get(w)
            if uid and uid in self.kernel.nodes:
                known_ids.append(uid)
            elif len(w) >= self._growth_min_len and w not in _STOP_WORDS:
                unknown_words.append(w)

        if not unknown_words or not known_ids:
            return

        grown = 0
        for w in unknown_words[:3]:  # Max 3 new nodes per query
            uid = self.lexicon.get_or_create_id(w)
            self.kernel.add_node(uid)
            # Weakly connect to known query words
            for kid in known_ids[:5]:
                self.kernel.add_connection(
                    uid, kid, 0.2,
                    f"[LEARNED] Co-occurred in query: {user_prompt[:60]}")
            grown += 1

        self.stats['nodes_grown'] += grown

    def _detect_abstractions(self):
        """Find repeated activation patterns and create abstract concept nodes.

        If the same cluster of 3+ nodes activates together across 3+ queries,
        that's an emergent concept. Create a node that binds them.
        """
        if len(self._activation_history) < 5:
            return

        # Find frequently co-activated node pairs
        pair_counts = defaultdict(int)
        for pattern in self._activation_history[-50:]:
            nodes = list(pattern)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pair = tuple(sorted([nodes[i], nodes[j]]))
                    pair_counts[pair] += 1

        # Find clusters that appear 3+ times
        frequent_pairs = {p for p, c in pair_counts.items() if c >= 3}
        if not frequent_pairs:
            return

        # Group into clusters (simple union-find)
        clusters = []
        used = set()
        for p in frequent_pairs:
            a, b = p
            if a in used or b in used:
                # Merge into existing cluster
                for cluster in clusters:
                    if a in cluster or b in cluster:
                        cluster.add(a)
                        cluster.add(b)
                        break
            else:
                clusters.append({a, b})
                used.add(a)
                used.add(b)

        formed = 0
        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Create an abstract node that represents this cluster
            words = []
            for nid in list(cluster)[:4]:
                w = self.lexicon.get_word(nid)
                if w:
                    words.append(w)
            if not words:
                continue

            abstract_name = "_".join(sorted(words[:3])) + "_concept"
            if self.lexicon.word_to_uuid.get(abstract_name):
                continue  # Already exists

            abstract_id = self.lexicon.get_or_create_id(abstract_name)
            self.kernel.add_node(abstract_id)

            # Connect abstract node to all cluster members
            for member_id in cluster:
                if member_id in self.kernel.nodes:
                    self.kernel.add_connection(
                        abstract_id, member_id, 0.6,
                        f"[ABSTRACTION] Emergent concept from repeated co-activation")
            formed += 1

        self.stats['abstractions_formed'] += formed

    def _consolidate(self):
        """Sleep-like consolidation: prune weak edges, strengthen strong ones."""
        if not self.kernel._rust:
            return

        rust = self.kernel._rust

        # Prune very weak edges
        pruned = rust.prune_weak_edges(0.05)

        # Decay myelin slightly (forgetting curve)
        rust.decay_myelin(0.95)

        # Cap edges per node (prevent super-hubs)
        capped = rust.prune_unmyelinated(200)

        self.stats['consolidations'] += 1

    def get_stats(self):
        """Return learning statistics."""
        result = dict(self.stats)
        if self.kernel._rust:
            rust_stats = self.kernel._rust.stats()
            result['total_myelin'] = rust_stats.get('total_myelin', 0)
            result['nodes'] = int(rust_stats.get('nodes', 0))
            result['edges'] = int(rust_stats.get('edges', 0))
        result['activation_history_size'] = len(self._activation_history)
        return result


# Stopwords for self-growing (don't create nodes for these)
_STOP_WORDS = {
    "what", "where", "when", "who", "why", "how", "which",
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "does", "did", "been", "being", "have", "there", "their",
    "here", "also", "just", "very", "really", "quite", "some",
    "any", "each", "every", "own", "same", "other", "such",
    "than", "about", "with", "from", "this", "that", "these",
    "those", "would", "could", "should", "will", "shall",
    "tell", "please", "know", "think", "want", "need",
    "make", "like", "get", "give", "take", "come", "say",
}
