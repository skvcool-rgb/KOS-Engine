"""
KOS V4.1 — Kernel (Spreading Activation Engine + VSA Backplane).

Priority-queue driven propagation with inline reset.
Provenance tracking for source evidence citations.
Save/load brain persistence via pickle.
VSA backplane: every node silently accumulates a 10,000-D state vector.
"""
import heapq
import pickle
from collections import defaultdict
from .node import ConceptNode


class KOSKernel:
    def __init__(self, enable_vsa: bool = True, vsa_dimensions: int = 10_000):
        self.nodes = {}
        self.provenance = defaultdict(set)
        self.contradictions = []  # FIX #9: detected contradictions log
        self._lexicon_ref = None  # Set via set_lexicon() for contradiction word lookup
        self.current_tick = 0
        self.max_ticks = 15
        self.tiebreaker = 0

        # VSA Backplane — silent hyperdimensional substrate
        self.vsa = None
        if enable_vsa:
            try:
                from kasm.bridge import VSABackplane
                self.vsa = VSABackplane(dimensions=vsa_dimensions)
            except ImportError:
                pass  # KASM not available, run in scalar-only mode

    def set_lexicon(self, lexicon):
        """Set lexicon reference for contradiction word lookup."""
        self._lexicon_ref = lexicon

    def add_node(self, concept_id: str):
        if concept_id not in self.nodes:
            self.nodes[concept_id] = ConceptNode(concept_id)
            # Silent VSA registration
            if self.vsa is not None:
                self.vsa.register_node(concept_id)
        return self.nodes[concept_id]

    def add_connection(self, source_id: str, target_id: str,
                       weight: float, source_text: str = ""):
        self.add_node(source_id)
        self.add_node(target_id)
        self.nodes[source_id].connections[target_id] = weight
        if source_text:
            pair = tuple(sorted([source_id, target_id]))
            self.provenance[pair].add(source_text)

        # FIX #9: Contradiction detection at ingestion
        # Check if the source already has a connection to an antonym of target
        if weight > 0.5 and source_id in self.nodes:
            contradiction = self._check_antonym_contradiction(
                source_id, target_id, weight)
            if contradiction:
                self.contradictions.append(contradiction)

        # Silent VSA binding
        if self.vsa is not None:
            self.vsa.on_edge_created(source_id, target_id, weight)

    # ── FIX #9: Contradiction Detection ──────────────────────

    def _check_antonym_contradiction(self, source_id: str,
                                      target_id: str,
                                      weight: float) -> dict:
        """
        Check if source already connects to an antonym of target.

        Uses WordNet antonym lookup + domain antonym map.
        Returns contradiction record or None.
        """
        try:
            from nltk.corpus import wordnet as wn
        except ImportError:
            return None

        # Get the plain word for the new target
        # Use lexicon reverse lookup if available
        target_word = target_id
        if hasattr(self, '_lexicon_ref') and self._lexicon_ref:
            target_word = self._lexicon_ref.get_word(target_id)
        elif '.' in target_id:
            target_word = target_id.split('.')[0]
        # Strip KASM_ prefix if present
        if target_word.startswith('KASM_'):
            target_word = target_word  # Can't decode — skip

        # Get WordNet antonyms for target word
        target_antonyms = set()
        for syn in wn.synsets(target_word):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    target_antonyms.add(ant.name().lower())

        # Supplement with common domain antonym pairs
        _DOMAIN_ANTONYMS = {
            "cheap": {"expensive", "costly", "pricey"},
            "expensive": {"cheap", "affordable", "inexpensive"},
            "affordable": {"expensive", "costly"},
            "fast": {"slow"},
            "slow": {"fast", "quick", "rapid"},
            "hot": {"cold", "cool"},
            "cold": {"hot", "warm"},
            "safe": {"dangerous", "unsafe", "hazardous"},
            "dangerous": {"safe", "harmless"},
            "efficient": {"inefficient", "wasteful"},
            "inefficient": {"efficient"},
            "large": {"small", "tiny"},
            "small": {"large", "big", "massive"},
            "strong": {"weak", "fragile"},
            "weak": {"strong", "robust"},
            "true": {"false"},
            "false": {"true"},
            "good": {"bad", "poor"},
            "bad": {"good"},
            "high": {"low"},
            "low": {"high"},
            "increase": {"decrease", "reduce"},
            "decrease": {"increase"},
        }
        if target_word.lower() in _DOMAIN_ANTONYMS:
            target_antonyms.update(_DOMAIN_ANTONYMS[target_word.lower()])

        if not target_antonyms:
            return None

        # Check if source already connects to any antonym
        node = self.nodes.get(source_id)
        if not node:
            return None

        for existing_target in node.connections:
            existing_word = existing_target
            if hasattr(self, '_lexicon_ref') and self._lexicon_ref:
                existing_word = self._lexicon_ref.get_word(existing_target)
            elif '.' in existing_target:
                existing_word = existing_target.split('.')[0]

            if existing_word.lower() in target_antonyms:
                return {
                    'source': source_id,
                    'existing_target': existing_target,
                    'new_target': target_id,
                    'existing_word': existing_word,
                    'new_word': target_word,
                    'type': 'antonym_contradiction',
                    'tick': self.current_tick,
                }

        return None

    def propagate(self, seed_ids: list,
                  seed_energy: float = 3.0) -> dict:
        activated = set()
        pq = []

        for cid in seed_ids:
            if cid in self.nodes:
                self.nodes[cid].receive_signal(seed_energy,
                                               self.current_tick)
                self.tiebreaker += 1
                heapq.heappush(pq, (-self.nodes[cid].fuel,
                                     self.tiebreaker, cid))
                activated.add(cid)

        ticks_run = 0
        while pq and ticks_run < self.max_ticks:
            _, _, nid = heapq.heappop(pq)
            if self.nodes[nid].fuel < 0.05:
                continue
            ticks_run += 1

            for tgt_id, passed_energy in self.nodes[nid].propagate(
                    self.current_tick):
                self.nodes[tgt_id].receive_signal(passed_energy,
                                                  self.current_tick)
                activated.add(tgt_id)
                if self.nodes[tgt_id].fuel >= 0.05:
                    self.tiebreaker += 1
                    heapq.heappush(pq, (-self.nodes[tgt_id].fuel,
                                         self.tiebreaker, tgt_id))

        # Collect results and inline reset
        results = {}
        for nid in activated:
            self.nodes[nid]._apply_lazy_decay(self.current_tick)
            if self.nodes[nid].activation > 0.1:
                results[nid] = self.nodes[nid].activation
            self.nodes[nid].fuel = 0.0
            self.nodes[nid].activation = 0.0

        return dict(sorted(results.items(),
                           key=lambda x: x[1], reverse=True))

    def query(self, concept_ids: list, top_k: int = 5):
        self.current_tick += 1
        results = self.propagate(concept_ids)
        return [(n, e) for n, e in results.items()
                if n not in concept_ids][:top_k]

    # ── FIX #8: Quantitative Comparison ─────────────────────

    def compare(self, node_a_id: str, node_b_id: str,
                property_name: str = None) -> dict:
        """
        Compare two nodes quantitatively.

        If property_name is given, compares that specific property.
        Otherwise, compares all shared properties.

        Returns: {property: {a_value, b_value, comparison, ratio}}
        """
        node_a = self.nodes.get(node_a_id)
        node_b = self.nodes.get(node_b_id)

        if not node_a or not node_b:
            return {"error": "One or both nodes not found"}

        props_a = node_a.properties
        props_b = node_b.properties

        if property_name:
            # Compare specific property
            if property_name in props_a and property_name in props_b:
                va, vb = props_a[property_name], props_b[property_name]
                return {property_name: self._compare_values(va, vb)}
            return {"error": f"Property '{property_name}' not found on both nodes"}

        # Compare all shared properties
        shared = set(props_a.keys()) & set(props_b.keys())
        # Exclude internal metadata keys
        shared = {k for k in shared if not k.startswith('_')}

        if not shared:
            return {"error": "No shared properties to compare"}

        results = {}
        for prop in shared:
            results[prop] = self._compare_values(
                props_a[prop], props_b[prop])
        return results

    @staticmethod
    def _compare_values(va, vb) -> dict:
        """Compare two numeric values."""
        try:
            va_f, vb_f = float(va), float(vb)
            if va_f > vb_f:
                comparison = "greater"
            elif va_f < vb_f:
                comparison = "less"
            else:
                comparison = "equal"
            ratio = va_f / vb_f if vb_f != 0 else float('inf')
            diff = va_f - vb_f
            return {
                "a": va_f, "b": vb_f,
                "comparison": comparison,
                "difference": diff,
                "ratio": round(ratio, 4),
            }
        except (ValueError, TypeError):
            return {"a": va, "b": vb, "comparison": "incomparable"}

    # ── VSA Operations (Fused KASM) ────────────────────────────

    def resonate_match(self, query_id: str,
                       threshold: float = 0.05) -> list:
        """
        Find concepts semantically similar to query_id using
        VSA cosine similarity on state vectors.

        Returns: [(node_id, similarity_score), ...]
        """
        if self.vsa is None:
            return []
        candidates = list(self.nodes.keys())
        return self.vsa.resonate_query(query_id, candidates, threshold)

    def export_vectors(self, filepath: str) -> int:
        """Export graph knowledge as compressed VSA vectors (.npz)."""
        if self.vsa is None:
            raise RuntimeError("VSA backplane not enabled")
        return self.vsa.export_vectors(filepath)

    def import_vectors(self, filepath: str, merge: bool = True) -> int:
        """Import VSA vectors from another KOS instance."""
        if self.vsa is None:
            raise RuntimeError("VSA backplane not enabled")
        return self.vsa.import_vectors(filepath, merge)

    # ── Persistence ───────────────────────────────────────────

    def save_brain(self, filepath: str):
        data = {
            'nodes': self.nodes,
            'provenance': dict(self.provenance),
            'current_tick': self.current_tick,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_brain(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.nodes = data['nodes']
        self.provenance = defaultdict(set, data['provenance'])
        self.current_tick = data['current_tick']

    # ── Visualization ─────────────────────────────────────────

    def export_graph_html(self, filepath: str = "kos_brain.html"):
        from pyvis.network import Network
        net = Network(height="750px", width="100%",
                      directed=True, bgcolor="#222222",
                      font_color="white")
        net.barnes_hut()

        for nid, node in self.nodes.items():
            size = 10 + len(node.connections) * 2
            net.add_node(nid, label=nid, size=size,
                         title=f"{nid}\nConnections: {len(node.connections)}")

        for nid, node in self.nodes.items():
            for target_id, weight in node.connections.items():
                color = "#ff4444" if weight < 0 else (
                    "#44ff44" if weight >= 0.8 else "#888888")
                pair = tuple(sorted([nid, target_id]))
                prov = self.provenance.get(pair, set())
                title = f"w={weight:.2f}"
                if prov:
                    title += f"\n{list(prov)[0][:80]}"
                net.add_edge(nid, target_id, value=abs(weight),
                             color=color, title=title)

        net.save_graph(filepath)
