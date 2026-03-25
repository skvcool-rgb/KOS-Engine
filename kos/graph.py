"""
KOS V7.0 — Kernel (Rust Arena Engine + Python Fallback + VSA Backplane).

Default: RustKernel via PyO3 (arena-based, contiguous memory, 10-50x faster).
Fallback: Pure Python priority-queue propagation (if kos_rust not installed).

The Python `nodes` dict is kept as a thin mirror for API compatibility —
router, weaver, lexicon, and daemon all access kernel.nodes directly.
The actual spreading activation runs in Rust when available.
"""
import heapq
import pickle
from collections import defaultdict
from .node import ConceptNode

# ── Try to import Rust backend ─────────────────────────────────────
_RUST_AVAILABLE = False
_RustKernel = None
try:
    from kos_rust import RustKernel as _RustKernel
    _RUST_AVAILABLE = True
except ImportError:
    pass


class KOSKernel:
    def __init__(self, enable_vsa: bool = True, vsa_dimensions: int = 10_000,
                 force_python: bool = False):
        self.nodes = {}
        self.provenance = defaultdict(set)
        self.contradictions = []
        self._lexicon_ref = None
        self.current_tick = 0
        self.max_ticks = 15
        self.tiebreaker = 0
        self._batch_mode = False  # Skip expensive checks during bulk ingestion
        self._working_memory = []  # Recent query seeds (last N)
        self._working_memory_max = 10
        self._neighborhoods = {}  # Precomputed 1-hop neighborhoods

        # ── Rust Backend ───────────────────────────────────────
        self._rust = None
        self._backend = "python"
        if _RUST_AVAILABLE and not force_python:
            self._rust = _RustKernel(
                dim=vsa_dimensions,
                temporal_decay=0.7,
                max_energy=3.0,
                seed=42,
            )
            self._backend = "rust"

        # VSA Backplane — silent hyperdimensional substrate
        # When Rust is active, RustKernel has its own VSA (built-in).
        # We still keep the Python VSA for export/import/resonate_query.
        self.vsa = None
        if enable_vsa:
            try:
                from kasm.bridge import VSABackplane
                self.vsa = VSABackplane(dimensions=vsa_dimensions)
            except ImportError:
                pass

    @property
    def backend(self) -> str:
        """Return which engine is active: 'rust' or 'python'."""
        return self._backend

    def set_lexicon(self, lexicon):
        """Set lexicon reference for contradiction word lookup."""
        self._lexicon_ref = lexicon

    def add_node(self, concept_id: str):
        if concept_id not in self.nodes:
            self.nodes[concept_id] = ConceptNode(concept_id)
            # Mirror to Rust arena
            if self._rust is not None:
                self._rust.add_node(concept_id)
            # Silent VSA registration
            if self.vsa is not None:
                self.vsa.register_node(concept_id)
        return self.nodes[concept_id]

    def add_connection(self, source_id: str, target_id: str,
                       weight: float, source_text: str = "",
                       edge_type: int = None):
        self.add_node(source_id)
        self.add_node(target_id)
        self.nodes[source_id].connections[target_id] = weight
        if source_text:
            pair = tuple(sorted([source_id, target_id]))
            self.provenance[pair].add(source_text)

        # Infer edge type from provenance if not provided
        if edge_type is None and source_text:
            from .edge_types import infer_type
            edge_type = infer_type(source_text)

        # Mirror to Rust arena (with edge type)
        if self._rust is not None:
            self._rust.add_connection(
                source_id, target_id, weight,
                source_text if source_text else None,
                edge_type if edge_type is not None else 0)

        # Contradiction detection at ingestion (skip in batch mode — WordNet is 3ms/call)
        if not self._batch_mode and weight > 0.5 and source_id in self.nodes:
            contradiction = self._check_antonym_contradiction(
                source_id, target_id, weight)
            if contradiction:
                self.contradictions.append(contradiction)

        # Silent VSA binding
        if self.vsa is not None:
            self.vsa.on_edge_created(source_id, target_id, weight)

    # ── Contradiction Detection ────────────────────────────────

    def _check_antonym_contradiction(self, source_id: str,
                                      target_id: str,
                                      weight: float) -> dict:
        try:
            from nltk.corpus import wordnet as wn
        except ImportError:
            return None

        target_word = target_id
        if hasattr(self, '_lexicon_ref') and self._lexicon_ref:
            target_word = self._lexicon_ref.get_word(target_id)
        elif '.' in target_id:
            target_word = target_id.split('.')[0]
        if target_word.startswith('KASM_'):
            target_word = target_word

        target_antonyms = set()
        for syn in wn.synsets(target_word):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    target_antonyms.add(ant.name().lower())

        _DOMAIN_ANTONYMS = {
            "cheap": {"expensive", "costly", "pricey"},
            "expensive": {"cheap", "affordable", "inexpensive"},
            "affordable": {"expensive", "costly"},
            "fast": {"slow"}, "slow": {"fast", "quick", "rapid"},
            "hot": {"cold", "cool"}, "cold": {"hot", "warm"},
            "safe": {"dangerous", "unsafe", "hazardous"},
            "dangerous": {"safe", "harmless"},
            "efficient": {"inefficient", "wasteful"},
            "inefficient": {"efficient"},
            "large": {"small", "tiny"}, "small": {"large", "big", "massive"},
            "strong": {"weak", "fragile"}, "weak": {"strong", "robust"},
            "true": {"false"}, "false": {"true"},
            "good": {"bad", "poor"}, "bad": {"good"},
            "high": {"low"}, "low": {"high"},
            "increase": {"decrease", "reduce"}, "decrease": {"increase"},
        }
        if target_word.lower() in _DOMAIN_ANTONYMS:
            target_antonyms.update(_DOMAIN_ANTONYMS[target_word.lower()])

        if not target_antonyms:
            return None

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

    # ── Spreading Activation ───────────────────────────────────

    def propagate(self, seed_ids: list,
                  seed_energy: float = 3.0) -> dict:
        """
        Spread activation from seed nodes.

        Rust path: delegates to RustKernel.query() — arena-based,
                   contiguous memory, 10-50x faster on large graphs.
        Python path: priority-queue propagation (original V4 engine).
        """
        if self._rust is not None:
            return self._propagate_rust(seed_ids)
        return self._propagate_python(seed_ids, seed_energy)

    def _propagate_rust(self, seed_ids: list) -> dict:
        """Delegate spreading activation to Rust arena engine."""
        # RustKernel.query() returns Vec<(String, f64)>
        # It handles tick increment, decay, fuel, myelination internally.
        results_list = self._rust.query(seed_ids, 500)
        # Convert to dict {node_id: activation}
        results = {}
        for node_id, activation in results_list:
            if activation > 0.1:
                results[node_id] = activation
        return dict(sorted(results.items(),
                           key=lambda x: x[1], reverse=True))

    def _propagate_python(self, seed_ids: list,
                          seed_energy: float = 3.0) -> dict:
        """Pure Python fallback — original V4 priority-queue engine."""
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

        results = {}
        for nid in activated:
            self.nodes[nid]._apply_lazy_decay(self.current_tick)
            if self.nodes[nid].activation > 0.1:
                results[nid] = self.nodes[nid].activation
            self.nodes[nid].fuel = 0.0
            self.nodes[nid].activation = 0.0

        return dict(sorted(results.items(),
                           key=lambda x: x[1], reverse=True))

    def _update_working_memory(self, seeds: list):
        """Track recent query seeds for working-memory bias."""
        for s in seeds:
            if s in self._working_memory:
                self._working_memory.remove(s)
            self._working_memory.append(s)
        # Trim to max size
        if len(self._working_memory) > self._working_memory_max:
            self._working_memory = self._working_memory[-self._working_memory_max:]

    def get_working_memory(self) -> list:
        """Return current working memory (recent query seeds)."""
        return list(self._working_memory)

    def precompute_neighborhood(self, node_id: str) -> list:
        """Cache 1-hop neighbors for a node. Returns list of (target, weight)."""
        if node_id in self._neighborhoods:
            return self._neighborhoods[node_id]
        if self._rust is not None:
            neighbors = self._rust.get_neighbors(node_id)
            result = [(n, w) for n, w, _, _ in neighbors]
        elif node_id in self.nodes:
            result = list(self.nodes[node_id].connections.items())
        else:
            result = []
        self._neighborhoods[node_id] = result
        return result

    def invalidate_neighborhoods(self):
        """Clear precomputed neighborhoods (call after graph changes)."""
        self._neighborhoods.clear()

    def query(self, concept_ids: list, top_k: int = 5):
        self.current_tick += 1
        self._update_working_memory(concept_ids)
        results = self.propagate(concept_ids)
        return [(n, e) for n, e in results.items()
                if n not in concept_ids][:top_k]

    def query_beam(self, concept_ids: list, top_k: int = 5,
                   beam_width: int = 32, max_depth: int = 5,
                   allowed_edge_types: list = None):
        """Beam-search retrieval (Rust only). Falls back to query()."""
        self.current_tick += 1
        self._update_working_memory(concept_ids)
        if self._rust is not None:
            results = self._rust.query_beam(
                concept_ids, top_k, beam_width, max_depth,
                allowed_edge_types)
            return [(n, s) for n, s in results if n not in concept_ids]
        return self.query(concept_ids, top_k)

    def query_causal(self, concept_ids: list, top_k: int = 5):
        """Causal-only retrieval: CAUSES + TEMPORAL edges only."""
        self.current_tick += 1
        if self._rust is not None:
            results = self._rust.query_causal(concept_ids, top_k)
            return [(n, s) for n, s in results if n not in concept_ids]
        # Python fallback: regular query (no edge type filtering)
        return self.query(concept_ids, top_k)

    def batch_add_connections(self, edges: list):
        """Batch-add edges: [(src, tgt, weight, text), ...].
        Skips contradiction checks. Pushes to Rust in one call."""
        old_batch = self._batch_mode
        self._batch_mode = True
        rust_edges = []
        for src, tgt, weight, text in edges:
            self.add_node(src)
            self.add_node(tgt)
            self.nodes[src].connections[tgt] = weight
            if text:
                pair = tuple(sorted([src, tgt]))
                self.provenance[pair].add(text)
            rust_edges.append((src, tgt, float(weight)))
        # Single Rust call for all edges
        if self._rust is not None and hasattr(self._rust, 'batch_add_edges'):
            self._rust.batch_add_edges(rust_edges)
        elif self._rust is not None:
            for src, tgt, w in rust_edges:
                self._rust.add_connection(src, tgt, w, None)
        self._batch_mode = old_batch

    # ── Quantitative Comparison ────────────────────────────────

    def compare(self, node_a_id: str, node_b_id: str,
                property_name: str = None) -> dict:
        node_a = self.nodes.get(node_a_id)
        node_b = self.nodes.get(node_b_id)

        if not node_a or not node_b:
            return {"error": "One or both nodes not found"}

        props_a = node_a.properties
        props_b = node_b.properties

        if property_name:
            if property_name in props_a and property_name in props_b:
                va, vb = props_a[property_name], props_b[property_name]
                return {property_name: self._compare_values(va, vb)}
            return {"error": f"Property '{property_name}' not found on both nodes"}

        shared = set(props_a.keys()) & set(props_b.keys())
        shared = {k for k in shared if not k.startswith('_')}
        if not shared:
            return {"error": "No shared properties to compare"}

        results = {}
        for prop in shared:
            results[prop] = self._compare_values(props_a[prop], props_b[prop])
        return results

    @staticmethod
    def _compare_values(va, vb) -> dict:
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

    # ── VSA Operations ─────────────────────────────────────────

    def resonate_match(self, query_id: str,
                       threshold: float = 0.05) -> list:
        # Try Rust VSA first (built into RustKernel)
        if self._rust is not None:
            try:
                return self._rust.resonate_search(query_id, 20, threshold)
            except Exception:
                pass
        # Fall back to Python VSA
        if self.vsa is None:
            return []
        candidates = list(self.nodes.keys())
        return self.vsa.resonate_query(query_id, candidates, threshold)

    def export_vectors(self, filepath: str) -> int:
        if self.vsa is None:
            raise RuntimeError("VSA backplane not enabled")
        return self.vsa.export_vectors(filepath)

    def import_vectors(self, filepath: str, merge: bool = True) -> int:
        if self.vsa is None:
            raise RuntimeError("VSA backplane not enabled")
        return self.vsa.import_vectors(filepath, merge)

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return kernel statistics."""
        n = len(self.nodes)
        e = sum(len(node.connections) for node in self.nodes.values())
        s = {"nodes": n, "edges": e, "backend": self._backend}
        if self._rust is not None:
            rust_stats = self._rust.stats()
            s["rust_nodes"] = int(rust_stats.get("nodes", 0))
            s["rust_edges"] = int(rust_stats.get("edges", 0))
            s["vsa_dim"] = int(rust_stats.get("vsa_dim", 0))
            s["vsa_mb"] = rust_stats.get("vsa_mb", 0)
            s["arena_contiguous"] = bool(rust_stats.get("arena_contiguous", 0))
        return s

    # ── Persistence ────────────────────────────────────────────

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
        # Rebuild Rust arena from loaded Python nodes
        if self._rust is not None:
            for nid, node in self.nodes.items():
                self._rust.add_node(nid)
            for nid, node in self.nodes.items():
                for tgt_id, weight in node.connections.items():
                    pair = tuple(sorted([nid, tgt_id]))
                    prov_texts = list(self.provenance.get(pair, set()))
                    text = prov_texts[0] if prov_texts else None
                    self._rust.add_connection(nid, tgt_id, weight, text)

    # ── Visualization ──────────────────────────────────────────

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
