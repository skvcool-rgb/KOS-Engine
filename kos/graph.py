"""
KOS V2.0 — Kernel (Spreading Activation Engine).

Priority-queue driven propagation with inline reset.
Provenance tracking for source evidence citations.
Save/load brain persistence via pickle.
"""
import heapq
import pickle
from collections import defaultdict
from .node import ConceptNode


class KOSKernel:
    def __init__(self):
        self.nodes = {}
        self.provenance = defaultdict(set)
        self.current_tick = 0
        self.max_ticks = 15
        self.tiebreaker = 0

    def add_node(self, concept_id: str):
        if concept_id not in self.nodes:
            self.nodes[concept_id] = ConceptNode(concept_id)
        return self.nodes[concept_id]

    def add_connection(self, source_id: str, target_id: str,
                       weight: float, source_text: str = ""):
        self.add_node(source_id)
        self.add_node(target_id)
        self.nodes[source_id].connections[target_id] = weight
        if source_text:
            pair = tuple(sorted([source_id, target_id]))
            self.provenance[pair].add(source_text)

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
