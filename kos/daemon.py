"""
KOS V5.0 — Autonomic Daemon (Background Brain Maintenance).

Five autonomic systems that run during idle cycles:
1. Garbage Collection: Prunes orphan nodes with zero connections.
2. Structural Deduplication: Merges isomorphic nodes via Jaccard similarity.
3. Predictive Reasoning: Triadic Closure — if A->B and B->C, infer A->C.
4. Analogical Abstraction: Layer 3 VSA metaphor detection (if enabled).
5. Proactive Attention: Self-generated curiosity, anticipation, staleness goals.
"""
import time
from collections import defaultdict


class KOSDaemon:
    def __init__(self, kernel, lexicon=None, forager=None,
                 attention_controller=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.forager = forager
        self.attention = attention_controller
        self._layer3 = None

    def _ensure_layer3(self):
        """Lazy-init Layer 3 abstraction engine."""
        if self._layer3 is None and hasattr(self.kernel, 'vsa') and self.kernel.vsa is not None:
            try:
                from kasm.abstraction import Layer3Daemon
                self._layer3 = Layer3Daemon(self.kernel, self.lexicon)
            except ImportError:
                pass

    def run_maintenance_cycle(self, enable_attention: bool = True,
                               max_forage_actions: int = 2) -> dict:
        """Executes the full suite of background autonomic processes."""
        print("\n[DAEMON] Waking up. Initiating deep-brain structural scan...")

        start_time = time.perf_counter()

        # Run the core autonomic protocols
        orphans_removed = self._prune_orphans()
        nodes_merged = self._merge_isomorphs()
        predictions_made = self._dream_triadic_closure()

        # Layer 3: Analogical Abstraction
        self._ensure_layer3()
        analogies_found = 0
        top_analogies = []
        if self._layer3:
            l3_report = self._layer3.run()
            analogies_found = l3_report["analogies_found"]
            top_analogies = l3_report.get("top", [])

        # Phase 2: Proactive Attention Controller
        attention_report = None
        if enable_attention and self.attention and self.forager:
            attention_report = self.attention.act_on_goals(
                self.forager, max_actions=max_forage_actions)

        elapsed = (time.perf_counter() - start_time) * 1000

        report = {
            "time_ms": elapsed,
            "orphans_pruned": orphans_removed,
            "isomorphs_merged": nodes_merged,
            "predicted_edges": predictions_made,
            "analogies_discovered": analogies_found,
            "top_analogies": top_analogies,
            "attention": attention_report,
        }

        print(f"\n[DAEMON] Cycle complete in {elapsed:.2f}ms.")
        print(f"   [-] Pruned {orphans_removed} memory leaks (Orphans)")
        print(f"   [v] Fused {nodes_merged} duplicate concepts (Graph Isomorphism)")
        print(f"   [+] Dreamt {predictions_made} new logical connections (Triadic Closure)")
        print(f"   [~] Discovered {analogies_found} structural analogies (Layer 3)")

        if attention_report:
            print(f"   [*] Attention: {attention_report['goals_generated']} goals, "
                  f"{attention_report['actions_taken']} actions, "
                  f"+{attention_report['concepts_acquired']} concepts")

        if top_analogies:
            lex = self.lexicon
            print(f"   [~] Top analogies:")
            for a, b, score in top_analogies[:5]:
                name_a = lex.get_word(a) if lex else a
                name_b = lex.get_word(b) if lex else b
                print(f"       {name_a} <=> {name_b}  (similarity: {score:+.4f})")

        return report

    def _prune_orphans(self) -> int:
        """O(V + E) Optimization: Scans edges once, avoiding O(N^2) loops."""
        inbound_tracker = set()

        for node in self.kernel.nodes.values():
            inbound_tracker.update(node.connections.keys())

        prune_list = [nid for nid, node in self.kernel.nodes.items()
                      if not node.connections and nid not in inbound_tracker]

        for node_id in prune_list:
            self.kernel.nodes.pop(node_id, None)

        return len(prune_list)

    def _merge_isomorphs(self, overlap_threshold: float = 0.85) -> int:
        """Degree-Bucketing Optimization: Kills the O(N^2) trap."""

        degree_buckets = defaultdict(list)
        for nid, node in self.kernel.nodes.items():
            if node.connections:
                degree_buckets[len(node.connections)].append((nid, node))

        merge_commands = []
        merges_executed = 0

        for bucket_size, group in degree_buckets.items():
            if len(group) < 2:
                continue

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    id_a, node_a = group[i]
                    id_b, node_b = group[j]

                    targets_a = set(node_a.connections.keys())
                    targets_b = set(node_b.connections.keys())

                    intersection = targets_a.intersection(targets_b)
                    similarity = len(intersection) / len(targets_a.union(targets_b))

                    if similarity >= overlap_threshold:
                        merge_commands.append((id_a, id_b))

        for id_a, id_b in merge_commands:
            if id_a in self.kernel.nodes and id_b in self.kernel.nodes:
                for target, data in self.kernel.nodes[id_b].connections.items():
                    if target not in self.kernel.nodes[id_a].connections:
                        self.kernel.nodes[id_a].connections[target] = data

                for n in self.kernel.nodes.values():
                    if id_b in n.connections:
                        n.connections[id_a] = n.connections.pop(id_b)

                self.kernel.nodes.pop(id_b, None)
                merges_executed += 1

        return merges_executed

    def _dream_triadic_closure(self) -> int:
        """
        The Inference Engine: If A->B and B->C, infer A->C.
        Wires a 'Predicted' edge while the system is idle.
        """
        new_edges = []

        for root_id, root_node in self.kernel.nodes.items():
            for hop1_id, weight1 in root_node.connections.items():

                if weight1 < 0.7 or hop1_id not in self.kernel.nodes:
                    continue

                hop1_node = self.kernel.nodes[hop1_id]
                for hop2_id, weight2 in hop1_node.connections.items():

                    if weight2 >= 0.7:
                        if hop2_id != root_id and hop2_id not in root_node.connections:
                            predicted_confidence = weight1 * weight2
                            new_edges.append((root_id, hop2_id, predicted_confidence))

        count = 0
        for A, C, conf in new_edges:
            safe_conf = min(0.5, conf)
            self.kernel.add_connection(
                A, C, safe_conf,
                source_text="[DAEMON INFERENCE] Automatically predicted via Triadic Closure."
            )
            count += 1

        return count
