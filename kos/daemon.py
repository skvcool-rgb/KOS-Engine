"""
KOS V2.0 — Autonomic Daemon (Background Brain Maintenance).

Three autonomic systems that run during idle cycles:
1. Garbage Collection: Prunes orphan nodes with zero connections (memory leaks).
2. Structural Deduplication: Merges isomorphic nodes via Jaccard similarity.
3. Predictive Reasoning: Triadic Closure — if A->B and B->C, infer A->C.
"""
import time
from collections import defaultdict


class KOSDaemon:
    def __init__(self, kernel):
        self.kernel = kernel

    def run_maintenance_cycle(self) -> dict:
        """Executes the full suite of background autonomic repairs."""
        print("\n[DAEMON] Waking up. Initiating deep-brain structural scan...")

        start_time = time.perf_counter()

        # Run the 3 core autonomic protocols
        orphans_removed = self._prune_orphans()
        nodes_merged = self._merge_isomorphs()
        predictions_made = self._dream_triadic_closure()

        elapsed = (time.perf_counter() - start_time) * 1000

        report = {
            "time_ms": elapsed,
            "orphans_pruned": orphans_removed,
            "isomorphs_merged": nodes_merged,
            "predicted_edges": predictions_made
        }

        print(f"[DAEMON] Cycle complete in {elapsed:.2f}ms.")
        print(f"   [-] Pruned {orphans_removed} memory leaks (Orphans)")
        print(f"   [v] Fused {nodes_merged} duplicate concepts (Graph Isomorphism)")
        print(f"   [+] Dreamt {predictions_made} new logical connections (Triadic Closure)")

        return report

    def _prune_orphans(self) -> int:
        """O(V + E) Optimization: Scans edges once, avoiding O(N^2) loops."""
        inbound_tracker = set()

        # 1. Single pass to log every node that is targeted by an edge
        for node in self.kernel.nodes.values():
            inbound_tracker.update(node.connections.keys())

        # 2. Excision: If it points to nothing, and nothing points to it.
        prune_list = [nid for nid, node in self.kernel.nodes.items()
                      if not node.connections and nid not in inbound_tracker]

        for node_id in prune_list:
            self.kernel.nodes.pop(node_id, None)

        return len(prune_list)

    def _merge_isomorphs(self, overlap_threshold: float = 0.85) -> int:
        """Degree-Bucketing Optimization: Kills the O(N^2) trap."""

        # 1. Group nodes by exactly how many outbound connections they have
        degree_buckets = defaultdict(list)
        for nid, node in self.kernel.nodes.items():
            if node.connections:  # Only group active nodes
                degree_buckets[len(node.connections)].append((nid, node))

        merge_commands = []
        merges_executed = 0

        # 2. Only compare nodes inside the exact same degree bucket
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

        # 3. Execute Merges
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
        The Inference Engine: If A->B and B->C, the logic strongly suggests A->C.
        Wires a 'Predicted' edge while the system is idle.
        """
        new_edges = []

        for root_id, root_node in self.kernel.nodes.items():
            for hop1_id, weight1 in root_node.connections.items():

                # Only trust strong, explicit excitatory edges for logic chains
                if weight1 < 0.7 or hop1_id not in self.kernel.nodes:
                    continue

                hop1_node = self.kernel.nodes[hop1_id]
                for hop2_id, weight2 in hop1_node.connections.items():

                    # If Hopper 1 points strongly to Hopper 2
                    if weight2 >= 0.7:
                        # If Root doesn't already know about Hopper 2... predict it!
                        if hop2_id != root_id and hop2_id not in root_node.connections:
                            # Synthesize the confidence (e.g., 0.9 * 0.9 = 0.81 probability)
                            predicted_confidence = weight1 * weight2
                            new_edges.append((root_id, hop2_id, predicted_confidence))

        # Wire the dreams into reality
        count = 0
        for A, C, conf in new_edges:
            # Wire a weak, predicted edge (we cap it at 0.5 so it doesn't override explicit facts)
            safe_conf = min(0.5, conf)
            self.kernel.add_connection(
                A, C, safe_conf,
                source_text="[DAEMON INFERENCE] Automatically predicted via Triadic Closure."
            )
            count += 1

        return count
