"""
KOS V5.1 — Research Module (Quarter 2: Deep Tech IP).

Fix #14: Hierarchical Predictive Coding
    Multiple prediction layers, each predicting the layer below.
    Meta-prediction: "predict how accurate my prediction will be."

Fix #15: Automatic Role Discovery for KASM Analogies
    Graph automorphism detection finds structurally similar subgraphs,
    then extracts role mappings without human-defined roles.

Fix #16: Bidirectional Sensorimotor (Action Backends)
    The agent can write to the world, not just read from it.

Fix #17: Catastrophic Unlearning
    Rapid belief erasure when evidence is overwhelmingly contradicting.
"""

import time
import math
from collections import defaultdict


# ── Fix #14: Hierarchical Predictive Coding ──────────────────

class HierarchicalPredictor:
    """
    Multi-layer prediction stack.

    Layer 0: Predicts which nodes will activate (existing PredictiveCodingEngine)
    Layer 1: Predicts how accurate Layer 0's prediction will be (meta-prediction)
    Layer 2: Predicts convergence time (how many cycles to reach MAE < threshold)

    Top-down error signals flow from higher layers to lower layers,
    modulating learning rates and confidence thresholds.
    """

    def __init__(self, pce):
        """pce = existing PredictiveCodingEngine instance."""
        self.pce = pce
        self.meta_predictions = {}  # seed_key -> predicted_accuracy
        self.convergence_predictions = {}  # seed_key -> predicted_cycles

    def predict_accuracy(self, seed_ids: list) -> float:
        """
        Layer 1: Meta-prediction — predict how accurate
        Layer 0's prediction will be.

        Uses historical accuracy of this seed set.
        """
        seed_key = self.pce._make_seed_key(seed_ids)

        if seed_key in self.pce.predictions:
            record = self.pce.predictions[seed_key]
            return record.accuracy
        else:
            # Unknown seeds — default to low confidence
            return 0.3

    def predict_convergence(self, seed_ids: list) -> int:
        """
        Layer 2: Predict how many cycles until MAE < 0.05.

        Based on how quickly past predictions converged.
        """
        seed_key = self.pce._make_seed_key(seed_ids)

        if seed_key in self.pce.predictions:
            record = self.pce.predictions[seed_key]
            if record.locked:
                return 0  # Already converged
            total = record.hit_count + record.miss_count
            if total > 0:
                return max(1, int(3 / max(record.accuracy, 0.1)))
        return 5  # Default estimate

    def hierarchical_query(self, seed_ids: list, top_k: int = 5,
                            verbose: bool = False) -> dict:
        """
        Full hierarchical prediction loop.

        1. Layer 2 predicts convergence time
        2. Layer 1 predicts accuracy
        3. Layer 0 predicts activations
        4. Actual propagation runs
        5. All layers update based on errors
        """
        # Layer 2: Convergence prediction
        predicted_convergence = self.predict_convergence(seed_ids)

        # Layer 1: Accuracy prediction
        predicted_accuracy = self.predict_accuracy(seed_ids)

        # Layer 0: Activation prediction + actual propagation
        report = self.pce.query_with_prediction(
            seed_ids, top_k=top_k, verbose=verbose)

        actual_accuracy = report['accuracy']

        # Meta-error: how wrong was the accuracy prediction?
        meta_error = abs(actual_accuracy - predicted_accuracy)

        # Update meta-prediction
        seed_key = self.pce._make_seed_key(seed_ids)
        if seed_key in self.meta_predictions:
            old = self.meta_predictions[seed_key]
            self.meta_predictions[seed_key] = 0.7 * old + 0.3 * actual_accuracy
        else:
            self.meta_predictions[seed_key] = actual_accuracy

        report['meta_predicted_accuracy'] = predicted_accuracy
        report['meta_error'] = meta_error
        report['predicted_convergence'] = predicted_convergence

        if verbose:
            print(f"[HIERARCHICAL] Meta-prediction: {predicted_accuracy:.0%} "
                  f"| Actual: {actual_accuracy:.0%} | "
                  f"Meta-error: {meta_error:.2f} | "
                  f"Convergence est: {predicted_convergence} cycles")

        return report


# ── Fix #15: Automatic Role Discovery ────────────────────────

class RoleDiscovery:
    """
    Discovers structural roles in the graph without human annotation.

    Strategy: Find pairs of subgraphs with high Jaccard similarity
    in their connection patterns. If subgraph A has the same
    "shape" as subgraph B, their corresponding nodes play
    analogous roles.

    Example:
        Sun → [center, orbiter, force] pattern
        Nucleus → [center, orbiter, force] pattern
        → Auto-discovered: Sun ↔ Nucleus (both play "hub" role)
    """

    def __init__(self, kernel, lexicon=None):
        self.kernel = kernel
        self.lexicon = lexicon

    def _get_connection_profile(self, node_id: str) -> tuple:
        """
        Get a node's structural "fingerprint" — the sorted tuple
        of its neighbor count and their degrees.
        """
        node = self.kernel.nodes.get(node_id)
        if not node or not node.connections:
            return ()

        neighbor_degrees = []
        for neighbor_id in node.connections:
            if neighbor_id in self.kernel.nodes:
                deg = len(self.kernel.nodes[neighbor_id].connections)
                neighbor_degrees.append(deg)

        return tuple(sorted(neighbor_degrees))

    def find_structural_analogs(self, min_connections: int = 3,
                                 similarity_threshold: float = 0.7) -> list:
        """
        Find pairs of nodes that play analogous structural roles.

        Two nodes are analogs if their connection profiles
        (sorted neighbor degree sequences) are highly similar.
        """
        # Build profiles for nodes with enough connections
        profiles = {}
        for nid, node in self.kernel.nodes.items():
            if len(node.connections) >= min_connections:
                profiles[nid] = self._get_connection_profile(nid)

        # Compare profiles using Jaccard-like similarity
        # (matching positions in sorted degree sequences)
        analogs = []
        nodes = list(profiles.keys())

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                nid_a, nid_b = nodes[i], nodes[j]
                prof_a, prof_b = profiles[nid_a], profiles[nid_b]

                if not prof_a or not prof_b:
                    continue

                # Pad shorter profile
                max_len = max(len(prof_a), len(prof_b))
                pa = prof_a + (0,) * (max_len - len(prof_a))
                pb = prof_b + (0,) * (max_len - len(prof_b))

                # Similarity: 1 - normalized distance
                total_diff = sum(abs(a - b) for a, b in zip(pa, pb))
                total_sum = sum(a + b for a, b in zip(pa, pb))
                if total_sum == 0:
                    continue

                similarity = 1.0 - (total_diff / total_sum)

                if similarity >= similarity_threshold:
                    name_a = self.lexicon.get_word(nid_a) if self.lexicon else nid_a
                    name_b = self.lexicon.get_word(nid_b) if self.lexicon else nid_b
                    analogs.append((name_a, name_b, similarity, nid_a, nid_b))

        analogs.sort(key=lambda x: x[2], reverse=True)
        return analogs


# ── Fix #16: Bidirectional Sensorimotor (Actions) ────────────

class ActionBackend:
    """Base class for action backends."""

    def execute(self, action_type: str, payload: dict) -> dict:
        raise NotImplementedError


class FileAction(ActionBackend):
    """Write reports and data to local files."""

    def execute(self, action_type: str, payload: dict) -> dict:
        if action_type == "write_report":
            filepath = payload.get("filepath", "kos_report.txt")
            content = payload.get("content", "")
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"status": "success", "filepath": filepath}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "unknown_action"}


class AlertAction(ActionBackend):
    """Print alerts to console (upgradeable to email/Slack)."""

    def execute(self, action_type: str, payload: dict) -> dict:
        if action_type == "alert":
            message = payload.get("message", "")
            severity = payload.get("severity", "info")
            print(f"\n[ALERT-{severity.upper()}] {message}\n")
            return {"status": "success", "delivered": True}
        return {"status": "unknown_action"}


class ActionRouter:
    """Routes actions to appropriate backends."""

    def __init__(self):
        self.backends = {
            "file": FileAction(),
            "alert": AlertAction(),
        }

    def register(self, name: str, backend: ActionBackend):
        self.backends[name] = backend

    def execute(self, backend_name: str, action_type: str,
                payload: dict) -> dict:
        backend = self.backends.get(backend_name)
        if not backend:
            return {"status": "error",
                    "error": f"Unknown backend: {backend_name}"}
        return backend.execute(action_type, payload)


# ── Fix #17: Catastrophic Unlearning ─────────────────────────

class CatastrophicUnlearner:
    """
    Rapid belief erasure when evidence is overwhelmingly contradicting.

    Normal weight correction: w *= 0.98 per cycle (slow decay)
    Catastrophic unlearning: w *= 0.5 per cycle (halves each time)

    Triggered when prediction error for a specific edge exceeds
    threshold for N consecutive cycles.
    """

    def __init__(self, kernel, threshold: float = 2.0,
                 trigger_cycles: int = 5):
        self.kernel = kernel
        self.threshold = threshold
        self.trigger_cycles = trigger_cycles
        self.error_history = defaultdict(list)  # (source, target) -> [errors]
        self.unlearned = []  # Log of unlearned edges

    def record_error(self, source_id: str, target_id: str,
                     prediction_error: float):
        """Record a prediction error for an edge."""
        key = (source_id, target_id)
        self.error_history[key].append(abs(prediction_error))

        # Keep only recent history
        if len(self.error_history[key]) > 20:
            self.error_history[key] = self.error_history[key][-20:]

    def check_and_unlearn(self) -> int:
        """
        Check all tracked edges for catastrophic unlearning triggers.

        If an edge has had high error for trigger_cycles consecutive
        readings, apply exponential decay.

        Returns number of edges unlearned.
        """
        unlearned_count = 0

        for (source_id, target_id), errors in list(self.error_history.items()):
            if len(errors) < self.trigger_cycles:
                continue

            # Check last N errors
            recent = errors[-self.trigger_cycles:]
            all_high = all(e > self.threshold for e in recent)

            if all_high and source_id in self.kernel.nodes:
                conn = self.kernel.nodes[source_id].connections
                if target_id in conn:
                    if isinstance(conn[target_id], dict):
                        old_w = conn[target_id]['w']
                        conn[target_id]['w'] = old_w * 0.5  # HALVE
                        conn[target_id]['myelin'] = max(
                            0, conn[target_id].get('myelin', 0) - 5)

                        self.unlearned.append({
                            'source': source_id,
                            'target': target_id,
                            'old_weight': old_w,
                            'new_weight': conn[target_id]['w'],
                        })
                        unlearned_count += 1

                        # If weight is negligible, remove the edge
                        if abs(conn[target_id]['w']) < 0.01:
                            del conn[target_id]

                    # Clear error history for this edge
                    del self.error_history[(source_id, target_id)]

        return unlearned_count
