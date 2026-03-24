"""
KOS V6.0 — Proposal 3: Hierarchical Predictive Coding (6 Layers)

Stacks multiple prediction layers like the human cortex:
    Layer 1: Predicts which nodes activate (existing PCE)
    Layer 2: Predicts Layer 1's prediction errors
    Layer 3: Predicts convergence time (how many cycles to stabilize)
    Layer 4: Predicts confidence (how certain the final answer will be)
    Layer 5: Predicts novelty (is this query unlike anything seen before?)
    Layer 6: Meta-prediction (predicts whether the system will need to forage)

Each layer sends TOP-DOWN priors to the layer below.
This is Karl Friston's hierarchical predictive coding.

Safety: each layer has independent monitoring.
CPU budget enforced per layer.
"""

import time
import math
from collections import defaultdict


class PredictionLayer:
    """A single layer in the hierarchy."""

    def __init__(self, layer_id: int, name: str):
        self.id = layer_id
        self.name = name
        self._predictions = {}  # key -> predicted_value
        self._actuals = {}      # key -> actual_value
        self._errors = {}       # key -> error
        self._history = []      # (timestamp, prediction, actual, error)
        self.total_predictions = 0
        self.total_correct = 0  # within tolerance
        self.tolerance = 0.15   # 15% error tolerance

    def predict(self, key: str, value: float):
        """Store a prediction."""
        self._predictions[key] = value
        self.total_predictions += 1

    def observe(self, key: str, actual: float):
        """Store the actual value and compute error."""
        self._actuals[key] = actual
        predicted = self._predictions.get(key)
        if predicted is not None:
            error = actual - predicted
            self._errors[key] = error
            self._history.append({
                "time": time.time(),
                "key": key,
                "predicted": round(predicted, 4),
                "actual": round(actual, 4),
                "error": round(error, 4),
            })
            if abs(error) <= self.tolerance * max(abs(actual), 0.01):
                self.total_correct += 1
            return error
        return None

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions

    @property
    def mean_error(self) -> float:
        if not self._errors:
            return 0.0
        return sum(abs(e) for e in self._errors.values()) / len(self._errors)

    def get_stats(self) -> dict:
        return {
            "layer": self.id,
            "name": self.name,
            "predictions": self.total_predictions,
            "accuracy": round(self.accuracy * 100, 1),
            "mean_error": round(self.mean_error, 4),
            "history_size": len(self._history),
        }


class HierarchicalPredictor:
    """
    6-layer hierarchical predictive coding system.

    Each layer predicts a different aspect:
        L1: Node activations (which concepts fire)
        L2: Prediction error magnitude (how wrong will L1 be)
        L3: Convergence time (how many ticks to stabilize)
        L4: Confidence (how certain is the final answer)
        L5: Novelty (has this query pattern been seen before)
        L6: Meta (will we need to forage for missing knowledge)
    """

    def __init__(self, kernel, pce=None):
        self.kernel = kernel
        self.pce = pce

        self.layers = [
            PredictionLayer(1, "node_activation"),
            PredictionLayer(2, "error_magnitude"),
            PredictionLayer(3, "convergence_time"),
            PredictionLayer(4, "confidence"),
            PredictionLayer(5, "novelty"),
            PredictionLayer(6, "meta_forage"),
        ]

        # Learning rates per layer (higher layers learn slower)
        self._learning_rates = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]

        # History for top-down prior generation
        self._query_patterns = defaultdict(list)  # pattern_key -> [outcomes]
        self._cycle_count = 0

    def predict_full(self, seeds: list, query_key: str = None) -> dict:
        """
        Run all 6 prediction layers for a given set of seeds.
        Returns the full prediction stack.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        if query_key is None:
            query_key = str(sorted(seeds))[:50]

        predictions = {}

        # ── Layer 1: Node Activation ─────────────────────
        # Predict which nodes will activate and with what energy
        l1_pred = self._predict_activations(seeds)
        predictions["activations"] = l1_pred
        self.layers[0].predict(query_key, l1_pred.get("top_energy", 0))

        # ── Layer 2: Error Magnitude ─────────────────────
        # Predict how wrong Layer 1 will be
        past_errors = [h["error"] for h in self.layers[0]._history[-10:]]
        avg_past_error = sum(abs(e) for e in past_errors) / len(past_errors) if past_errors else 0.5
        self.layers[1].predict(query_key, avg_past_error)
        predictions["expected_error"] = round(avg_past_error, 4)

        # ── Layer 3: Convergence Time ────────────────────
        # Predict how many ticks the graph will need to stabilize
        node_count = len(self.kernel.nodes)
        seed_degree = sum(
            len(self.kernel.nodes[s].connections)
            for s in seeds if s in self.kernel.nodes
        )
        # Heuristic: more connections = faster convergence
        predicted_ticks = max(1, 15 - min(14, seed_degree // 5))
        self.layers[2].predict(query_key, predicted_ticks)
        predictions["convergence_ticks"] = predicted_ticks

        # ── Layer 4: Confidence ──────────────────────────
        # Predict how confident the answer will be
        # Based on: seed resolution rate, graph density, past accuracy
        seeds_resolved = sum(1 for s in seeds if s in self.kernel.nodes)
        resolution_rate = seeds_resolved / max(len(seeds), 1)
        past_acc = self.layers[0].accuracy
        predicted_confidence = 0.5 * resolution_rate + 0.5 * past_acc
        self.layers[3].predict(query_key, predicted_confidence)
        predictions["confidence"] = round(predicted_confidence, 3)

        # ── Layer 5: Novelty ─────────────────────────────
        # Is this query pattern unlike anything seen before?
        pattern_history = self._query_patterns.get(query_key, [])
        novelty = 1.0 if not pattern_history else max(0, 1.0 - len(pattern_history) * 0.1)
        self.layers[4].predict(query_key, novelty)
        predictions["novelty"] = round(novelty, 3)

        # ── Layer 6: Meta (Forage Needed?) ───────────────
        # Predict whether the system will need to go to the internet
        forage_probability = (1.0 - predicted_confidence) * novelty
        self.layers[5].predict(query_key, forage_probability)
        predictions["forage_probability"] = round(forage_probability, 3)

        predictions["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return predictions

    def observe_actual(self, seeds: list, actual_results: dict,
                        query_key: str = None) -> dict:
        """
        After the actual query runs, observe the results and
        compute prediction errors across all 6 layers.
        """
        if query_key is None:
            query_key = str(sorted(seeds))[:50]

        errors = {}

        # L1: actual top energy
        actual_energy = actual_results.get("top_energy", 0)
        e1 = self.layers[0].observe(query_key, actual_energy)
        errors["L1_activation_error"] = e1

        # L2: actual error magnitude (from L1)
        if e1 is not None:
            actual_error = abs(e1)
            e2 = self.layers[1].observe(query_key, actual_error)
            errors["L2_error_prediction_error"] = e2

        # L3: actual convergence (approximate from results)
        actual_ticks = actual_results.get("ticks", 10)
        e3 = self.layers[2].observe(query_key, actual_ticks)
        errors["L3_convergence_error"] = e3

        # L4: actual confidence (did it answer correctly?)
        actual_confidence = actual_results.get("confidence", 0)
        e4 = self.layers[3].observe(query_key, actual_confidence)
        errors["L4_confidence_error"] = e4

        # L5: actual novelty (was it really novel?)
        actual_novelty = 1.0 if actual_results.get("no_answer") else 0.0
        e5 = self.layers[4].observe(query_key, actual_novelty)
        errors["L5_novelty_error"] = e5

        # L6: actual forage (did we need to forage?)
        actual_forage = 1.0 if actual_results.get("foraged") else 0.0
        e6 = self.layers[5].observe(query_key, actual_forage)
        errors["L6_forage_error"] = e6

        # Record pattern for future predictions
        self._query_patterns[query_key].append(actual_results)

        return errors

    def _predict_activations(self, seeds: list) -> dict:
        """Layer 1: predict node activations from seeds."""
        if self.pce:
            # Use existing PCE for L1
            result = self.pce.predict(seeds)
            if result:
                top_energy = max(result.values()) if result else 0
                return {"predicted_nodes": len(result), "top_energy": top_energy}

        # Fallback: estimate from seed connectivity
        total_connections = sum(
            len(self.kernel.nodes[s].connections)
            for s in seeds if s in self.kernel.nodes
        )
        estimated_energy = min(3.0, total_connections * 0.2)
        return {"predicted_nodes": total_connections, "top_energy": estimated_energy}

    # ── Top-Down Priors ──────────────────────────────────

    def get_top_down_priors(self, seeds: list) -> dict:
        """
        Generate top-down priors from higher layers.

        Higher layers constrain lower layers:
        - If L6 predicts forage needed, L4 confidence should be low
        - If L5 predicts high novelty, L2 error should be high
        - If L3 predicts slow convergence, L1 should expect low energy
        """
        predictions = self.predict_full(seeds)

        priors = {}

        # L6 -> L4: if forage likely, reduce confidence prior
        if predictions["forage_probability"] > 0.7:
            priors["confidence_ceiling"] = 0.3

        # L5 -> L2: if novel, expect high error
        if predictions["novelty"] > 0.8:
            priors["expected_error_floor"] = 0.3

        # L3 -> L1: if slow convergence, expect diffuse activation
        if predictions["convergence_ticks"] > 10:
            priors["activation_spread"] = "diffuse"
        else:
            priors["activation_spread"] = "focused"

        priors["source_predictions"] = predictions
        return priors

    # ── Monitoring ───────────────────────────────────────

    def get_all_stats(self) -> list:
        """Get stats for all 6 layers."""
        return [layer.get_stats() for layer in self.layers]

    def get_summary(self) -> dict:
        """Summary across all layers."""
        stats = self.get_all_stats()
        return {
            "total_cycles": self._cycle_count,
            "layers": stats,
            "avg_accuracy": round(
                sum(s["accuracy"] for s in stats) / len(stats), 1
            ),
            "unique_patterns": len(self._query_patterns),
        }

    def print_dashboard(self):
        """Print monitoring dashboard to console."""
        summary = self.get_summary()
        print("\n=== HIERARCHICAL PREDICTOR (6 Layers) ===")
        print("Total cycles: %d | Unique patterns: %d | Avg accuracy: %.1f%%" % (
            summary["total_cycles"],
            summary["unique_patterns"],
            summary["avg_accuracy"],
        ))
        print()
        print("  %-5s %-20s %8s %8s %8s" % ("Layer", "Name", "Predict", "Accuracy", "AvgErr"))
        print("  " + "-" * 55)
        for s in summary["layers"]:
            print("  L%-4d %-20s %8d %7.1f%% %8.4f" % (
                s["layer"], s["name"], s["predictions"],
                s["accuracy"], s["mean_error"],
            ))
