"""
KOS V5.0 — Predictive Coding Engine (Phase 3).

Implementation of Karl Friston's Free Energy Principle applied to
spreading activation graphs. The system learns to predict its own
behavior and corrects itself when predictions fail.

The Loop:
    1. PREDICT: Before propagation, predict which nodes will activate
       and their expected energy levels (based on learned priors)
    2. PROPAGATE: Run the actual spreading activation
    3. COMPARE: Compute prediction error (expected vs actual)
    4. UPDATE: Adjust edge weights to minimize future prediction error

This is how the cortex works (Friston, 2010):
    - Top-down signals carry PREDICTIONS ("I expect to see X")
    - Bottom-up signals carry PREDICTION ERRORS ("I saw Y instead")
    - The brain adjusts its internal model to minimize surprise

Biological analog:
    - Hearing "The doctor picked up the..." → brain predicts "stethoscope"
    - If actual word is "gun" → massive prediction error → model update
    - Next time in similar context, prediction is more accurate

Key insight: This is NOT gradient descent. It's a local Hebbian rule:
    - If predicted activation matches actual → strengthen the path
    - If predicted activation misses → weaken the prediction path
    - If unexpected activation occurs → create new prediction path
"""

import math
import time
from collections import defaultdict


class PredictionRecord:
    """A single prediction: expected activation pattern for a seed set."""
    __slots__ = ['seed_key', 'predicted_activations', 'confidence',
                 'hit_count', 'miss_count', 'created_tick',
                 'consecutive_low_error', 'locked']

    def __init__(self, seed_key: tuple, predicted: dict,
                 confidence: float, tick: int):
        self.seed_key = seed_key
        self.predicted_activations = predicted  # {node_id: expected_energy}
        self.confidence = confidence
        self.hit_count = 0
        self.miss_count = 0
        self.created_tick = tick
        self.consecutive_low_error = 0  # Tracks stable convergence
        self.locked = False  # True = prediction snapped to actual

    @property
    def accuracy(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class PredictiveCodingEngine:
    """
    The prediction-error minimization loop.

    Wraps the kernel's propagation cycle with:
    1. A prediction cache (learned priors)
    2. An error signal calculator
    3. A weight adjustment rule

    The engine gets better with every query — it learns which
    activation patterns follow which seed sets.
    """

    def __init__(self, kernel, learning_rate: float = 0.02,
                 prediction_threshold: float = 0.3,
                 max_predictions: int = 500):
        self.kernel = kernel
        self.lr = learning_rate
        self.threshold = prediction_threshold
        self.max_predictions = max_predictions

        # Prediction memory: seed_key -> PredictionRecord
        # seed_key is a frozenset of seed node IDs
        self.predictions = {}

        # Running statistics
        self.total_predictions = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_surprises = 0  # Unexpected activations
        self.total_updates = 0

        # Per-query log for the current cycle
        self._last_report = {}

    def _make_seed_key(self, seed_ids: list) -> tuple:
        """Create a hashable key from seed IDs."""
        return tuple(sorted(seed_ids))

    # ── PREDICT ──────────────────────────────────────────────────

    def predict(self, seed_ids: list) -> dict:
        """
        Generate a predicted activation pattern for the given seeds.

        Strategy 1: If we've seen these exact seeds before, use
        the cached prediction (adjusted by accuracy).

        Strategy 2: If we haven't seen these seeds, generate a
        naive prediction based on edge weights (1-hop lookahead).
        """
        seed_key = self._make_seed_key(seed_ids)

        # Strategy 1: Cached prediction from prior experience
        if seed_key in self.predictions:
            record = self.predictions[seed_key]
            # Scale predictions by confidence
            return {nid: energy * record.confidence
                    for nid, energy in record.predicted_activations.items()}

        # Strategy 2: Naive 1-hop prediction from edge weights
        predicted = {}
        for sid in seed_ids:
            if sid not in self.kernel.nodes:
                continue
            node = self.kernel.nodes[sid]
            for tgt_id, data in node.connections.items():
                if tgt_id in seed_ids:
                    continue  # Skip self-loops
                if isinstance(data, dict):
                    w = data['w'] * (1 + data.get('myelin', 0) * 0.01)
                else:
                    w = data
                # Predicted energy = seed_energy * weight * spatial_decay
                pred_energy = 3.0 * w * 0.8
                if abs(pred_energy) >= self.threshold:
                    if tgt_id in predicted:
                        predicted[tgt_id] = max(predicted[tgt_id], pred_energy)
                    else:
                        predicted[tgt_id] = pred_energy

        return predicted

    # ── COMPARE ──────────────────────────────────────────────────

    def compute_error(self, predicted: dict, actual: dict) -> dict:
        """
        Compute the prediction error signal.

        Returns a dict with three categories:
        - hits: nodes predicted AND activated (prediction correct)
        - misses: nodes predicted BUT NOT activated (false positive)
        - surprises: nodes NOT predicted BUT activated (false negative)

        The error signal drives weight updates.
        """
        predicted_set = set(predicted.keys())
        actual_set = set(actual.keys())

        hits = {}      # Correct predictions
        misses = {}    # Predicted but didn't happen
        surprises = {} # Happened but wasn't predicted

        # Hits: predicted and activated
        for nid in predicted_set & actual_set:
            error = actual[nid] - predicted[nid]
            hits[nid] = {
                'predicted': predicted[nid],
                'actual': actual[nid],
                'error': error,
                'magnitude': abs(error),
            }

        # Misses: predicted but not activated
        for nid in predicted_set - actual_set:
            misses[nid] = {
                'predicted': predicted[nid],
                'actual': 0.0,
                'error': -predicted[nid],
                'magnitude': abs(predicted[nid]),
            }

        # Surprises: activated but not predicted
        for nid in actual_set - predicted_set:
            surprises[nid] = {
                'predicted': 0.0,
                'actual': actual[nid],
                'error': actual[nid],
                'magnitude': actual[nid],
            }

        return {'hits': hits, 'misses': misses, 'surprises': surprises}

    # ── UPDATE ───────────────────────────────────────────────────

    def update_weights(self, seed_ids: list, error_signal: dict) -> int:
        """
        Adjust edge weights based on prediction error.

        Rules (local Hebbian, no backpropagation):

        1. HITS with small error → strengthen the prediction path
           (the edge correctly predicted this activation)

        2. MISSES → weaken the edge that predicted a false activation
           (the edge overestimated its importance)

        3. SURPRISES → if a strong surprise connects to a seed,
           that edge was underweighted → strengthen it

        Returns the number of weight adjustments made.
        """
        adjustments = 0

        # Rule 1: Hits with small error → reinforce
        for nid, info in error_signal['hits'].items():
            if info['magnitude'] < 1.0:
                # Good prediction — strengthen the path
                for sid in seed_ids:
                    if sid in self.kernel.nodes:
                        conn = self.kernel.nodes[sid].connections
                        if nid in conn:
                            if isinstance(conn[nid], dict):
                                # Boost myelin (Hebbian reinforcement)
                                conn[nid]['myelin'] = conn[nid].get('myelin', 0) + 2
                            adjustments += 1

        # Rule 2: Misses → weaken the false prediction edge
        for nid, info in error_signal['misses'].items():
            if info['magnitude'] > 0.5:
                for sid in seed_ids:
                    if sid in self.kernel.nodes:
                        conn = self.kernel.nodes[sid].connections
                        if nid in conn:
                            if isinstance(conn[nid], dict):
                                # Reduce weight slightly
                                old_w = conn[nid]['w']
                                conn[nid]['w'] = old_w * (1.0 - self.lr)
                                # Demyelinate (weaken learned path)
                                conn[nid]['myelin'] = max(
                                    0, conn[nid].get('myelin', 0) - 1)
                            else:
                                conn[nid] = conn[nid] * (1.0 - self.lr)
                            adjustments += 1

        # Rule 3: Surprises → strengthen underweighted edges
        for nid, info in error_signal['surprises'].items():
            if info['magnitude'] > 1.0:
                # This node activated strongly but wasn't predicted
                # Find if any seed connects to it
                for sid in seed_ids:
                    if sid in self.kernel.nodes:
                        conn = self.kernel.nodes[sid].connections
                        if nid in conn:
                            if isinstance(conn[nid], dict):
                                # Boost weight slightly
                                old_w = conn[nid]['w']
                                conn[nid]['w'] = min(1.0, old_w * (1.0 + self.lr))
                                conn[nid]['myelin'] = conn[nid].get('myelin', 0) + 1
                            else:
                                conn[nid] = min(1.0, conn[nid] * (1.0 + self.lr))
                            adjustments += 1

        return adjustments

    # ── LEARN ────────────────────────────────────────────────────

    def update_prediction_cache(self, seed_ids: list, actual: dict,
                                 error_signal: dict):
        """
        Update the prediction memory with this experience.

        If we've seen these seeds before, adjust the cached prediction
        toward the actual outcome (exponential moving average).

        If this is a new seed set, create a new prediction record.
        """
        seed_key = self._make_seed_key(seed_ids)

        hits = len(error_signal['hits'])
        misses = len(error_signal['misses'])
        surprises = len(error_signal['surprises'])

        if seed_key in self.predictions:
            record = self.predictions[seed_key]
            record.hit_count += hits
            record.miss_count += misses + surprises

            # ── FIX 1: Adaptive Learning Rate ────────────────────
            # Alpha increases with accuracy: uncertain predictions
            # update cautiously, confident predictions converge fast.
            # Biology: the brain locks in stable patterns quickly
            # but is cautious with novel/conflicting signals.
            base_alpha = 0.3
            alpha = min(0.95, base_alpha + record.accuracy * 0.65)

            # ── FIX 2: Snap-to-Lock ─────────────────────────────
            # If prediction error has been below threshold for 3+
            # consecutive cycles, snap predictions to actual values.
            # Biology: stable neural patterns "crystallize" —
            # the brain stops updating what it's confident about.
            current_mae = 0.0
            mae_count = 0
            for nid, energy in actual.items():
                if nid in record.predicted_activations:
                    current_mae += abs(energy - record.predicted_activations[nid])
                    mae_count += 1
            if mae_count > 0:
                current_mae /= mae_count

            LOCK_THRESHOLD = 0.05  # MAE below this = stable
            LOCK_CYCLES = 3        # Consecutive stable cycles to lock

            if current_mae < LOCK_THRESHOLD:
                record.consecutive_low_error += 1
            else:
                record.consecutive_low_error = 0
                record.locked = False  # Unlock if error spikes (new evidence)

            if record.consecutive_low_error >= LOCK_CYCLES:
                # SNAP: set all predictions to exact actual values
                record.predicted_activations = dict(actual)
                record.locked = True
                record.confidence = 1.0
            else:
                # ── Standard EMA update with adaptive alpha ──────
                for nid, energy in actual.items():
                    if nid in record.predicted_activations:
                        old = record.predicted_activations[nid]
                        record.predicted_activations[nid] = (
                            (1 - alpha) * old + alpha * energy)
                    else:
                        # ── FIX 3: Full absorption of new nodes ──
                        # New nodes get actual energy immediately,
                        # not alpha-scaled. The brain doesn't
                        # partially notice a new stimulus — it
                        # either detects it or it doesn't.
                        record.predicted_activations[nid] = energy

            # Remove predictions that consistently miss
            to_remove = []
            for nid in list(record.predicted_activations.keys()):
                if nid not in actual and nid in error_signal['misses']:
                    record.predicted_activations[nid] *= 0.5
                    if abs(record.predicted_activations[nid]) < 0.1:
                        to_remove.append(nid)
            for nid in to_remove:
                del record.predicted_activations[nid]

            # Update confidence based on accuracy
            record.confidence = min(1.0, record.accuracy + 0.1)

        else:
            # New experience — store it
            record = PredictionRecord(
                seed_key=seed_key,
                predicted=dict(actual),  # First time: actual = prediction
                confidence=0.5,          # Start at 50% confidence
                tick=self.kernel.current_tick
            )
            record.hit_count = hits
            record.miss_count = misses + surprises
            self.predictions[seed_key] = record

        # Evict old predictions if cache is full
        if len(self.predictions) > self.max_predictions:
            # Remove least accurate predictions
            sorted_preds = sorted(
                self.predictions.items(),
                key=lambda x: x[1].accuracy)
            for key, _ in sorted_preds[:len(self.predictions) - self.max_predictions]:
                del self.predictions[key]

    # ── MAIN LOOP ────────────────────────────────────────────────

    def query_with_prediction(self, seed_ids: list, top_k: int = 5,
                                verbose: bool = False) -> dict:
        """
        The complete predictive coding loop:
            PREDICT → PROPAGATE → COMPARE → UPDATE → LEARN

        Returns the query results plus the prediction report.
        """
        t0 = time.perf_counter()

        # 1. PREDICT
        predicted = self.predict(seed_ids)
        self.total_predictions += 1

        # 2. PROPAGATE (actual spreading activation)
        self.kernel.current_tick += 1
        actual_raw = self.kernel.propagate(seed_ids)
        # Filter to non-seed results
        actual = {n: e for n, e in actual_raw.items() if n not in seed_ids}

        # 3. COMPARE
        error_signal = self.compute_error(predicted, actual)

        hits = len(error_signal['hits'])
        misses = len(error_signal['misses'])
        surprises = len(error_signal['surprises'])
        self.total_hits += hits
        self.total_misses += misses
        self.total_surprises += surprises

        # 4. UPDATE weights based on error
        adjustments = self.update_weights(seed_ids, error_signal)
        self.total_updates += adjustments

        # 5. LEARN — update prediction cache
        self.update_prediction_cache(seed_ids, actual, error_signal)

        elapsed = (time.perf_counter() - t0) * 1000

        # Calculate prediction accuracy for this query
        total_predicted = len(predicted)
        accuracy = hits / total_predicted if total_predicted > 0 else 0.0

        # Mean absolute error for hits
        if error_signal['hits']:
            mae = sum(h['magnitude'] for h in error_signal['hits'].values()) / hits
        else:
            mae = float('inf')

        report = {
            'results': sorted(actual.items(), key=lambda x: x[1],
                               reverse=True)[:top_k],
            'predicted_count': total_predicted,
            'hits': hits,
            'misses': misses,
            'surprises': surprises,
            'accuracy': accuracy,
            'mae': mae,
            'adjustments': adjustments,
            'elapsed_ms': elapsed,
        }

        self._last_report = report

        if verbose:
            print(f"[PREDICT] Predicted {total_predicted} nodes | "
                  f"Hits={hits} Misses={misses} Surprises={surprises} | "
                  f"Accuracy={accuracy:.0%} | MAE={mae:.3f} | "
                  f"Adjustments={adjustments} | {elapsed:.1f}ms")

        return report

    # ── STATISTICS ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return cumulative prediction statistics."""
        total = self.total_hits + self.total_misses + self.total_surprises
        return {
            'total_predictions': self.total_predictions,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'total_surprises': self.total_surprises,
            'overall_accuracy': (self.total_hits / total if total > 0 else 0.0),
            'total_weight_adjustments': self.total_updates,
            'cached_predictions': len(self.predictions),
        }
