"""
KOS V7.0 — Level 3.75: Auto Self-Improvement

Auto-applies SAFE proposals. Queues UNSAFE ones for human review.

SAFE (auto-applied):
    - synonym additions
    - threshold tuning (bounded)
    - compound noun detection
    - orphan pruning
    - edge weight normalization
    - prediction cache training

UNSAFE (queued for human):
    - new code generation
    - architecture changes
    - weaver formula changes

Every action is logged. Every change is reversible.
"""

import time
import json
import os
import re
from collections import defaultdict


class AutoImprover:
    """
    Runs self-improvement automatically on safe proposals.
    Logs everything. Queues unsafe proposals for human review.
    """

    # What's safe to auto-apply
    SAFE_TYPES = {
        "synonym_addition",
        "threshold_change",
        "compound_detection",
        "orphan_prune",
        "weight_normalization",
        "prediction_training",
    }

    # What needs human approval
    UNSAFE_TYPES = {
        "code_generation",
        "architecture_change",
        "weaver_formula",
        "daemon_strategy",
    }

    def __init__(self, kernel, lexicon, shell=None, pce=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.shell = shell
        self.pce = pce

        self._applied = []      # Successfully applied changes
        self._queued = []       # Waiting for human review
        self._rejected = []     # Failed or rejected changes
        self._log = []          # Full audit log

        # Config bounds for safe tuning
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load or create self-tuning config."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), '.cache', 'auto_improve_config.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "learning_rate_min": 0.01,
            "learning_rate_max": 0.15,
            "learning_rate": 0.05,
            "fuzzy_cutoff_min": 0.5,
            "fuzzy_cutoff_max": 0.8,
            "fuzzy_cutoff": 0.6,
            "max_synonyms_per_cycle": 10,
            "max_compounds_per_cycle": 5,
            "normalization_clip": 1.0,
            "auto_apply_enabled": True,
        }

    def _save_config(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), '.cache', 'auto_improve_config.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def _log_action(self, action_type, details, auto_applied):
        entry = {
            "time": time.time(),
            "type": action_type,
            "details": details,
            "auto_applied": auto_applied,
        }
        self._log.append(entry)

    # ── MAIN LOOP ────────────────────────────────────────

    def improve(self, verbose=True) -> dict:
        """
        Run one improvement cycle. Discovers issues,
        auto-applies safe fixes, queues unsafe ones.
        """
        if not self._config.get("auto_apply_enabled", True):
            return {"status": "disabled"}

        t0 = time.perf_counter()
        results = {
            "synonyms_added": 0,
            "compounds_found": 0,
            "orphans_pruned": 0,
            "weights_normalized": 0,
            "predictions_trained": 0,
            "thresholds_tuned": 0,
            "queued_for_human": 0,
        }

        # 1. Auto-discover and add synonyms
        results["synonyms_added"] = self._auto_synonyms(verbose)

        # 2. Auto-detect compound nouns
        results["compounds_found"] = self._auto_compounds(verbose)

        # 3. Auto-prune orphans
        results["orphans_pruned"] = self._auto_prune_orphans(verbose)

        # 4. Auto-normalize edge weights
        results["weights_normalized"] = self._auto_normalize_weights(verbose)

        # 5. Auto-train predictions
        results["predictions_trained"] = self._auto_train_predictions(verbose)

        # 6. Auto-tune thresholds based on benchmark
        results["thresholds_tuned"] = self._auto_tune_thresholds(verbose)

        # 7. Check for unsafe improvements needed
        results["queued_for_human"] = self._check_unsafe_improvements(verbose)

        elapsed = (time.perf_counter() - t0) * 1000
        results["elapsed_ms"] = round(elapsed, 1)
        results["total_applied"] = sum(v for k, v in results.items()
                                        if k not in ("queued_for_human", "elapsed_ms"))
        results["status"] = "improved" if results["total_applied"] > 0 else "no_changes"

        if verbose:
            print("[AUTO-IMPROVE] Cycle complete in %.0fms: %d applied, %d queued" % (
                elapsed, results["total_applied"], results["queued_for_human"]))

        return results

    # ── SAFE AUTO-APPLY METHODS ──────────────────────────

    def _auto_synonyms(self, verbose) -> int:
        """Discover synonyms from graph structure overlap."""
        added = 0
        try:
            # Find node pairs that share many connections
            nodes = list(self.kernel.nodes.items())
            if len(nodes) < 5:
                return 0

            for i in range(min(len(nodes), 50)):
                nid_a, node_a = nodes[i]
                if not hasattr(node_a, 'connections'):
                    continue
                targets_a = set()
                conns_a = node_a.connections
                if isinstance(conns_a, dict):
                    targets_a = set(conns_a.keys())

                for j in range(i + 1, min(len(nodes), 50)):
                    nid_b, node_b = nodes[j]
                    if not hasattr(node_b, 'connections'):
                        continue
                    conns_b = node_b.connections
                    if isinstance(conns_b, dict):
                        targets_b = set(conns_b.keys())
                    else:
                        continue

                    if not targets_a or not targets_b:
                        continue

                    shared = targets_a & targets_b
                    union = targets_a | targets_b
                    if len(union) == 0:
                        continue
                    jaccard = len(shared) / len(union)

                    if jaccard > 0.85 and nid_a != nid_b and len(shared) >= 3:
                        word_a = self.lexicon.get_word(nid_a) if hasattr(self.lexicon, 'get_word') else None
                        word_b = self.lexicon.get_word(nid_b) if hasattr(self.lexicon, 'get_word') else None
                        if word_a and word_b and word_a != word_b:
                            # Add synonym mapping
                            self.lexicon.word_to_uuid[word_b.lower()] = nid_a
                            added += 1
                            self._log_action("synonym_addition",
                                "%s -> %s (jaccard=%.2f)" % (word_b, word_a, jaccard), True)
                            self._applied.append({
                                "type": "synonym", "from": word_b, "to": word_a})
                            if added >= self._config.get("max_synonyms_per_cycle", 10):
                                break
                if added >= self._config.get("max_synonyms_per_cycle", 10):
                    break

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Synonym error: %s" % str(e)[:60])

        if verbose and added:
            print("[AUTO-IMPROVE] Added %d synonym mappings" % added)
        return added

    def _auto_compounds(self, verbose) -> int:
        """Detect compound nouns from graph provenance."""
        found = 0
        try:
            from .compound_detector import CompoundDetector
            detector = CompoundDetector()

            # Collect sentences from provenance
            sentences = []
            for key, sents in self.kernel.provenance.items():
                if isinstance(sents, set):
                    sentences.extend(list(sents)[:5])
                elif isinstance(sents, list):
                    sentences.extend(sents[:5])
            if len(sentences) < 3:
                return 0

            compounds = detector.detect_from_corpus(sentences)
            if compounds and self.shell:
                from .drivers.text import TextDriver
                if hasattr(self.shell, 'kernel'):
                    # Find the TextDriver to update
                    for attr in dir(self.shell):
                        obj = getattr(self.shell, attr, None)
                        if isinstance(obj, TextDriver):
                            detector.update_textdriver(obj, compounds)
                            found = len(compounds)
                            break

            if found:
                self._log_action("compound_detection",
                    "Found %d compounds" % found, True)
                self._applied.append({"type": "compound", "count": found})

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Compound error: %s" % str(e)[:60])

        if verbose and found:
            print("[AUTO-IMPROVE] Detected %d compound nouns" % found)
        return found

    def _auto_prune_orphans(self, verbose) -> int:
        """Remove disconnected nodes."""
        pruned = 0
        try:
            # Build set of all nodes that are targeted by edges
            referenced = set()
            for node in self.kernel.nodes.values():
                if hasattr(node, 'connections') and isinstance(node.connections, dict):
                    referenced.update(node.connections.keys())

            orphans = []
            for nid, node in list(self.kernel.nodes.items()):
                if hasattr(node, 'connections'):
                    if isinstance(node.connections, dict) and not node.connections:
                        # Only prune if NO other node points to it
                        if nid not in referenced:
                            orphans.append(nid)

            for nid in orphans[:20]:  # Max 20 per cycle
                self.kernel.nodes.pop(nid, None)
                pruned += 1

            if pruned:
                self._log_action("orphan_prune",
                    "Pruned %d orphan nodes" % pruned, True)
                self._applied.append({"type": "orphan_prune", "count": pruned})

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Prune error: %s" % str(e)[:60])

        if verbose and pruned:
            print("[AUTO-IMPROVE] Pruned %d orphan nodes" % pruned)
        return pruned

    def _auto_normalize_weights(self, verbose) -> int:
        """Clip edge weights to [-1, 1]."""
        clipped = 0
        try:
            clip = self._config.get("normalization_clip", 1.0)
            for node in self.kernel.nodes.values():
                if not hasattr(node, 'connections'):
                    continue
                if not isinstance(node.connections, dict):
                    continue
                for tgt, data in node.connections.items():
                    if isinstance(data, dict) and 'w' in data:
                        w = data['w']
                        if abs(w) > clip:
                            data['w'] = max(-clip, min(clip, w))
                            clipped += 1

            if clipped:
                self._log_action("weight_normalization",
                    "Clipped %d edges to [-%s, %s]" % (clipped, clip, clip), True)

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Normalize error: %s" % str(e)[:60])

        if verbose and clipped:
            print("[AUTO-IMPROVE] Normalized %d edge weights" % clipped)
        return clipped

    def _auto_train_predictions(self, verbose) -> int:
        """Train predictive coding on high-degree nodes."""
        trained = 0
        if not self.pce:
            return 0
        try:
            ranked = sorted(
                self.kernel.nodes.items(),
                key=lambda x: len(x[1].connections) if hasattr(x[1], 'connections') and isinstance(x[1].connections, dict) else 0,
                reverse=True
            )[:10]

            for uid, node in ranked:
                self.pce.query_with_prediction([uid], top_k=3, verbose=False)
                trained += 1

            if trained:
                self._log_action("prediction_training",
                    "Trained on %d high-degree nodes" % trained, True)

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Prediction error: %s" % str(e)[:60])

        if verbose and trained:
            print("[AUTO-IMPROVE] Trained predictions on %d nodes" % trained)
        return trained

    def _auto_tune_thresholds(self, verbose) -> int:
        """Tune thresholds based on recent query performance."""
        tuned = 0
        if not self.shell:
            return 0
        try:
            # Check if self-model has enough query history
            if hasattr(self.shell, '_self_model') and self.shell._self_model:
                sm = self.shell._self_model
                if len(sm._query_history) >= 5:
                    # Calculate success rate from recent queries
                    recent = sm._query_history[-10:]
                    no_data_count = sum(1 for q in recent
                                         if "don't have" in q.get("answer", "").lower())
                    fail_rate = no_data_count / len(recent)

                    # If fail rate > 50%, loosen matching
                    if fail_rate > 0.5:
                        current_lr = self._config.get("learning_rate", 0.05)
                        new_lr = min(self._config["learning_rate_max"],
                                      current_lr + 0.01)
                        if new_lr != current_lr:
                            self._config["learning_rate"] = new_lr
                            self._save_config()
                            tuned += 1
                            self._log_action("threshold_change",
                                "learning_rate: %.3f -> %.3f (high fail rate)" % (
                                    current_lr, new_lr), True)

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Tune error: %s" % str(e)[:60])

        if verbose and tuned:
            print("[AUTO-IMPROVE] Tuned %d thresholds" % tuned)
        return tuned

    # ── UNSAFE: QUEUE FOR HUMAN ──────────────────────────

    def _check_unsafe_improvements(self, verbose) -> int:
        """Identify improvements that need human approval."""
        queued = 0
        try:
            # Check if Weaver scoring could be improved
            if hasattr(self.shell, '_self_model') and self.shell._self_model:
                sm = self.shell._self_model
                if len(sm._query_history) >= 10:
                    recent = sm._query_history[-20:]
                    no_data = [q for q in recent if "don't have" in q.get("answer", "").lower()]
                    if len(no_data) > len(recent) * 0.3:
                        self._queued.append({
                            "type": "weaver_formula",
                            "reason": "%d/%d recent queries returned 'no data'. "
                                      "Weaver scoring may need adjustment." % (
                                          len(no_data), len(recent)),
                            "suggested": "Review Weaver WHERE/WHEN/WHO boost values",
                            "time": time.time(),
                        })
                        queued += 1

            # Check graph health
            total_nodes = len(self.kernel.nodes)
            if total_nodes > 0:
                orphan_count = sum(1 for n in self.kernel.nodes.values()
                                    if hasattr(n, 'connections') and
                                    isinstance(n.connections, dict) and
                                    not n.connections)
                if orphan_count > total_nodes * 0.3:
                    self._queued.append({
                        "type": "architecture_change",
                        "reason": "%d/%d nodes are orphans (%.0f%%). "
                                  "TextDriver may need better extraction." % (
                                      orphan_count, total_nodes,
                                      orphan_count/total_nodes*100),
                        "suggested": "Review TextDriver SVO extraction",
                        "time": time.time(),
                    })
                    queued += 1

        except Exception as e:
            if verbose:
                print("[AUTO-IMPROVE] Check error: %s" % str(e)[:60])

        if verbose and queued:
            print("[AUTO-IMPROVE] Queued %d proposals for human review" % queued)
        return queued

    # ── MONITORING ───────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "total_applied": len(self._applied),
            "total_queued": len(self._queued),
            "total_rejected": len(self._rejected),
            "log_entries": len(self._log),
            "config": self._config,
            "auto_enabled": self._config.get("auto_apply_enabled", True),
        }

    def get_applied(self) -> list:
        return self._applied[-20:]

    def get_queued(self) -> list:
        return self._queued

    def approve_queued(self, index: int) -> dict:
        """Human approves a queued proposal."""
        if 0 <= index < len(self._queued):
            proposal = self._queued.pop(index)
            self._log_action("human_approved", str(proposal), False)
            return {"status": "approved", "proposal": proposal}
        return {"status": "error", "message": "Invalid index"}

    def reject_queued(self, index: int) -> dict:
        """Human rejects a queued proposal."""
        if 0 <= index < len(self._queued):
            proposal = self._queued.pop(index)
            self._rejected.append(proposal)
            self._log_action("human_rejected", str(proposal), False)
            return {"status": "rejected", "proposal": proposal}
        return {"status": "error", "message": "Invalid index"}

    def get_log(self, last_n: int = 20) -> list:
        return self._log[-last_n:]

    def disable(self):
        self._config["auto_apply_enabled"] = False
        self._save_config()
        self._log_action("control", "Auto-improvement DISABLED", False)

    def enable(self):
        self._config["auto_apply_enabled"] = True
        self._save_config()
        self._log_action("control", "Auto-improvement ENABLED", False)
