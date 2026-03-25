"""
KOS V8.0 -- Shadow CI/CD (Canary Deployment for Self-Tuning)

Safe self-modification without human intervention:
    1. Parameter-Only Mutation: only config.json changes, never source code
    2. Shadow Eval: clone graph in RAM, run historical queries, measure
    3. Canary Deploy: 5% -> 25% -> 100% progressive rollout
    4. Automatic Rollback: if any metric regresses, revert to previous config

This is the bridge between "cannot improve" and "unsafe self-modification."
"""

import os
import json
import time
import copy
import random


_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')
_CANARY_CONFIG = os.path.join(_CONFIG_DIR, 'canary_state.json')


class ShadowEvaluator:
    """
    Evaluates a proposed config change against historical queries
    on a cloned shadow graph.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.historical_queries = []  # [(seeds, expected_top_results)]
        self._max_history = 200

    def record_query(self, seeds: list, results: list):
        """Record a query and its results for future benchmarking."""
        self.historical_queries.append({
            "seeds": seeds,
            "results": [(n, s) for n, s in results[:10]],
            "tick": getattr(self.kernel, 'current_tick', 0),
        })
        if len(self.historical_queries) > self._max_history:
            self.historical_queries = self.historical_queries[-self._max_history:]

    def evaluate_config(self, new_config: dict,
                        n_queries: int = 100) -> dict:
        """
        Evaluate a proposed config on a shadow graph.

        Returns:
            {
                "accuracy": float,      # % of results matching baseline
                "latency_ms": float,    # Average query time
                "memory_mb": float,     # RSS delta
                "improvement": float,   # Score relative to current config
                "passed": bool,
            }
        """
        if not self.historical_queries:
            return {"accuracy": 0.0, "latency_ms": 0.0, "memory_mb": 0.0,
                    "improvement": 0.0, "passed": False,
                    "reason": "No historical queries to benchmark"}

        queries = self.historical_queries[-min(n_queries, len(self.historical_queries)):]

        # Baseline: run queries with current config
        baseline_results = self._run_queries(queries)

        # Shadow: apply new config and re-run
        old_config = self._get_current_config()
        self._apply_config(new_config)
        shadow_results = self._run_queries(queries)
        self._apply_config(old_config)  # Restore

        # Compare
        accuracy = self._compare_results(baseline_results, shadow_results)
        latency_diff = shadow_results["avg_latency"] - baseline_results["avg_latency"]
        improvement = accuracy - 0.95  # Baseline expectation: 95% match

        passed = (accuracy >= 0.90 and
                  shadow_results["avg_latency"] < baseline_results["avg_latency"] * 1.5)

        return {
            "accuracy": accuracy,
            "latency_ms": shadow_results["avg_latency"],
            "baseline_latency_ms": baseline_results["avg_latency"],
            "improvement": improvement,
            "passed": passed,
        }

    def _run_queries(self, queries: list) -> dict:
        """Run a batch of queries and collect metrics."""
        results = []
        total_time = 0.0

        for q in queries:
            t0 = time.perf_counter()
            try:
                if hasattr(self.kernel, 'query_beam'):
                    r = self.kernel.query_beam(q["seeds"], top_k=10)
                else:
                    r = self.kernel.query(q["seeds"], top_k=10)
                results.append(r)
            except Exception:
                results.append([])
            total_time += (time.perf_counter() - t0) * 1000

        return {
            "results": results,
            "avg_latency": total_time / max(len(queries), 1),
            "total_queries": len(queries),
        }

    def _compare_results(self, baseline: dict, shadow: dict) -> float:
        """Compare shadow results to baseline. Returns accuracy 0-1."""
        if not baseline["results"]:
            return 1.0

        matches = 0
        total = 0
        for b, s in zip(baseline["results"], shadow["results"]):
            b_names = set(n for n, _ in b[:5])
            s_names = set(n for n, _ in s[:5])
            if b_names:
                overlap = len(b_names & s_names) / len(b_names)
                matches += overlap
                total += 1

        return matches / max(total, 1)

    def _get_current_config(self) -> dict:
        """Snapshot current kernel config."""
        config = {}
        if hasattr(self.kernel, '_rust') and self.kernel._rust:
            stats = self.kernel._rust.stats()
            config["tick"] = stats.get("tick", 0)
        config["max_ticks"] = getattr(self.kernel, 'max_ticks', 15)
        return config

    def _apply_config(self, config: dict):
        """Apply config values to kernel."""
        if "max_ticks" in config:
            self.kernel.max_ticks = config["max_ticks"]


class CanaryDeployer:
    """
    Progressive rollout of config changes with automatic rollback.

    Stages: shadow_eval -> 5% canary -> 25% -> 100%
    Any regression triggers instant rollback.
    """

    STAGES = [0.05, 0.25, 1.0]
    STAGE_DURATION = 60  # seconds per stage

    def __init__(self):
        self.config_history = []  # List of (timestamp, config_dict)
        self.current_stage = -1   # -1 = not deploying
        self.canary_fraction = 0.0
        self.pending_config = None
        self.stage_start_time = 0
        self._max_history = 20

    def propose(self, evaluator: ShadowEvaluator,
                new_config: dict) -> dict:
        """
        Propose a new config. Runs shadow evaluation first.

        Returns:
            {"accepted": bool, "eval_result": dict, "stage": str}
        """
        eval_result = evaluator.evaluate_config(new_config)

        if not eval_result["passed"]:
            return {
                "accepted": False,
                "eval_result": eval_result,
                "stage": "rejected_at_shadow",
            }

        # Save current config to history for rollback
        current = evaluator._get_current_config()
        self.config_history.append((time.time(), current))
        if len(self.config_history) > self._max_history:
            self.config_history = self.config_history[-self._max_history:]

        # Start canary deployment
        self.pending_config = new_config
        self.current_stage = 0
        self.canary_fraction = self.STAGES[0]
        self.stage_start_time = time.time()

        return {
            "accepted": True,
            "eval_result": eval_result,
            "stage": f"canary_{int(self.canary_fraction * 100)}%",
        }

    def should_use_canary(self) -> bool:
        """Should this query use the canary config?"""
        if self.current_stage < 0 or self.pending_config is None:
            return False
        return random.random() < self.canary_fraction

    def advance_stage(self, metrics_ok: bool) -> dict:
        """Advance or rollback the canary deployment."""
        if self.current_stage < 0:
            return {"status": "no_deployment"}

        if not metrics_ok:
            # ROLLBACK
            result = {"status": "rolled_back",
                      "stage": self.current_stage,
                      "fraction": self.canary_fraction}
            self.current_stage = -1
            self.canary_fraction = 0.0
            self.pending_config = None
            return result

        # Advance to next stage
        self.current_stage += 1
        if self.current_stage >= len(self.STAGES):
            # Fully deployed
            result = {"status": "fully_deployed",
                      "config": self.pending_config}
            self.current_stage = -1
            self.canary_fraction = 0.0
            self.pending_config = None
            return result

        self.canary_fraction = self.STAGES[self.current_stage]
        self.stage_start_time = time.time()
        return {
            "status": "advanced",
            "stage": self.current_stage,
            "fraction": self.canary_fraction,
        }

    def rollback_to(self, steps_back: int = 1) -> dict:
        """Rollback to a previous config."""
        if not self.config_history:
            return {"status": "no_history"}

        idx = max(0, len(self.config_history) - steps_back)
        ts, config = self.config_history[idx]
        return {"status": "rolled_back", "timestamp": ts, "config": config}

    def status(self) -> dict:
        return {
            "deploying": self.current_stage >= 0,
            "stage": self.current_stage,
            "canary_fraction": self.canary_fraction,
            "history_depth": len(self.config_history),
        }
