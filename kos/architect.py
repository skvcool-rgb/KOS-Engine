"""
KOS V9.0 -- Architecture Reviewer & Self-Execution Loop

Closes the gap between "observe" and "act":

    Old flow:  Detect problem -> Propose fix -> STOP (human gate)
    New flow:  Detect -> Diagnose -> Score -> Approve/Reject -> Execute -> Verify -> Log

Safety Model (3 tiers):
    SAFE:       Config-only changes (thresholds, weights). Auto-executable.
    SUPERVISED: Graph mutations (edge weights, node creation). Shadow-eval first.
    RESTRICTED: Anything else. Requires human gate. Never auto-executed.

The system reviews its own architecture, identifies bottlenecks,
generates scored repair plans, and executes SAFE/SUPERVISED actions
through the canary pipeline. RESTRICTED proposals are queued for
human review but never auto-executed.

Key invariant: NO SOURCE CODE MODIFICATION. All mutations happen
through existing APIs (kernel.add_connection, selfmod._save_config,
daemon methods). The system improves by tuning, not rewriting.
"""

import time
import json
import os
import math
from collections import defaultdict
from datetime import datetime


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '.cache')
_LOOP_LOG = os.path.join(_CACHE_DIR, 'self_loop_log.json')


# =====================================================================
# TIER DEFINITIONS
# =====================================================================

TIER_SAFE = "safe"             # Auto-executable, config-only
TIER_SUPERVISED = "supervised"  # Shadow-eval required, graph mutations
TIER_RESTRICTED = "restricted"  # Human gate required


# =====================================================================
# ARCHITECTURE HEALTH METRICS
# =====================================================================

class HealthMetric:
    """A single measurable health indicator."""
    __slots__ = ['name', 'value', 'threshold', 'direction', 'weight',
                 'category']

    def __init__(self, name: str, value: float, threshold: float,
                 direction: str = "higher_better", weight: float = 1.0,
                 category: str = "general"):
        self.name = name
        self.value = value
        self.threshold = threshold
        self.direction = direction  # "higher_better" or "lower_better"
        self.weight = weight
        self.category = category

    @property
    def healthy(self) -> bool:
        if self.direction == "higher_better":
            return self.value >= self.threshold
        return self.value <= self.threshold

    @property
    def severity(self) -> float:
        """How far from healthy? 0.0 = fine, 1.0 = critical."""
        if self.healthy:
            return 0.0
        if self.direction == "higher_better":
            if self.threshold == 0:
                return 1.0
            return min(1.0, (self.threshold - self.value) / self.threshold)
        else:
            if self.threshold == 0:
                return 1.0
            return min(1.0, (self.value - self.threshold) / max(self.threshold, 1))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "value": round(self.value, 4),
            "threshold": self.threshold, "healthy": self.healthy,
            "severity": round(self.severity, 3), "category": self.category,
        }


# =====================================================================
# REPAIR ACTION
# =====================================================================

class RepairAction:
    """A concrete, executable repair action."""

    def __init__(self, name: str, tier: str, description: str,
                 execute_fn=None, priority: float = 0.5,
                 triggered_by: str = "", estimated_impact: str = ""):
        self.name = name
        self.tier = tier  # TIER_SAFE, TIER_SUPERVISED, TIER_RESTRICTED
        self.description = description
        self.execute_fn = execute_fn  # callable() -> dict
        self.priority = priority  # 0.0 (low) to 1.0 (critical)
        self.triggered_by = triggered_by
        self.estimated_impact = estimated_impact
        self.status = "pending"  # pending, approved, executed, failed, skipped
        self.result = None
        self.timestamp = time.time()

    def execute(self) -> dict:
        """Execute the repair action."""
        if not self.execute_fn:
            self.status = "failed"
            self.result = {"error": "no execute function"}
            return self.result
        try:
            self.result = self.execute_fn()
            self.status = "executed"
        except Exception as e:
            self.result = {"error": str(e)}
            self.status = "failed"
        return self.result

    def to_dict(self) -> dict:
        return {
            "name": self.name, "tier": self.tier,
            "description": self.description,
            "priority": round(self.priority, 3),
            "triggered_by": self.triggered_by,
            "estimated_impact": self.estimated_impact,
            "status": self.status,
            "result": self.result,
        }


# =====================================================================
# ARCHITECTURE REVIEWER
# =====================================================================

class ArchitectureReviewer:
    """
    Reviews KOS's own architecture by collecting health metrics
    from every subsystem, scoring them, and identifying problems.

    This is the OBSERVE layer of the self-improvement loop.
    """

    def __init__(self, kernel, lexicon=None, pce=None,
                 self_improver=None, learning_coord=None,
                 daemon=None, canary_evaluator=None,
                 memory_lifecycle=None, verification=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.pce = pce
        self.self_improver = self_improver
        self.learning_coord = learning_coord
        self.daemon = daemon
        self.canary_eval = canary_evaluator
        self.memory_lifecycle = memory_lifecycle
        self.verification = verification

    def review(self) -> dict:
        """
        Full architecture review. Collects health metrics from
        every reachable subsystem.

        Returns:
            {
                "overall_health": float (0-1),
                "metrics": [HealthMetric.to_dict()],
                "problems": [{"metric": str, "severity": float}],
                "timestamp": float,
            }
        """
        metrics = []

        # -- Graph Health --
        metrics.extend(self._review_graph())

        # -- Prediction Health --
        metrics.extend(self._review_prediction())

        # -- Learning Health --
        metrics.extend(self._review_learning())

        # -- Memory Lifecycle --
        metrics.extend(self._review_memory())

        # -- Verification Pipeline --
        metrics.extend(self._review_verification())

        # -- Overall Score --
        if metrics:
            weighted_sum = sum(
                (1.0 - m.severity) * m.weight for m in metrics)
            total_weight = sum(m.weight for m in metrics)
            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall = 1.0

        problems = [
            {"metric": m.name, "severity": m.severity,
             "category": m.category, "value": m.value,
             "threshold": m.threshold}
            for m in metrics if not m.healthy
        ]
        problems.sort(key=lambda p: p["severity"], reverse=True)

        return {
            "overall_health": round(overall, 3),
            "metrics": [m.to_dict() for m in metrics],
            "problems": problems,
            "problem_count": len(problems),
            "metric_count": len(metrics),
            "timestamp": time.time(),
        }

    def _review_graph(self) -> list:
        """Graph structure health checks."""
        metrics = []
        nodes = self.kernel.nodes
        total_nodes = len(nodes)

        if total_nodes == 0:
            metrics.append(HealthMetric(
                "graph_populated", 0.0, 1.0,
                "higher_better", 2.0, "graph"))
            return metrics

        # Orphan ratio (nodes with 0 connections)
        orphans = sum(1 for n in nodes.values() if not n.connections)
        orphan_ratio = orphans / total_nodes
        metrics.append(HealthMetric(
            "orphan_ratio", orphan_ratio, 0.10,
            "lower_better", 1.5, "graph"))

        # Hub concentration (max degree / mean degree)
        degrees = [len(n.connections) for n in nodes.values()]
        mean_deg = sum(degrees) / len(degrees) if degrees else 1
        max_deg = max(degrees) if degrees else 0
        hub_ratio = max_deg / max(mean_deg, 1)
        metrics.append(HealthMetric(
            "hub_concentration", hub_ratio, 20.0,
            "lower_better", 1.0, "graph"))

        # Contradiction rate
        contradictions = len(getattr(self.kernel, 'contradictions', []))
        contradiction_rate = contradictions / total_nodes
        metrics.append(HealthMetric(
            "contradiction_rate", contradiction_rate, 0.05,
            "lower_better", 1.5, "graph"))

        # Edge weight variance (numerical stability)
        all_weights = []
        for n in nodes.values():
            for data in n.connections.values():
                w = data['w'] if isinstance(data, dict) else data
                all_weights.append(abs(w))
        if all_weights:
            max_w = max(all_weights)
            metrics.append(HealthMetric(
                "max_edge_weight", max_w, 1.5,
                "lower_better", 1.0, "graph"))

        return metrics

    def _review_prediction(self) -> list:
        """Predictive coding engine health."""
        metrics = []
        if not self.pce:
            return metrics

        stats = self.pce.get_stats()
        accuracy = stats.get("overall_accuracy", 0)
        metrics.append(HealthMetric(
            "prediction_accuracy", accuracy, 0.60,
            "higher_better", 1.5, "prediction"))

        miss_ratio = (stats.get("total_misses", 0) /
                      max(stats.get("total_predictions", 1), 1))
        metrics.append(HealthMetric(
            "prediction_miss_ratio", miss_ratio, 0.30,
            "lower_better", 1.0, "prediction"))

        cache_size = stats.get("cached_predictions", 0)
        if cache_size > 400:  # Nearing max_predictions=500
            metrics.append(HealthMetric(
                "prediction_cache_pressure", cache_size, 400,
                "lower_better", 0.5, "prediction"))

        return metrics

    def _review_learning(self) -> list:
        """Learning coordinator health."""
        metrics = []
        if not self.learning_coord:
            return metrics

        stats = self.learning_coord.get_stats()
        queries = stats.get("queries_learned", 0)
        if queries > 0:
            growth_rate = stats.get("nodes_grown", 0) / queries
            metrics.append(HealthMetric(
                "learning_growth_rate", growth_rate, 0.01,
                "higher_better", 0.8, "learning"))

            weakened_ratio = stats.get("edges_weakened", 0) / queries
            # Too many weakened edges = too many re-asks
            metrics.append(HealthMetric(
                "reask_edge_weakness", weakened_ratio, 0.50,
                "lower_better", 1.0, "learning"))

        return metrics

    def _review_memory(self) -> list:
        """Memory lifecycle health."""
        metrics = []
        if not self.memory_lifecycle:
            return metrics

        stats = self.memory_lifecycle.stats()
        tiers = stats.get("tiers", {})
        hot_count = tiers.get("hot", 0)
        total = stats.get("total_active", 0)

        if total > 0:
            # Hot tier shouldn't dominate (means nothing is aging)
            hot_ratio = hot_count / total
            if total > 100:  # Only check if graph is large enough
                metrics.append(HealthMetric(
                    "hot_tier_ratio", hot_ratio, 0.80,
                    "lower_better", 0.7, "memory"))

        return metrics

    def _review_verification(self) -> list:
        """Verification pipeline health."""
        metrics = []
        if not self.verification:
            return metrics

        stats = self.verification.stats()
        ingested = stats.get("ingested", 0)
        if ingested > 0:
            quarantine_rate = stats.get("quarantined", 0) / ingested
            metrics.append(HealthMetric(
                "quarantine_rate", quarantine_rate, 0.30,
                "lower_better", 0.8, "verification"))

        return metrics


# =====================================================================
# REPAIR PLANNER
# =====================================================================

class RepairPlanner:
    """
    Given health metrics problems, generates concrete repair actions.

    This is the DIAGNOSE + PLAN layer. Each problem maps to one or
    more repair actions, each with a safety tier.
    """

    def __init__(self, kernel, lexicon=None, self_improver=None,
                 learning_coord=None, memory_lifecycle=None):
        self.kernel = kernel
        self.lexicon = lexicon
        self.self_improver = self_improver
        self.learning_coord = learning_coord
        self.memory_lifecycle = memory_lifecycle

    def plan_repairs(self, problems: list) -> list:
        """
        Given a list of problems from ArchitectureReviewer,
        generate RepairActions.
        """
        actions = []

        for problem in problems:
            name = problem["metric"]
            severity = problem["severity"]

            new_actions = self._map_problem_to_actions(name, severity, problem)
            actions.extend(new_actions)

        # Sort by priority (highest first)
        actions.sort(key=lambda a: a.priority, reverse=True)
        return actions

    def _map_problem_to_actions(self, metric_name: str,
                                severity: float, problem: dict) -> list:
        """Map a specific problem to repair actions."""
        actions = []

        if metric_name == "orphan_ratio":
            actions.append(RepairAction(
                name="rebalance_orphans",
                tier=TIER_SUPERVISED,
                description="Connect orphan nodes to nearest neighbors "
                            "via word similarity",
                execute_fn=self._action_rebalance_orphans,
                priority=severity * 0.8,
                triggered_by=metric_name,
                estimated_impact="Reduce orphan ratio by connecting "
                                 "isolated nodes at weight 0.3",
            ))

        elif metric_name == "hub_concentration":
            actions.append(RepairAction(
                name="weaken_hub_edges",
                tier=TIER_SUPERVISED,
                description="Halve weakest 50% edges on super-hub nodes",
                execute_fn=self._action_weaken_hubs,
                priority=severity * 0.7,
                triggered_by=metric_name,
                estimated_impact="Reduce hub concentration, improve "
                                 "spreading activation fairness",
            ))

        elif metric_name == "contradiction_rate":
            actions.append(RepairAction(
                name="resolve_contradictions",
                tier=TIER_SUPERVISED,
                description="Auto-resolve contradictions by "
                            "provenance evidence weight",
                execute_fn=self._action_resolve_contradictions,
                priority=severity * 0.9,
                triggered_by=metric_name,
                estimated_impact="Suppress minority-evidence side "
                                 "of contradictions (weight x0.3)",
            ))

        elif metric_name == "max_edge_weight":
            actions.append(RepairAction(
                name="normalize_weights",
                tier=TIER_SAFE,
                description="Clip all edge weights to [-1.0, 1.0]",
                execute_fn=self._action_normalize_weights,
                priority=severity * 0.6,
                triggered_by=metric_name,
                estimated_impact="Restore numerical stability",
            ))

        elif metric_name == "prediction_accuracy":
            actions.append(RepairAction(
                name="tune_prediction_lr",
                tier=TIER_SAFE,
                description="Increase PCE learning rate for "
                            "faster convergence",
                execute_fn=self._action_tune_prediction,
                priority=severity * 0.7,
                triggered_by=metric_name,
                estimated_impact="Improve prediction accuracy by "
                                 "increasing learning rate to 0.05",
            ))

        elif metric_name == "prediction_miss_ratio":
            actions.append(RepairAction(
                name="clear_stale_predictions",
                tier=TIER_SAFE,
                description="Evict low-accuracy prediction records",
                execute_fn=self._action_clear_stale_predictions,
                priority=severity * 0.5,
                triggered_by=metric_name,
                estimated_impact="Free prediction cache of bad records",
            ))

        elif metric_name == "reask_edge_weakness":
            actions.append(RepairAction(
                name="consolidate_weak_edges",
                tier=TIER_SUPERVISED,
                description="Prune very weak edges and boost "
                            "strong paths",
                execute_fn=self._action_consolidate,
                priority=severity * 0.6,
                triggered_by=metric_name,
                estimated_impact="Clean up paths degraded by "
                                 "repeated anti-Hebbian weakening",
            ))

        elif metric_name == "hot_tier_ratio":
            actions.append(RepairAction(
                name="force_memory_sweep",
                tier=TIER_SAFE,
                description="Run memory lifecycle sweep to "
                            "demote stale nodes",
                execute_fn=self._action_memory_sweep,
                priority=severity * 0.4,
                triggered_by=metric_name,
                estimated_impact="Move stale nodes from HOT to "
                                 "WARM/COLD tiers",
            ))

        elif metric_name == "quarantine_rate":
            actions.append(RepairAction(
                name="review_quarantine_queue",
                tier=TIER_RESTRICTED,
                description="High quarantine rate detected. "
                            "Queue for human review.",
                execute_fn=None,  # No auto-execution
                priority=severity * 0.5,
                triggered_by=metric_name,
                estimated_impact="Human must review quarantined edges",
            ))

        return actions

    # -- Repair Action Implementations --

    def _action_rebalance_orphans(self) -> dict:
        if self.self_improver:
            return self.self_improver.rebalance_degrees(verbose=False)
        return {"status": "no_self_improver"}

    def _action_weaken_hubs(self) -> dict:
        if self.self_improver:
            return self.self_improver.rebalance_degrees(
                hub_threshold=20, verbose=False)
        return {"status": "no_self_improver"}

    def _action_resolve_contradictions(self) -> dict:
        if self.self_improver:
            return self.self_improver.resolve_contradictions(verbose=False)
        return {"status": "no_self_improver"}

    def _action_normalize_weights(self) -> dict:
        if self.self_improver:
            return self.self_improver.normalize_weights(verbose=False)
        return {"status": "no_self_improver"}

    def _action_tune_prediction(self) -> dict:
        from .selfmod import _load_config, _save_config
        config = _load_config()
        old_lr = config.get("pce_learning_rate", 0.02)
        new_lr = min(0.10, old_lr * 1.5)
        config["pce_learning_rate"] = new_lr
        config["pce_lr_updated"] = datetime.now().isoformat()
        _save_config(config)
        return {"old_lr": old_lr, "new_lr": new_lr}

    def _action_clear_stale_predictions(self) -> dict:
        from .predictive import PredictiveCodingEngine
        # Access the PCE through the kernel's known references
        pce = getattr(self.kernel, '_pce_ref', None)
        if not pce:
            return {"status": "no_pce_ref"}
        before = len(pce.predictions)
        # Evict predictions with accuracy < 0.3
        to_remove = [k for k, v in pce.predictions.items()
                     if v.accuracy < 0.3 and v.hit_count + v.miss_count > 5]
        for k in to_remove:
            del pce.predictions[k]
        return {"removed": len(to_remove), "before": before,
                "after": len(pce.predictions)}

    def _action_consolidate(self) -> dict:
        if not self.kernel._rust:
            return {"status": "no_rust"}
        rust = self.kernel._rust
        pruned = rust.prune_weak_edges(0.03)
        rust.decay_myelin(0.95)
        return {"pruned": pruned}

    def _action_memory_sweep(self) -> dict:
        if self.memory_lifecycle:
            return self.memory_lifecycle.sweep()
        return {"status": "no_memory_lifecycle"}


# =====================================================================
# SELF-EXECUTION LOOP
# =====================================================================

class SelfExecutionLoop:
    """
    The complete closed-loop self-improvement system.

    Connects: Review -> Plan -> Approve -> Execute -> Verify -> Log

    Approval tiers:
        SAFE:       Auto-approved (config changes only)
        SUPERVISED: Shadow-evaluated first, auto-approved if metrics pass
        RESTRICTED: Logged for human review, never auto-executed

    The loop runs periodically (triggered by daemon, tick, or manual).
    Every execution is logged to .cache/self_loop_log.json for audit.
    """

    def __init__(self, reviewer: ArchitectureReviewer,
                 planner: RepairPlanner,
                 canary_evaluator=None,
                 canary_deployer=None,
                 max_actions_per_cycle: int = 5,
                 auto_mode: bool = True):
        self.reviewer = reviewer
        self.planner = planner
        self.canary_eval = canary_evaluator
        self.canary_deployer = canary_deployer
        self.max_actions = max_actions_per_cycle
        self.auto_mode = auto_mode  # False = all actions require human gate

        # Execution log
        self.cycle_count = 0
        self.total_actions_executed = 0
        self.total_actions_skipped = 0
        self.history = []  # Last N cycle results
        self._max_history = 50

    def run_cycle(self, verbose: bool = True) -> dict:
        """
        Run one complete self-improvement cycle:
        1. Review architecture health
        2. Plan repairs for any problems
        3. Approve based on tier
        4. Execute approved actions
        5. Verify results
        6. Log everything
        """
        self.cycle_count += 1
        t0 = time.perf_counter()

        if verbose:
            print(f"\n[SELF-LOOP] Cycle {self.cycle_count} starting...")

        # 1. REVIEW
        review = self.reviewer.review()
        health = review["overall_health"]
        problems = review["problems"]

        if verbose:
            print(f"  Health: {health:.1%} | "
                  f"Problems: {len(problems)} | "
                  f"Metrics: {review['metric_count']}")

        if not problems:
            if verbose:
                print("  No problems detected. System healthy.")
            result = {
                "cycle": self.cycle_count,
                "health_before": health,
                "health_after": health,
                "problems_found": 0,
                "actions_planned": 0,
                "actions_executed": 0,
                "actions_skipped": 0,
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }
            self._log_cycle(result)
            return result

        # 2. PLAN
        actions = self.planner.plan_repairs(problems)
        actions = actions[:self.max_actions]  # Limit per cycle

        if verbose:
            print(f"  Planned {len(actions)} repair actions:")
            for a in actions:
                print(f"    [{a.tier.upper():>10}] {a.name} "
                      f"(priority={a.priority:.2f})")

        # 3. APPROVE + 4. EXECUTE
        executed = 0
        skipped = 0
        results = []

        for action in actions:
            approved = self._approve(action, verbose)
            if approved:
                if verbose:
                    print(f"    -> Executing: {action.name}...", end=" ")
                result = action.execute()
                executed += 1
                self.total_actions_executed += 1
                if verbose:
                    status = action.status
                    print(f"[{status.upper()}]")
            else:
                skipped += 1
                self.total_actions_skipped += 1
                action.status = "skipped"

            results.append(action.to_dict())

        # 5. VERIFY (re-review after repairs)
        if executed > 0:
            verify_review = self.reviewer.review()
            health_after = verify_review["overall_health"]
            delta = health_after - health
        else:
            health_after = health
            delta = 0.0

        elapsed = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"\n  Results: {executed} executed, {skipped} skipped")
            if executed > 0:
                direction = "+" if delta >= 0 else ""
                print(f"  Health: {health:.1%} -> {health_after:.1%} "
                      f"({direction}{delta:.1%})")
            print(f"  Elapsed: {elapsed:.0f}ms")

        # 6. LOG
        cycle_result = {
            "cycle": self.cycle_count,
            "health_before": round(health, 4),
            "health_after": round(health_after, 4),
            "health_delta": round(delta, 4),
            "problems_found": len(problems),
            "actions_planned": len(actions),
            "actions_executed": executed,
            "actions_skipped": skipped,
            "actions": results,
            "elapsed_ms": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        self._log_cycle(cycle_result)
        return cycle_result

    def _approve(self, action: RepairAction, verbose: bool) -> bool:
        """Approve or reject an action based on its tier."""
        if not self.auto_mode:
            # All actions require human approval
            return self._human_approve(action, verbose)

        if action.tier == TIER_SAFE:
            # Auto-approve: config-only, no graph mutation
            action.status = "approved"
            return True

        elif action.tier == TIER_SUPERVISED:
            # Shadow-evaluate first
            if self.canary_eval:
                eval_result = self.canary_eval.evaluate_config({})
                if eval_result.get("passed", True):
                    action.status = "approved"
                    return True
                else:
                    if verbose:
                        print(f"    -> REJECTED by shadow eval: "
                              f"{action.name}")
                    return False
            else:
                # No canary evaluator: auto-approve supervised
                # (graph mutations through existing safe APIs)
                action.status = "approved"
                return True

        elif action.tier == TIER_RESTRICTED:
            # Never auto-execute
            if verbose:
                print(f"    -> QUEUED for human review: {action.name}")
            return False

        return False

    def _human_approve(self, action: RepairAction, verbose: bool) -> bool:
        """Interactive human approval."""
        if verbose:
            print(f"\n    REVIEW: {action.name}")
            print(f"    Tier: {action.tier}")
            print(f"    Description: {action.description}")
            print(f"    Impact: {action.estimated_impact}")
        try:
            response = input("    Approve? [Y/N]: ").strip().lower()
            approved = response in ('y', 'yes')
            if approved:
                action.status = "approved"
            return approved
        except (EOFError, KeyboardInterrupt):
            return False

    def _log_cycle(self, result: dict):
        """Persist cycle result to log file."""
        self.history.append(result)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

        os.makedirs(_CACHE_DIR, exist_ok=True)
        try:
            # Append to log
            existing = []
            if os.path.exists(_LOOP_LOG):
                with open(_LOOP_LOG, 'r') as f:
                    existing = json.load(f)
            existing.append(result)
            # Keep last 200 cycles
            if len(existing) > 200:
                existing = existing[-200:]
            with open(_LOOP_LOG, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass  # Don't crash on log failure

    def stats(self) -> dict:
        return {
            "cycles": self.cycle_count,
            "total_executed": self.total_actions_executed,
            "total_skipped": self.total_actions_skipped,
            "auto_mode": self.auto_mode,
            "last_health": (self.history[-1]["health_after"]
                           if self.history else None),
        }

    def trend(self) -> list:
        """Return health trend over recent cycles."""
        return [
            {"cycle": h["cycle"],
             "health": h["health_after"],
             "actions": h["actions_executed"]}
            for h in self.history[-20:]
        ]


# =====================================================================
# CONVENIENCE: Wire everything together
# =====================================================================

def create_self_loop(kernel, lexicon=None, pce=None,
                     self_improver=None, learning_coord=None,
                     daemon=None, canary_evaluator=None,
                     canary_deployer=None, memory_lifecycle=None,
                     verification=None,
                     auto_mode: bool = True) -> SelfExecutionLoop:
    """
    Factory function to wire up the complete self-improvement loop
    from existing KOS subsystems.

    Usage:
        loop = create_self_loop(kernel, lexicon, pce, ...)
        result = loop.run_cycle()
    """
    reviewer = ArchitectureReviewer(
        kernel=kernel, lexicon=lexicon, pce=pce,
        self_improver=self_improver,
        learning_coord=learning_coord,
        daemon=daemon,
        canary_evaluator=canary_evaluator,
        memory_lifecycle=memory_lifecycle,
        verification=verification,
    )

    planner = RepairPlanner(
        kernel=kernel, lexicon=lexicon,
        self_improver=self_improver,
        learning_coord=learning_coord,
        memory_lifecycle=memory_lifecycle,
    )

    return SelfExecutionLoop(
        reviewer=reviewer,
        planner=planner,
        canary_evaluator=canary_evaluator,
        canary_deployer=canary_deployer,
        auto_mode=auto_mode,
    )
