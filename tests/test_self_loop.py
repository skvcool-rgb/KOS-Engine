"""
Test: KOS V9.0 Self-Improvement Execution Loop

Validates the complete observe -> diagnose -> plan -> execute -> verify cycle.
Tests all 3 safety tiers (SAFE, SUPERVISED, RESTRICTED).
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kos.graph import KOSKernel
from kos.architect import (
    ArchitectureReviewer, RepairPlanner, SelfExecutionLoop,
    HealthMetric, RepairAction, create_self_loop,
    TIER_SAFE, TIER_SUPERVISED, TIER_RESTRICTED,
)

# ---- Helpers ----

class MockLexicon:
    def __init__(self):
        self.word_to_uuid = {}
        self._counter = 0

    def get_or_create_id(self, word):
        if word not in self.word_to_uuid:
            self._counter += 1
            self.word_to_uuid[word] = f"uuid_{self._counter}"
        return self.word_to_uuid[word]

    def get_word(self, uid):
        for w, u in self.word_to_uuid.items():
            if u == uid:
                return w
        return uid


class MockSelfImprover:
    """Mimics SelfImprover for testing."""
    def __init__(self):
        self.rebalance_called = False
        self.contradictions_called = False
        self.normalize_called = False

    def rebalance_degrees(self, hub_threshold=20, verbose=False):
        self.rebalance_called = True
        return {"hubs_fixed": 2, "orphans_fixed": 5}

    def resolve_contradictions(self, verbose=False):
        self.contradictions_called = True
        return {"contradictions_total": 3, "resolved": 2}

    def normalize_weights(self, verbose=False):
        self.normalize_called = True
        return {"clipped": 10, "max_weight": 0.95}


def make_test_kernel(n_nodes=50, orphan_count=10, hub_degree=30):
    """Create a kernel with known structural issues."""
    kernel = KOSKernel(enable_vsa=False, force_python=True)
    lexicon = MockLexicon()

    uids = []
    for i in range(n_nodes):
        uid = lexicon.get_or_create_id(f"node_{i}")
        kernel.add_node(uid)
        uids.append(uid)

    # Wire most nodes (skip orphan_count to leave orphans)
    for i in range(orphan_count, n_nodes - 1):
        kernel.add_connection(uids[i], uids[i + 1], 0.5,
                              f"test edge {i}->{i+1}")

    # Create a super-hub
    if hub_degree > 0 and n_nodes > hub_degree + 1:
        hub = uids[orphan_count]
        for i in range(orphan_count + 1,
                       min(orphan_count + hub_degree + 1, n_nodes)):
            kernel.add_connection(hub, uids[i], 0.3,
                                  f"hub edge to {i}")

    return kernel, lexicon, uids


# ---- Tests ----

passed = 0
failed = 0
total = 0


def run_test(name, test_fn):
    global passed, failed, total
    total += 1
    try:
        test_fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")


# -- Test 1: HealthMetric scoring --
def test_health_metric():
    # Healthy metric (higher is better)
    m = HealthMetric("accuracy", 0.85, 0.70, "higher_better")
    assert m.healthy, "Should be healthy"
    assert m.severity == 0.0, f"Severity should be 0, got {m.severity}"

    # Unhealthy metric (lower is better)
    m2 = HealthMetric("orphan_ratio", 0.25, 0.10, "lower_better")
    assert not m2.healthy, "Should be unhealthy"
    assert m2.severity > 0, "Severity should be > 0"

run_test("1. HealthMetric scoring", test_health_metric)


# -- Test 2: RepairAction execution --
def test_repair_action():
    executed = {"called": False}
    def my_fix():
        executed["called"] = True
        return {"fixed": 42}

    action = RepairAction("test_fix", TIER_SAFE, "test",
                          execute_fn=my_fix, priority=0.8)
    result = action.execute()
    assert executed["called"], "Execute function should be called"
    assert result["fixed"] == 42, f"Expected 42, got {result}"
    assert action.status == "executed"

run_test("2. RepairAction execution", test_repair_action)


# -- Test 3: RepairAction failure handling --
def test_repair_action_failure():
    def bad_fix():
        raise ValueError("intentional")

    action = RepairAction("bad_fix", TIER_SAFE, "test",
                          execute_fn=bad_fix)
    result = action.execute()
    assert action.status == "failed"
    assert "intentional" in result["error"]

run_test("3. RepairAction failure handling", test_repair_action_failure)


# -- Test 4: Architecture review detects orphans --
def test_review_orphans():
    kernel, lexicon, uids = make_test_kernel(
        n_nodes=50, orphan_count=15, hub_degree=0)
    reviewer = ArchitectureReviewer(kernel=kernel, lexicon=lexicon)
    review = reviewer.review()

    orphan_problem = [p for p in review["problems"]
                      if p["metric"] == "orphan_ratio"]
    assert len(orphan_problem) > 0, \
        f"Should detect orphan problem. Problems: {review['problems']}"

run_test("4. Review detects orphans", test_review_orphans)


# -- Test 5: Architecture review detects hub concentration --
def test_review_hubs():
    kernel, lexicon, uids = make_test_kernel(
        n_nodes=50, orphan_count=0, hub_degree=40)
    reviewer = ArchitectureReviewer(kernel=kernel, lexicon=lexicon)
    review = reviewer.review()

    hub_problem = [p for p in review["problems"]
                   if p["metric"] == "hub_concentration"]
    assert len(hub_problem) > 0, \
        f"Should detect hub problem. Problems: {review['problems']}"

run_test("5. Review detects hub concentration", test_review_hubs)


# -- Test 6: Healthy graph passes review --
def test_healthy_graph():
    kernel = KOSKernel(enable_vsa=False, force_python=True)
    lexicon = MockLexicon()

    # Small well-connected graph
    uids = []
    for i in range(20):
        uid = lexicon.get_or_create_id(f"n{i}")
        kernel.add_node(uid)
        uids.append(uid)
    for i in range(19):
        kernel.add_connection(uids[i], uids[i + 1], 0.5, f"edge {i}")
    # Close the loop
    kernel.add_connection(uids[19], uids[0], 0.5, "loop edge")

    reviewer = ArchitectureReviewer(kernel=kernel, lexicon=lexicon)
    review = reviewer.review()

    assert review["overall_health"] >= 0.8, \
        f"Healthy graph should score >= 0.8, got {review['overall_health']}"

run_test("6. Healthy graph passes review", test_healthy_graph)


# -- Test 7: RepairPlanner generates correct actions --
def test_repair_planner():
    kernel, lexicon, _ = make_test_kernel()
    improver = MockSelfImprover()
    planner = RepairPlanner(kernel=kernel, lexicon=lexicon,
                            self_improver=improver)

    problems = [
        {"metric": "orphan_ratio", "severity": 0.6,
         "category": "graph", "value": 0.3, "threshold": 0.1},
        {"metric": "max_edge_weight", "severity": 0.3,
         "category": "graph", "value": 2.0, "threshold": 1.5},
    ]

    actions = planner.plan_repairs(problems)
    assert len(actions) >= 2, f"Expected >= 2 actions, got {len(actions)}"

    # Check tiers assigned correctly
    orphan_action = [a for a in actions if a.name == "rebalance_orphans"]
    assert orphan_action[0].tier == TIER_SUPERVISED

    weight_action = [a for a in actions if a.name == "normalize_weights"]
    assert weight_action[0].tier == TIER_SAFE

    # Verify sorted by priority (highest first)
    for i in range(len(actions) - 1):
        assert actions[i].priority >= actions[i + 1].priority

run_test("7. RepairPlanner generates correct actions", test_repair_planner)


# -- Test 8: SAFE tier auto-executes --
def test_safe_auto_execute():
    kernel, lexicon, _ = make_test_kernel(n_nodes=30, orphan_count=0,
                                          hub_degree=0)
    improver = MockSelfImprover()

    # Inject a weight problem
    uids = list(kernel.nodes.keys())
    if len(uids) >= 2:
        kernel.nodes[uids[0]].connections[uids[1]] = 2.5  # Over 1.5

    reviewer = ArchitectureReviewer(kernel=kernel, self_improver=improver)
    planner = RepairPlanner(kernel=kernel, self_improver=improver)

    loop = SelfExecutionLoop(reviewer=reviewer, planner=planner,
                             auto_mode=True)
    result = loop.run_cycle(verbose=False)

    assert result["actions_executed"] >= 1, \
        f"SAFE action should auto-execute. Result: {result}"
    assert improver.normalize_called, \
        "normalize_weights should have been called"

run_test("8. SAFE tier auto-executes", test_safe_auto_execute)


# -- Test 9: RESTRICTED tier never auto-executes --
def test_restricted_skipped():
    kernel = KOSKernel(enable_vsa=False, force_python=True)

    # Create a fake restricted-only problem
    planner = RepairPlanner(kernel=kernel)
    problems = [
        {"metric": "quarantine_rate", "severity": 0.8,
         "category": "verification", "value": 0.5, "threshold": 0.3},
    ]
    actions = planner.plan_repairs(problems)
    assert len(actions) > 0, "Should generate quarantine review action"
    assert actions[0].tier == TIER_RESTRICTED

    # Run through loop
    reviewer = ArchitectureReviewer(kernel=kernel)

    # Override reviewer to inject our problems
    class FakeReviewer:
        def review(self):
            return {
                "overall_health": 0.5,
                "metrics": [],
                "problems": problems,
                "problem_count": 1,
                "metric_count": 1,
                "timestamp": time.time(),
            }

    loop = SelfExecutionLoop(
        reviewer=FakeReviewer(), planner=planner, auto_mode=True)
    result = loop.run_cycle(verbose=False)

    assert result["actions_skipped"] >= 1, \
        f"RESTRICTED should be skipped. Result: {result}"
    assert result["actions_executed"] == 0, \
        "RESTRICTED should NOT auto-execute"

run_test("9. RESTRICTED tier never auto-executes", test_restricted_skipped)


# -- Test 10: Full loop improves health --
def test_full_loop_improvement():
    kernel, lexicon, uids = make_test_kernel(
        n_nodes=50, orphan_count=15, hub_degree=30)

    # Add some contradictions
    kernel.contradictions = [
        {"source": uids[20], "existing_target": uids[21],
         "new_target": uids[22], "type": "antonym_contradiction"},
    ] * 6  # 6 contradictions to trigger the metric

    improver = MockSelfImprover()

    reviewer = ArchitectureReviewer(
        kernel=kernel, lexicon=lexicon, self_improver=improver)
    planner = RepairPlanner(
        kernel=kernel, lexicon=lexicon, self_improver=improver)

    loop = SelfExecutionLoop(
        reviewer=reviewer, planner=planner, auto_mode=True)

    result = loop.run_cycle(verbose=False)

    assert result["problems_found"] > 0, "Should find problems"
    assert result["actions_planned"] > 0, "Should plan actions"
    assert result["actions_executed"] > 0, "Should execute actions"

run_test("10. Full loop detects and repairs", test_full_loop_improvement)


# -- Test 11: create_self_loop factory --
def test_factory():
    kernel, lexicon, _ = make_test_kernel(n_nodes=20, orphan_count=0,
                                          hub_degree=0)
    loop = create_self_loop(kernel=kernel, lexicon=lexicon, auto_mode=True)
    assert isinstance(loop, SelfExecutionLoop)
    result = loop.run_cycle(verbose=False)
    assert "health_before" in result
    assert "health_after" in result

run_test("11. create_self_loop factory", test_factory)


# -- Test 12: Cycle logging --
def test_cycle_logging():
    kernel, lexicon, _ = make_test_kernel(n_nodes=20, orphan_count=5,
                                          hub_degree=0)
    loop = create_self_loop(kernel=kernel, lexicon=lexicon, auto_mode=True)

    # Run multiple cycles
    loop.run_cycle(verbose=False)
    loop.run_cycle(verbose=False)
    loop.run_cycle(verbose=False)

    assert loop.cycle_count == 3, f"Expected 3 cycles, got {loop.cycle_count}"
    assert len(loop.history) == 3, f"Expected 3 history entries"

    trend = loop.trend()
    assert len(trend) == 3
    for t in trend:
        assert "health" in t
        assert "cycle" in t

run_test("12. Cycle logging and trend", test_cycle_logging)


# -- Test 13: Stats reporting --
def test_stats():
    kernel, lexicon, _ = make_test_kernel(n_nodes=30, orphan_count=8,
                                          hub_degree=0)
    loop = create_self_loop(kernel=kernel, lexicon=lexicon, auto_mode=True)
    loop.run_cycle(verbose=False)

    stats = loop.stats()
    assert stats["cycles"] == 1
    assert "total_executed" in stats
    assert "total_skipped" in stats
    assert "auto_mode" in stats

run_test("13. Stats reporting", test_stats)


# -- Test 14: SUPERVISED action executes without canary --
def test_supervised_no_canary():
    """SUPERVISED actions should still execute if no canary evaluator."""
    kernel, lexicon, _ = make_test_kernel(
        n_nodes=50, orphan_count=15, hub_degree=0)
    improver = MockSelfImprover()

    reviewer = ArchitectureReviewer(
        kernel=kernel, lexicon=lexicon, self_improver=improver)
    planner = RepairPlanner(
        kernel=kernel, lexicon=lexicon, self_improver=improver)

    loop = SelfExecutionLoop(
        reviewer=reviewer, planner=planner,
        canary_evaluator=None,  # No canary
        auto_mode=True)

    result = loop.run_cycle(verbose=False)
    # Orphan problem should be detected and repaired
    assert improver.rebalance_called, \
        "Rebalance should execute even without canary"

run_test("14. SUPERVISED executes without canary", test_supervised_no_canary)


# -- Test 15: Contradiction detection + auto-resolution --
def test_contradiction_auto_resolve():
    kernel, lexicon, uids = make_test_kernel(
        n_nodes=30, orphan_count=0, hub_degree=0)

    # Inject many contradictions
    kernel.contradictions = [
        {"source": uids[i], "existing_target": uids[i + 1],
         "new_target": uids[i + 2],
         "type": "antonym_contradiction"}
        for i in range(0, 6, 3)
    ] * 3  # 6 contradictions

    improver = MockSelfImprover()

    reviewer = ArchitectureReviewer(
        kernel=kernel, lexicon=lexicon, self_improver=improver)
    planner = RepairPlanner(
        kernel=kernel, lexicon=lexicon, self_improver=improver)

    loop = SelfExecutionLoop(
        reviewer=reviewer, planner=planner, auto_mode=True)
    result = loop.run_cycle(verbose=False)

    assert improver.contradictions_called, \
        "Contradiction resolver should be called"

run_test("15. Contradiction auto-resolution", test_contradiction_auto_resolve)


# -- Test 16: Empty kernel doesn't crash --
def test_empty_kernel():
    kernel = KOSKernel(enable_vsa=False, force_python=True)
    loop = create_self_loop(kernel=kernel, auto_mode=True)
    result = loop.run_cycle(verbose=False)
    # Should detect graph_populated problem or handle gracefully
    assert result is not None
    assert "health_before" in result

run_test("16. Empty kernel handled gracefully", test_empty_kernel)


# ---- Summary ----
print(f"\n{'='*50}")
print(f"Self-Loop Tests: {passed}/{total} PASSED, {failed} FAILED")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
