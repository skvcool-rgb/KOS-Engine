"""
Test: Autonomous Agent + Self-Execution Loop + Canary Deployer Integration

Validates that the self-loop fires every N cycles inside the autonomous agent,
and that canary stage advancement is automatic.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kos.graph import KOSKernel
from kos.autonomous import AutonomousAgent
from kos.architect import (
    SelfExecutionLoop, ArchitectureReviewer, RepairPlanner,
    create_self_loop, TIER_SAFE, TIER_SUPERVISED,
)
from kos.canary import ShadowEvaluator, CanaryDeployer


# ---- Helpers ----

class MockLexicon:
    def __init__(self):
        self.word_to_uuid = {}
        self._counter = 0
    def get_or_create_id(self, word):
        if word not in self.word_to_uuid:
            self._counter += 1
            self.word_to_uuid[word] = "uuid_%d" % self._counter
        return self.word_to_uuid[word]
    def get_word(self, uid):
        for w, u in self.word_to_uuid.items():
            if u == uid:
                return w
        return uid


class MockShell:
    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon
    def chat(self, msg):
        return "mock answer"


class MockDriver:
    def ingest(self, text):
        pass


class MockSelfImprover:
    def __init__(self):
        self.rebalance_count = 0
        self.normalize_count = 0
        self.contradiction_count = 0
    def rebalance_degrees(self, hub_threshold=20, verbose=False):
        self.rebalance_count += 1
        return {"hubs_fixed": 1, "orphans_fixed": 2}
    def resolve_contradictions(self, verbose=False):
        self.contradiction_count += 1
        return {"contradictions_total": 1, "resolved": 1}
    def normalize_weights(self, verbose=False):
        self.normalize_count += 1
        return {"clipped": 5, "max_weight": 0.9}
    def improve(self, verbose=False):
        return {"total_applied": 0}


def make_kernel_with_issues(n_nodes=40, orphan_count=10):
    """Create a kernel with structural problems for the self-loop to fix."""
    kernel = KOSKernel(enable_vsa=False, force_python=True)
    lexicon = MockLexicon()

    uids = []
    for i in range(n_nodes):
        uid = lexicon.get_or_create_id("node_%d" % i)
        kernel.add_node(uid)
        uids.append(uid)

    # Wire non-orphans
    for i in range(orphan_count, n_nodes - 1):
        kernel.add_connection(uids[i], uids[i + 1], 0.5,
                              "edge %d->%d" % (i, i + 1))

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
        print("  PASS  %s" % name)
    except Exception as e:
        failed += 1
        print("  FAIL  %s: %s" % (name, e))


# -- Test 1: Self-loop wires into autonomous agent --
def test_self_loop_wired():
    kernel, lexicon, uids = make_kernel_with_issues()
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver_legacy = MockSelfImprover()

    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver_legacy, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop)

    assert agent._self_loop is self_loop, "Self-loop should be set"
    assert agent.self_loop_interval == 5, "Default interval should be 5"

run_test("1. Self-loop wires into agent", test_self_loop_wired)


# -- Test 2: Self-loop fires at correct interval --
def test_self_loop_interval():
    kernel, lexicon, uids = make_kernel_with_issues()
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop)
    agent.self_loop_interval = 3  # Every 3 cycles
    agent.cycle_interval_sec = 0  # No sleep

    # Run 9 cycles (should trigger self-loop at cycles 3, 6, 9)
    agent.persistence_interval = 0
    agent.run(max_cycles=9, cycle_interval=0, verbose=False)

    # Self-loop should have run 3 times (cycles 3, 6, 9)
    assert len(agent._self_loop_results) == 3, \
        "Expected 3 self-loop runs, got %d" % len(agent._self_loop_results)

run_test("2. Self-loop fires at correct interval", test_self_loop_interval)


# -- Test 3: Self-loop detects and repairs orphans --
def test_self_loop_repairs():
    kernel, lexicon, uids = make_kernel_with_issues(
        n_nodes=40, orphan_count=12)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop)
    agent.self_loop_interval = 1  # Every cycle

    agent.persistence_interval = 0
    agent.run(max_cycles=1, cycle_interval=0, verbose=False)

    assert len(agent._self_loop_results) == 1
    result = agent._self_loop_results[0]
    assert result["problems_found"] > 0, \
        "Should detect orphan problem. Result: %s" % result
    assert result["actions_executed"] > 0, \
        "Should execute repair. Result: %s" % result

run_test("3. Self-loop detects and repairs orphans", test_self_loop_repairs)


# -- Test 4: Lazy self-loop creation from subsystem refs --
def test_lazy_self_loop():
    kernel, lexicon, uids = make_kernel_with_issues(
        n_nodes=30, orphan_count=8)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    # Don't pass self_loop, pass subsystem refs instead
    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_improver_legacy=improver)

    assert agent._self_loop is None, "Should start as None"
    agent.self_loop_interval = 1

    agent.persistence_interval = 0
    agent.run(max_cycles=1, cycle_interval=0, verbose=False)

    # Should have lazily created self-loop
    assert agent._self_loop is not None, \
        "Self-loop should be lazily created"
    assert len(agent._self_loop_results) == 1, \
        "Should have run one self-loop cycle"

run_test("4. Lazy self-loop creation", test_lazy_self_loop)


# -- Test 5: Canary deployer wires in --
def test_canary_wired():
    kernel, lexicon, _ = make_kernel_with_issues(n_nodes=20, orphan_count=0)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()

    deployer = CanaryDeployer()
    evaluator = ShadowEvaluator(kernel)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        canary_deployer=deployer, canary_evaluator=evaluator)

    assert agent._canary_deployer is deployer
    assert agent._canary_eval is evaluator

run_test("5. Canary deployer wires in", test_canary_wired)


# -- Test 6: Canary auto-advances when stage duration elapsed --
def test_canary_auto_advance():
    kernel, lexicon, _ = make_kernel_with_issues(n_nodes=20, orphan_count=0)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()

    deployer = CanaryDeployer()
    deployer.STAGE_DURATION = 0  # Instant for testing
    evaluator = ShadowEvaluator(kernel)

    # Manually put deployer in canary state
    deployer.current_stage = 0
    deployer.canary_fraction = 0.05
    deployer.pending_config = {"max_ticks": 20}
    deployer.stage_start_time = time.time() - 100  # Already elapsed

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        canary_deployer=deployer)
    agent.canary_check_interval = 1  # Every cycle

    agent.persistence_interval = 0
    agent.run(max_cycles=1, cycle_interval=0, verbose=False)

    assert len(agent._canary_results) == 1, \
        "Canary should have advanced. Results: %s" % agent._canary_results
    assert agent._canary_results[0]["status"] == "advanced", \
        "Status should be 'advanced', got: %s" % agent._canary_results[0]

run_test("6. Canary auto-advances", test_canary_auto_advance)


# -- Test 7: Canary rollback on bad health --
def test_canary_rollback():
    kernel, lexicon, uids = make_kernel_with_issues(
        n_nodes=40, orphan_count=15)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    deployer = CanaryDeployer()
    deployer.STAGE_DURATION = 0

    # Put in canary state
    deployer.current_stage = 0
    deployer.canary_fraction = 0.05
    deployer.pending_config = {"max_ticks": 20}
    deployer.stage_start_time = time.time() - 100

    # Create self-loop but DON'T run it this cycle (interval=999)
    # so the pre-populated low health persists for the canary check
    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    # Pre-populate self-loop history with low health
    self_loop.history = [{"health_after": 0.3, "cycle": 0}]

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop, canary_deployer=deployer)
    agent.canary_check_interval = 1
    agent.self_loop_interval = 999  # Don't run self-loop this cycle

    agent.persistence_interval = 0
    agent.run(max_cycles=1, cycle_interval=0, verbose=False)

    # The canary check should see low health (0.3 < 0.7) and rollback
    canary_results = [r for r in agent._canary_results
                      if r.get("status") == "rolled_back"]
    assert len(canary_results) > 0, \
        "Canary should rollback on low health. Results: %s" % agent._canary_results

run_test("7. Canary rollback on bad health", test_canary_rollback)


# -- Test 8: Status includes self-loop and canary data --
def test_status_includes_new_fields():
    kernel, lexicon, _ = make_kernel_with_issues()
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    deployer = CanaryDeployer()
    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop, canary_deployer=deployer)
    agent.self_loop_interval = 1

    agent.persistence_interval = 0
    agent.run(max_cycles=2, cycle_interval=0, verbose=False)

    status = agent.get_status()
    assert "self_loop_cycles" in status, "Missing self_loop_cycles"
    assert "self_loop_actions" in status, "Missing self_loop_actions"
    assert "last_health" in status, "Missing last_health"
    assert "canary" in status, "Missing canary"
    assert "self_loop_interval" in status, "Missing self_loop_interval"
    assert status["self_loop_cycles"] == 2, \
        "Expected 2, got %s" % status["self_loop_cycles"]

run_test("8. Status includes self-loop/canary", test_status_includes_new_fields)


# -- Test 9: Self-loop doesn't crash without subsystems --
def test_no_subsystems():
    kernel = KOSKernel(enable_vsa=False, force_python=True)
    lexicon = MockLexicon()
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver)
    agent.self_loop_interval = 1

    # Should not crash even with no subsystems
    agent.persistence_interval = 0
    agent.run(max_cycles=3, cycle_interval=0, verbose=False)

    # Lazy creation should have built a self-loop from kernel alone
    assert agent._self_loop is not None, "Should lazily create self-loop"

run_test("9. No crash without subsystems", test_no_subsystems)


# -- Test 10: Multiple self-loop cycles show health trend --
def test_health_trend():
    kernel, lexicon, uids = make_kernel_with_issues(
        n_nodes=40, orphan_count=10)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop)
    agent.self_loop_interval = 2  # Every 2 cycles

    agent.persistence_interval = 0
    agent.run(max_cycles=10, cycle_interval=0, verbose=False)

    # Should have 5 self-loop results (cycles 2,4,6,8,10)
    assert len(agent._self_loop_results) == 5, \
        "Expected 5, got %d" % len(agent._self_loop_results)

    # Each should have health data
    for r in agent._self_loop_results:
        assert r["health_before"] is not None
        assert r["health_after"] is not None

    # Trend available through self-loop
    trend = self_loop.trend()
    assert len(trend) == 5, "Expected 5 trend points"

run_test("10. Health trend over cycles", test_health_trend)


# -- Test 11: Canary full deployment through stages --
def test_canary_full_deployment():
    kernel, lexicon, _ = make_kernel_with_issues(n_nodes=20, orphan_count=0)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()

    deployer = CanaryDeployer()
    deployer.STAGE_DURATION = 0  # Instant for testing

    # Start at stage 0
    deployer.current_stage = 0
    deployer.canary_fraction = 0.05
    deployer.pending_config = {"max_ticks": 25}
    deployer.stage_start_time = time.time() - 100

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        canary_deployer=deployer)
    agent.canary_check_interval = 1

    # Run 3 cycles to advance through all stages: 5% -> 25% -> 100%
    agent.persistence_interval = 0
    agent.run(max_cycles=3, cycle_interval=0, verbose=False)

    # Should have advanced through all stages
    statuses = [r["status"] for r in agent._canary_results]
    assert "fully_deployed" in statuses, \
        "Should reach fully_deployed. Got: %s" % statuses

run_test("11. Canary full deployment", test_canary_full_deployment)


# -- Test 12: Cycle events include self-loop data --
def test_events_include_self_loop():
    kernel, lexicon, _ = make_kernel_with_issues(
        n_nodes=30, orphan_count=8)
    shell = MockShell(kernel, lexicon)
    driver = MockDriver()
    improver = MockSelfImprover()

    self_loop = create_self_loop(
        kernel=kernel, lexicon=lexicon,
        self_improver=improver, auto_mode=True)

    agent = AutonomousAgent(
        kernel=kernel, lexicon=lexicon, shell=shell, driver=driver,
        self_loop=self_loop)
    agent.self_loop_interval = 1

    agent.persistence_interval = 0
    agent.run(max_cycles=1, cycle_interval=0, verbose=False)

    events = agent.get_events()
    # Find the cycle event (not a log message)
    cycle_events = [e for e in events if "self_loop" in e]
    assert len(cycle_events) >= 1, \
        "Cycle event should contain self_loop key. Events: %s" % events
    assert cycle_events[0]["self_loop"] is not None, \
        "self_loop should have data"

run_test("12. Events include self-loop data", test_events_include_self_loop)


# ---- Summary ----
print("\n" + "=" * 50)
print("Autonomous+SelfLoop Integration: %d/%d PASSED, %d FAILED" % (
    passed, total, failed))
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("FAILURES: %d" % failed)
    sys.exit(1)
