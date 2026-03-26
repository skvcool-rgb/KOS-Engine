"""
KOS v0.8 Mission Resilience Tests

5 test categories targeting the failure modes that matter now:
  1. Failure Injection    -- forced failures, correct cascading, no fake success
  2. Dependency Integrity -- execution order enforced, no premature synthesis
  3. Replanning on Failure -- retry logic, score thresholds, escalation
  4. Concurrent Missions  -- no state leakage, correct isolation
  5. Audit Trail          -- timestamps, scores, trust, errors, deliverables

Runs against MissionManager directly (no server, no LLM).
Uses mock query_fn to control outcomes precisely.
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.mission import (
    MissionManager, Mission, Goal, GoalStatus, GoalType,
    MissionStatus, CheckpointStatus, DeliverableType,
)

PASS = 0
FAIL = 0
TOTAL = 0


def check(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} -- {detail}")


# ── Mock query functions ───────────────────────────────────────

def mock_good(prompt):
    """Always returns a good answer."""
    return {
        "answer": f"Good answer for: {prompt}",
        "relevance_score": 0.82,
        "trust_label": "verified",
        "latency_ms": 150,
        "source": "mock",
    }


def mock_low_score(prompt):
    """Returns an answer below the pass threshold (0.46)."""
    return {
        "answer": f"Weak answer for: {prompt}",
        "relevance_score": 0.30,
        "trust_label": "low-confidence",
        "latency_ms": 200,
        "source": "mock",
    }


def mock_no_data(prompt):
    """Returns a no-data response."""
    return {
        "answer": "I don't have data on this topic.",
        "relevance_score": 0.0,
        "trust_label": "unverified",
        "latency_ms": 50,
        "source": "mock",
    }


def mock_exception(prompt):
    """Raises an exception (simulates timeout/crash)."""
    raise TimeoutError(f"LLM timeout for: {prompt}")


call_count = 0
def mock_fail_then_succeed(prompt):
    """Fails first 2 calls, succeeds on 3rd (tests retry)."""
    global call_count
    call_count += 1
    if call_count <= 2:
        return {
            "answer": f"Poor answer attempt {call_count}",
            "relevance_score": 0.20,
            "trust_label": "low-confidence",
            "latency_ms": 300,
            "source": "mock",
        }
    return {
        "answer": f"Good answer on attempt {call_count}",
        "relevance_score": 0.78,
        "trust_label": "verified",
        "latency_ms": 150,
        "source": "mock",
    }


# Track which prompts were queried and in what order
query_log = []
def mock_logging(prompt):
    """Logs every call and returns good answer."""
    query_log.append({
        "prompt": prompt,
        "timestamp": time.time(),
    })
    return {
        "answer": f"Answer for: {prompt}",
        "relevance_score": 0.80,
        "trust_label": "verified",
        "latency_ms": 100,
        "source": "mock",
    }


# Per-goal outcome control
goal_outcomes = {}
def mock_controlled(prompt):
    """Returns different results based on prompt keywords."""
    for keyword, result in goal_outcomes.items():
        if keyword.lower() in prompt.lower():
            if isinstance(result, Exception):
                raise result
            return result
    return mock_good(prompt)


# ═══════════════════════════════════════════════════════════════
# TEST 1: FAILURE INJECTION
# ═══════════════════════════════════════════════════════════════

def test_1_failure_injection():
    print("\n" + "=" * 60)
    print("TEST 1: FAILURE INJECTION")
    print("=" * 60)

    # 1a: Goal fails after max retries -> downstream goals SKIPPED
    print("\n  1a: Goal failure cascades to dependents")
    mm = MissionManager(query_fn=mock_no_data, persist_path=None)
    m = mm.create_mission("Test failure cascade",
                          description="Compare apples and oranges")
    mm.plan(m.id)

    # Execute all goals (all will fail due to mock_no_data returning 0.0)
    for _ in range(20):  # safety cap
        r = mm.execute_step(m.id)
        if r.get("status") in ("mission_completed", "no_ready_goals",
                                "mission_failed"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state = mm.get_mission(m.id)
    goals = state["goals"]
    retrieve_goals = [g for g in goals if g["goal_type"] == "retrieve"]
    dependent_goals = [g for g in goals
                       if g["goal_type"] in ("compare", "synthesize")]

    # Retrieve goals should be FAILED (exhausted retries)
    for rg in retrieve_goals:
        check(f"retrieve '{rg['description'][:30]}' is failed",
              rg["status"] == "failed",
              f"got {rg['status']}")

    # Dependent goals should be SKIPPED (dep failed)
    for dg in dependent_goals:
        check(f"dependent '{dg['description'][:30]}' is skipped",
              dg["status"] == "skipped",
              f"got {dg['status']}")

    # No deliverables should be generated
    check("no deliverables on failure",
          len(state["deliverables"]) == 0,
          f"got {len(state['deliverables'])} deliverables")

    # Error log should be populated
    check("error log records failures",
          len(state["error_log"]) > 0,
          "error_log is empty")

    # 1b: Exception in query_fn is caught, retried, then fails
    print("\n  1b: Exception handling (simulated timeout)")
    mm2 = MissionManager(query_fn=mock_exception, persist_path=None)
    m2 = mm2.create_mission("Exception test", description="Analyze widgets")
    mm2.plan(m2.id)

    for _ in range(15):
        r = mm2.execute_step(m2.id)
        if r.get("status") in ("mission_completed", "no_ready_goals",
                                "mission_failed"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state2 = mm2.get_mission(m2.id)
    for g in state2["goals"]:
        if g["goal_type"] == "retrieve":
            check("exception goal has error recorded",
                  g["error"] is not None and len(g["error"]) > 0,
                  f"error: {g['error']}")
            check("exception goal is failed",
                  g["status"] == "failed",
                  f"got {g['status']}")
            check("exception goal used all attempts",
                  g["attempts"] == g["max_attempts"],
                  f"attempts={g['attempts']}/{g['max_attempts']}")

    # 1c: Low-score answer does NOT generate deliverable
    print("\n  1c: Low-score answer produces no deliverable")
    mm3 = MissionManager(query_fn=mock_low_score, persist_path=None)
    m3 = mm3.create_mission("Low score test", description="What is Toronto?")
    mm3.plan(m3.id)

    for _ in range(15):
        r = mm3.execute_step(m3.id)
        if r.get("status") in ("mission_completed", "no_ready_goals",
                                "mission_failed"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state3 = mm3.get_mission(m3.id)
    check("low-score mission has no deliverables",
          len(state3["deliverables"]) == 0,
          f"got {len(state3['deliverables'])}")


# ═══════════════════════════════════════════════════════════════
# TEST 2: DEPENDENCY INTEGRITY
# ═══════════════════════════════════════════════════════════════

def test_2_dependency_integrity():
    print("\n" + "=" * 60)
    print("TEST 2: DEPENDENCY INTEGRITY")
    print("=" * 60)

    global query_log
    query_log = []

    mm = MissionManager(query_fn=mock_logging, persist_path=None)
    m = mm.create_mission("Dep test",
                          description="Compare Toronto and Montreal")
    mm.plan(m.id)

    state = mm.get_mission(m.id)
    goals = state["goals"]
    check("4-goal comparison decomposition",
          len(goals) == 4, f"got {len(goals)} goals")

    # Check initial readiness: only retrieve goals should be READY
    ready_types = [g["goal_type"] for g in goals if g["status"] == "ready"]
    pending_types = [g["goal_type"] for g in goals if g["status"] == "pending"]
    check("only retrieve goals are initially ready",
          set(ready_types) == {"retrieve"},
          f"ready: {ready_types}")
    check("compare and synthesize are pending",
          "compare" in pending_types and "synthesize" in pending_types,
          f"pending: {pending_types}")

    # Execute step-by-step and verify ordering
    execution_order = []
    for step in range(10):
        r = mm.execute_step(m.id)
        gid = r.get("goal_id")
        if not gid:
            break
        # Find goal type by ID
        for g in mm.get_mission(m.id)["goals"]:
            if g["id"] == gid:
                execution_order.append(g["goal_type"])
                break

    # Verify: retrieves must come before compare, compare before synthesize
    check("execution order has 4 steps",
          len(execution_order) == 4,
          f"got {len(execution_order)}: {execution_order}")

    if len(execution_order) == 4:
        retrieve_indices = [i for i, t in enumerate(execution_order)
                            if t == "retrieve"]
        compare_idx = execution_order.index("compare") if "compare" in execution_order else -1
        synth_idx = execution_order.index("synthesize") if "synthesize" in execution_order else -1

        check("both retrieves execute before compare",
              all(ri < compare_idx for ri in retrieve_indices),
              f"order: {execution_order}")
        check("compare executes before synthesize",
              compare_idx < synth_idx,
              f"compare@{compare_idx} synth@{synth_idx}")

    # Verify query log matches execution order
    check("query log has 4 entries",
          len(query_log) == 4,
          f"got {len(query_log)}")

    # 2b: Manual goal with circular dep should not execute
    print("\n  2b: Synthesize cannot run before dependencies")
    mm2 = MissionManager(query_fn=mock_good, persist_path=None)
    m2 = mm2.create_mission("Manual dep test")

    # Create goals manually with explicit dependencies
    g1 = Goal(description="Step A", goal_type=GoalType.RETRIEVE,
              query="What is A?", priority=1)
    g2 = Goal(description="Step B", goal_type=GoalType.SYNTHESIZE,
              query="Synthesize A+B", dependencies=[g1.id], priority=2)

    mm2.plan(m2.id, goals=[
        {"description": g1.description, "goal_type": "retrieve",
         "query": g1.query, "priority": 1},
    ])

    # Now add dependent goal
    mm2.add_goal(m2.id, description="Synthesize result",
                 goal_type="synthesize", query="Synthesize",
                 dependencies=[mm2.get_mission(m2.id)["goals"][0]["id"]])

    state2 = mm2.get_mission(m2.id)
    synth_g = [g for g in state2["goals"] if g["goal_type"] == "synthesize"][0]
    check("synthesize starts as pending (not ready)",
          synth_g["status"] == "pending",
          f"got {synth_g['status']}")

    # Execute first step (should be retrieve, not synthesize)
    r = mm2.execute_step(m2.id)
    executed_goal = None
    for g in mm2.get_mission(m2.id)["goals"]:
        if g["id"] == r.get("goal_id"):
            executed_goal = g
    check("first executed goal is retrieve, not synthesize",
          executed_goal and executed_goal["goal_type"] == "retrieve",
          f"executed: {executed_goal['goal_type'] if executed_goal else 'none'}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: RETRY AND RECOVERY
# ═══════════════════════════════════════════════════════════════

def test_3_retry_recovery():
    print("\n" + "=" * 60)
    print("TEST 3: RETRY AND RECOVERY")
    print("=" * 60)

    # 3a: Goal retries on low score, eventually succeeds
    global call_count
    call_count = 0

    mm = MissionManager(query_fn=mock_fail_then_succeed, persist_path=None)
    m = mm.create_mission("Retry test", description="What is Toronto?")
    mm.plan(m.id)

    results = []
    for _ in range(10):
        r = mm.execute_step(m.id)
        results.append(r)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state = mm.get_mission(m.id)
    retrieve_g = [g for g in state["goals"] if g["goal_type"] == "retrieve"][0]

    check("retrieve goal eventually completes",
          retrieve_g["status"] == "completed",
          f"got {retrieve_g['status']}")
    check("retrieve took multiple attempts",
          retrieve_g["attempts"] >= 2,
          f"attempts={retrieve_g['attempts']}")
    check("final score is good",
          retrieve_g["result"].get("score", 0) >= 0.46,
          f"score={retrieve_g['result'].get('score', 0)}")

    # 3b: max_attempts honored (goal fails after 3 tries)
    print("\n  3b: Max attempts enforced")
    mm2 = MissionManager(query_fn=mock_low_score, persist_path=None)
    m2 = mm2.create_mission("Max retry test", description="What is XYZ?")
    mm2.plan(m2.id)

    for _ in range(15):
        r = mm2.execute_step(m2.id)
        if r.get("status") in ("no_ready_goals", "mission_completed"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state2 = mm2.get_mission(m2.id)
    for g in state2["goals"]:
        if g["goal_type"] == "retrieve":
            check(f"goal '{g['description'][:25]}' capped at max_attempts",
                  g["attempts"] == g["max_attempts"],
                  f"attempts={g['attempts']}/{g['max_attempts']}")
            check("goal is failed after max attempts",
                  g["status"] == "failed",
                  f"got {g['status']}")

    # 3c: Score threshold boundary test
    print("\n  3c: Score threshold boundary (0.46)")
    def mock_borderline(prompt):
        return {"answer": "Borderline", "relevance_score": 0.46,
                "trust_label": "best-effort", "latency_ms": 100,
                "source": "mock"}

    mm3 = MissionManager(query_fn=mock_borderline, persist_path=None)
    m3 = mm3.create_mission("Threshold test", description="What is ABC?")
    mm3.plan(m3.id)
    mm3.execute_step(m3.id)  # Should complete (>= 0.46)

    state3 = mm3.get_mission(m3.id)
    rg = [g for g in state3["goals"] if g["goal_type"] == "retrieve"][0]
    check("score 0.46 passes threshold (completes in 1 attempt)",
          rg["status"] == "completed" and rg["attempts"] == 1,
          f"status={rg['status']} attempts={rg['attempts']}")

    def mock_just_below(prompt):
        return {"answer": "Just below", "relevance_score": 0.459,
                "trust_label": "low-confidence", "latency_ms": 100,
                "source": "mock"}

    mm4 = MissionManager(query_fn=mock_just_below, persist_path=None)
    m4 = mm4.create_mission("Below threshold", description="What is DEF?")
    mm4.plan(m4.id)
    mm4.execute_step(m4.id)  # Should retry (< 0.46)

    state4 = mm4.get_mission(m4.id)
    rg2 = [g for g in state4["goals"] if g["goal_type"] == "retrieve"][0]
    check("score 0.459 triggers retry (status=ready after 1 attempt)",
          rg2["status"] == "ready" and rg2["attempts"] == 1,
          f"status={rg2['status']} attempts={rg2['attempts']}")


# ═══════════════════════════════════════════════════════════════
# TEST 4: CONCURRENT MISSIONS
# ═══════════════════════════════════════════════════════════════

def test_4_concurrent_missions():
    print("\n" + "=" * 60)
    print("TEST 4: CONCURRENT MISSIONS")
    print("=" * 60)

    # 4a: Multiple missions, no state leakage
    mm = MissionManager(query_fn=mock_good, persist_path=None)

    missions = []
    for i in range(5):
        m = mm.create_mission(
            f"Mission {i}",
            description=f"Compare entity{i}a and entity{i}b",
            tags=[f"batch-{i}"])
        mm.plan(m.id)
        missions.append(m.id)

    check("5 missions created",
          len(mm.list_missions()) == 5,
          f"got {len(mm.list_missions())}")

    # Execute all missions completely
    for mid in missions:
        for _ in range(20):
            r = mm.execute_step(mid)
            if r.get("status") in ("mission_completed", "no_ready_goals"):
                break
            if "error" in r and "Mission is" in r.get("error", ""):
                break

    # Verify each mission independently
    for i, mid in enumerate(missions):
        state = mm.get_mission(mid)
        goals = state["goals"]
        completed = [g for g in goals if g["status"] == "completed"]

        check(f"mission {i} has correct tag",
              f"batch-{i}" in state["tags"],
              f"tags={state['tags']}")
        check(f"mission {i} all goals completed ({len(completed)}/{len(goals)})",
              len(completed) == len(goals),
              f"{len(completed)}/{len(goals)}")

    # 4b: Verify deliverables don't cross missions
    all_deliverable_ids = set()
    for mid in missions:
        state = mm.get_mission(mid)
        for d in state["deliverables"]:
            check(f"deliverable '{d['id']}' is unique across missions",
                  d["id"] not in all_deliverable_ids,
                  f"duplicate: {d['id']}")
            all_deliverable_ids.add(d["id"])

            # Deliverable's source_goals should belong to this mission
            mission_goal_ids = {g["id"] for g in state["goals"]}
            for sg in d["source_goals"]:
                check(f"deliverable source goal belongs to same mission",
                      sg in mission_goal_ids,
                      f"goal {sg} not in mission {mid}")

    # 4c: Status filtering with multiple missions
    # Completed missions auto-transition to COMPLETED status
    completed_list = mm.list_missions(status="completed")
    check("completed filter returns 5 (all missions done)",
          len(completed_list) == 5,
          f"got {len(completed_list)}")

    # Create fresh missions for cancel/pause testing
    m_cancel = mm.create_mission("Cancel target", description="Analyze foo")
    mm.plan(m_cancel.id)
    mm.cancel(m_cancel.id)

    m_pause = mm.create_mission("Pause target", description="Analyze bar")
    mm.plan(m_pause.id)
    mm.pause(m_pause.id)

    cancelled = mm.list_missions(status="cancelled")
    paused = mm.list_missions(status="paused")

    check("cancelled filter returns 1",
          len(cancelled) == 1,
          f"got {len(cancelled)}")
    check("paused filter returns 1",
          len(paused) == 1,
          f"got {len(paused)}")

    # 4d: Thread safety - concurrent execution
    print("\n  4d: Thread-safe concurrent execution")
    mm2 = MissionManager(query_fn=mock_good, persist_path=None)

    thread_missions = []
    for i in range(3):
        m = mm2.create_mission(f"Thread mission {i}",
                               description=f"Analyze topic{i}")
        mm2.plan(m.id)
        thread_missions.append(m.id)

    errors = []
    def run_mission(mid):
        try:
            for _ in range(20):
                r = mm2.execute_step(mid)
                if r.get("status") in ("mission_completed", "no_ready_goals"):
                    break
                if "error" in r and "Mission is" in r.get("error", ""):
                    break
        except Exception as e:
            errors.append(f"Mission {mid}: {e}")

    threads = [threading.Thread(target=run_mission, args=(mid,))
               for mid in thread_missions]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    check("no thread safety errors",
          len(errors) == 0,
          f"errors: {errors}")

    for i, mid in enumerate(thread_missions):
        state = mm2.get_mission(mid)
        completed = sum(1 for g in state["goals"]
                        if g["status"] == "completed")
        check(f"thread mission {i} goals completed ({completed}/{len(state['goals'])})",
              completed == len(state["goals"]),
              f"{completed}/{len(state['goals'])}")


# ═══════════════════════════════════════════════════════════════
# TEST 5: AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════

def test_5_audit_trail():
    print("\n" + "=" * 60)
    print("TEST 5: AUDIT TRAIL")
    print("=" * 60)

    global query_log
    query_log = []

    mm = MissionManager(query_fn=mock_logging, persist_path=None)
    m = mm.create_mission("Audit test",
                          description="Compare Toronto and Montreal",
                          tags=["audit"])
    mm.plan(m.id)

    # Add a checkpoint
    goals = mm.get_mission(m.id)["goals"]
    retrieve_ids = [g["id"] for g in goals if g["goal_type"] == "retrieve"]
    mm.add_checkpoint(m.id, description="Both entities retrieved",
                      required_goals=retrieve_ids)

    # Execute all
    for _ in range(10):
        r = mm.execute_step(m.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state = mm.get_mission(m.id)

    # 5a: Every goal has timestamps
    print("\n  5a: Goal timestamps")
    for g in state["goals"]:
        check(f"goal '{g['description'][:25]}' has created_at",
              g["created_at"] is not None and g["created_at"] > 0,
              f"created_at={g['created_at']}")
        if g["status"] == "completed":
            check(f"goal '{g['description'][:25]}' has completed_at",
                  g["completed_at"] is not None and g["completed_at"] > 0,
                  f"completed_at={g['completed_at']}")
            check(f"completed_at > created_at",
                  g["completed_at"] >= g["created_at"],
                  f"created={g['created_at']} completed={g['completed_at']}")

    # 5b: Every goal has scores and trust labels
    print("\n  5b: Goal scores and trust")
    for g in state["goals"]:
        if g["status"] == "completed" and g["result"]:
            result = g["result"]
            check(f"goal '{g['description'][:25]}' has score",
                  "score" in result and result["score"] > 0,
                  f"result={result}")
            check(f"goal '{g['description'][:25]}' has trust label",
                  "trust" in result and len(result["trust"]) > 0,
                  f"trust={result.get('trust')}")
            check(f"goal '{g['description'][:25]}' has latency",
                  "latency_ms" in result,
                  f"keys={list(result.keys())}")

    # 5c: Attempts tracked
    print("\n  5c: Attempt counts")
    for g in state["goals"]:
        check(f"goal '{g['description'][:25]}' attempts >= 1",
              g["attempts"] >= 1,
              f"attempts={g['attempts']}")

    # 5d: Deliverables have metadata
    print("\n  5d: Deliverable metadata")
    check("has deliverables", len(state["deliverables"]) > 0,
          f"got {len(state['deliverables'])}")
    for d in state["deliverables"]:
        check(f"deliverable '{d['title'][:25]}' has content",
              len(d.get("content", "")) > 0,
              "empty content")
        check(f"deliverable has source_goals",
              len(d.get("source_goals", [])) > 0,
              "no source_goals")
        check(f"deliverable has created_at",
              d.get("created_at") is not None,
              f"created_at={d.get('created_at')}")
        meta = d.get("metadata", {})
        check(f"deliverable has score in metadata",
              "score" in meta,
              f"metadata={meta}")
        check(f"deliverable has trust in metadata",
              "trust" in meta,
              f"metadata={meta}")

    # 5e: Checkpoint reached
    print("\n  5e: Checkpoint tracking")
    check("has checkpoints", len(state["checkpoints"]) > 0,
          f"got {len(state['checkpoints'])}")
    for cp in state["checkpoints"]:
        check(f"checkpoint '{cp['description'][:25]}' reached",
              cp["status"] == "reached",
              f"status={cp['status']}")
        check(f"checkpoint has reached_at timestamp",
              cp["reached_at"] is not None and cp["reached_at"] > 0,
              f"reached_at={cp['reached_at']}")

    # 5f: Mission-level audit
    print("\n  5f: Mission-level metadata")
    check("mission has created_at",
          state["created_at"] is not None,
          f"created_at={state['created_at']}")
    check("mission has updated_at",
          state["updated_at"] is not None,
          f"updated_at={state['updated_at']}")
    check("updated_at >= created_at",
          state["updated_at"] >= state["created_at"],
          f"created={state['created_at']} updated={state['updated_at']}")
    check("mission has goal_summary",
          "goal_summary" in state and len(state["goal_summary"]) > 0,
          f"goal_summary={state.get('goal_summary')}")
    check("mission has tags",
          "audit" in state["tags"],
          f"tags={state['tags']}")

    # 5g: Query log matches execution (from mock_logging)
    print("\n  5g: Query execution log")
    check("query log has 4 entries (4 goals executed)",
          len(query_log) == 4,
          f"got {len(query_log)}")
    if len(query_log) >= 2:
        check("queries are in chronological order",
              all(query_log[i]["timestamp"] <= query_log[i+1]["timestamp"]
                  for i in range(len(query_log)-1)),
              "timestamps out of order")


# ═══════════════════════════════════════════════════════════════
# TEST 6 (BONUS): PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def test_6_persistence():
    print("\n" + "=" * 60)
    print("TEST 6 (BONUS): PERSISTENCE")
    print("=" * 60)

    import tempfile
    persist_file = os.path.join(tempfile.gettempdir(), "kos_test_missions.json")

    # Clean up from previous runs
    if os.path.exists(persist_file):
        os.remove(persist_file)

    # Create and execute a mission
    mm1 = MissionManager(query_fn=mock_good, persist_path=persist_file)
    m = mm1.create_mission("Persist test", description="Analyze widgets")
    mm1.plan(m.id)
    mm1.execute_step(m.id)  # Execute first goal

    state_before = mm1.get_mission(m.id)
    goals_before = {g["id"]: g["status"] for g in state_before["goals"]}

    # Simulate server restart: create new MissionManager from same file
    mm2 = MissionManager(query_fn=mock_good, persist_path=persist_file)

    check("mission survives restart",
          len(mm2.list_missions()) == 1,
          f"got {len(mm2.list_missions())} missions")

    state_after = mm2.get_mission(m.id)
    check("mission ID preserved",
          state_after["id"] == m.id,
          f"expected {m.id}, got {state_after['id']}")
    check("mission name preserved",
          state_after["name"] == "Persist test",
          f"got {state_after['name']}")
    check("mission status preserved",
          state_after["status"] == state_before["status"],
          f"expected {state_before['status']}, got {state_after['status']}")

    goals_after = {g["id"]: g["status"] for g in state_after["goals"]}
    check("goal states preserved",
          goals_before == goals_after,
          f"before={goals_before} after={goals_after}")

    # Clean up
    if os.path.exists(persist_file):
        os.remove(persist_file)


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KOS v0.8 MISSION RESILIENCE TESTS")
    print("=" * 60)

    test_1_failure_injection()
    test_2_dependency_integrity()
    test_3_retry_recovery()
    test_4_concurrent_missions()
    test_5_audit_trail()
    test_6_persistence()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print(f"\nFAILED TESTS: {FAIL}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)
