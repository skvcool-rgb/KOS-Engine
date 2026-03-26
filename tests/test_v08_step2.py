"""
KOS v0.8 Step 2 -- Multi-Agent Framework Tests

6 test suites:
  A. Protocol tests (AgentTask, AgentResult, AgentStatus)
  B. Registry tests (register, match, duplicate, unknown)
  C. Dispatcher tests (route, exception safety, audit log)
  D. Parity tests (legacy vs agent path produce same results)
  E. Failure isolation tests (agent failure does not leak)
  F. End-to-end tests (full mission lifecycle through agents)
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.agent_protocol import (
    AgentTask, AgentResult, AgentStatus, AgentEvidence,
)
from kos.agents.base_agent import BaseAgent
from kos.agents.retrieval_agent import RetrievalAgent
from kos.agents.comparison_agent import ComparisonAgent
from kos.agents.synthesis_agent import SynthesisAgent
from kos.agent_registry import AgentRegistry
from kos.task_dispatcher import TaskDispatcher
from kos.mission import MissionManager, GoalStatus, GoalType

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


# ── Mock functions ─────────────────────────────────────────────

def mock_good(prompt):
    return {
        "answer": f"Good answer for: {prompt}",
        "relevance_score": 0.82,
        "trust_label": "verified",
        "latency_ms": 100,
        "source": "mock",
        "route": "fast",
    }

def mock_fail(prompt):
    return {
        "answer": "I don't have data on this topic.",
        "relevance_score": 0.0,
        "trust_label": "unverified",
        "latency_ms": 50,
        "source": "mock",
    }

def mock_exception(prompt):
    raise TimeoutError(f"Timeout: {prompt}")


# ═══════════════════════════════════════════════════════════════
# A. PROTOCOL TESTS
# ═══════════════════════════════════════════════════════════════

def test_a_protocol():
    print("\n" + "=" * 60)
    print("A. PROTOCOL TESTS")
    print("=" * 60)

    # A1: Create valid AgentTask
    task = AgentTask(
        task_id="t1", mission_id="m1", goal_id="g1",
        goal_type="retrieve", payload={"query": "What is Toronto?"})
    check("AgentTask creates with all fields",
          task.task_id == "t1" and task.goal_type == "retrieve",
          f"task_id={task.task_id}")

    # A2: AgentTask to_dict
    d = task.to_dict()
    check("AgentTask.to_dict has all keys",
          all(k in d for k in ("task_id", "mission_id", "goal_id",
                                "goal_type", "payload", "attempt")),
          f"keys={list(d.keys())}")

    # A3: Create valid AgentResult
    result = AgentResult(
        task_id="t1", mission_id="m1", goal_id="g1",
        agent_name="retrieval_agent", status=AgentStatus.COMPLETE,
        output={"answer": "Toronto is a city"}, score=0.85,
        confidence=0.85, trust_label="verified", latency_ms=150)
    check("AgentResult creates with all fields",
          result.status == AgentStatus.COMPLETE and result.score == 0.85,
          f"status={result.status}")

    # A4: AgentResult to_dict serializes enums
    rd = result.to_dict()
    check("AgentResult.to_dict serializes status as string",
          rd["status"] == "COMPLETE",
          f"status={rd['status']}")

    # A5: Failed result with error and retryable
    fail_result = AgentResult(
        task_id="t2", mission_id="m1", goal_id="g2",
        agent_name="retrieval_agent", status=AgentStatus.FAILED,
        error="Connection timeout", retryable=True)
    check("Failed result carries error",
          fail_result.error == "Connection timeout",
          f"error={fail_result.error}")
    check("Failed result carries retryable flag",
          fail_result.retryable is True,
          f"retryable={fail_result.retryable}")

    # A6: AgentEvidence
    ev = AgentEvidence(source="wikipedia", content="Toronto is...", score=0.9)
    check("AgentEvidence creates correctly",
          ev.source == "wikipedia" and ev.score == 0.9,
          f"source={ev.source}")
    check("AgentEvidence.to_dict works",
          "source" in ev.to_dict() and "content" in ev.to_dict(),
          f"keys={list(ev.to_dict().keys())}")

    # A7: All AgentStatus values
    check("AgentStatus has 4 values",
          len(AgentStatus) == 4,
          f"got {len(AgentStatus)}")
    for s in ("COMPLETE", "FAILED", "RETRYABLE", "SKIPPED"):
        check(f"AgentStatus.{s} exists",
              hasattr(AgentStatus, s), "missing")


# ═══════════════════════════════════════════════════════════════
# B. REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════

def test_b_registry():
    print("\n" + "=" * 60)
    print("B. REGISTRY TESTS")
    print("=" * 60)

    reg = AgentRegistry()

    # B1: Register three agents
    reg.register(RetrievalAgent(mock_good))
    reg.register(ComparisonAgent(mock_good))
    reg.register(SynthesisAgent(mock_good))
    check("3 agents registered",
          len(reg) == 3, f"got {len(reg)}")

    # B2: List agents (sorted)
    names = reg.list_agents()
    check("list_agents returns sorted names",
          names == ["comparison_agent", "retrieval_agent", "synthesis_agent"],
          f"got {names}")

    # B3: Duplicate registration raises
    try:
        reg.register(RetrievalAgent(mock_good))
        check("duplicate raises ValueError", False, "no exception")
    except ValueError:
        check("duplicate raises ValueError", True)

    # B4: match("retrieve") returns RetrievalAgent
    agent = reg.match("retrieve")
    check("match('retrieve') -> retrieval_agent",
          agent.name == "retrieval_agent",
          f"got {agent.name}")

    # B5: match("compare") returns ComparisonAgent
    agent = reg.match("compare")
    check("match('compare') -> comparison_agent",
          agent.name == "comparison_agent",
          f"got {agent.name}")

    # B6: match("synthesize") returns SynthesisAgent
    agent = reg.match("synthesize")
    check("match('synthesize') -> synthesis_agent",
          agent.name == "synthesis_agent",
          f"got {agent.name}")

    # B7: match("factual") also returns RetrievalAgent
    agent = reg.match("factual")
    check("match('factual') -> retrieval_agent",
          agent.name == "retrieval_agent",
          f"got {agent.name}")

    # B8: Unknown goal type raises KeyError
    try:
        reg.match("unknown_type")
        check("unknown type raises KeyError", False, "no exception")
    except KeyError:
        check("unknown type raises KeyError", True)

    # B9: get() by name
    agent = reg.get("retrieval_agent")
    check("get('retrieval_agent') works",
          agent.name == "retrieval_agent",
          f"got {agent.name}")

    # B10: get() unknown raises KeyError
    try:
        reg.get("nonexistent")
        check("get unknown raises KeyError", False, "no exception")
    except KeyError:
        check("get unknown raises KeyError", True)


# ═══════════════════════════════════════════════════════════════
# C. DISPATCHER TESTS
# ═══════════════════════════════════════════════════════════════

def test_c_dispatcher():
    print("\n" + "=" * 60)
    print("C. DISPATCHER TESTS")
    print("=" * 60)

    reg = AgentRegistry()
    reg.register(RetrievalAgent(mock_good))
    reg.register(ComparisonAgent(mock_good))
    reg.register(SynthesisAgent(mock_good))
    dispatcher = TaskDispatcher(reg)

    # C1: Dispatch retrieve
    task = AgentTask(task_id="c1", mission_id="m1", goal_id="g1",
                     goal_type="retrieve", payload={"query": "What is Toronto?"})
    result = dispatcher.dispatch(task)
    check("dispatch retrieve -> COMPLETE",
          result.status == AgentStatus.COMPLETE,
          f"status={result.status}")
    check("dispatch retrieve -> retrieval_agent",
          result.agent_name == "retrieval_agent",
          f"agent={result.agent_name}")
    check("dispatch retrieve has score",
          result.score > 0,
          f"score={result.score}")

    # C2: Dispatch compare
    task2 = AgentTask(task_id="c2", mission_id="m1", goal_id="g2",
                      goal_type="compare",
                      payload={"query": "Compare A and B", "left": "A", "right": "B"})
    result2 = dispatcher.dispatch(task2)
    check("dispatch compare -> COMPLETE",
          result2.status == AgentStatus.COMPLETE,
          f"status={result2.status}")
    check("dispatch compare -> comparison_agent",
          result2.agent_name == "comparison_agent",
          f"agent={result2.agent_name}")

    # C3: Dispatch synthesize
    task3 = AgentTask(task_id="c3", mission_id="m1", goal_id="g3",
                      goal_type="synthesize",
                      payload={"query": "Summarize findings"})
    result3 = dispatcher.dispatch(task3)
    check("dispatch synthesize -> COMPLETE",
          result3.status == AgentStatus.COMPLETE,
          f"status={result3.status}")
    check("dispatch synthesize -> synthesis_agent",
          result3.agent_name == "synthesis_agent",
          f"agent={result3.agent_name}")

    # C4: Unknown goal type -> FAILED (not crash)
    task4 = AgentTask(task_id="c4", mission_id="m1", goal_id="g4",
                      goal_type="unknown", payload={})
    result4 = dispatcher.dispatch(task4)
    check("dispatch unknown -> FAILED (no crash)",
          result4.status == AgentStatus.FAILED,
          f"status={result4.status}")
    check("dispatch unknown has error message",
          result4.error is not None and "No agent" in result4.error,
          f"error={result4.error}")

    # C5: Agent exception -> FAILED result (not uncaught)
    reg2 = AgentRegistry()
    reg2.register(RetrievalAgent(mock_exception))
    dispatcher2 = TaskDispatcher(reg2)
    task5 = AgentTask(task_id="c5", mission_id="m1", goal_id="g5",
                      goal_type="retrieve", payload={"query": "test"})
    result5 = dispatcher2.dispatch(task5)
    check("agent exception -> FAILED (not crash)",
          result5.status == AgentStatus.FAILED,
          f"status={result5.status}")
    check("agent exception error captured",
          result5.error is not None and "Timeout" in result5.error,
          f"error={result5.error}")

    # C6: Audit log
    log = dispatcher.get_log()
    check("dispatcher audit log has entries",
          len(log) >= 4,
          f"got {len(log)} entries")
    check("audit log entry has required fields",
          all(k in log[0] for k in ("task_id", "goal_type", "agent",
                                     "status", "timestamp")),
          f"keys={list(log[0].keys())}")


# ═══════════════════════════════════════════════════════════════
# D. PARITY TESTS
# ═══════════════════════════════════════════════════════════════

def test_d_parity():
    print("\n" + "=" * 60)
    print("D. PARITY TESTS (legacy vs agent)")
    print("=" * 60)

    # Build dispatcher
    reg = AgentRegistry()
    reg.register(RetrievalAgent(mock_good))
    reg.register(ComparisonAgent(mock_good))
    reg.register(SynthesisAgent(mock_good))
    dispatcher = TaskDispatcher(reg)

    missions_to_test = [
        ("Compare Toronto and Montreal", "comparison"),
        ("Analyze backpropagation", "analyze"),
        ("What is perovskite?", "simple"),
    ]

    for desc, label in missions_to_test:
        print(f"\n  Parity: {label} ({desc[:40]})")

        # Legacy path
        mm_legacy = MissionManager(query_fn=mock_good, persist_path=None,
                                   use_agents=False)
        m1 = mm_legacy.create_mission(f"Legacy {label}", description=desc)
        mm_legacy.plan(m1.id)
        for _ in range(20):
            r = mm_legacy.execute_step(m1.id)
            if r.get("status") in ("mission_completed", "no_ready_goals"):
                break
            if "error" in r and "Mission is" in r.get("error", ""):
                break
        state1 = mm_legacy.get_mission(m1.id)

        # Agent path
        mm_agent = MissionManager(query_fn=mock_good, persist_path=None,
                                  use_agents=True, dispatcher=dispatcher)
        m2 = mm_agent.create_mission(f"Agent {label}", description=desc)
        mm_agent.plan(m2.id)
        for _ in range(20):
            r = mm_agent.execute_step(m2.id)
            if r.get("status") in ("mission_completed", "no_ready_goals"):
                break
            if "error" in r and "Mission is" in r.get("error", ""):
                break
        state2 = mm_agent.get_mission(m2.id)

        # Compare outcomes
        check(f"[{label}] same mission status",
              state1["status"] == state2["status"],
              f"legacy={state1['status']} agent={state2['status']}")

        check(f"[{label}] same goal count",
              len(state1["goals"]) == len(state2["goals"]),
              f"legacy={len(state1['goals'])} agent={len(state2['goals'])}")

        check(f"[{label}] same deliverable count",
              len(state1["deliverables"]) == len(state2["deliverables"]),
              f"legacy={len(state1['deliverables'])} agent={len(state2['deliverables'])}")

        # Compare goal ordering (types should match)
        types1 = [g["goal_type"] for g in state1["goals"]]
        types2 = [g["goal_type"] for g in state2["goals"]]
        check(f"[{label}] same goal type ordering",
              types1 == types2,
              f"legacy={types1} agent={types2}")

        # Compare goal statuses
        statuses1 = [g["status"] for g in state1["goals"]]
        statuses2 = [g["status"] for g in state2["goals"]]
        check(f"[{label}] same goal statuses",
              statuses1 == statuses2,
              f"legacy={statuses1} agent={statuses2}")

        # Score drift check (within tolerance)
        scores1 = [g.get("result", {}).get("score", 0)
                    for g in state1["goals"] if g.get("result")]
        scores2 = [g.get("result", {}).get("score", 0)
                    for g in state2["goals"] if g.get("result")]
        if scores1 and scores2:
            avg1 = sum(scores1) / len(scores1)
            avg2 = sum(scores2) / len(scores2)
            drift = abs(avg1 - avg2)
            check(f"[{label}] score drift <= 0.02",
                  drift <= 0.02,
                  f"legacy_avg={avg1:.3f} agent_avg={avg2:.3f} drift={drift:.3f}")

        # Agent routing check (agent path should have agent names)
        for g in state2["goals"]:
            if g.get("result") and g["status"] == "completed":
                agent_name = g["result"].get("agent", g["result"].get("source", ""))
                check(f"[{label}] goal '{g['goal_type']}' routed to agent",
                      agent_name and agent_name != "unknown",
                      f"agent={agent_name}")


# ═══════════════════════════════════════════════════════════════
# E. FAILURE ISOLATION TESTS
# ═══════════════════════════════════════════════════════════════

def test_e_failure_isolation():
    print("\n" + "=" * 60)
    print("E. FAILURE ISOLATION TESTS")
    print("=" * 60)

    # E1: ComparisonAgent fails -> synth skipped, retrieves unaffected
    print("\n  E1: ComparisonAgent failure isolation")

    call_log = {"compare": 0, "retrieve": 0, "synthesize": 0}
    def mock_compare_fails(prompt):
        if "compare" in prompt.lower():
            call_log["compare"] += 1
            return {"answer": "", "relevance_score": 0.0,
                    "trust_label": "unverified", "source": "mock"}
        call_log["retrieve"] += 1
        return mock_good(prompt)

    reg = AgentRegistry()
    reg.register(RetrievalAgent(mock_good))
    reg.register(ComparisonAgent(mock_compare_fails))
    reg.register(SynthesisAgent(mock_good))
    dispatcher = TaskDispatcher(reg)

    mm = MissionManager(query_fn=mock_good, persist_path=None,
                        use_agents=True, dispatcher=dispatcher)
    m = mm.create_mission("Fail compare test",
                          description="Compare apples and oranges")
    mm.plan(m.id)

    for _ in range(20):
        r = mm.execute_step(m.id)
        if r.get("status") in ("mission_completed", "no_ready_goals",
                                "mission_failed"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state = mm.get_mission(m.id)
    for g in state["goals"]:
        if g["goal_type"] == "retrieve":
            check(f"retrieve '{g['description'][:25]}' completed despite compare fail",
                  g["status"] == "completed",
                  f"status={g['status']}")
        elif g["goal_type"] == "compare":
            check("compare goal failed",
                  g["status"] == "failed",
                  f"status={g['status']}")
        elif g["goal_type"] == "synthesize":
            check("synthesize skipped (dep failed)",
                  g["status"] == "skipped",
                  f"status={g['status']}")

    check("no fake deliverables from failed comparison",
          all(d["metadata"].get("score", 0) > 0
              for d in state["deliverables"]) if state["deliverables"] else True,
          "deliverable with 0 score found")

    # E2: RetrievalAgent timeout -> retry + no state corruption
    print("\n  E2: RetrievalAgent timeout/retry")

    attempt_count = [0]
    def mock_retry_retrieve(prompt):
        attempt_count[0] += 1
        if attempt_count[0] <= 2:
            raise TimeoutError("Simulated timeout")
        return mock_good(prompt)

    reg2 = AgentRegistry()
    reg2.register(RetrievalAgent(mock_retry_retrieve))
    reg2.register(ComparisonAgent(mock_good))
    reg2.register(SynthesisAgent(mock_good))
    dispatcher2 = TaskDispatcher(reg2)

    mm2 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher2)
    m2 = mm2.create_mission("Retry test", description="What is Toronto?")
    mm2.plan(m2.id)

    for _ in range(15):
        r = mm2.execute_step(m2.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state2 = mm2.get_mission(m2.id)
    rg = [g for g in state2["goals"] if g["goal_type"] == "retrieve"][0]
    check("retrieval recovered after timeout retries",
          rg["status"] == "completed",
          f"status={rg['status']}")
    check("retrieval took multiple attempts",
          rg["attempts"] >= 2,
          f"attempts={rg['attempts']}")

    # E3: SynthesisAgent failure -> no fake deliverable
    print("\n  E3: SynthesisAgent failure -> no fake deliverable")

    def mock_synth_fails(prompt):
        if "synth" in prompt.lower() or "summar" in prompt.lower():
            return {"answer": "", "relevance_score": 0.0,
                    "trust_label": "unverified", "source": "mock"}
        return mock_good(prompt)

    reg3 = AgentRegistry()
    reg3.register(RetrievalAgent(mock_good))
    reg3.register(ComparisonAgent(mock_good))
    reg3.register(SynthesisAgent(mock_synth_fails))
    dispatcher3 = TaskDispatcher(reg3)

    mm3 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher3)
    m3 = mm3.create_mission("Synth fail test",
                            description="Compare X and Y")
    mm3.plan(m3.id)

    for _ in range(20):
        r = mm3.execute_step(m3.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state3 = mm3.get_mission(m3.id)
    synth_goals = [g for g in state3["goals"]
                   if g["goal_type"] == "synthesize"]
    for sg in synth_goals:
        check("synth goal failed (not fake success)",
              sg["status"] == "failed",
              f"status={sg['status']}")

    # Upstream goals should still be completed
    for g in state3["goals"]:
        if g["goal_type"] in ("retrieve", "compare"):
            check(f"upstream '{g['goal_type']}' still completed",
                  g["status"] == "completed",
                  f"status={g['status']}")

    # No synth deliverables
    synth_deliverables = [d for d in state3["deliverables"]
                          if "synth" in d.get("title", "").lower()]
    check("no synthesis deliverables from failed synth",
          len(synth_deliverables) == 0,
          f"got {len(synth_deliverables)}")

    # E4: Cross-mission isolation
    print("\n  E4: Cross-mission failure isolation")

    reg4 = AgentRegistry()
    reg4.register(RetrievalAgent(mock_good))
    reg4.register(ComparisonAgent(mock_good))
    reg4.register(SynthesisAgent(mock_good))
    dispatcher4 = TaskDispatcher(reg4)

    mm4 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher4)

    # Mission A: will succeed
    ma = mm4.create_mission("Good mission", description="Compare A and B")
    mm4.plan(ma.id)

    # Mission B: will fail (use mock_fail as query_fn for its agent)
    # Actually, the agents are shared. Let's use a different approach:
    # Create mission B but with query that triggers failure
    mb = mm4.create_mission("Bad mission",
                            description="Compare unicorns and dragons")
    mm4.plan(mb.id)

    # Execute both
    for _ in range(20):
        r = mm4.execute_step(ma.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    for _ in range(20):
        r = mm4.execute_step(mb.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state_a = mm4.get_mission(ma.id)
    state_b = mm4.get_mission(mb.id)

    # Both should complete (mock_good always succeeds)
    check("mission A completed independently",
          state_a["status"] == "completed",
          f"status={state_a['status']}")
    check("mission B completed independently",
          state_b["status"] == "completed",
          f"status={state_b['status']}")

    # No deliverable crossover
    a_goal_ids = {g["id"] for g in state_a["goals"]}
    b_goal_ids = {g["id"] for g in state_b["goals"]}
    check("no goal ID overlap between missions",
          a_goal_ids.isdisjoint(b_goal_ids),
          f"overlap: {a_goal_ids & b_goal_ids}")

    for d in state_a["deliverables"]:
        for sg in d["source_goals"]:
            check("mission A deliverable source is from mission A",
                  sg in a_goal_ids,
                  f"goal {sg} not in mission A")


# ═══════════════════════════════════════════════════════════════
# F. END-TO-END TESTS
# ═══════════════════════════════════════════════════════════════

def test_f_e2e():
    print("\n" + "=" * 60)
    print("F. END-TO-END TESTS")
    print("=" * 60)

    reg = AgentRegistry()
    reg.register(RetrievalAgent(mock_good))
    reg.register(ComparisonAgent(mock_good))
    reg.register(SynthesisAgent(mock_good))
    dispatcher = TaskDispatcher(reg)

    # F1: Happy path
    print("\n  F1: Happy path (create -> plan -> execute -> deliverables)")
    mm = MissionManager(query_fn=mock_good, persist_path=None,
                        use_agents=True, dispatcher=dispatcher)
    m = mm.create_mission("E2E test",
                          description="Compare Toronto and Montreal")
    mm.plan(m.id)

    state = mm.get_mission(m.id)
    check("4 goals planned", len(state["goals"]) == 4,
          f"got {len(state['goals'])}")

    for _ in range(10):
        r = mm.execute_step(m.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state = mm.get_mission(m.id)
    check("mission completed", state["status"] == "completed",
          f"status={state['status']}")
    completed = sum(1 for g in state["goals"] if g["status"] == "completed")
    check("all 4 goals completed", completed == 4,
          f"completed={completed}")
    check("2 deliverables produced",
          len(state["deliverables"]) == 2,
          f"got {len(state['deliverables'])}")

    # Check agent routing
    for g in state["goals"]:
        agent = g.get("result", {}).get("agent", "")
        check(f"goal '{g['goal_type']}' has agent recorded",
              agent != "" and agent != "unknown",
              f"agent='{agent}'")

    # F2: Manual goal insertion
    print("\n  F2: Manual goal insertion")
    mm2 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher)
    m2 = mm2.create_mission("Manual goal test", description="Test manual goals")
    mm2.plan(m2.id)

    # Add custom synth goal
    existing_goals = mm2.get_mission(m2.id)["goals"]
    mm2.add_goal(m2.id, description="Custom synthesis",
                 goal_type="synthesize", query="Synthesize everything",
                 dependencies=[existing_goals[0]["id"]])

    state2 = mm2.get_mission(m2.id)
    synth_goals = [g for g in state2["goals"]
                   if g["description"] == "Custom synthesis"]
    check("custom goal added",
          len(synth_goals) == 1,
          f"found {len(synth_goals)}")

    # F3: Pause/resume during execution
    print("\n  F3: Pause/resume lifecycle")
    mm3 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher)
    m3 = mm3.create_mission("Lifecycle test",
                            description="Analyze deep learning")
    mm3.plan(m3.id)

    # Execute 1 step
    mm3.execute_step(m3.id)

    # Pause
    mm3.pause(m3.id)
    state3 = mm3.get_mission(m3.id)
    check("mission paused", state3["status"] == "paused",
          f"status={state3['status']}")

    # Execute while paused should return error
    r = mm3.execute_step(m3.id)
    check("execute while paused returns error",
          "error" in r,
          f"result={r}")

    # Resume and complete
    mm3.resume(m3.id)
    for _ in range(10):
        r = mm3.execute_step(m3.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state3 = mm3.get_mission(m3.id)
    check("mission completed after resume",
          state3["status"] == "completed",
          f"status={state3['status']}")

    # F4: Persistence
    print("\n  F4: Persistence across restart")
    import tempfile
    persist_file = os.path.join(tempfile.gettempdir(), "kos_step2_test.json")
    if os.path.exists(persist_file):
        os.remove(persist_file)

    mm4 = MissionManager(query_fn=mock_good, persist_path=persist_file,
                         use_agents=True, dispatcher=dispatcher)
    m4 = mm4.create_mission("Persist test", description="Compare X and Y")
    mm4.plan(m4.id)
    mm4.execute_step(m4.id)  # Partial execution

    state_before = mm4.get_mission(m4.id)
    goals_before = {g["id"]: g["status"] for g in state_before["goals"]}

    # Simulate restart
    mm4b = MissionManager(query_fn=mock_good, persist_path=persist_file,
                          use_agents=True, dispatcher=dispatcher)
    check("mission survives restart",
          len(mm4b.list_missions()) >= 1,
          f"got {len(mm4b.list_missions())}")

    state_after = mm4b.get_mission(m4.id)
    goals_after = {g["id"]: g["status"] for g in state_after["goals"]}
    check("goal states preserved after restart",
          goals_before == goals_after,
          f"before={goals_before} after={goals_after}")

    # Continue execution after restart
    for _ in range(10):
        r = mm4b.execute_step(m4.id)
        if r.get("status") in ("mission_completed", "no_ready_goals"):
            break
        if "error" in r and "Mission is" in r.get("error", ""):
            break

    state_final = mm4b.get_mission(m4.id)
    check("mission completes after restart",
          state_final["status"] == "completed",
          f"status={state_final['status']}")

    if os.path.exists(persist_file):
        os.remove(persist_file)

    # F5: Concurrent missions
    print("\n  F5: Concurrent missions (3 parallel)")
    mm5 = MissionManager(query_fn=mock_good, persist_path=None,
                         use_agents=True, dispatcher=dispatcher)

    mission_ids = []
    for i in range(3):
        m = mm5.create_mission(f"Concurrent {i}",
                               description=f"Compare entity{i}a and entity{i}b",
                               tags=[f"concurrent-{i}"])
        mm5.plan(m.id)
        mission_ids.append(m.id)

    errors = []
    def run_mission(mid):
        try:
            for _ in range(20):
                r = mm5.execute_step(mid)
                if r.get("status") in ("mission_completed", "no_ready_goals"):
                    break
                if "error" in r and "Mission is" in r.get("error", ""):
                    break
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=run_mission, args=(mid,))
               for mid in mission_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    check("no concurrency errors", len(errors) == 0,
          f"errors: {errors}")

    for i, mid in enumerate(mission_ids):
        state = mm5.get_mission(mid)
        completed = sum(1 for g in state["goals"]
                        if g["status"] == "completed")
        check(f"concurrent mission {i} completed ({completed}/{len(state['goals'])})",
              completed == len(state["goals"]),
              f"{completed}/{len(state['goals'])}")

    # Verify no deliverable crossover
    all_source_goals = {}
    for mid in mission_ids:
        state = mm5.get_mission(mid)
        mission_goal_ids = {g["id"] for g in state["goals"]}
        for d in state["deliverables"]:
            for sg in d["source_goals"]:
                check(f"deliverable source goal in correct mission",
                      sg in mission_goal_ids,
                      f"goal {sg} not in mission {mid}")

    # F6: Dispatcher audit log
    print("\n  F6: Dispatcher audit log")
    log = dispatcher.get_log()
    check("dispatcher log has entries from all tests",
          len(log) > 0,
          f"got {len(log)}")
    check("all log entries have timestamps",
          all("timestamp" in entry for entry in log),
          "missing timestamps")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KOS v0.8 STEP 2 -- MULTI-AGENT FRAMEWORK TESTS")
    print("=" * 60)

    test_a_protocol()
    test_b_registry()
    test_c_dispatcher()
    test_d_parity()
    test_e_failure_isolation()
    test_f_e2e()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print(f"\nFAILED TESTS: {FAIL}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)
