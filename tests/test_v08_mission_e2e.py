"""
KOS v0.8 Mission System -- End-to-End Tests

Tests the full mission lifecycle through the API:
  1. Create mission
  2. Auto-decompose into goal graph
  3. Execute goals in dependency order
  4. Verify deliverables generated
  5. Pause/resume/cancel
  6. List/filter missions
"""
import json
import urllib.request
import time

BASE = "http://localhost:8080"


def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


def test_missions_list_empty():
    """List missions on fresh server should return empty."""
    r, code = api("GET", "/api/missions")
    assert code == 200, f"Expected 200, got {code}"
    assert r["missions"] == [], f"Expected empty list, got {r['missions']}"
    print("  PASS: empty mission list")


def test_create_mission():
    """Create a simple mission."""
    r, code = api("POST", "/api/missions", {
        "name": "Test Mission",
        "description": "Compare Toronto and Montreal",
        "tags": ["test"]
    })
    assert code == 200, f"Expected 200, got {code}: {r}"
    m = r["mission"]
    assert m["name"] == "Test Mission"
    assert m["status"] == "planning"
    assert "id" in m
    print(f"  PASS: created mission {m['id'][:11]}")
    return m["id"]


def test_plan_comparison(mid):
    """Plan should auto-decompose comparison into 4 goals."""
    r, code = api("POST", f"/api/missions/{mid}/plan")
    assert code == 200, f"Expected 200, got {code}: {r}"
    m = r["mission"]
    goals = m["goals"]
    print(f"  Goals decomposed: {len(goals)}")
    for g in goals:
        print(f"    [{g['status']:9s}] {g['goal_type']:10s} | {g['description']}")

    assert len(goals) == 4, f"Expected 4 goals (comparison), got {len(goals)}"
    types = [g["goal_type"] for g in goals]
    assert types.count("retrieve") == 2, f"Expected 2 retrieve goals"
    assert types.count("compare") == 1, f"Expected 1 compare goal"
    assert types.count("synthesize") == 1, f"Expected 1 synthesize goal"
    print("  PASS: 4-goal comparison decomposition")
    return m


def test_execute_all(mid):
    """Execute goals one at a time to avoid timeout."""
    max_steps = 10
    for step_num in range(1, max_steps + 1):
        print(f"  Step {step_num}:")
        r, code = api("POST", f"/api/missions/{mid}/execute")
        assert code == 200, f"Expected 200, got {code}: {r}"

        result = r.get("result", r)
        gid = result.get("goal_id", "?")
        if gid != "?":
            gid = gid[:11]
        status = result.get("status", "?")
        score = result.get("score", 0)
        progress = result.get("mission_progress", 0)
        print(f"    [{status:9s}] score={score:.3f} progress={progress:.1%} | {gid}")

        if status in ("mission_completed", "no_ready_goals", "mission_failed"):
            break
        if "error" in result:
            err = result.get("error", "")
            if err:
                print(f"    Stopped: {err}")
                break
        # Stop when all goals done (progress == 100%)
        if progress >= 1.0:
            print(f"    All goals complete (progress=100%)")
            break

    # Get final state for deliverables
    r2, _ = api("GET", f"/api/missions/{mid}")
    m = r2["mission"]
    deliverables = m.get("deliverables", [])
    print(f"  Deliverables: {len(deliverables)}")
    for d in deliverables:
        print(f"    [{d.get('dtype', '?')}] {d.get('title', '?')}")

    return m


def test_get_mission(mid):
    """Fetch full mission state."""
    r, code = api("GET", f"/api/missions/{mid}")
    assert code == 200, f"Expected 200, got {code}: {r}"
    m = r["mission"]
    print(f"  Status: {m['status']}")
    print(f"  Goals: {len(m['goals'])}")
    for g in m["goals"]:
        score = g.get("result", {}).get("score", 0) if g.get("result") else 0
        print(f"    [{g['status']:9s}] score={score:.3f} | {g['description']}")
    print(f"  Deliverables: {len(m['deliverables'])}")
    return m


def test_pause_resume(mid):
    """Create + pause + resume cycle."""
    r, code = api("POST", "/api/missions", {
        "name": "Pause Test",
        "description": "Analyze backpropagation",
        "tags": ["test"]
    })
    mid2 = r["mission"]["id"]

    # Plan it
    api("POST", f"/api/missions/{mid2}/plan")

    # Pause
    r, code = api("POST", f"/api/missions/{mid2}/pause")
    assert code == 200, f"Pause failed: {r}"
    r2, _ = api("GET", f"/api/missions/{mid2}")
    assert r2["mission"]["status"] == "paused", "Expected paused"
    print("  PASS: pause works")

    # Resume
    r, code = api("POST", f"/api/missions/{mid2}/resume")
    assert code == 200, f"Resume failed: {r}"
    r2, _ = api("GET", f"/api/missions/{mid2}")
    assert r2["mission"]["status"] == "active", "Expected active"
    print("  PASS: resume works")

    # Cancel
    r, code = api("POST", f"/api/missions/{mid2}/cancel")
    assert code == 200, f"Cancel failed: {r}"
    r2, _ = api("GET", f"/api/missions/{mid2}")
    assert r2["mission"]["status"] == "cancelled", "Expected cancelled"
    print("  PASS: cancel works")

    return mid2


def test_list_filter():
    """List missions with status filter."""
    r, code = api("GET", "/api/missions")
    total = len(r["missions"])
    print(f"  Total missions: {total}")

    r2, _ = api("GET", "/api/missions?status=cancelled")
    cancelled = len(r2["missions"])
    print(f"  Cancelled: {cancelled}")
    assert cancelled >= 1, "Should have at least 1 cancelled mission"
    print("  PASS: status filter works")


def test_add_goal(mid):
    """Add a manual goal to existing mission."""
    r, code = api("POST", "/api/missions", {
        "name": "Goal Add Test",
        "description": "Test adding goals manually"
    })
    mid3 = r["mission"]["id"]
    api("POST", f"/api/missions/{mid3}/plan")

    # Add a custom goal
    r, code = api("POST", f"/api/missions/{mid3}/goals", {
        "description": "Custom: check weather in Toronto",
        "goal_type": "retrieve",
        "query": "Weather in Toronto"
    })
    assert code == 200, f"Add goal failed: {r}"
    print("  PASS: custom goal added")
    return mid3


# ── Run all tests ──────────────────────────────────────────────
print("=" * 72)
print("KOS v0.8 MISSION E2E TEST")
print("=" * 72)

print("\n1. List (empty):")
test_missions_list_empty()

print("\n2. Create mission:")
mission_id = test_create_mission()

print("\n3. Plan (auto-decompose):")
test_plan_comparison(mission_id)

print("\n4. Execute all (step by step):")
final_state = test_execute_all(mission_id)

print("\n5. Get mission state:")
final_state = test_get_mission(mission_id)

print("\n6. Pause / Resume / Cancel:")
test_pause_resume(mission_id)

print("\n7. List + filter:")
test_list_filter()

print("\n8. Add custom goal:")
test_add_goal(mission_id)

# ── Summary ────────────────────────────────────────────────────
completed_goals = sum(
    1 for g in final_state["goals"] if g["status"] == "completed"
)
total_goals = len(final_state["goals"])
has_deliverables = len(final_state["deliverables"]) > 0

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  Mission status:  {final_state['status']}")
print(f"  Goals completed: {completed_goals}/{total_goals}")
print(f"  Deliverables:    {len(final_state['deliverables'])}")
print(f"  Has output:      {'YES' if has_deliverables else 'NO'}")
print("=" * 72)
