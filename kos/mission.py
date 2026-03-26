"""
KOS v0.8 -- Mission Manager (Adaptive Mission System)

The core of v0.8. Turns single-query agents into sustained mission execution.

A mission is a persistent, multi-step goal like:
  - "Monitor this topic for 30 days and alert on changes"
  - "Build a comparison report across 5 entities"
  - "Track evolving evidence and produce staged deliverables"

Architecture:
    User Goal -> Mission Manager -> Goal Graph -> Agent Dispatch -> Verify -> Deliver

Key concepts:
  - Mission: top-level container with goals, checkpoints, deliverables
  - Goal: a single objective with dependencies, assigned agent, and completion criteria
  - Checkpoint: a progress gate that must be reached by a deadline
  - MissionManager: orchestrates planning, execution, monitoring, and completion
"""

import uuid
import time
import threading
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ── Enums ───────────────────────────────────────────────────────

class MissionStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalStatus(Enum):
    PENDING = "pending"
    READY = "ready"       # All dependencies met, can execute
    ACTIVE = "active"     # Currently being worked on
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoalType(Enum):
    RETRIEVE = "retrieve"         # Gather information on a topic
    COMPARE = "compare"           # Compare two or more entities
    ANALYZE = "analyze"           # Deep analysis of a topic
    MONITOR = "monitor"           # Watch for changes over time
    SYNTHESIZE = "synthesize"     # Produce a deliverable from evidence
    VERIFY = "verify"             # Fact-check or validate claims
    FORAGE = "forage"             # Acquire missing knowledge
    CUSTOM = "custom"             # User-defined goal


class CheckpointStatus(Enum):
    PENDING = "pending"
    REACHED = "reached"
    MISSED = "missed"


class DeliverableType(Enum):
    ANSWER = "answer"             # Quick answer
    SUMMARY = "summary"           # Executive summary
    REPORT = "report"             # Detailed report
    COMPARISON = "comparison"     # Side-by-side comparison
    MEMO = "memo"                 # Decision memo
    EVIDENCE_MATRIX = "evidence_matrix"
    CHANGE_LOG = "change_log"
    ACTION_CHECKLIST = "action_checklist"


# ── Data Models ─────────────────────────────────────────────────

@dataclass
class Goal:
    """A single objective within a mission."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    goal_type: GoalType = GoalType.RETRIEVE
    status: GoalStatus = GoalStatus.PENDING
    dependencies: list = field(default_factory=list)  # List of goal IDs
    priority: int = 1             # 1 = highest
    assigned_agent: str = None    # Agent type or ID
    query: str = ""               # The actual query to execute
    result: dict = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    completed_at: float = None
    error: str = None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "assigned_agent": self.assigned_agent,
            "query": self.query,
            "result": self.result,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


@dataclass
class Checkpoint:
    """A progress gate in a mission."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    status: CheckpointStatus = CheckpointStatus.PENDING
    required_goals: list = field(default_factory=list)  # Goal IDs that must complete
    deadline: float = None         # Unix timestamp
    reached_at: float = None
    condition: str = ""            # Human-readable condition

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "required_goals": self.required_goals,
            "deadline": self.deadline,
            "reached_at": self.reached_at,
            "condition": self.condition,
        }


@dataclass
class Deliverable:
    """An output artifact from a mission."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    deliverable_type: DeliverableType = DeliverableType.ANSWER
    title: str = ""
    content: str = ""
    source_goals: list = field(default_factory=list)  # Goal IDs that produced this
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.deliverable_type.value,
            "title": self.title,
            "content": self.content,
            "source_goals": self.source_goals,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class Mission:
    """A persistent, multi-step goal."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    status: MissionStatus = MissionStatus.PLANNING
    goals: list = field(default_factory=list)           # List of Goal objects
    checkpoints: list = field(default_factory=list)     # List of Checkpoint objects
    deliverables: list = field(default_factory=list)    # List of Deliverable objects
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = None
    deadline: float = None
    tags: list = field(default_factory=list)
    progress: float = 0.0         # 0.0 - 1.0
    error_log: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "goals": [g.to_dict() for g in self.goals],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "deliverables": [d.to_dict() for d in self.deliverables],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "deadline": self.deadline,
            "tags": self.tags,
            "progress": round(self.progress, 3),
            "error_log": self.error_log[-10:],  # Last 10 errors
            "goal_summary": self._goal_summary(),
        }

    def _goal_summary(self):
        by_status = {}
        for g in self.goals:
            s = g.status.value
            by_status[s] = by_status.get(s, 0) + 1
        return by_status

    def update_progress(self):
        """Recompute progress from goal completion.

        Auto-transitions to COMPLETED when all goals are done,
        or FAILED when all remaining goals are blocked.
        """
        if not self.goals:
            self.progress = 0.0
            return
        completed = sum(1 for g in self.goals
                        if g.status == GoalStatus.COMPLETED)
        self.progress = completed / len(self.goals)
        self.updated_at = time.time()

        # Auto-complete: all goals finished (completed or skipped)
        if self.status == MissionStatus.ACTIVE:
            terminal = sum(1 for g in self.goals
                           if g.status in (GoalStatus.COMPLETED,
                                           GoalStatus.SKIPPED,
                                           GoalStatus.FAILED))
            if terminal == len(self.goals):
                if completed > 0:
                    self.status = MissionStatus.COMPLETED
                    self.completed_at = time.time()
                else:
                    self.status = MissionStatus.FAILED
                    self.completed_at = time.time()


# ── Mission Manager ─────────────────────────────────────────────

class MissionManager:
    """
    Orchestrates mission lifecycle: plan, execute, monitor, complete.

    Thread-safe. Persists missions to disk.

    Usage:
        mm = MissionManager(query_fn=pipeline.query)
        mission = mm.create_mission("Compare Toronto and Montreal in depth")
        mm.plan(mission.id)
        mm.execute_step(mission.id)  # Run next ready goal
        mm.get_status(mission.id)
    """

    def __init__(self, query_fn=None, persist_path=".cache/missions.json",
                 use_agents=False, dispatcher=None):
        """
        Args:
            query_fn: callable(prompt) -> dict (the pipeline query function)
            persist_path: where to save mission state
            use_agents: if True, route execution through agent dispatcher
            dispatcher: TaskDispatcher instance (required if use_agents=True)
        """
        self._missions = {}  # id -> Mission
        self._lock = threading.Lock()
        self._query_fn = query_fn
        self._persist_path = persist_path
        self.use_agents = use_agents
        self._dispatcher = dispatcher
        self._load()

    # ── Creation ────────────────────────────────────────────────

    def create_mission(self, name, description="", tags=None,
                       deadline=None) -> Mission:
        """Create a new mission in PLANNING status."""
        mission = Mission(
            name=name,
            description=description or name,
            tags=tags or [],
            deadline=deadline,
        )
        with self._lock:
            self._missions[mission.id] = mission
        self._save()
        return mission

    # ── Planning ────────────────────────────────────────────────

    def plan(self, mission_id, goals=None) -> Mission:
        """
        Decompose a mission into goals. If goals not provided,
        auto-decompose from the mission description.

        Args:
            mission_id: mission to plan
            goals: optional list of goal dicts with keys:
                   description, goal_type, query, dependencies, priority

        Returns:
            Updated mission
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if not mission:
                raise ValueError(f"Mission {mission_id} not found")

            if goals:
                # User-provided goal list
                for g_spec in goals:
                    goal = Goal(
                        description=g_spec.get("description", ""),
                        goal_type=GoalType(g_spec.get("goal_type", "retrieve")),
                        query=g_spec.get("query", g_spec.get("description", "")),
                        dependencies=g_spec.get("dependencies", []),
                        priority=g_spec.get("priority", 1),
                        assigned_agent=g_spec.get("agent"),
                    )
                    mission.goals.append(goal)
            else:
                # Auto-decompose from description
                mission.goals = self._auto_decompose(mission)

            # Mark goals with no dependencies as READY
            for goal in mission.goals:
                if not goal.dependencies:
                    goal.status = GoalStatus.READY

            mission.status = MissionStatus.ACTIVE
            mission.updated_at = time.time()

        self._save()
        return mission

    def _auto_decompose(self, mission) -> list:
        """
        Heuristic decomposition of a mission description into goals.

        Patterns recognized:
          - "compare X and Y" -> retrieve X, retrieve Y, compare, synthesize
          - "monitor X" -> retrieve X, set up monitoring goal
          - "analyze X" -> retrieve X, analyze, synthesize report
          - default -> retrieve, synthesize
        """
        desc = mission.description.lower()
        goals = []

        # Comparison mission
        import re
        comp_m = re.search(
            r'(?:compare|comparison\s+of)\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)(?:\s|$|\.)',
            desc, re.I)
        if comp_m:
            entity_a = comp_m.group(1).strip()
            entity_b = comp_m.group(2).strip()

            g1 = Goal(description=f"Retrieve information about {entity_a}",
                      goal_type=GoalType.RETRIEVE,
                      query=f"What is {entity_a}?", priority=1)
            g2 = Goal(description=f"Retrieve information about {entity_b}",
                      goal_type=GoalType.RETRIEVE,
                      query=f"What is {entity_b}?", priority=1)
            g3 = Goal(description=f"Compare {entity_a} and {entity_b}",
                      goal_type=GoalType.COMPARE,
                      query=f"Compare {entity_a} and {entity_b}",
                      dependencies=[g1.id, g2.id], priority=2)
            g4 = Goal(description=f"Synthesize comparison report",
                      goal_type=GoalType.SYNTHESIZE,
                      query=f"Summarize comparison of {entity_a} vs {entity_b}",
                      dependencies=[g3.id], priority=3)
            return [g1, g2, g3, g4]

        # Monitor mission
        if 'monitor' in desc or 'track' in desc or 'watch' in desc:
            topic_m = re.search(
                r'(?:monitor|track|watch)\s+(.+?)(?:\s+for|\s+over|$)', desc)
            topic = topic_m.group(1).strip() if topic_m else mission.name

            g1 = Goal(description=f"Retrieve baseline for {topic}",
                      goal_type=GoalType.RETRIEVE,
                      query=f"What is {topic}?", priority=1)
            g2 = Goal(description=f"Monitor {topic} for changes",
                      goal_type=GoalType.MONITOR,
                      query=f"Latest updates on {topic}",
                      dependencies=[g1.id], priority=2)
            g3 = Goal(description=f"Produce monitoring summary",
                      goal_type=GoalType.SYNTHESIZE,
                      query=f"Summary of {topic} changes",
                      dependencies=[g2.id], priority=3)
            return [g1, g2, g3]

        # Analyze mission
        if 'analyze' in desc or 'research' in desc or 'investigate' in desc:
            topic_m = re.search(
                r'(?:analyze|research|investigate)\s+(.+?)(?:\s+and|$)', desc)
            topic = topic_m.group(1).strip() if topic_m else mission.name

            g1 = Goal(description=f"Retrieve data on {topic}",
                      goal_type=GoalType.RETRIEVE,
                      query=f"Tell me about {topic}", priority=1)
            g2 = Goal(description=f"Deep analysis of {topic}",
                      goal_type=GoalType.ANALYZE,
                      query=f"Explain {topic} in detail",
                      dependencies=[g1.id], priority=2)
            g3 = Goal(description=f"Verify findings on {topic}",
                      goal_type=GoalType.VERIFY,
                      query=f"Verify: {topic}",
                      dependencies=[g2.id], priority=3)
            g4 = Goal(description=f"Produce analysis report",
                      goal_type=GoalType.SYNTHESIZE,
                      query=f"Analysis report on {topic}",
                      dependencies=[g3.id], priority=4)
            return [g1, g2, g3, g4]

        # Default: simple retrieve + synthesize
        g1 = Goal(description=f"Retrieve: {mission.name}",
                  goal_type=GoalType.RETRIEVE,
                  query=mission.description, priority=1)
        g2 = Goal(description=f"Synthesize answer",
                  goal_type=GoalType.SYNTHESIZE,
                  query=mission.description,
                  dependencies=[g1.id], priority=2)
        return [g1, g2]

    # ── Execution ───────────────────────────────────────────────

    def execute_step(self, mission_id, verbose=False) -> dict:
        """
        Execute the next ready goal in priority order.

        Returns dict with:
            goal_id, status, result, mission_progress
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if not mission:
                raise ValueError(f"Mission {mission_id} not found")
            if mission.status != MissionStatus.ACTIVE:
                return {"error": f"Mission is {mission.status.value}",
                        "mission_progress": mission.progress}

            # Update goal readiness
            self._update_goal_readiness(mission)

            # Find next ready goal (highest priority = lowest number)
            ready = [g for g in mission.goals
                     if g.status == GoalStatus.READY]
            if not ready:
                # Check if all done
                if all(g.status in (GoalStatus.COMPLETED, GoalStatus.SKIPPED)
                       for g in mission.goals):
                    mission.status = MissionStatus.COMPLETED
                    mission.completed_at = time.time()
                    mission.update_progress()
                    self._save()
                    return {"status": "mission_completed",
                            "mission_progress": mission.progress}
                # Check if stuck (all remaining goals have failed deps)
                active = [g for g in mission.goals
                          if g.status == GoalStatus.ACTIVE]
                if not active:
                    return {"status": "no_ready_goals",
                            "mission_progress": mission.progress}
                return {"status": "goals_in_progress",
                        "mission_progress": mission.progress}

            ready.sort(key=lambda g: g.priority)
            goal = ready[0]

        # Execute outside the lock (can be slow)
        return self._execute_goal(mission, goal, verbose)

    def execute_all(self, mission_id, verbose=False) -> dict:
        """
        Execute all goals in dependency order until mission completes or stalls.

        Returns final mission status.
        """
        results = []
        max_steps = 50  # Safety cap
        for _ in range(max_steps):
            result = self.execute_step(mission_id, verbose=verbose)
            results.append(result)

            status = result.get("status", "")
            if status in ("mission_completed", "no_ready_goals", "mission_failed"):
                break
            if "error" in result and "not found" in result.get("error", ""):
                break

        with self._lock:
            mission = self._missions.get(mission_id)
            return {
                "mission_id": mission_id,
                "status": mission.status.value if mission else "not_found",
                "progress": mission.progress if mission else 0,
                "steps_executed": len(results),
                "results": results,
                "deliverables": [d.to_dict() for d in mission.deliverables]
                                if mission else [],
            }

    def _build_goal_payload(self, mission, goal) -> dict:
        """Build agent task payload based on goal type."""
        payload = {"query": goal.query}

        if goal.goal_type == GoalType.COMPARE:
            # Extract entities from query (e.g., "Compare toronto and montreal")
            import re
            m = re.search(
                r'(?:compare|comparison\s+of)\s+(.+?)\s+(?:and|vs\.?|versus|with)\s+(.+?)$',
                goal.query, re.I)
            if m:
                payload["left"] = m.group(1).strip()
                payload["right"] = m.group(2).strip()

        elif goal.goal_type == GoalType.SYNTHESIZE:
            # Collect outputs from completed dependency goals
            inputs = []
            completed_goals = {g.id: g for g in mission.goals
                               if g.status == GoalStatus.COMPLETED}
            for dep_id in goal.dependencies:
                dep_goal = completed_goals.get(dep_id)
                if dep_goal and dep_goal.result:
                    inputs.append({
                        "goal_id": dep_goal.id,
                        "output": dep_goal.result,
                    })
            if inputs:
                payload["inputs"] = inputs
                payload["mode"] = "comparison_summary" if any(
                    g.goal_type == GoalType.COMPARE
                    for g in mission.goals
                    if g.status == GoalStatus.COMPLETED
                ) else "summary"

        return payload

    def _apply_agent_result(self, mission, goal, agent_result) -> tuple:
        """Apply an AgentResult to goal state. Returns (score, answer, trust)."""
        from kos.agent_protocol import AgentStatus

        output = agent_result.output or {}
        score = agent_result.score
        answer = output.get("answer", "")
        trust = agent_result.trust_label

        goal.result = {
            "answer": answer,
            "score": score,
            "trust": trust,
            "latency_ms": agent_result.latency_ms,
            "source": agent_result.agent_name,
            "agent": agent_result.agent_name,
        }

        if agent_result.status == AgentStatus.COMPLETE and score >= 0.46 \
                and "don't have" not in answer.lower():
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()

            # Auto-generate deliverable
            if goal.goal_type == GoalType.SYNTHESIZE:
                mission.deliverables.append(Deliverable(
                    deliverable_type=DeliverableType.SUMMARY,
                    title=goal.description, content=answer,
                    source_goals=[goal.id],
                    metadata={"score": score, "trust": trust},
                ))
            elif goal.goal_type == GoalType.COMPARE:
                mission.deliverables.append(Deliverable(
                    deliverable_type=DeliverableType.COMPARISON,
                    title=goal.description, content=answer,
                    source_goals=[goal.id],
                    metadata={"score": score, "trust": trust},
                ))

        elif agent_result.status == AgentStatus.FAILED and not agent_result.retryable:
            goal.status = GoalStatus.FAILED
            goal.error = agent_result.error or "Agent failed (non-retryable)"
            mission.error_log.append(f"Goal {goal.id} failed: {goal.error}")

        elif goal.attempts >= goal.max_attempts:
            goal.status = GoalStatus.FAILED
            goal.error = f"Failed after {goal.attempts} attempts (score={score:.3f})"
            mission.error_log.append(f"Goal {goal.id} failed: {goal.error}")

        else:
            goal.status = GoalStatus.READY
            goal.error = f"Attempt {goal.attempts} scored {score:.3f}, retrying"

        return score, answer, trust

    def _execute_goal(self, mission, goal, verbose=False) -> dict:
        """Execute a single goal using the query pipeline or agent dispatcher."""
        with self._lock:
            goal.status = GoalStatus.ACTIVE
            goal.attempts += 1

        t0 = time.time()

        try:
            if verbose:
                print(f"[MISSION {mission.id}] Executing goal {goal.id}: "
                      f"{goal.description}")

            # ── Agent path ──
            if self.use_agents and self._dispatcher:
                from kos.agent_protocol import AgentTask
                task = AgentTask(
                    task_id=f"{mission.id}:{goal.id}:attempt{goal.attempts}",
                    mission_id=mission.id,
                    goal_id=goal.id,
                    goal_type=goal.goal_type.value,
                    payload=self._build_goal_payload(mission, goal),
                    dependencies=goal.dependencies,
                    attempt=goal.attempts,
                )
                agent_result = self._dispatcher.dispatch(task)

                with self._lock:
                    score, answer, trust = self._apply_agent_result(
                        mission, goal, agent_result)
                    mission.update_progress()
                    self._update_checkpoints(mission)

                self._save()
                return {
                    "goal_id": goal.id,
                    "status": goal.status.value,
                    "score": score,
                    "trust": trust,
                    "agent": agent_result.agent_name,
                    "mission_progress": mission.progress,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                }

            # ── Legacy path ──
            if not self._query_fn:
                raise RuntimeError("No query function configured")

            # Run the query
            result = self._query_fn(goal.query)

            # Check result quality
            score = result.get("relevance_score", 0)
            answer = result.get("answer", "")
            trust = result.get("trust_label", "unverified")

            with self._lock:
                goal.result = {
                    "answer": answer,
                    "score": score,
                    "trust": trust,
                    "latency_ms": result.get("latency_ms", 0),
                    "source": result.get("source", "unknown"),
                }

                if score >= 0.46 and "don't have" not in answer.lower():
                    goal.status = GoalStatus.COMPLETED
                    goal.completed_at = time.time()

                    # Auto-generate deliverable for synthesis goals
                    if goal.goal_type == GoalType.SYNTHESIZE:
                        deliverable = Deliverable(
                            deliverable_type=DeliverableType.SUMMARY,
                            title=goal.description,
                            content=answer,
                            source_goals=[goal.id],
                            metadata={"score": score, "trust": trust},
                        )
                        mission.deliverables.append(deliverable)
                    elif goal.goal_type == GoalType.COMPARE:
                        deliverable = Deliverable(
                            deliverable_type=DeliverableType.COMPARISON,
                            title=goal.description,
                            content=answer,
                            source_goals=[goal.id],
                            metadata={"score": score, "trust": trust},
                        )
                        mission.deliverables.append(deliverable)

                elif goal.attempts >= goal.max_attempts:
                    goal.status = GoalStatus.FAILED
                    goal.error = f"Failed after {goal.attempts} attempts (score={score:.3f})"
                    mission.error_log.append(
                        f"Goal {goal.id} failed: {goal.error}")
                else:
                    # Retry: set back to READY
                    goal.status = GoalStatus.READY
                    goal.error = f"Attempt {goal.attempts} scored {score:.3f}, retrying"

                mission.update_progress()
                self._update_checkpoints(mission)

            self._save()

            return {
                "goal_id": goal.id,
                "status": goal.status.value,
                "score": score,
                "trust": trust,
                "mission_progress": mission.progress,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }

        except Exception as e:
            with self._lock:
                goal.error = str(e)
                if goal.attempts >= goal.max_attempts:
                    goal.status = GoalStatus.FAILED
                    mission.error_log.append(
                        f"Goal {goal.id} exception: {e}")
                else:
                    goal.status = GoalStatus.READY

                mission.update_progress()

            self._save()
            return {
                "goal_id": goal.id,
                "status": "error",
                "error": str(e),
                "mission_progress": mission.progress,
            }

    def _update_goal_readiness(self, mission):
        """Mark goals as READY if all their dependencies are met.

        Cascade: if a dependency is FAILED or SKIPPED, dependents are SKIPPED.
        This runs repeatedly until no more changes (handles multi-level cascades).
        """
        changed = True
        while changed:
            changed = False
            completed_ids = {g.id for g in mission.goals
                             if g.status == GoalStatus.COMPLETED}
            blocked_ids = {g.id for g in mission.goals
                           if g.status in (GoalStatus.FAILED, GoalStatus.SKIPPED)}

            for goal in mission.goals:
                if goal.status != GoalStatus.PENDING:
                    continue
                dep_set = set(goal.dependencies)
                if dep_set <= completed_ids:
                    goal.status = GoalStatus.READY
                    changed = True
                elif dep_set & blocked_ids:
                    # A dependency failed or was skipped — cascade skip
                    goal.status = GoalStatus.SKIPPED
                    goal.error = "Dependency failed or skipped"
                    changed = True

    def _update_checkpoints(self, mission):
        """Check if any checkpoints have been reached or missed."""
        completed_ids = {g.id for g in mission.goals
                         if g.status == GoalStatus.COMPLETED}
        now = time.time()

        for cp in mission.checkpoints:
            if cp.status != CheckpointStatus.PENDING:
                continue
            required = set(cp.required_goals)
            if required <= completed_ids:
                cp.status = CheckpointStatus.REACHED
                cp.reached_at = now
            elif cp.deadline and now > cp.deadline:
                cp.status = CheckpointStatus.MISSED

    # ── Query ───────────────────────────────────────────────────

    def get_mission(self, mission_id) -> dict:
        """Get full mission state."""
        with self._lock:
            mission = self._missions.get(mission_id)
            if not mission:
                raise ValueError(f"Mission {mission_id} not found")
            return mission.to_dict()

    def list_missions(self, status=None) -> list:
        """List all missions, optionally filtered by status."""
        with self._lock:
            missions = list(self._missions.values())
            if status:
                try:
                    s = MissionStatus(status)
                    missions = [m for m in missions if m.status == s]
                except ValueError:
                    pass
            return [m.to_dict() for m in missions]

    # ── Control ─────────────────────────────────────────────────

    def pause(self, mission_id):
        with self._lock:
            m = self._missions.get(mission_id)
            if m and m.status == MissionStatus.ACTIVE:
                m.status = MissionStatus.PAUSED
                m.updated_at = time.time()
        self._save()

    def resume(self, mission_id):
        with self._lock:
            m = self._missions.get(mission_id)
            if m and m.status == MissionStatus.PAUSED:
                m.status = MissionStatus.ACTIVE
                m.updated_at = time.time()
        self._save()

    def cancel(self, mission_id):
        with self._lock:
            m = self._missions.get(mission_id)
            if m and m.status in (MissionStatus.PLANNING,
                                   MissionStatus.ACTIVE,
                                   MissionStatus.PAUSED):
                m.status = MissionStatus.CANCELLED
                m.updated_at = time.time()
        self._save()

    def add_goal(self, mission_id, description, goal_type="retrieve",
                 query=None, dependencies=None, priority=1) -> dict:
        """Add a goal to an existing mission."""
        with self._lock:
            m = self._missions.get(mission_id)
            if not m:
                raise ValueError(f"Mission {mission_id} not found")
            goal = Goal(
                description=description,
                goal_type=GoalType(goal_type),
                query=query or description,
                dependencies=dependencies or [],
                priority=priority,
            )
            if not goal.dependencies:
                goal.status = GoalStatus.READY
            m.goals.append(goal)
            m.updated_at = time.time()
        self._save()
        return goal.to_dict()

    def add_checkpoint(self, mission_id, description, required_goals=None,
                       deadline=None, condition="") -> dict:
        """Add a checkpoint to a mission."""
        with self._lock:
            m = self._missions.get(mission_id)
            if not m:
                raise ValueError(f"Mission {mission_id} not found")
            cp = Checkpoint(
                description=description,
                required_goals=required_goals or [],
                deadline=deadline,
                condition=condition,
            )
            m.checkpoints.append(cp)
            m.updated_at = time.time()
        self._save()
        return cp.to_dict()

    # ── Persistence ─────────────────────────────────────────────

    def _save(self):
        """Persist missions to JSON."""
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with self._lock:
                data = {mid: m.to_dict()
                        for mid, m in self._missions.items()}
            with open(self._persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        """Load missions from disk."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for mid, mdict in data.items():
                mission = Mission(
                    id=mdict.get("id", mid),
                    name=mdict.get("name", ""),
                    description=mdict.get("description", ""),
                    status=MissionStatus(mdict.get("status", "planning")),
                    created_at=mdict.get("created_at", 0),
                    updated_at=mdict.get("updated_at", 0),
                    completed_at=mdict.get("completed_at"),
                    deadline=mdict.get("deadline"),
                    tags=mdict.get("tags", []),
                    progress=mdict.get("progress", 0),
                    error_log=mdict.get("error_log", []),
                )
                # Restore goals
                for gd in mdict.get("goals", []):
                    goal = Goal(
                        id=gd.get("id", ""),
                        description=gd.get("description", ""),
                        goal_type=GoalType(gd.get("goal_type", "retrieve")),
                        status=GoalStatus(gd.get("status", "pending")),
                        dependencies=gd.get("dependencies", []),
                        priority=gd.get("priority", 1),
                        assigned_agent=gd.get("assigned_agent"),
                        query=gd.get("query", ""),
                        result=gd.get("result", {}),
                        attempts=gd.get("attempts", 0),
                        max_attempts=gd.get("max_attempts", 3),
                        created_at=gd.get("created_at", 0),
                        completed_at=gd.get("completed_at"),
                        error=gd.get("error"),
                    )
                    mission.goals.append(goal)
                # Restore checkpoints
                for cd in mdict.get("checkpoints", []):
                    cp = Checkpoint(
                        id=cd.get("id", ""),
                        description=cd.get("description", ""),
                        status=CheckpointStatus(cd.get("status", "pending")),
                        required_goals=cd.get("required_goals", []),
                        deadline=cd.get("deadline"),
                        reached_at=cd.get("reached_at"),
                        condition=cd.get("condition", ""),
                    )
                    mission.checkpoints.append(cp)
                # Restore deliverables
                for dd in mdict.get("deliverables", []):
                    d = Deliverable(
                        id=dd.get("id", ""),
                        deliverable_type=DeliverableType(dd.get("type", "answer")),
                        title=dd.get("title", ""),
                        content=dd.get("content", ""),
                        source_goals=dd.get("source_goals", []),
                        created_at=dd.get("created_at", 0),
                        metadata=dd.get("metadata", {}),
                    )
                    mission.deliverables.append(d)

                self._missions[mission.id] = mission
        except Exception:
            pass

    def __len__(self):
        with self._lock:
            return len(self._missions)
