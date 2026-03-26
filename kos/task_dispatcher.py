"""
KOS v0.8 Step 2 -- Task Dispatcher

Single authority for agent execution. The dispatcher:
  - Receives an AgentTask
  - Finds the right agent via the registry
  - Executes and returns the AgentResult
  - Never lets exceptions escape (wraps in FAILED result)

No agent-to-agent communication. All routing is centralized here.
"""
from __future__ import annotations

import time

from kos.agent_protocol import AgentTask, AgentResult, AgentStatus
from kos.agent_registry import AgentRegistry


class TaskDispatcher:
    """Central dispatch authority. Routes tasks to agents."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        self.dispatch_log: list = []  # Audit trail of dispatches

    def dispatch(self, task: AgentTask) -> AgentResult:
        """Dispatch a task to the matching agent.

        Guarantees:
          - Always returns an AgentResult (never raises)
          - Unhandled exceptions become FAILED results
          - Dispatch is logged for audit
        """
        t0 = time.perf_counter()

        try:
            agent = self.registry.match(task.goal_type)
        except KeyError as e:
            latency = int((time.perf_counter() - t0) * 1000)
            result = AgentResult(
                task_id=task.task_id,
                mission_id=task.mission_id,
                goal_id=task.goal_id,
                agent_name="none",
                status=AgentStatus.FAILED,
                error=f"No agent for goal_type={task.goal_type}",
                latency_ms=latency,
                retryable=False,
            )
            self._log(task, result)
            return result

        try:
            result = agent.execute(task)
        except Exception as e:
            # Agent violated protocol (should not raise). Wrap the error.
            latency = int((time.perf_counter() - t0) * 1000)
            result = AgentResult(
                task_id=task.task_id,
                mission_id=task.mission_id,
                goal_id=task.goal_id,
                agent_name=agent.name,
                status=AgentStatus.FAILED,
                error=f"Agent {agent.name} raised: {e}",
                latency_ms=latency,
                retryable=True,
            )

        self._log(task, result)
        return result

    def _log(self, task: AgentTask, result: AgentResult) -> None:
        """Record dispatch for audit trail."""
        self.dispatch_log.append({
            "task_id": task.task_id,
            "goal_type": task.goal_type,
            "agent": result.agent_name,
            "status": result.status.value,
            "score": result.score,
            "latency_ms": result.latency_ms,
            "error": result.error,
            "timestamp": time.time(),
        })

    def get_log(self) -> list:
        """Return the dispatch audit log."""
        return list(self.dispatch_log)

    def clear_log(self) -> None:
        """Clear the audit log."""
        self.dispatch_log.clear()
