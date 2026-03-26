"""
KOS v0.8 Step 2 -- Comparison Agent

Wraps the query pipeline for comparison goal types.
Accepts AgentTask with payload {"query": "...", "left": "...", "right": "..."}.
Falls back to query-based comparison if no dedicated engine available.
"""
from __future__ import annotations

import time

from kos.agent_protocol import AgentResult, AgentStatus, AgentTask, AgentEvidence
from kos.agents.base_agent import BaseAgent


class ComparisonAgent(BaseAgent):
    """Agent for comparison queries."""

    name = "comparison_agent"
    capabilities = ["compare", "comparison"]

    def __init__(self, query_fn):
        """
        Args:
            query_fn: callable(prompt) -> dict with pipeline response
        """
        self._query_fn = query_fn

    def execute(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        try:
            # Build comparison query from payload
            query = task.payload.get("query", "")
            left = task.payload.get("left", "")
            right = task.payload.get("right", "")

            if not query and left and right:
                query = f"Compare {left} and {right}"
            if not query:
                return AgentResult(
                    task_id=task.task_id,
                    mission_id=task.mission_id,
                    goal_id=task.goal_id,
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    error="No query or entities in payload",
                    latency_ms=0,
                    retryable=False,
                )

            response = self._query_fn(query)
            latency_ms = int((time.perf_counter() - started) * 1000)

            score = float(response.get("relevance_score", 0.0))
            answer = response.get("answer", "")
            trust = response.get("trust_label", "unverified")

            evidence = []
            if answer and score > 0:
                evidence.append(AgentEvidence(
                    source=response.get("source", "pipeline"),
                    content=answer[:500],
                    score=score,
                    metadata={
                        "trust": trust,
                        "entities": [left, right] if left else [],
                    },
                ))

            return AgentResult(
                task_id=task.task_id,
                mission_id=task.mission_id,
                goal_id=task.goal_id,
                agent_name=self.name,
                status=AgentStatus.COMPLETE,
                output=response,
                score=score,
                confidence=score,
                trust_label=trust,
                latency_ms=latency_ms,
                evidence=evidence,
                metadata={"entities": [left, right] if left else []},
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return AgentResult(
                task_id=task.task_id,
                mission_id=task.mission_id,
                goal_id=task.goal_id,
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error=str(e),
                latency_ms=latency_ms,
                retryable=True,
            )
