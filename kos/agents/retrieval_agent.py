"""
KOS v0.8 Step 2 -- Retrieval Agent

Wraps the existing query pipeline for factual/retrieve goal types.
Accepts AgentTask with payload {"query": "..."}.
Returns AgentResult with pipeline output.
"""
from __future__ import annotations

import time

from kos.agent_protocol import AgentResult, AgentStatus, AgentTask, AgentEvidence
from kos.agents.base_agent import BaseAgent


class RetrievalAgent(BaseAgent):
    """Agent for factual retrieval queries."""

    name = "retrieval_agent"
    capabilities = ["retrieve", "factual", "verify", "analyze", "monitor"]

    def __init__(self, query_fn):
        """
        Args:
            query_fn: callable(prompt) -> dict with keys:
                      answer, relevance_score, trust_label, latency_ms, source
        """
        self._query_fn = query_fn

    def execute(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        try:
            query = task.payload.get("query", "")
            if not query:
                return AgentResult(
                    task_id=task.task_id,
                    mission_id=task.mission_id,
                    goal_id=task.goal_id,
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    error="No query in payload",
                    latency_ms=0,
                    retryable=False,
                )

            response = self._query_fn(query)
            latency_ms = int((time.perf_counter() - started) * 1000)

            score = float(response.get("relevance_score", 0.0))
            answer = response.get("answer", "")
            trust = response.get("trust_label", "unverified")

            # Build evidence from response
            evidence = []
            if answer and score > 0:
                evidence.append(AgentEvidence(
                    source=response.get("source", "pipeline"),
                    content=answer[:500],
                    score=score,
                    metadata={"trust": trust},
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
                metadata={"route": response.get("route", "unknown")},
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
