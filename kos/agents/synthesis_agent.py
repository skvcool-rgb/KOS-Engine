"""
KOS v0.8 Step 2 -- Synthesis Agent

Synthesizes from prior goal outputs, not re-query.
Accepts AgentTask with payload {"inputs": [...], "mode": "...", "query": "..."}.

When inputs are available from upstream goals, synthesizes from them.
Falls back to query_fn if no inputs provided.
"""
from __future__ import annotations

import time

from kos.agent_protocol import AgentResult, AgentStatus, AgentTask, AgentEvidence
from kos.agents.base_agent import BaseAgent


class SynthesisAgent(BaseAgent):
    """Agent for synthesis/summary goal types."""

    name = "synthesis_agent"
    capabilities = ["synthesize", "summary"]

    def __init__(self, query_fn):
        """
        Args:
            query_fn: callable(prompt) -> dict with pipeline response
        """
        self._query_fn = query_fn

    def execute(self, task: AgentTask) -> AgentResult:
        started = time.perf_counter()
        try:
            inputs = task.payload.get("inputs", [])
            mode = task.payload.get("mode", "summary")
            query = task.payload.get("query", "")

            # If we have upstream goal outputs, build a synthesis prompt
            if inputs:
                parts = []
                for inp in inputs:
                    goal_id = inp.get("goal_id", "?")
                    output = inp.get("output", {})
                    answer = output.get("answer", "") if isinstance(output, dict) else str(output)
                    if answer:
                        parts.append(answer)

                if parts and query:
                    # Synthesize by querying with context
                    synth_query = f"{query}. Context: {' '.join(parts[:3])}"
                    response = self._query_fn(synth_query[:500])
                elif query:
                    response = self._query_fn(query)
                else:
                    response = {
                        "answer": " ".join(parts),
                        "relevance_score": 0.7,
                        "trust_label": "best-effort",
                        "source": "synthesis",
                    }
            elif query:
                response = self._query_fn(query)
            else:
                return AgentResult(
                    task_id=task.task_id,
                    mission_id=task.mission_id,
                    goal_id=task.goal_id,
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    error="No inputs or query in payload",
                    latency_ms=0,
                    retryable=False,
                )

            latency_ms = int((time.perf_counter() - started) * 1000)

            score = float(response.get("relevance_score", 0.0))
            answer = response.get("answer", "")
            trust = response.get("trust_label", "unverified")

            evidence = []
            if answer and score > 0:
                evidence.append(AgentEvidence(
                    source="synthesis",
                    content=answer[:500],
                    score=score,
                    metadata={"mode": mode, "input_count": len(inputs)},
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
                metadata={"mode": mode, "input_count": len(inputs)},
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
                retryable=False,
            )
