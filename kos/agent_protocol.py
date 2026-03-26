"""
KOS v0.8 Step 2 -- Agent Protocol

Strict contracts for agent communication. Every agent must:
  - Accept one AgentTask
  - Return one AgentResult
  - Never mutate shared mission state directly
  - Never call another agent directly

The dispatcher is the ONLY authority that routes tasks to agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"
    SKIPPED = "SKIPPED"


@dataclass
class AgentTask:
    """Immutable work unit dispatched to an agent."""
    task_id: str
    mission_id: str
    goal_id: str
    goal_type: str
    payload: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    attempt: int = 1

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "payload": self.payload,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "attempt": self.attempt,
        }


@dataclass
class AgentEvidence:
    """A single piece of evidence collected by an agent."""
    source: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    """Strict output contract from any agent."""
    task_id: str
    mission_id: str
    goal_id: str
    agent_name: str
    status: AgentStatus
    output: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    confidence: float = 0.0
    trust_label: str = "unknown"
    latency_ms: int = 0
    evidence: List[AgentEvidence] = field(default_factory=list)
    error: Optional[str] = None
    retryable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "goal_id": self.goal_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "output": self.output,
            "score": self.score,
            "confidence": self.confidence,
            "trust_label": self.trust_label,
            "latency_ms": self.latency_ms,
            "evidence": [e.to_dict() for e in self.evidence],
            "error": self.error,
            "retryable": self.retryable,
            "metadata": self.metadata,
        }
