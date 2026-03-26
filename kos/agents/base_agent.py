"""
KOS v0.8 Step 2 -- Base Agent

Abstract base class for all KOS agents. Enforces protocol compliance:
  - accept one AgentTask
  - return one AgentResult
  - never mutate shared state
  - never call another agent
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from kos.agent_protocol import AgentTask, AgentResult


class BaseAgent(ABC):
    """Abstract base for all KOS agents."""

    name: str = "base"
    capabilities: List[str] = []

    def can_handle(self, goal_type: str) -> bool:
        """Check if this agent can handle a given goal type."""
        return goal_type in self.capabilities

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task and return a result. Must not raise."""
        raise NotImplementedError
