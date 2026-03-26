"""
KOS v0.8 Step 2 -- Agent Registry

Dumb and deterministic. Maps goal types to agents.
No intelligence here -- just a lookup table.
"""
from __future__ import annotations

from typing import Dict, List

from kos.agents.base_agent import BaseAgent


class AgentRegistry:
    """Registry of available agents. Deterministic matching."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent. Rejects duplicates."""
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        """Get an agent by name. Raises KeyError if not found."""
        if name not in self._agents:
            raise KeyError(f"Agent not found: {name}")
        return self._agents[name]

    def list_agents(self) -> List[str]:
        """List all registered agent names, sorted."""
        return sorted(self._agents.keys())

    def match(self, goal_type: str) -> BaseAgent:
        """Find the agent that can handle a goal type.

        Returns the first matching agent (sorted by name for determinism).
        Raises KeyError if no agent can handle the goal type.
        """
        matches = [a for a in self._agents.values()
                    if a.can_handle(goal_type)]
        if not matches:
            raise KeyError(f"No agent found for goal_type={goal_type}")
        if len(matches) > 1:
            matches.sort(key=lambda a: a.name)
        return matches[0]

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
