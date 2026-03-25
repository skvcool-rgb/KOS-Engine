"""
KOS V8.0 -- Action Schema Registry

Every action the system can take is defined as a schema with:
    - name: unique identifier
    - inputs: required parameters
    - preconditions: what must be true before execution
    - effects: what changes after execution
    - permissions: what authorization is needed
    - rollback: how to undo this action
    - cost: resource cost estimate

Actions are composable: complex behaviors are built by chaining
registered actions. The registry enforces permissions and tracks
execution history for rollback.
"""

import time


class ActionSchema:
    """A registered action with full metadata."""

    def __init__(self, name: str, inputs: list = None,
                 preconditions: list = None, effects: list = None,
                 permissions: list = None, rollback_fn=None,
                 cost: float = 1.0, description: str = ""):
        self.name = name
        self.inputs = inputs or []
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.permissions = permissions or []  # e.g., ["network", "file_write"]
        self.rollback_fn = rollback_fn
        self.cost = cost
        self.description = description
        self.execution_count = 0
        self.last_executed = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "permissions": self.permissions,
            "cost": self.cost,
            "description": self.description,
            "executions": self.execution_count,
        }


class ActionExecution:
    """Record of a single action execution."""

    def __init__(self, action_name: str, inputs: dict,
                 result: dict, timestamp: float):
        self.action_name = action_name
        self.inputs = inputs
        self.result = result
        self.timestamp = timestamp
        self.rolled_back = False


class ActionRegistry:
    """
    Central registry for all system actions.
    Enforces permissions, tracks history, supports rollback.
    """

    def __init__(self):
        self.actions = {}           # name -> ActionSchema
        self.execution_history = [] # List of ActionExecution
        self.granted_permissions = set()  # Currently granted permissions
        self._max_history = 100

        # Register default KOS actions
        self._register_defaults()

    def _register_defaults(self):
        """Register the standard KOS actions."""
        defaults = [
            ActionSchema(
                "forage_web",
                inputs=["query", "source_url"],
                preconditions=["network_available"],
                effects=["knowledge_acquired"],
                permissions=["network"],
                cost=5.0,
                description="Search the web for knowledge",
            ),
            ActionSchema(
                "ingest_file",
                inputs=["file_path"],
                preconditions=["file_exists"],
                effects=["knowledge_acquired"],
                permissions=["file_read"],
                cost=3.0,
                description="Ingest a local file into the graph",
            ),
            ActionSchema(
                "query_graph",
                inputs=["seeds", "top_k"],
                preconditions=[],
                effects=["results_available"],
                permissions=[],
                cost=1.0,
                description="Query the knowledge graph",
            ),
            ActionSchema(
                "synthesize_response",
                inputs=["evidence", "intent"],
                preconditions=["results_available"],
                effects=["response_ready"],
                permissions=[],
                cost=2.0,
                description="Synthesize a response from evidence",
            ),
            ActionSchema(
                "speak",
                inputs=["text"],
                preconditions=["response_ready"],
                effects=["response_delivered"],
                permissions=["output"],
                cost=1.0,
                description="Deliver response to user",
            ),
            ActionSchema(
                "sleep",
                inputs=[],
                preconditions=[],
                effects=["rested"],
                permissions=[],
                cost=0.1,
                description="Enter low-power idle state",
            ),
            ActionSchema(
                "run_daemon",
                inputs=[],
                preconditions=[],
                effects=["maintenance_complete"],
                permissions=[],
                cost=2.0,
                description="Run maintenance daemon cycle",
            ),
            ActionSchema(
                "self_tune",
                inputs=["parameter", "value"],
                preconditions=[],
                effects=["config_updated"],
                permissions=["config_write"],
                cost=3.0,
                description="Tune a system parameter",
            ),
        ]
        for action in defaults:
            self.actions[action.name] = action

    def register(self, action: ActionSchema):
        """Register a new action."""
        self.actions[action.name] = action

    def grant_permission(self, permission: str):
        """Grant a permission to the system."""
        self.granted_permissions.add(permission)

    def revoke_permission(self, permission: str):
        """Revoke a permission."""
        self.granted_permissions.discard(permission)

    def can_execute(self, action_name: str) -> dict:
        """Check if an action can be executed."""
        if action_name not in self.actions:
            return {"allowed": False, "reason": "action not registered"}

        action = self.actions[action_name]
        missing = [p for p in action.permissions
                   if p not in self.granted_permissions]

        if missing:
            return {"allowed": False, "reason": f"missing permissions: {missing}"}

        return {"allowed": True}

    def execute(self, action_name: str, inputs: dict = None) -> dict:
        """
        Execute an action (checks permissions first).

        Returns execution result.
        """
        check = self.can_execute(action_name)
        if not check["allowed"]:
            return {"status": "denied", "reason": check["reason"]}

        action = self.actions[action_name]
        action.execution_count += 1
        action.last_executed = time.time()

        result = {
            "status": "executed",
            "action": action_name,
            "effects": action.effects,
            "cost": action.cost,
        }

        # Record execution
        execution = ActionExecution(
            action_name, inputs or {}, result, time.time())
        self.execution_history.append(execution)

        if len(self.execution_history) > self._max_history:
            self.execution_history = self.execution_history[-self._max_history:]

        return result

    def rollback_last(self) -> dict:
        """Rollback the last action (if rollback function exists)."""
        if not self.execution_history:
            return {"status": "no_history"}

        last = self.execution_history[-1]
        if last.rolled_back:
            return {"status": "already_rolled_back"}

        action = self.actions.get(last.action_name)
        if action and action.rollback_fn:
            try:
                action.rollback_fn(last.inputs)
                last.rolled_back = True
                return {"status": "rolled_back", "action": last.action_name}
            except Exception as e:
                return {"status": "rollback_failed", "error": str(e)}

        return {"status": "no_rollback_available", "action": last.action_name}

    def list_actions(self) -> list:
        """List all registered actions."""
        return [a.to_dict() for a in self.actions.values()]

    def stats(self) -> dict:
        return {
            "registered_actions": len(self.actions),
            "granted_permissions": list(self.granted_permissions),
            "total_executions": len(self.execution_history),
            "most_used": max(self.actions.values(),
                            key=lambda a: a.execution_count).name
                        if self.actions else None,
        }
