"""
KOS V8.0 -- HTN Task Planner (Hierarchical Task Network)

The missing piece between "reactive Q&A" and "goal-directed behavior."
KOS provides salience (what matters), the planner provides sequencing
(what order to do things).

Components:
    - Goal Stack: ordered list of goals with priorities
    - Operators: atomic actions with preconditions and effects
    - HTN Decomposition: break complex goals into operator sequences
    - Plan Execution: step through plan, handle failures
"""


class Operator:
    """An atomic action the system can perform."""

    def __init__(self, name: str, preconditions: dict = None,
                 effects: dict = None, cost: float = 1.0):
        self.name = name
        self.preconditions = preconditions or {}  # {state_var: required_value}
        self.effects = effects or {}              # {state_var: new_value}
        self.cost = cost

    def applicable(self, state: dict) -> bool:
        """Can this operator be applied in the given state?"""
        for var, val in self.preconditions.items():
            if state.get(var) != val:
                return False
        return True

    def apply(self, state: dict) -> dict:
        """Apply this operator, returning new state."""
        new_state = dict(state)
        new_state.update(self.effects)
        return new_state

    def __repr__(self):
        return f"Op({self.name})"


class Goal:
    """A goal with target state and priority."""

    def __init__(self, name: str, target_state: dict,
                 priority: float = 1.0):
        self.name = name
        self.target_state = target_state  # {state_var: desired_value}
        self.priority = priority
        self.status = "pending"  # pending, active, achieved, failed

    def satisfied(self, state: dict) -> bool:
        """Is this goal satisfied in the given state?"""
        for var, val in self.target_state.items():
            if state.get(var) != val:
                return False
        return True

    def __repr__(self):
        return f"Goal({self.name}, pri={self.priority}, status={self.status})"


class Plan:
    """An ordered sequence of operators to achieve a goal."""

    def __init__(self, goal: Goal, steps: list = None):
        self.goal = goal
        self.steps = steps or []  # List of Operator
        self.current_step = 0
        self.status = "ready"  # ready, executing, completed, failed

    def next_step(self) -> Operator:
        """Get the next operator to execute."""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def advance(self):
        """Mark current step as done, advance."""
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.status = "completed"

    def total_cost(self) -> float:
        return sum(op.cost for op in self.steps)

    def __repr__(self):
        return (f"Plan({self.goal.name}, {len(self.steps)} steps, "
                f"at step {self.current_step})")


class HTNPlanner:
    """Hierarchical Task Network planner over KOS memory."""

    def __init__(self):
        self.operators = {}      # name -> Operator
        self.goal_stack = []     # Sorted by priority (highest first)
        self.current_plan = None
        self.plan_history = []

        # Register default KOS operators
        self._register_defaults()

    def _register_defaults(self):
        """Register the default KOS action operators."""
        defaults = [
            Operator("forage", {"has_query": True}, {"knowledge_acquired": True}, cost=5.0),
            Operator("query_graph", {"has_seeds": True}, {"results_available": True}, cost=1.0),
            Operator("synthesize", {"results_available": True}, {"answer_ready": True}, cost=2.0),
            Operator("verify", {"answer_ready": True}, {"answer_verified": True}, cost=3.0),
            Operator("respond", {"answer_verified": True}, {"response_sent": True}, cost=1.0),
            Operator("sleep", {}, {"rested": True}, cost=0.1),
            Operator("ingest_file", {"file_path_set": True}, {"knowledge_acquired": True}, cost=4.0),
            Operator("run_maintenance", {}, {"graph_clean": True}, cost=2.0),
        ]
        for op in defaults:
            self.operators[op.name] = op

    def register_operator(self, op: Operator):
        """Register a custom operator."""
        self.operators[op.name] = op

    def push_goal(self, goal: Goal):
        """Add a goal to the stack, maintaining priority order."""
        self.goal_stack.append(goal)
        self.goal_stack.sort(key=lambda g: g.priority, reverse=True)

    def pop_goal(self) -> Goal:
        """Get the highest-priority unsatisfied goal."""
        for goal in self.goal_stack:
            if goal.status == "pending":
                return goal
        return None

    def plan(self, goal: Goal, state: dict) -> Plan:
        """
        Generate a plan to achieve the goal from the current state.
        Uses forward-chaining search over operators.
        """
        if goal.satisfied(state):
            goal.status = "achieved"
            return Plan(goal, [])

        # BFS over operator sequences (bounded)
        max_depth = 10
        queue = [(state, [])]
        visited = set()

        for _ in range(1000):  # Max iterations
            if not queue:
                break

            current_state, steps = queue.pop(0)

            state_key = frozenset(current_state.items())
            if state_key in visited:
                continue
            visited.add(state_key)

            if len(steps) >= max_depth:
                continue

            for op in self.operators.values():
                if op.applicable(current_state):
                    new_state = op.apply(current_state)
                    new_steps = steps + [op]

                    if goal.satisfied(new_state):
                        plan = Plan(goal, new_steps)
                        plan.status = "ready"
                        return plan

                    queue.append((new_state, new_steps))

        # No plan found
        return Plan(goal, [])

    def plan_and_execute(self, goal: Goal, state: dict) -> dict:
        """Plan and step through execution."""
        plan = self.plan(goal, state)
        self.current_plan = plan

        if not plan.steps:
            if goal.satisfied(state):
                goal.status = "achieved"
                return {"status": "already_achieved", "plan": plan}
            goal.status = "failed"
            return {"status": "no_plan_found", "plan": plan}

        goal.status = "active"
        plan.status = "executing"

        results = {
            "status": "plan_ready",
            "plan": plan,
            "steps": [op.name for op in plan.steps],
            "cost": plan.total_cost(),
        }

        self.plan_history.append(plan)
        return results

    def execute_next_step(self, state: dict) -> dict:
        """Execute the next step in the current plan."""
        if not self.current_plan:
            return {"status": "no_plan"}

        op = self.current_plan.next_step()
        if not op:
            self.current_plan.status = "completed"
            self.current_plan.goal.status = "achieved"
            return {"status": "plan_completed"}

        if not op.applicable(state):
            return {"status": "precondition_failed", "operator": op.name,
                    "needed": op.preconditions}

        new_state = op.apply(state)
        self.current_plan.advance()

        return {
            "status": "step_executed",
            "operator": op.name,
            "new_state": new_state,
            "remaining_steps": len(self.current_plan.steps) - self.current_plan.current_step,
        }

    def status(self) -> dict:
        return {
            "goals": len(self.goal_stack),
            "pending_goals": sum(1 for g in self.goal_stack if g.status == "pending"),
            "operators": len(self.operators),
            "current_plan": str(self.current_plan) if self.current_plan else None,
            "plan_history": len(self.plan_history),
        }
