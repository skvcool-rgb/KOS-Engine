"""
KOS V8.0 -- Reasoning Workspace (Multi-Step Deliberation)

The graph nominates candidates. The workspace deliberates.

Components:
    - Scratchpad: temporary working memory for a reasoning session
    - Subproblem Tree: decompose complex queries into sub-queries
    - Iterative Solver: loop until convergence or resource limit
    - N parallel hypothesis workspaces with independent state
"""


class Scratchpad:
    """Temporary working memory for a single reasoning session."""

    def __init__(self):
        self.facts = []       # Accumulated facts
        self.inferences = []  # Derived conclusions
        self.queries = []     # Sub-queries generated
        self.notes = []       # Free-form reasoning notes
        self._step = 0

    def add_fact(self, fact: str, source: str = ""):
        self.facts.append({"text": fact, "source": source, "step": self._step})

    def add_inference(self, inference: str, from_facts: list = None):
        self.inferences.append({
            "text": inference,
            "derived_from": from_facts or [],
            "step": self._step,
        })

    def add_query(self, query: str, reason: str = ""):
        self.queries.append({"query": query, "reason": reason, "step": self._step})

    def note(self, text: str):
        self.notes.append({"text": text, "step": self._step})

    def step(self):
        self._step += 1

    def summary(self) -> dict:
        return {
            "facts": len(self.facts),
            "inferences": len(self.inferences),
            "pending_queries": len(self.queries),
            "steps": self._step,
        }

    def all_conclusions(self) -> list:
        """Return all inferences as strings."""
        return [i["text"] for i in self.inferences]


class SubproblemTree:
    """Decompose a complex query into a tree of sub-problems."""

    def __init__(self, root_query: str):
        self.root = {"query": root_query, "children": [], "answer": None,
                     "status": "pending"}

    def decompose(self, parent_query: str, sub_queries: list):
        """Add sub-queries as children of a parent query."""
        node = self._find(self.root, parent_query)
        if node:
            for sq in sub_queries:
                child = {"query": sq, "children": [], "answer": None,
                         "status": "pending"}
                node["children"].append(child)

    def answer(self, query: str, answer: str):
        """Provide an answer to a sub-query."""
        node = self._find(self.root, query)
        if node:
            node["answer"] = answer
            node["status"] = "answered"

    def pending_queries(self) -> list:
        """Get all unanswered leaf queries."""
        leaves = []
        self._collect_pending(self.root, leaves)
        return leaves

    def is_complete(self) -> bool:
        """Are all sub-problems answered?"""
        return len(self.pending_queries()) == 0

    def synthesize(self) -> str:
        """Combine all sub-answers into a root answer."""
        answers = []
        self._collect_answers(self.root, answers)
        return " ".join(answers) if answers else ""

    def _find(self, node, query):
        if node["query"] == query:
            return node
        for child in node["children"]:
            result = self._find(child, query)
            if result:
                return result
        return None

    def _collect_pending(self, node, result):
        if node["status"] == "pending" and not node["children"]:
            result.append(node["query"])
        for child in node["children"]:
            self._collect_pending(child, result)

    def _collect_answers(self, node, result):
        if node["answer"]:
            result.append(node["answer"])
        for child in node["children"]:
            self._collect_answers(child, result)


class HypothesisWorkspace:
    """
    A single hypothesis workspace with its own scratchpad and state.
    N of these run in parallel to explore competing theories.
    """

    def __init__(self, hypothesis_id: str, initial_claim: str):
        self.id = hypothesis_id
        self.claim = initial_claim
        self.scratchpad = Scratchpad()
        self.confidence = 0.5
        self.supporting_evidence = []
        self.contradicting_evidence = []
        self.status = "active"  # active, confirmed, refuted, merged

    def add_support(self, evidence: str, strength: float = 0.5):
        self.supporting_evidence.append((evidence, strength))
        self.confidence = min(1.0, self.confidence + strength * 0.1)

    def add_contradiction(self, evidence: str, strength: float = 0.5):
        self.contradicting_evidence.append((evidence, strength))
        self.confidence = max(0.0, self.confidence - strength * 0.15)

    def evaluate(self) -> str:
        """Evaluate: should this hypothesis be kept, merged, or discarded?"""
        if self.confidence > 0.8:
            self.status = "confirmed"
        elif self.confidence < 0.2:
            self.status = "refuted"
        return self.status

    def summary(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "confidence": self.confidence,
            "support": len(self.supporting_evidence),
            "contradictions": len(self.contradicting_evidence),
            "status": self.status,
        }


class ReasoningEngine:
    """
    Orchestrates multi-step reasoning with parallel hypothesis workspaces.

    Flow:
    1. Receive complex query
    2. Decompose into sub-problems
    3. For each sub-problem, query KOS graph
    4. Accumulate evidence in scratchpad
    5. Fork hypotheses when contradictions arise
    6. Iterate until convergence
    7. Return ranked conclusions
    """

    MAX_ITERATIONS = 10
    MAX_WORKSPACES = 5

    def __init__(self, kernel):
        self.kernel = kernel
        self.workspaces = []
        self.scratchpad = Scratchpad()
        self.problem_tree = None

    def reason(self, query: str, seeds: list = None,
               max_iterations: int = None) -> dict:
        """
        Run a full reasoning session.

        Args:
            query: the complex question
            seeds: initial graph seeds
            max_iterations: override iteration limit

        Returns:
            {
                "conclusions": [str],
                "confidence": float,
                "hypotheses": [dict],
                "iterations": int,
                "scratchpad": dict,
            }
        """
        max_iter = max_iterations or self.MAX_ITERATIONS
        seeds = seeds or []

        # Step 1: Initialize
        self.problem_tree = SubproblemTree(query)
        self.scratchpad = Scratchpad()
        self.workspaces = []

        # Step 2: Initial graph query
        if seeds and hasattr(self.kernel, 'query_beam'):
            results = self.kernel.query_beam(seeds, top_k=20)
        elif seeds:
            results = self.kernel.query(seeds, top_k=20)
        else:
            results = []

        # Step 3: Accumulate initial evidence
        for node_id, score in results:
            self.scratchpad.add_fact(
                f"{node_id} (relevance={score:.2f})",
                source="graph_query")

        # Step 4: Create primary hypothesis
        if results:
            primary = HypothesisWorkspace("h0", query)
            for node_id, score in results[:5]:
                primary.add_support(node_id, score)
            self.workspaces.append(primary)

        # Step 5: Iterative refinement
        for i in range(max_iter):
            self.scratchpad.step()

            # Check for contradictions -> fork
            self._check_and_fork()

            # Evaluate hypotheses
            for ws in self.workspaces:
                ws.evaluate()

            # Remove refuted hypotheses
            self.workspaces = [ws for ws in self.workspaces
                               if ws.status != "refuted"]

            # Check convergence
            if all(ws.status == "confirmed" for ws in self.workspaces):
                break
            if not self.workspaces:
                break

        # Step 6: Synthesize conclusions
        conclusions = []
        for ws in sorted(self.workspaces,
                         key=lambda w: w.confidence, reverse=True):
            conclusions.append(ws.claim)

        avg_conf = (sum(ws.confidence for ws in self.workspaces)
                    / max(len(self.workspaces), 1))

        return {
            "conclusions": conclusions,
            "confidence": avg_conf,
            "hypotheses": [ws.summary() for ws in self.workspaces],
            "iterations": self.scratchpad._step,
            "scratchpad": self.scratchpad.summary(),
        }

    def _check_and_fork(self):
        """Check for contradictions and fork new hypothesis workspaces."""
        contradictions = getattr(self.kernel, 'contradictions', [])
        if not contradictions or len(self.workspaces) >= self.MAX_WORKSPACES:
            return

        for c in contradictions[-3:]:  # Only check recent contradictions
            existing = c.get('existing_target', '')
            new = c.get('new_target', '')
            if existing and new:
                # Check if both appear in any workspace
                for ws in self.workspaces:
                    support_nodes = set(e for e, _ in ws.supporting_evidence)
                    if existing in support_nodes and new not in support_nodes:
                        # Fork: create alternative workspace
                        alt = HypothesisWorkspace(
                            f"h{len(self.workspaces)}",
                            f"Alternative: {new} instead of {existing}")
                        alt.add_support(new, 0.5)
                        alt.add_contradiction(existing, 0.3)
                        self.workspaces.append(alt)
                        break
