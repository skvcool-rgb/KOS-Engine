"""
KOS v0.6 -- Agent Router (Fast Path vs Agentic Path)

Decides in <10ms whether a query needs:
  - FAST PATH: single-hop factual, definition, math -> direct pipeline
  - AGENTIC PATH: multi-step, compare/contrast, ambiguous -> planner loop

Also detects:
  - Modality: text / math / image / file / audio
  - Answer type: factual / procedural / comparison / opinion / computation
  - Routing confidence: how sure are we about the path choice

Design principle: Do NOT make every query agentic.
Fast by default. Agentic only when needed.
"""

import re


# ── Answer Type Classification ─────────────────────────────────────

_FACTUAL_PATTERNS = [
    r'^what\s+is\b',
    r'^who\s+(?:is|was|are)\b',
    r'^where\s+(?:is|was|are)\b',
    r'^when\s+(?:was|did|is)\b',
    r'^how\s+(?:many|much|old|far|long|tall|big)\b',
    r'^define\b',
    r'^tell\s+me\s+about\b',
    r'distance\s+(?:of|from|between)',
    r'population\s+of',
    r'capital\s+of',
]

_MULTI_STEP_PATTERNS = [
    r'^compare\b',
    r'^contrast\b',
    r'\bcompare\s+to\b',
    r'\bcompared?\s+(?:to|with)\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\bdifference\s+between\b',
    r'\bhow\s+does\s+\w+\s+compare\b',
    r'\bwhich\s+is\s+(?:better|faster|stronger|bigger|cheaper|more)\b',
    r'^explain\s+(?:how|why|the\s+difference)\b',
    r'^what\s+(?:are\s+the\s+differences?|is\s+the\s+(?:relationship|difference))\b',
    r'^analyze\b',
    r'^summarize\b',
    r'^if\b.*then\b',
    r'^why\s+(?:does|do|is|are|did|was)\b',
    r'step\s+by\s+step',
    r'pros?\s+and\s+cons?',
    r'\band\b.*\bcompare\b',
]

_COMPUTATION_KEYWORDS = {
    'calculate', 'compute', 'solve', 'evaluate', 'integrate',
    'derivative', 'differentiate', 'factorial', 'sqrt', 'log',
}

_FILE_PATTERNS = [
    r'(?:read|parse|extract|analyze)\s+(?:this|the)\s+(?:pdf|file|document|image|chart)',
    r'\.(?:pdf|docx?|xlsx?|csv|png|jpg|jpeg)\b',
]


class RouteDecision:
    """Immutable routing decision."""
    __slots__ = ['path', 'modality', 'answer_type', 'confidence',
                 'sub_questions', 'solver', 'reason']

    def __init__(self, path, modality, answer_type, confidence,
                 sub_questions=None, solver=None, reason=""):
        self.path = path              # "fast" or "agentic"
        self.modality = modality      # "text", "math", "image", "file", "audio"
        self.answer_type = answer_type  # "factual", "procedural", "comparison", "computation", "opinion"
        self.confidence = confidence  # 0.0-1.0
        self.sub_questions = sub_questions or []
        self.solver = solver          # "graph", "math", "file", "web", None
        self.reason = reason

    def to_dict(self):
        return {
            "path": self.path,
            "modality": self.modality,
            "answer_type": self.answer_type,
            "confidence": round(self.confidence, 3),
            "sub_questions": self.sub_questions,
            "solver": self.solver,
            "reason": self.reason,
        }


class AgentRouter:
    """Route queries to fast path or agentic path."""

    def __init__(self, math_driver=None):
        self.math_driver = math_driver
        self._factual_re = [re.compile(p, re.I) for p in _FACTUAL_PATTERNS]
        self._multi_step_re = [re.compile(p, re.I) for p in _MULTI_STEP_PATTERNS]
        self._file_re = [re.compile(p, re.I) for p in _FILE_PATTERNS]

    def route(self, query: str) -> RouteDecision:
        """
        Decide routing in <10ms.

        Returns RouteDecision with path, modality, answer_type, and confidence.
        """
        lower = query.lower().strip()
        words = lower.split()

        # ── 1. Math detection (deterministic solver, always fast) ──
        if self.math_driver and self.math_driver.is_math_query(query):
            return RouteDecision(
                path="fast",
                modality="math",
                answer_type="computation",
                confidence=0.95,
                solver="math",
                reason="Math expression detected"
            )
        # Check computation keywords even without math_driver
        if _COMPUTATION_KEYWORDS & set(words):
            return RouteDecision(
                path="fast",
                modality="math",
                answer_type="computation",
                confidence=0.85,
                solver="math",
                reason="Computation keyword detected"
            )

        # ── 2. File/multimodal detection ──
        for pattern in self._file_re:
            if pattern.search(query):
                return RouteDecision(
                    path="agentic",
                    modality="file",
                    answer_type="procedural",
                    confidence=0.80,
                    solver="file",
                    reason="File/document reference detected"
                )

        # ── 3. Multi-step detection ──
        multi_score = 0
        for pattern in self._multi_step_re:
            if pattern.search(query):
                multi_score += 1
        # Also check for multiple question marks or conjunctions
        if query.count('?') > 1:
            multi_score += 1
        if ' and ' in lower and any(w in lower for w in ['what', 'how', 'why']):
            multi_score += 0.5
        # Long queries are more likely multi-step
        if len(words) > 20:
            multi_score += 0.5

        if multi_score >= 1:
            sub_qs = self._decompose(query) if multi_score >= 1.5 else []
            return RouteDecision(
                path="agentic",
                modality="text",
                answer_type="comparison" if any(k in lower for k in ['compar', ' vs', 'versus', 'difference between', 'which is better', 'pros and cons']) else "procedural",
                confidence=min(0.5 + multi_score * 0.15, 0.90),
                sub_questions=sub_qs,
                solver=None,
                reason=f"Multi-step signals: {multi_score}"
            )

        # ── 4. Factual detection (single-hop, fast path) ──
        factual_score = 0
        for pattern in self._factual_re:
            if pattern.search(query):
                factual_score += 1

        if factual_score > 0:
            return RouteDecision(
                path="fast",
                modality="text",
                answer_type="factual",
                confidence=min(0.6 + factual_score * 0.1, 0.95),
                solver="graph",
                reason=f"Factual pattern matched ({factual_score})"
            )

        # ── 5. Default: fast path with moderate confidence ──
        return RouteDecision(
            path="fast",
            modality="text",
            answer_type="factual",
            confidence=0.50,
            solver="graph",
            reason="Default fast path"
        )

    def _decompose(self, query: str) -> list:
        """
        Decompose a complex query into sub-questions.
        Simple heuristic: split on conjunctions and question marks.
        """
        # Split on " and " between clauses
        parts = re.split(r'\s+and\s+(?=\w+\s+(?:is|are|was|does|do|how|what|why))',
                          query, flags=re.I)
        if len(parts) > 1:
            return [p.strip().rstrip('?') + '?' for p in parts if len(p.strip()) > 10]

        # Split on multiple question marks
        parts = [p.strip() + '?' for p in query.split('?') if len(p.strip()) > 10]
        if len(parts) > 1:
            return parts

        return []
