"""
KOS v0.6 -- Decision Gate (Policy Engine)

Upgraded from simple confidence threshold to a full policy engine.
Decides:
  - SPEAK: answer now (sufficient evidence)
  - STREAM_PARTIAL: give partial answer while continuing
  - ESCALATE: send to agent loop for deeper research
  - FORAGE: enqueue background internet search
  - REFUSE: unsafe or unanswerable

Uses the 4-layer relevance scorer plus evidence quality metrics.
"""

from enum import Enum


class Decision(Enum):
    SPEAK = "speak"                     # Answer immediately
    STREAM_PARTIAL = "stream_partial"   # Give partial answer, continue
    ESCALATE = "escalate"               # Send to agentic planner
    FORAGE = "forage"                   # Search internet for missing knowledge
    REFUSE = "refuse"                   # Cannot answer safely


class GateResult:
    """Immutable decision result."""
    __slots__ = ['decision', 'confidence', 'reason', 'evidence_count',
                 'relevance_score', 'missing_concepts', 'partial_answer']

    def __init__(self, decision, confidence=0.0, reason="",
                 evidence_count=0, relevance_score=0.0,
                 missing_concepts=None, partial_answer=None):
        self.decision = decision
        self.confidence = confidence
        self.reason = reason
        self.evidence_count = evidence_count
        self.relevance_score = relevance_score
        self.missing_concepts = missing_concepts or []
        self.partial_answer = partial_answer

    def to_dict(self):
        return {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "relevance_score": round(self.relevance_score, 3),
            "missing_concepts": self.missing_concepts,
        }


class DecisionGate:
    """
    Policy engine that decides what to do with a candidate answer.

    Inputs:
        - candidate answer text
        - evidence store (how much evidence do we have?)
        - relevance score (from 4-layer scorer)
        - route decision (fast or agentic?)
        - query complexity

    Outputs:
        - GateResult with decision + reasoning
    """

    # Thresholds (tuned on 28-query test set)
    SPEAK_THRESHOLD = 0.46          # Relevance score above this = speak
    PARTIAL_THRESHOLD = 0.30        # Below speak but above this = stream partial
    MIN_EVIDENCE = 1                # Need at least 1 evidence item
    HIGH_CONFIDENCE = 0.70          # Above this = high confidence answer

    def __init__(self, relevance_scorer=None):
        self.scorer = relevance_scorer

    def decide(self, query: str, answer: str, evidence_count: int = 0,
               relevance_score: float = None, route_path: str = "fast",
               is_math: bool = False) -> GateResult:
        """
        Main decision function. Runs in <5ms.

        Args:
            query: original user query
            answer: candidate answer from pipeline
            evidence_count: number of evidence items found
            relevance_score: pre-computed relevance (or None to compute)
            route_path: "fast" or "agentic"
            is_math: True if math solver was used

        Returns:
            GateResult with decision
        """
        # ── Math answers are always trusted (deterministic) ──
        if is_math and answer and len(answer.strip()) > 0:
            return GateResult(
                decision=Decision.SPEAK,
                confidence=0.99,
                reason="Deterministic math solver",
                evidence_count=1,
                relevance_score=1.0,
            )

        # ── No answer at all ──
        if not answer or len(answer.strip()) < 5:
            return GateResult(
                decision=Decision.FORAGE,
                confidence=0.0,
                reason="No answer generated",
                evidence_count=evidence_count,
                relevance_score=0.0,
            )

        # ── Compute relevance if not provided ──
        if relevance_score is None and self.scorer:
            try:
                relevance_score, _ = self.scorer.score(query, answer)
            except Exception:
                relevance_score = 0.5

        if relevance_score is None:
            relevance_score = 0.5

        # ── Check for "no data" phrases ──
        _NO_DATA = ["i don't have data", "no data on this", "i don't have information",
                     "no information on", "i don't know", "cannot answer",
                     "no answer found", "no relevant information"]
        answer_lower = answer.lower().strip()
        if any(phrase in answer_lower for phrase in _NO_DATA):
            return GateResult(
                decision=Decision.FORAGE,
                confidence=0.1,
                reason="Answer indicates no data",
                evidence_count=evidence_count,
                relevance_score=relevance_score,
            )

        # ── High confidence: speak immediately ──
        if relevance_score >= self.SPEAK_THRESHOLD and evidence_count >= self.MIN_EVIDENCE:
            return GateResult(
                decision=Decision.SPEAK,
                confidence=relevance_score,
                reason=f"Relevance {relevance_score:.2f} >= {self.SPEAK_THRESHOLD}",
                evidence_count=evidence_count,
                relevance_score=relevance_score,
            )

        # ── Medium confidence: depends on route ──
        if relevance_score >= self.PARTIAL_THRESHOLD:
            if route_path == "agentic":
                return GateResult(
                    decision=Decision.ESCALATE,
                    confidence=relevance_score,
                    reason=f"Medium relevance ({relevance_score:.2f}) on agentic path",
                    evidence_count=evidence_count,
                    relevance_score=relevance_score,
                    partial_answer=answer,
                )
            else:
                # Fast path: stream partial and forage
                return GateResult(
                    decision=Decision.STREAM_PARTIAL,
                    confidence=relevance_score,
                    reason=f"Medium relevance ({relevance_score:.2f}), streaming partial",
                    evidence_count=evidence_count,
                    relevance_score=relevance_score,
                    partial_answer=answer,
                )

        # ── Low confidence: forage ──
        return GateResult(
            decision=Decision.FORAGE,
            confidence=relevance_score,
            reason=f"Low relevance ({relevance_score:.2f} < {self.PARTIAL_THRESHOLD})",
            evidence_count=evidence_count,
            relevance_score=relevance_score,
        )
