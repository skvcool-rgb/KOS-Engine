"""
KOS V8.0 -- Unified Utility-Constrained Drive System

Governs ALL autonomous behavior through a single mathematical score.
Prevents goal drift, wireheading, and resource waste.

DriveScore = (0.30 * KnowledgeGap)
           + (0.25 * RiskRelevance)
           + (0.20 * TaskContext)
           + (0.15 * RecencyDecay)
           + (0.10 * Novelty)

If DriveScore < DO_NOTHING_THRESHOLD: the system sleeps.
If DriveScore < mission alignment threshold: the action is suppressed.
"""

import math
import time


# ---- Thresholds ----------------------------------------------------------
DO_NOTHING_THRESHOLD = 0.65
MISSION_ALIGNMENT_MIN = 0.40

# ---- Drive Weights -------------------------------------------------------
W_KNOWLEDGE_GAP = 0.30
W_RISK_RELEVANCE = 0.25
W_TASK_CONTEXT = 0.20
W_RECENCY_DECAY = 0.15
W_NOVELTY = 0.10


class Mission:
    """
    Defines the system's core mission. All drives are subordinate to this.
    Without a mission, the system defaults to general knowledge acquisition.
    """

    def __init__(self, description: str = "general knowledge acquisition",
                 keywords: list = None, domain: str = "general"):
        self.description = description
        self.keywords = set(k.lower() for k in (keywords or []))
        self.domain = domain
        self.priority_topics = set()   # Topics with elevated priority
        self.blocked_topics = set()    # Topics to never forage

    def alignment_score(self, query: str, target_topic: str = "") -> float:
        """How well does a proposed action align with the mission?"""
        text = (query + " " + target_topic).lower()
        words = set(text.split())

        if not self.keywords:
            return 0.7  # No mission = neutral alignment

        # Direct keyword overlap
        overlap = len(self.keywords & words)
        base = min(overlap / max(len(self.keywords), 1), 1.0)

        # Priority topic boost
        if target_topic.lower() in self.priority_topics:
            base = min(base + 0.3, 1.0)

        # Blocked topic suppression
        if target_topic.lower() in self.blocked_topics:
            return 0.0

        return base

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "keywords": list(self.keywords),
            "domain": self.domain,
            "priority_topics": list(self.priority_topics),
            "blocked_topics": list(self.blocked_topics),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Mission":
        m = cls(d.get("description", ""),
                d.get("keywords", []),
                d.get("domain", "general"))
        m.priority_topics = set(d.get("priority_topics", []))
        m.blocked_topics = set(d.get("blocked_topics", []))
        return m


class DriveScorer:
    """
    Computes a unified utility score for any proposed autonomous action.
    Returns (score, breakdown_dict) so decisions are auditable.
    """

    def __init__(self, mission: Mission = None):
        self.mission = mission or Mission()

    def score(self, kernel, query: str, target_topic: str = "",
              is_active_task: bool = False) -> tuple:
        """
        Compute DriveScore for a proposed action.

        Args:
            kernel: KOSKernel instance
            query: the proposed search/action query
            target_topic: node or topic being targeted
            is_active_task: True if this relates to an active user task

        Returns:
            (total_score: float, breakdown: dict)
        """
        breakdown = {}

        # 1. Knowledge Gap: how much don't we know about this topic?
        breakdown["knowledge_gap"] = self._knowledge_gap(kernel, target_topic)

        # 2. Risk Relevance: does this reduce uncertainty in critical areas?
        breakdown["risk_relevance"] = self._risk_relevance(
            kernel, target_topic, query)

        # 3. Task Context: does this relate to what the user is working on?
        breakdown["task_context"] = self._task_context(
            kernel, query, is_active_task)

        # 4. Recency Decay: is existing knowledge stale?
        breakdown["recency_decay"] = self._recency_decay(kernel, target_topic)

        # 5. Novelty: is this genuinely new information?
        breakdown["novelty"] = self._novelty(kernel, target_topic)

        # Weighted sum
        raw_score = (
            W_KNOWLEDGE_GAP * breakdown["knowledge_gap"]
            + W_RISK_RELEVANCE * breakdown["risk_relevance"]
            + W_TASK_CONTEXT * breakdown["task_context"]
            + W_RECENCY_DECAY * breakdown["recency_decay"]
            + W_NOVELTY * breakdown["novelty"]
        )

        # Mission alignment gate
        mission_score = self.mission.alignment_score(query, target_topic)
        breakdown["mission_alignment"] = mission_score

        # Final score = raw * mission_alignment (mission suppresses off-topic)
        if mission_score < MISSION_ALIGNMENT_MIN:
            total = raw_score * 0.2  # Heavily penalized
        else:
            total = raw_score * (0.5 + 0.5 * mission_score)

        breakdown["raw_score"] = raw_score
        breakdown["total_score"] = total

        # Decision
        if total < DO_NOTHING_THRESHOLD:
            breakdown["decision"] = "DO_NOTHING"
        else:
            breakdown["decision"] = "EXECUTE"

        return total, breakdown

    def _knowledge_gap(self, kernel, target_topic: str) -> float:
        """How much don't we know? 1.0 = topic completely unknown."""
        if not target_topic:
            return 0.5

        # Check if topic exists in graph
        if target_topic in kernel.nodes:
            node = kernel.nodes[target_topic]
            conn_count = len(node.connections)
            # More connections = less gap
            return max(0.0, 1.0 - (conn_count / 20.0))
        return 1.0  # Completely unknown

    def _risk_relevance(self, kernel, target_topic: str,
                        query: str) -> float:
        """Does this reduce uncertainty in high-stakes areas?"""
        risk_words = {"risk", "danger", "failure", "critical", "safety",
                      "compliance", "regulation", "breach", "vulnerability",
                      "error", "loss", "threat", "hazard", "incident"}
        query_words = set(query.lower().split())
        overlap = len(risk_words & query_words)
        return min(overlap / 2.0, 1.0)

    def _task_context(self, kernel, query: str,
                      is_active_task: bool) -> float:
        """Does this relate to the user's current work?"""
        if is_active_task:
            return 1.0

        # Check working memory overlap
        wm = getattr(kernel, '_working_memory', [])
        if not wm:
            return 0.3

        query_words = set(query.lower().split())
        wm_words = set(w.lower().replace('.', ' ').replace('_', ' ')
                        for w in wm)
        wm_flat = set()
        for w in wm_words:
            wm_flat.update(w.split())

        overlap = len(query_words & wm_flat)
        return min(overlap / max(len(query_words), 1), 1.0)

    def _recency_decay(self, kernel, target_topic: str) -> float:
        """How stale is our knowledge? 1.0 = very stale, needs refresh."""
        if not target_topic or target_topic not in kernel.nodes:
            return 0.5

        node = kernel.nodes[target_topic]
        last_tick = getattr(node, 'last_tick', 0)
        current_tick = getattr(kernel, 'current_tick', 0)
        age = current_tick - last_tick

        if age < 50:
            return 0.1   # Fresh
        elif age < 200:
            return 0.5   # Warming
        else:
            return 0.9   # Stale

    def _novelty(self, kernel, target_topic: str) -> float:
        """Is this genuinely new? 1.0 = never seen before."""
        if not target_topic:
            return 0.5
        if target_topic not in kernel.nodes:
            return 1.0  # Brand new
        return 0.2  # Already known


class AdaptiveTickController:
    """
    Controls the organism's tick rate based on system activity.

    Active mode: 100Hz (10ms ticks) -- user is interacting
    Idle mode:   1Hz (1000ms ticks) -- no activity
    Sleep mode:  0.1Hz (10s ticks) -- prolonged inactivity

    Prevents CPU waste when there's nothing meaningful to do.
    """

    ACTIVE_INTERVAL = 0.01    # 100Hz
    IDLE_INTERVAL = 1.0       # 1Hz
    SLEEP_INTERVAL = 10.0     # 0.1Hz

    IDLE_AFTER = 30.0         # seconds of no activity -> idle
    SLEEP_AFTER = 300.0       # seconds of no activity -> sleep

    def __init__(self):
        self.last_activity = time.time()
        self.current_mode = "active"
        self._forced_mode = None

    def record_activity(self):
        """Call when user interacts or a meaningful event occurs."""
        self.last_activity = time.time()
        self.current_mode = "active"

    def get_tick_interval(self) -> float:
        """Return the current tick interval in seconds."""
        if self._forced_mode:
            return {"active": self.ACTIVE_INTERVAL,
                    "idle": self.IDLE_INTERVAL,
                    "sleep": self.SLEEP_INTERVAL}[self._forced_mode]

        elapsed = time.time() - self.last_activity

        if elapsed < self.IDLE_AFTER:
            self.current_mode = "active"
            return self.ACTIVE_INTERVAL
        elif elapsed < self.SLEEP_AFTER:
            self.current_mode = "idle"
            return self.IDLE_INTERVAL
        else:
            self.current_mode = "sleep"
            return self.SLEEP_INTERVAL

    def force_mode(self, mode: str):
        """Override automatic mode selection."""
        if mode in ("active", "idle", "sleep", None):
            self._forced_mode = mode

    def status(self) -> dict:
        interval = self.get_tick_interval()
        return {
            "mode": self.current_mode,
            "tick_interval_ms": interval * 1000,
            "hz": 1.0 / max(interval, 0.001),
            "seconds_since_activity": time.time() - self.last_activity,
        }
