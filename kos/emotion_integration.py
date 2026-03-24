"""
KOS V6.1 — Emotion-Decision Integration Bridge.

Connects the EmotionEngine neurochemical state to actual KOS behavior:
    - Modulates confidence thresholds (high cortisol = more cautious)
    - Reinforces scoring paths via myelin (dopamine reward)
    - Flags system quality degradation (low serotonin)
    - Triggers Active Inference foraging (high entropy + cortisol)
    - Adjusts Weaver evidence scores based on emotional state

The bridge does NOT generate emotions — it reads the EmotionEngine
state and translates it into actionable parameter adjustments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .emotion import EmotionEngine, NeurochemicalState


# ---------------------------------------------------------------------------
# Decision record — logged after every emotion-modulated decision
# ---------------------------------------------------------------------------

@dataclass
class EmotionDecisionRecord:
    """Single record of an emotion-modulated decision."""
    timestamp: float
    action: str
    base_value: float
    modulated_value: float
    emotion_snapshot: Dict[str, float]
    emotion_label: str
    reason: str


# ---------------------------------------------------------------------------
# Emotion-Decision Bridge
# ---------------------------------------------------------------------------

class EmotionDecisionBridge:
    """Translates EmotionEngine neurochemical state into KOS behavior changes.

    The bridge reads the current neurochemical vector and modulates:
    - Confidence thresholds (more cautious under stress)
    - Weaver evidence scores (sharper focus under arousal)
    - Myelin reinforcement (reward pathway strengthening)
    - System health flags (degradation warnings)
    - Active Inference triggers (foraging urgency)

    Example::

        engine = EmotionEngine()
        bridge = EmotionDecisionBridge(engine)
        adjusted = bridge.modulate_confidence(0.7, engine.state)
        # Under high cortisol: adjusted < 0.7 (more cautious)
    """

    # Thresholds for behavioral triggers
    CORTISOL_HIGH = 40.0       # Stress threshold → lower confidence
    CORTISOL_CRISIS = 60.0     # Extreme stress → aggressive caution
    DOPAMINE_REWARD = 65.0     # Reward threshold → reinforce pathways
    SEROTONIN_LOW = 25.0       # Quality degradation flag
    ENTROPY_HIGH = 0.7         # High uncertainty (normalised 0-1)
    MYELIN_REWARD_INCREMENT = 3  # Myelin boost on reward
    MYELIN_PUNISH_DECREMENT = 2  # Myelin reduction on punishment

    def __init__(self, emotion_engine: EmotionEngine,
                 max_history: int = 500) -> None:
        """Initialise the bridge with a reference to the EmotionEngine.

        Args:
            emotion_engine: The EmotionEngine whose state drives modulation.
            max_history: Maximum decision records to retain.
        """
        self.engine = emotion_engine
        self._history: List[EmotionDecisionRecord] = []
        self._max_history = max_history
        self._last_used_edges: List[Tuple[str, str]] = []
        self._quality_flags: List[Dict] = []

    # ----- confidence modulation ------------------------------------------

    def modulate_confidence(self, base_confidence: float,
                            emotion_state: Optional[NeurochemicalState] = None
                            ) -> float:
        """Adjust a confidence score based on emotional state.

        High cortisol lowers confidence (system becomes more cautious,
        requiring stronger evidence before committing to an answer).
        High dopamine slightly boosts confidence (recent success).
        Low serotonin applies a small penalty (quality concern).

        Args:
            base_confidence: Raw confidence score (0.0 - 1.0).
            emotion_state: Override state; defaults to engine's current state.

        Returns:
            Adjusted confidence, clamped to [0.0, 1.0].
        """
        state = emotion_state or self.engine.state
        adjusted = base_confidence

        # High cortisol → lower confidence (more cautious)
        if state.cortisol > self.CORTISOL_HIGH:
            # Scale: cortisol 40-100 maps to 0.05-0.25 reduction
            stress_factor = (state.cortisol - self.CORTISOL_HIGH) / 60.0
            reduction = 0.05 + stress_factor * 0.20
            adjusted -= reduction
            reason = "cortisol_caution"
        # High dopamine → slight confidence boost
        elif state.dopamine > self.DOPAMINE_REWARD:
            boost = (state.dopamine - self.DOPAMINE_REWARD) / 100.0 * 0.10
            adjusted += boost
            reason = "dopamine_boost"
        else:
            reason = "neutral"

        # Low serotonin penalty (stacks with cortisol)
        if state.serotonin < self.SEROTONIN_LOW:
            serotonin_penalty = (self.SEROTONIN_LOW - state.serotonin) / 50.0 * 0.08
            adjusted -= serotonin_penalty
            if reason == "neutral":
                reason = "serotonin_degradation"
            else:
                reason += "+serotonin_degradation"

        adjusted = max(0.0, min(1.0, adjusted))

        self._record(
            action="modulate_confidence",
            base_value=base_confidence,
            modulated_value=adjusted,
            state=state,
            reason=reason,
        )

        return adjusted

    # ----- weaver score modulation ----------------------------------------

    def modulate_weaver_scores(
        self, scores: List[Tuple[float, float, str]],
        emotion_state: Optional[NeurochemicalState] = None
    ) -> List[Tuple[float, float, str]]:
        """Adjust Weaver evidence scores based on emotional state.

        Scores are (score, tiebreak, sentence) tuples from AlgorithmicWeaver.

        Under stress (high cortisol):
            - Boost high-scoring sentences (focus on strongest evidence)
            - Penalise low-scoring sentences (prune weak leads)
        Under reward (high dopamine):
            - Slight uniform boost (more permissive evidence gathering)
        Under low serotonin:
            - Compress score range (less discrimination = safety mode)

        Args:
            scores: List of (score, tiebreak, sentence) from Weaver.
            emotion_state: Override state; defaults to engine's current state.

        Returns:
            New list of (score, tiebreak, sentence) with adjusted scores.
        """
        if not scores:
            return scores

        state = emotion_state or self.engine.state
        result = []

        for score, tiebreak, sentence in scores:
            adj_score = score

            # High cortisol → sharpen discrimination
            if state.cortisol > self.CORTISOL_HIGH:
                stress_factor = (state.cortisol - self.CORTISOL_HIGH) / 60.0
                if score > 0:
                    # Boost strong evidence under stress
                    adj_score *= (1.0 + stress_factor * 0.3)
                else:
                    # Penalise weak evidence under stress
                    adj_score *= (1.0 + stress_factor * 0.5)

            # High dopamine → slight uniform boost
            if state.dopamine > self.DOPAMINE_REWARD:
                dopamine_factor = (state.dopamine - self.DOPAMINE_REWARD) / 50.0
                adj_score += dopamine_factor * 5.0

            # Low serotonin → compress range toward mean
            if state.serotonin < self.SEROTONIN_LOW:
                serotonin_factor = (self.SEROTONIN_LOW - state.serotonin) / 25.0
                mean_score = sum(s for s, _, _ in scores) / len(scores)
                adj_score = adj_score + (mean_score - adj_score) * serotonin_factor * 0.3

            result.append((adj_score, tiebreak, sentence))

        # Record the aggregate modulation
        if scores:
            orig_top = max(s for s, _, _ in scores)
            new_top = max(s for s, _, _ in result)
            self._record(
                action="modulate_weaver_scores",
                base_value=orig_top,
                modulated_value=new_top,
                state=state,
                reason=f"adjusted_{len(scores)}_sentences",
            )

        return result

    # ----- reward / punishment (myelin reinforcement) ---------------------

    def reward(self, stimulus_name: str = "reward",
               kernel=None) -> Dict[str, float]:
        """Called after a correct answer — reinforces used pathways.

        Applies the 'reward' stimulus to the EmotionEngine (dopamine spike)
        and increases myelin on recently used graph edges.

        Args:
            stimulus_name: Stimulus to apply (default 'reward').
            kernel: KOSKernel reference for myelin adjustment.

        Returns:
            The neurochemical deltas applied.
        """
        deltas = self.engine.apply_stimulus(stimulus_name)

        # Reinforce myelin on last-used edges
        if kernel and self._last_used_edges:
            reinforced = 0
            for source_id, target_id in self._last_used_edges:
                if source_id in kernel.nodes:
                    conn = kernel.nodes[source_id].connections.get(target_id)
                    if isinstance(conn, dict) and 'myelin' in conn:
                        conn['myelin'] += self.MYELIN_REWARD_INCREMENT
                        reinforced += 1
            self._record(
                action="reward_myelin",
                base_value=0.0,
                modulated_value=float(reinforced),
                state=self.engine.state,
                reason=f"reinforced_{reinforced}_edges",
            )

        self._record(
            action="reward_stimulus",
            base_value=0.0,
            modulated_value=deltas.get('dopamine', 0.0),
            state=self.engine.state,
            reason=stimulus_name,
        )

        return deltas

    def punish(self, stimulus_name: str = "social_rejection",
               kernel=None) -> Dict[str, float]:
        """Called after an incorrect answer — weakens used pathways.

        Applies a negative stimulus to the EmotionEngine (cortisol spike,
        dopamine dip) and decreases myelin on recently used edges.

        Args:
            stimulus_name: Stimulus to apply (default 'social_rejection').
            kernel: KOSKernel reference for myelin adjustment.

        Returns:
            The neurochemical deltas applied.
        """
        deltas = self.engine.apply_stimulus(stimulus_name)

        # Weaken myelin on last-used edges
        if kernel and self._last_used_edges:
            weakened = 0
            for source_id, target_id in self._last_used_edges:
                if source_id in kernel.nodes:
                    conn = kernel.nodes[source_id].connections.get(target_id)
                    if isinstance(conn, dict) and 'myelin' in conn:
                        conn['myelin'] = max(0, conn['myelin'] - self.MYELIN_PUNISH_DECREMENT)
                        weakened += 1
            self._record(
                action="punish_myelin",
                base_value=0.0,
                modulated_value=float(weakened),
                state=self.engine.state,
                reason=f"weakened_{weakened}_edges",
            )

        self._record(
            action="punish_stimulus",
            base_value=0.0,
            modulated_value=deltas.get('cortisol', 0.0),
            state=self.engine.state,
            reason=stimulus_name,
        )

        return deltas

    # ----- edge tracking (for reward/punish) ------------------------------

    def track_edges(self, edges: List[Tuple[str, str]]) -> None:
        """Record which graph edges were used in the latest query.

        Call this after propagation so that reward/punish knows which
        pathways to reinforce or weaken.

        Args:
            edges: List of (source_id, target_id) tuples used in query.
        """
        self._last_used_edges = list(edges)

    # ----- system health flags --------------------------------------------

    def check_system_health(self,
                            entropy: float = 0.0) -> List[Dict]:
        """Check emotional state for system health flags.

        Flags raised:
        - Low serotonin (<25): quality degradation warning
        - High cortisol + high entropy: urgent foraging trigger
        - Cortisol crisis (>60): system overload warning

        Args:
            entropy: Current system entropy (0.0 - 1.0), from PCE or
                     similar uncertainty measure.

        Returns:
            List of flag dicts with 'type', 'severity', 'message', 'action'.
        """
        state = self.engine.state
        flags = []

        # Low serotonin → quality degradation
        if state.serotonin < self.SEROTONIN_LOW:
            flags.append({
                'type': 'quality_degradation',
                'severity': 'warning',
                'message': (
                    f"Serotonin critically low ({state.serotonin:.1f}). "
                    "System quality may be degrading — answers may be less "
                    "reliable. Consider rest/consolidation cycle."
                ),
                'action': 'recommend_sleep_cycle',
                'serotonin': state.serotonin,
            })

        # High entropy + high cortisol → urgent foraging
        if entropy > self.ENTROPY_HIGH and state.cortisol > self.CORTISOL_HIGH:
            flags.append({
                'type': 'urgent_foraging',
                'severity': 'critical',
                'message': (
                    f"High uncertainty (entropy={entropy:.2f}) combined with "
                    f"high stress (cortisol={state.cortisol:.1f}). "
                    "Active Inference foraging urgently recommended."
                ),
                'action': 'trigger_foraging',
                'entropy': entropy,
                'cortisol': state.cortisol,
            })

        # Cortisol crisis → system overload
        if state.cortisol > self.CORTISOL_CRISIS:
            flags.append({
                'type': 'system_overload',
                'severity': 'critical',
                'message': (
                    f"Cortisol at crisis level ({state.cortisol:.1f}). "
                    "System is in overload — consider decay/meditation "
                    "stimulus before processing more queries."
                ),
                'action': 'apply_meditation',
                'cortisol': state.cortisol,
            })

        self._quality_flags = flags
        return flags

    def should_forage(self, entropy: float = 0.0) -> bool:
        """Quick check: should Active Inference foraging be triggered?

        Returns True if entropy is high AND cortisol is elevated,
        meaning the system is both uncertain and stressed about it.

        Args:
            entropy: Current uncertainty measure (0.0 - 1.0).

        Returns:
            True if foraging should be urgently triggered.
        """
        state = self.engine.state
        return (entropy > self.ENTROPY_HIGH
                and state.cortisol > self.CORTISOL_HIGH)

    # ----- history / monitoring -------------------------------------------

    def get_history(self, last_n: int = 50) -> List[EmotionDecisionRecord]:
        """Return the last N emotion-modulated decision records.

        Args:
            last_n: Number of records to return.

        Returns:
            List of EmotionDecisionRecord instances.
        """
        return self._history[-last_n:]

    def get_quality_flags(self) -> List[Dict]:
        """Return the most recent system health flags."""
        return list(self._quality_flags)

    def get_stats(self) -> Dict:
        """Return summary statistics of emotion-decision interactions."""
        if not self._history:
            return {
                'total_decisions': 0,
                'avg_modulation': 0.0,
                'reward_count': 0,
                'punish_count': 0,
                'current_emotion': self.engine.current_emotion(),
            }

        modulations = [
            abs(r.modulated_value - r.base_value)
            for r in self._history
            if r.action.startswith('modulate_')
        ]
        reward_count = sum(1 for r in self._history if r.action == 'reward_stimulus')
        punish_count = sum(1 for r in self._history if r.action == 'punish_stimulus')

        return {
            'total_decisions': len(self._history),
            'avg_modulation': sum(modulations) / len(modulations) if modulations else 0.0,
            'reward_count': reward_count,
            'punish_count': punish_count,
            'current_emotion': self.engine.current_emotion(),
            'cortisol': self.engine.state.cortisol,
            'dopamine': self.engine.state.dopamine,
            'serotonin': self.engine.state.serotonin,
        }

    # ----- internal -------------------------------------------------------

    def _record(self, action: str, base_value: float,
                modulated_value: float, state: NeurochemicalState,
                reason: str) -> None:
        """Record a decision to the history log."""
        record = EmotionDecisionRecord(
            timestamp=time.time(),
            action=action,
            base_value=base_value,
            modulated_value=modulated_value,
            emotion_snapshot=state.as_dict(),
            emotion_label=self.engine.current_emotion(),
            reason=reason,
        )
        self._history.append(record)

        # Trim history if over limit
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def __repr__(self) -> str:
        emotion = self.engine.current_emotion()
        decisions = len(self._history)
        return (f"EmotionDecisionBridge(emotion={emotion!r}, "
                f"decisions={decisions})")
