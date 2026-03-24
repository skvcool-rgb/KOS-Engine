"""
EmotionEngine — models emotions as neurochemical state vectors.

NOT conscious. Behaviorally computable. Maps measurable neurochemical
concentrations to named emotional states using clinical thresholds,
half-life decay kinetics, and stimulus-response profiles derived from
psychopharmacology literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Neurochemical state vector
# ---------------------------------------------------------------------------

@dataclass
class NeurochemicalState:
    """Instantaneous neurochemical concentrations.

    Each field models a real neuroactive substance with physiological
    units where applicable. Values are clamped to their valid range
    on every mutation.

    Attributes:
        cortisol:      Stress hormone (0-100 ug/dL).
        adrenaline:    Epinephrine, fight-or-flight (0-500 pg/mL).
        dopamine:      Reward / motivation (0-100, normalised index).
        serotonin:     Mood stabilisation (0-100, normalised index).
        oxytocin:      Social bonding (0-100, normalised index).
        testosterone:  Dominance / aggression (0-100, normalised index).
        gaba:          Inhibitory calm (0-100, normalised index).
        endorphin:     Endogenous analgesia (0-100, normalised index).
    """

    cortisol: float = 15.0
    adrenaline: float = 50.0
    dopamine: float = 50.0
    serotonin: float = 50.0
    oxytocin: float = 30.0
    testosterone: float = 50.0
    gaba: float = 50.0
    endorphin: float = 20.0

    # Valid ranges per chemical — (min, max).
    _RANGES: dict = field(default_factory=lambda: {
        "cortisol":     (0.0, 100.0),
        "adrenaline":   (0.0, 500.0),
        "dopamine":     (0.0, 100.0),
        "serotonin":    (0.0, 100.0),
        "oxytocin":     (0.0, 100.0),
        "testosterone": (0.0, 100.0),
        "gaba":         (0.0, 100.0),
        "endorphin":    (0.0, 100.0),
    }, repr=False)

    def clamp(self) -> None:
        """Clamp every field to its physiological range."""
        for name, (lo, hi) in self._RANGES.items():
            setattr(self, name, max(lo, min(hi, getattr(self, name))))

    def as_dict(self) -> Dict[str, float]:
        """Return chemical concentrations as a plain dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "_RANGES"
        }

    def copy(self) -> "NeurochemicalState":
        """Return an independent copy of this state."""
        return NeurochemicalState(**self.as_dict())


# ---------------------------------------------------------------------------
# Baseline — resting healthy adult
# ---------------------------------------------------------------------------

BASELINE = NeurochemicalState(
    cortisol=15.0,       # ~15 ug/dL morning resting cortisol
    adrenaline=50.0,     # ~50 pg/mL resting epinephrine
    dopamine=50.0,       # midpoint normalised index
    serotonin=50.0,      # midpoint normalised index
    oxytocin=30.0,       # low-moderate resting level
    testosterone=50.0,   # midpoint normalised index
    gaba=50.0,           # midpoint normalised index
    endorphin=20.0,      # low resting endorphin
)

# ---------------------------------------------------------------------------
# Half-lives (minutes) — how fast each chemical returns toward baseline
# ---------------------------------------------------------------------------

HALF_LIVES: Dict[str, float] = {
    "cortisol":     60.0,    # ~60-90 min biological half-life
    "adrenaline":   2.0,     # ~2 min plasma half-life
    "dopamine":     30.0,    # synaptic clearance modelled at ~30 min
    "serotonin":    45.0,    # slower turnover
    "oxytocin":     20.0,    # ~20 min plasma half-life
    "testosterone": 120.0,   # slow diurnal variation
    "gaba":         40.0,    # moderate clearance
    "endorphin":    25.0,    # ~20-30 min
}

# ---------------------------------------------------------------------------
# Stimulus → neurochemical delta map
# ---------------------------------------------------------------------------

STIMULUS_RESPONSES: Dict[str, Dict[str, float]] = {
    "threat": {
        "cortisol": +30.0,
        "adrenaline": +250.0,
    },
    "reward": {
        "dopamine": +40.0,
        "serotonin": +15.0,
    },
    "social_bond": {
        "oxytocin": +40.0,
        "serotonin": +10.0,
    },
    "pain": {
        "cortisol": +20.0,
        "adrenaline": +100.0,
        "endorphin": +50.0,
    },
    "exercise": {
        "endorphin": +40.0,
        "dopamine": +20.0,
        "cortisol": -10.0,
    },
    "sleep_deprivation": {
        "cortisol": +25.0,
        "serotonin": -30.0,
        "dopamine": -20.0,
    },
    "sexual_arousal": {
        "dopamine": +35.0,
        "oxytocin": +30.0,
        "testosterone": +20.0,
    },
    "social_rejection": {
        "cortisol": +25.0,
        "serotonin": -20.0,
        "oxytocin": -15.0,
        "dopamine": -15.0,
    },
    "meditation": {
        "gaba": +30.0,
        "cortisol": -15.0,
        "serotonin": +10.0,
    },
    "conflict": {
        "cortisol": +35.0,
        "adrenaline": +180.0,
        "testosterone": +25.0,
    },
    "music_pleasure": {
        "dopamine": +25.0,
        "endorphin": +15.0,
        "serotonin": +10.0,
    },
    "starvation": {
        "cortisol": +30.0,
        "dopamine": -25.0,
        "serotonin": -15.0,
        "gaba": -20.0,
    },
    "laughter": {
        "endorphin": +30.0,
        "dopamine": +15.0,
        "cortisol": -10.0,
    },
    "grief_event": {
        "serotonin": -35.0,
        "oxytocin": -20.0,
        "cortisol": +30.0,
        "dopamine": -20.0,
    },
    "novelty": {
        "dopamine": +30.0,
        "adrenaline": +60.0,
    },
}

# ---------------------------------------------------------------------------
# Drug → neurochemical delta map
# ---------------------------------------------------------------------------

DRUG_EFFECTS: Dict[str, Dict[str, float]] = {
    "ssri": {
        "serotonin": +30.0,
        "dopamine": +5.0,
    },
    "anxiolytic": {
        "gaba": +40.0,
        "cortisol": -15.0,
    },
    "beta_blocker": {
        "adrenaline": -150.0,
        "cortisol": -10.0,
    },
    "opioid": {
        "endorphin": +60.0,
        "dopamine": +30.0,
        "gaba": +10.0,
    },
    "stimulant": {
        "dopamine": +45.0,
        "adrenaline": +120.0,
        "cortisol": +10.0,
    },
    "antipsychotic": {
        "dopamine": -30.0,
        "serotonin": +10.0,
    },
    "alcohol": {
        "gaba": +30.0,
        "dopamine": +15.0,
        "serotonin": -10.0,
        "cortisol": +10.0,
    },
    "caffeine": {
        "adrenaline": +80.0,
        "dopamine": +10.0,
        "cortisol": +5.0,
    },
    "thc": {
        "dopamine": +20.0,
        "gaba": +15.0,
        "cortisol": -5.0,
        "serotonin": +5.0,
    },
    "mdma": {
        "serotonin": +50.0,
        "dopamine": +35.0,
        "oxytocin": +40.0,
    },
}


# ---------------------------------------------------------------------------
# EmotionEngine
# ---------------------------------------------------------------------------

class EmotionEngine:
    """Neurochemical emotion model.

    Maintains a mutable ``NeurochemicalState`` and exposes methods to
    apply stimuli, decay concentrations toward baseline, classify the
    current emotional label, and run simple clinical diagnostics.

    Example::

        engine = EmotionEngine()
        engine.apply_stimulus("threat")
        print(engine.current_emotion())   # 'fear'
        engine.decay(minutes=10)
        print(engine.state.as_dict())
    """

    def __init__(self, state: Optional[NeurochemicalState] = None) -> None:
        self.state = state.copy() if state else BASELINE.copy()
        self._history: List[Tuple[str, Dict[str, float]]] = []

    # ----- stimulus application -------------------------------------------

    def apply_stimulus(self, stimulus_name: str) -> Dict[str, float]:
        """Apply a named stimulus and return the delta that was applied.

        After applying the delta the state is clamped to valid ranges.

        Args:
            stimulus_name: Key in ``STIMULUS_RESPONSES``.

        Returns:
            Dict of chemical deltas that were actually applied.

        Raises:
            ValueError: If *stimulus_name* is not recognised.
        """
        if stimulus_name not in STIMULUS_RESPONSES:
            raise ValueError(
                f"Unknown stimulus '{stimulus_name}'. "
                f"Known stimuli: {sorted(STIMULUS_RESPONSES)}"
            )
        deltas = STIMULUS_RESPONSES[stimulus_name]
        for chem, delta in deltas.items():
            current = getattr(self.state, chem)
            setattr(self.state, chem, current + delta)
        self.state.clamp()
        self._history.append((stimulus_name, deltas))
        return deltas

    # ----- half-life decay ------------------------------------------------

    def decay(self, minutes: float) -> None:
        """Decay all neurochemicals toward baseline over *minutes*.

        Uses exponential half-life decay::

            value(t) = baseline + (current - baseline) * 0.5^(t / half_life)

        Args:
            minutes: Elapsed time in minutes (must be >= 0).
        """
        if minutes < 0:
            raise ValueError("minutes must be >= 0")
        baseline = BASELINE.as_dict()
        for chem, hl in HALF_LIVES.items():
            current = getattr(self.state, chem)
            base = baseline[chem]
            decayed = base + (current - base) * math.pow(0.5, minutes / hl)
            setattr(self.state, chem, decayed)
        self.state.clamp()

    # ----- emotion classification -----------------------------------------

    def current_emotion(self) -> str:
        """Map the neurochemical vector to a single named emotion.

        Emotions are tested in priority order; the first matching rule
        wins.  If no rule matches the default label is ``'neutral'``.

        Returns:
            One of: ``'euphoria'``, ``'fear'``, ``'anger'``, ``'love'``,
            ``'joy'``, ``'depression'``, ``'grief'``, ``'anxiety'``,
            ``'calm'``, ``'neutral'``.
        """
        s = self.state
        # Order matters — most specific / extreme states first.
        if s.dopamine > 85 and s.endorphin > 60:
            return "euphoria"
        if s.cortisol > 40 and s.adrenaline > 200:
            return "fear"
        if s.cortisol > 50 and s.testosterone > 70 and s.adrenaline > 150:
            return "anger"
        if s.oxytocin > 70 and s.dopamine > 50:
            return "love"
        if s.dopamine > 70 and s.serotonin > 60:
            return "joy"
        if s.serotonin < 20 and s.dopamine < 20:
            return "depression"
        if s.serotonin < 25 and s.oxytocin < 20 and s.cortisol > 40:
            return "grief"
        if s.cortisol > 35 and s.gaba < 30:
            return "anxiety"
        if s.gaba > 70 and s.cortisol < 15:
            return "calm"
        return "neutral"

    # ----- clinical diagnostics -------------------------------------------

    def diagnose(self) -> List[str]:
        """Return a list of clinical-range deviation warnings.

        Each entry is a human-readable string describing a deviation
        that would be clinically noteworthy (e.g. depression risk,
        anxiety risk, HPA axis dysregulation).

        Returns:
            List of diagnostic strings. Empty if within normal range.
        """
        warnings: List[str] = []
        s = self.state

        if s.serotonin < 20 and s.dopamine < 20:
            warnings.append(
                "DEPRESSION RISK: critically low serotonin "
                f"({s.serotonin:.1f}) and dopamine ({s.dopamine:.1f})"
            )
        if s.cortisol > 50:
            warnings.append(
                f"HPA AXIS DYSREGULATION: cortisol at {s.cortisol:.1f} ug/dL "
                "(Cushing-range; chronic stress or pathology)"
            )
        if s.cortisol > 35 and s.gaba < 30:
            warnings.append(
                "ANXIETY RISK: elevated cortisol "
                f"({s.cortisol:.1f}) with low GABAergic inhibition ({s.gaba:.1f})"
            )
        if s.adrenaline > 350:
            warnings.append(
                f"ADRENERGIC CRISIS: adrenaline at {s.adrenaline:.1f} pg/mL "
                "(pheochromocytoma-range; acute danger)"
            )
        if s.serotonin > 90 and s.dopamine > 80:
            warnings.append(
                "SEROTONIN SYNDROME RISK: serotonin "
                f"({s.serotonin:.1f}) and dopamine ({s.dopamine:.1f}) both critically high"
            )
        if s.dopamine > 90:
            warnings.append(
                f"DOPAMINERGIC EXCESS: dopamine at {s.dopamine:.1f} "
                "(mania / psychosis risk)"
            )
        if s.gaba > 85:
            warnings.append(
                f"EXCESSIVE SEDATION: GABA at {s.gaba:.1f} "
                "(respiratory depression risk if pharmacological)"
            )
        if s.endorphin > 85:
            warnings.append(
                f"OPIOIDERGIC EXCESS: endorphin at {s.endorphin:.1f} "
                "(tolerance / dependence trajectory)"
            )
        if s.serotonin < 25 and s.oxytocin < 20 and s.cortisol > 40:
            warnings.append(
                "GRIEF/BEREAVEMENT PATTERN: low serotonin, low oxytocin, "
                "elevated cortisol — monitor for prolonged grief disorder"
            )
        if s.testosterone > 85 and s.cortisol > 40:
            warnings.append(
                "AGGRESSION RISK: high testosterone "
                f"({s.testosterone:.1f}) with elevated cortisol ({s.cortisol:.1f})"
            )
        return warnings

    # ----- drug effects ---------------------------------------------------

    def drug_effect(self, drug_name: str) -> Dict[str, float]:
        """Apply a pharmacological agent and return the delta.

        Args:
            drug_name: Key in ``DRUG_EFFECTS`` (case-insensitive).

        Returns:
            Dict of chemical deltas that were applied.

        Raises:
            ValueError: If *drug_name* is not recognised.
        """
        key = drug_name.lower()
        if key not in DRUG_EFFECTS:
            raise ValueError(
                f"Unknown drug '{drug_name}'. "
                f"Known drugs: {sorted(DRUG_EFFECTS)}"
            )
        deltas = DRUG_EFFECTS[key]
        for chem, delta in deltas.items():
            current = getattr(self.state, chem)
            setattr(self.state, chem, current + delta)
        self.state.clamp()
        self._history.append((f"drug:{key}", deltas))
        return deltas

    # ----- utilities ------------------------------------------------------

    def reset(self) -> None:
        """Reset the neurochemical state to baseline."""
        self.state = BASELINE.copy()
        self._history.clear()

    def history(self) -> List[Tuple[str, Dict[str, float]]]:
        """Return the ordered list of (event, deltas) applied so far."""
        return list(self._history)

    def __repr__(self) -> str:
        emotion = self.current_emotion()
        return f"EmotionEngine(emotion={emotion!r}, state={self.state.as_dict()})"
