"""
SocialEngine — models social behaviour as game theory.

Implements classical games (Prisoner's Dilemma, Ultimatum, Public Goods),
iterated strategies (Tit-for-Tat, Pavlov), Bayesian trust updates,
Hamilton's kin selection rule, emotional contagion (SIR-like), Shapley
value coalition formation, free-rider detection, and Elo-style hierarchy
ranking.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

Action = Literal["cooperate", "defect"]


# =========================================================================
# Agent
# =========================================================================

@dataclass
class Agent:
    """A social agent with reputation, trust memory, and history.

    Attributes:
        id:                  Unique identifier.
        reputation:          Public reputation score in [0, 1].
        trust_scores:        Per-peer Bayesian trust estimates {peer_id: float}.
        cooperation_history: Ordered list of (round, opponent_id, my_action,
                             opponent_action) tuples.
    """

    id: str
    reputation: float = 0.5
    trust_scores: Dict[str, float] = field(default_factory=dict)
    cooperation_history: List[Tuple[int, str, Action, Action]] = field(
        default_factory=list
    )

    def record(self, round_num: int, opponent_id: str,
               my_action: Action, their_action: Action) -> None:
        """Append an interaction to cooperation history."""
        self.cooperation_history.append(
            (round_num, opponent_id, my_action, their_action)
        )

    def history_with(self, opponent_id: str) -> List[Tuple[Action, Action]]:
        """Return (my_action, their_action) pairs for a specific opponent."""
        return [
            (my, them)
            for _, oid, my, them in self.cooperation_history
            if oid == opponent_id
        ]


# =========================================================================
# Games
# =========================================================================

# --- Prisoner's Dilemma payoff matrix (T > R > P > S, 2R > T + S) --------
#   Canonical values: T=5, R=3, P=1, S=0

_PD_PAYOFFS: Dict[Tuple[Action, Action], Tuple[int, int]] = {
    ("cooperate", "cooperate"): (3, 3),   # mutual cooperation (R, R)
    ("cooperate", "defect"):    (0, 5),   # sucker / temptation (S, T)
    ("defect",    "cooperate"): (5, 0),   # temptation / sucker (T, S)
    ("defect",    "defect"):    (1, 1),   # mutual defection (P, P)
}


def prisoners_dilemma(action_a: Action, action_b: Action) -> Tuple[int, int]:
    """Classic one-shot Prisoner's Dilemma.

    Args:
        action_a: Agent A's action ('cooperate' or 'defect').
        action_b: Agent B's action ('cooperate' or 'defect').

    Returns:
        (payoff_a, payoff_b) from the standard payoff matrix
        T=5, R=3, P=1, S=0.
    """
    return _PD_PAYOFFS[(action_a, action_b)]


def ultimatum_game(proposer_offer: float,
                   responder_threshold: float) -> Tuple[float, float]:
    """Ultimatum Game with a [0, 1] normalised offer.

    The proposer offers a fraction of a unit pie.  The responder accepts
    if the offer meets or exceeds their threshold; otherwise both get 0.

    Args:
        proposer_offer:       Fraction offered (0-1).
        responder_threshold:  Minimum acceptable fraction (0-1).

    Returns:
        (proposer_payoff, responder_payoff).
    """
    if proposer_offer < 0 or proposer_offer > 1:
        raise ValueError("proposer_offer must be in [0, 1]")
    if responder_threshold < 0 or responder_threshold > 1:
        raise ValueError("responder_threshold must be in [0, 1]")

    if proposer_offer >= responder_threshold:
        return (1.0 - proposer_offer, proposer_offer)
    return (0.0, 0.0)


def public_goods(contributions: List[float],
                 multiplier: float = 1.6) -> List[float]:
    """Public Goods Game.

    Each player contributes from an endowment of 1.0.  The total
    contribution is multiplied and shared equally.

    Args:
        contributions: Per-player contributions (each in [0, 1]).
        multiplier:    Public pool multiplier (>1 for social dilemma).

    Returns:
        List of per-player net payoffs (kept + share - contribution).
    """
    n = len(contributions)
    if n == 0:
        return []
    total = sum(contributions)
    share = (total * multiplier) / n
    return [(1.0 - c) + share for c in contributions]


# =========================================================================
# Strategies (for iterated Prisoner's Dilemma)
# =========================================================================

def tit_for_tat(history: List[Tuple[Action, Action]]) -> Action:
    """Tit-for-Tat: cooperate first, then mirror opponent's last move.

    Args:
        history: List of (my_action, opponent_action) from past rounds.

    Returns:
        'cooperate' or 'defect'.
    """
    if not history:
        return "cooperate"
    return history[-1][1]  # opponent's last action


def generous_tit_for_tat(history: List[Tuple[Action, Action]],
                         forgiveness: float = 0.1) -> Action:
    """Generous Tit-for-Tat: like TFT but forgives defections with probability.

    Args:
        history:     Past interaction pairs.
        forgiveness: Probability of cooperating despite opponent defection.

    Returns:
        'cooperate' or 'defect'.
    """
    if not history:
        return "cooperate"
    if history[-1][1] == "defect":
        return "cooperate" if random.random() < forgiveness else "defect"
    return "cooperate"


def always_cooperate(history: List[Tuple[Action, Action]]) -> Action:
    """Always cooperate regardless of history."""
    return "cooperate"


def always_defect(history: List[Tuple[Action, Action]]) -> Action:
    """Always defect regardless of history."""
    return "defect"


def pavlov(history: List[Tuple[Action, Action]]) -> Action:
    """Pavlov (Win-Stay, Lose-Shift).

    Cooperate on the first move.  Thereafter repeat the previous action
    if it yielded a 'win' (mutual cooperation or exploiting a cooperator),
    otherwise switch.

    Args:
        history: Past (my_action, opponent_action) pairs.

    Returns:
        'cooperate' or 'defect'.
    """
    if not history:
        return "cooperate"
    my_last, opp_last = history[-1]
    # A "win" is getting R (3) or T (5) — happens when opponent cooperated
    # or when I defected and they cooperated.  Simplification: win if
    # payoff >= R.
    payoff_a, _ = prisoners_dilemma(my_last, opp_last)
    if payoff_a >= 3:  # R or T — win → stay
        return my_last
    # lose → shift
    return "defect" if my_last == "cooperate" else "cooperate"


# =========================================================================
# Social dynamics
# =========================================================================

def update_trust(agent_a: Agent, agent_b: Agent, outcome: Action) -> float:
    """Bayesian trust update for agent_a's trust in agent_b.

    Uses a Beta-distribution model where 'cooperate' is a success
    observation and 'defect' is a failure.  Trust is the posterior mean
    alpha / (alpha + beta), initialised from the current trust score.

    Args:
        agent_a:  The trusting agent (whose trust_scores are updated).
        agent_b:  The agent being evaluated.
        outcome:  agent_b's observed action ('cooperate' or 'defect').

    Returns:
        Updated trust value in (0, 1).
    """
    prior = agent_a.trust_scores.get(agent_b.id, 0.5)
    # Derive pseudo-counts from prior (assume effective sample size = 10)
    n = 10.0
    alpha = prior * n
    beta = (1.0 - prior) * n

    if outcome == "cooperate":
        alpha += 1.0
    else:
        beta += 1.0

    posterior = alpha / (alpha + beta)
    agent_a.trust_scores[agent_b.id] = posterior
    return posterior


def calculate_reputation(agent: Agent,
                         community: List[Agent]) -> float:
    """Calculate an agent's reputation as the weighted mean of peer trust.

    Each peer's trust score toward *agent* is weighted by the peer's own
    reputation (more reputable peers have more influence).

    Args:
        agent:     The agent whose reputation is being calculated.
        community: All agents (may include *agent* itself, which is skipped).

    Returns:
        Reputation in [0, 1].  Returns 0.5 if no peers have an opinion.
    """
    total_weight = 0.0
    weighted_trust = 0.0
    for peer in community:
        if peer.id == agent.id:
            continue
        if agent.id in peer.trust_scores:
            w = max(peer.reputation, 0.01)  # floor to avoid zero weight
            weighted_trust += peer.trust_scores[agent.id] * w
            total_weight += w
    if total_weight == 0:
        return 0.5
    new_rep = weighted_trust / total_weight
    agent.reputation = new_rep
    return new_rep


def kin_selection(relatedness: float, benefit: float, cost: float) -> bool:
    """Hamilton's rule: altruism evolves when rB > C.

    Args:
        relatedness: Coefficient of relatedness r (0-1).
        benefit:     Fitness benefit to the recipient.
        cost:        Fitness cost to the altruist.

    Returns:
        True if the altruistic act is favoured by kin selection.
    """
    return relatedness * benefit > cost


def emotional_contagion(source_emotion: float,
                        target_state: float,
                        susceptibility: float = 0.3,
                        recovery_rate: float = 0.05) -> float:
    """SIR-inspired emotional contagion step.

    Models emotion as a scalar intensity in [0, 1].  Each time step the
    target is "infected" proportionally to the gap between source and
    target, modulated by susceptibility, and simultaneously "recovers"
    toward a neutral baseline of 0.5.

    Args:
        source_emotion:  Source agent's emotional intensity (0-1).
        target_state:    Target agent's current emotional intensity (0-1).
        susceptibility:  How easily the target is influenced (0-1).
        recovery_rate:   Rate of return toward 0.5 neutral baseline.

    Returns:
        Updated target emotional intensity, clamped to [0, 1].
    """
    infection = susceptibility * (source_emotion - target_state)
    recovery = recovery_rate * (0.5 - target_state)
    new_state = target_state + infection + recovery
    return max(0.0, min(1.0, new_state))


# =========================================================================
# Group dynamics
# =========================================================================

def form_coalition(agents: List[Agent],
                   task_value: float) -> Dict[str, float]:
    """Allocate task_value among agents using the Shapley value.

    The marginal contribution of each agent to every possible coalition
    is computed.  Agent value is approximated by reputation (higher
    reputation = more marginal value in a coalition).

    Args:
        agents:     Participating agents.
        task_value: Total value to be divided.

    Returns:
        {agent_id: share} dict summing to *task_value*.
    """
    n = len(agents)
    if n == 0:
        return {}
    if n == 1:
        return {agents[0].id: task_value}

    # Coalition value function: v(S) = task_value * (sum of reputations
    # in S / sum of all reputations) — simple additive model.
    total_rep = sum(a.reputation for a in agents) or 1.0

    def coalition_value(member_ids: set) -> float:
        rep = sum(a.reputation for a in agents if a.id in member_ids)
        return task_value * (rep / total_rep)

    shapley: Dict[str, float] = {a.id: 0.0 for a in agents}
    agent_ids = [a.id for a in agents]

    for i, agent in enumerate(agents):
        # Enumerate all subsets not containing agent
        others = [aid for aid in agent_ids if aid != agent.id]
        m = len(others)
        for size in range(0, m + 1):
            for combo in combinations(others, size):
                s = set(combo)
                s_with = s | {agent.id}
                marginal = coalition_value(s_with) - coalition_value(s)
                # Shapley weight: |S|!(n-|S|-1)! / n!
                weight = (
                    math.factorial(len(s))
                    * math.factorial(n - len(s) - 1)
                    / math.factorial(n)
                )
                shapley[agent.id] += weight * marginal

    return shapley


def detect_free_riders(contributions: Dict[str, float],
                       threshold: float = 0.5) -> List[str]:
    """Identify agents contributing below *threshold* fraction of the mean.

    Args:
        contributions: {agent_id: contribution_amount}.
        threshold:     Fraction of the mean below which an agent is a
                       free rider (0-1, default 0.5 = half the mean).

    Returns:
        List of free-rider agent IDs, sorted by contribution ascending.
    """
    if not contributions:
        return []
    mean_c = sum(contributions.values()) / len(contributions)
    cutoff = mean_c * threshold
    riders = [aid for aid, c in contributions.items() if c < cutoff]
    riders.sort(key=lambda aid: contributions[aid])
    return riders


def hierarchy_rank(agents: List[Agent],
                   k: float = 32.0) -> List[Tuple[str, float]]:
    """Compute Elo-like rankings from pairwise interaction history.

    Every recorded interaction where one agent cooperated and the other
    defected is treated as a "win" for the defector (dominance) in a
    standard Elo update.  Mutual cooperation/defection is a draw.

    Args:
        agents: All agents with populated cooperation_history.
        k:      Elo K-factor (sensitivity to individual results).

    Returns:
        List of (agent_id, elo_rating) sorted descending by rating.
    """
    ratings: Dict[str, float] = {a.id: 1000.0 for a in agents}

    # Collect all interactions
    seen: set = set()
    for agent in agents:
        for rnd, opp_id, my_act, opp_act in agent.cooperation_history:
            key = (min(agent.id, opp_id), max(agent.id, opp_id), rnd)
            if key in seen:
                continue
            seen.add(key)

            ra = ratings[agent.id]
            rb = ratings.get(opp_id, 1000.0)
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            eb = 1.0 - ea

            if my_act == "defect" and opp_act == "cooperate":
                sa, sb = 1.0, 0.0  # agent dominates
            elif my_act == "cooperate" and opp_act == "defect":
                sa, sb = 0.0, 1.0  # opponent dominates
            else:
                sa, sb = 0.5, 0.5  # draw

            ratings[agent.id] = ra + k * (sa - ea)
            if opp_id in ratings:
                ratings[opp_id] = rb + k * (sb - eb)

    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    return ranked


# =========================================================================
# SocialEngine facade
# =========================================================================

class SocialEngine:
    """Facade coordinating agents, games, trust, and group dynamics.

    Example::

        engine = SocialEngine()
        a = engine.create_agent("alice")
        b = engine.create_agent("bob")
        engine.play_prisoners_dilemma(a, b, "cooperate", "defect")
        print(a.trust_scores)
        print(engine.rankings())
    """

    def __init__(self) -> None:
        self.agents: Dict[str, Agent] = {}
        self._round: int = 0

    def create_agent(self, agent_id: str,
                     reputation: float = 0.5) -> Agent:
        """Create and register a new Agent."""
        agent = Agent(id=agent_id, reputation=reputation)
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        """Retrieve a registered agent by ID."""
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' not found")
        return self.agents[agent_id]

    def play_prisoners_dilemma(
        self,
        agent_a: Agent,
        agent_b: Agent,
        action_a: Action,
        action_b: Action,
    ) -> Tuple[int, int]:
        """Play one round of Prisoner's Dilemma between two agents.

        Records history, updates trust, and advances the round counter.

        Returns:
            (payoff_a, payoff_b).
        """
        payoff = prisoners_dilemma(action_a, action_b)
        self._round += 1
        agent_a.record(self._round, agent_b.id, action_a, action_b)
        agent_b.record(self._round, agent_a.id, action_b, action_a)
        update_trust(agent_a, agent_b, action_b)
        update_trust(agent_b, agent_a, action_a)
        return payoff

    def play_ultimatum(
        self,
        proposer: Agent,
        responder: Agent,
        offer: float,
        threshold: float,
    ) -> Tuple[float, float]:
        """Play one Ultimatum Game round.

        Returns:
            (proposer_payoff, responder_payoff).
        """
        payoff = ultimatum_game(offer, threshold)
        outcome: Action = "cooperate" if payoff[1] > 0 else "defect"
        update_trust(proposer, responder, outcome)
        update_trust(responder, proposer, "cooperate" if offer >= 0.4 else "defect")
        return payoff

    def play_public_goods(
        self,
        agent_contributions: Dict[str, float],
        multiplier: float = 1.6,
    ) -> Dict[str, float]:
        """Play a Public Goods Game and return per-agent payoffs.

        Args:
            agent_contributions: {agent_id: contribution}.
            multiplier:          Pool multiplier.

        Returns:
            {agent_id: net_payoff}.
        """
        ids = list(agent_contributions.keys())
        contribs = [agent_contributions[aid] for aid in ids]
        payoffs = public_goods(contribs, multiplier)
        return dict(zip(ids, payoffs))

    def update_reputations(self) -> Dict[str, float]:
        """Recalculate reputation for every registered agent.

        Returns:
            {agent_id: new_reputation}.
        """
        community = list(self.agents.values())
        return {
            a.id: calculate_reputation(a, community)
            for a in community
        }

    def rankings(self, k: float = 32.0) -> List[Tuple[str, float]]:
        """Return Elo-style hierarchy rankings for all agents.

        Returns:
            Sorted list of (agent_id, elo_rating) descending.
        """
        return hierarchy_rank(list(self.agents.values()), k=k)

    def find_free_riders(
        self,
        contributions: Dict[str, float],
        threshold: float = 0.5,
    ) -> List[str]:
        """Identify free riders from a contributions dict.

        Returns:
            List of agent IDs contributing below *threshold* of the mean.
        """
        return detect_free_riders(contributions, threshold)

    def form_coalition(self, agent_ids: List[str],
                       task_value: float) -> Dict[str, float]:
        """Allocate *task_value* among agents via Shapley values.

        Returns:
            {agent_id: share}.
        """
        agents = [self.get_agent(aid) for aid in agent_ids]
        return form_coalition(agents, task_value)

    def __repr__(self) -> str:
        return (
            f"SocialEngine(agents={len(self.agents)}, "
            f"rounds={self._round})"
        )
