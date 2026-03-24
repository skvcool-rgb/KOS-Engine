"""
KOS V5.0 — Proactive Attention Controller (Phase 2 Complete).

The missing piece between "reactive Q&A" and "autonomous intelligence."
Instead of waiting for user queries, this module generates its own
goals based on three biological drives:

1. CURIOSITY — Super-hubs with few sub-topics should be expanded.
   "I know Toronto has 50 connections but only 2 sub-categories.
    I should learn more about its geography, culture, economy."

2. ANTICIPATION — If the user keeps asking about topic X, pre-load
   related topics Y and Z before they ask.
   "User asked about Toronto climate, then Toronto weather.
    They'll probably ask about Toronto transportation next."

3. STALENESS — Knowledge decays. Nodes that were wired long ago
   and never accessed should be refreshed or pruned.
   "The perovskite cost data is from tick 5. We're at tick 500.
    That fact might be outdated."

The Attention Controller runs as a daemon alongside the existing
maintenance cycle. It outputs a ranked list of "attention goals"
that the Forager can act on.
"""

import time
import math
from collections import Counter, defaultdict


class AttentionGoal:
    """A single self-generated goal with priority and rationale."""
    __slots__ = ['goal_type', 'target', 'priority', 'rationale', 'query']

    def __init__(self, goal_type: str, target: str, priority: float,
                 rationale: str, query: str):
        self.goal_type = goal_type      # "curiosity" | "anticipation" | "staleness"
        self.target = target            # node_id or topic string
        self.priority = priority        # 0.0 to 100.0
        self.rationale = rationale      # human-readable reason
        self.query = query              # search query for the forager

    def __repr__(self):
        return (f"Goal({self.goal_type}, priority={self.priority:.1f}, "
                f"query='{self.query}')")


class AttentionController:
    """
    The proactive brain. Generates attention goals without user input.

    Biological analog: the Default Mode Network (DMN) — the brain
    regions active when you're NOT focused on a task. The DMN
    consolidates memories, plans future actions, and daydreams.
    """

    def __init__(self, kernel, lexicon, max_goals: int = 10):
        self.kernel = kernel
        self.lexicon = lexicon
        self.max_goals = max_goals

        # Track user query history for anticipation
        self.query_history = []         # list of (tick, [seed_ids])
        self.topic_frequency = Counter()  # node_id -> query count

        # Track when nodes were last accessed or wired
        self.node_birth_tick = {}       # node_id -> tick when created
        self.node_last_access = {}      # node_id -> tick when last queried

    def record_query(self, seed_ids: list, current_tick: int):
        """
        Called after every user query. Builds the interaction history
        that drives anticipation.
        """
        self.query_history.append((current_tick, seed_ids))
        for sid in seed_ids:
            self.topic_frequency[sid] += 1
            self.node_last_access[sid] = current_tick

        # Keep history bounded (last 100 queries)
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]

    def record_node_creation(self, node_id: str, current_tick: int):
        """Track when a node was born (for staleness detection)."""
        if node_id not in self.node_birth_tick:
            self.node_birth_tick[node_id] = current_tick

    # ── Drive 1: CURIOSITY ───────────────────────────────────────

    def _generate_curiosity_goals(self) -> list:
        """
        Find super-hubs that are broad but shallow.

        A node with 20+ connections but whose neighbors mostly
        connect to different topics = a hub that needs sub-topic
        expansion.

        Biological analog: "I know this word but can't explain it
        in detail" = the tip-of-the-tongue state.
        """
        goals = []

        for nid, node in self.kernel.nodes.items():
            conn_count = len(node.connections)
            if conn_count < 5:
                continue  # Not a hub

            # Count unique "neighbor neighborhoods" — how many
            # distinct clusters does this hub connect to?
            neighbor_topics = set()
            for target_id in list(node.connections.keys())[:50]:
                if target_id in self.kernel.nodes:
                    # Use the target's top connection as a proxy for its "topic"
                    target_node = self.kernel.nodes[target_id]
                    if target_node.connections:
                        top_neighbor = max(target_node.connections.items(),
                                           key=lambda x: x[1] if not isinstance(x[1], dict)
                                           else x[1].get('w', 0))
                        neighbor_topics.add(top_neighbor[0])

            # Diversity ratio: how many unique topics vs total connections
            if neighbor_topics:
                diversity = len(neighbor_topics) / min(conn_count, 50)
            else:
                diversity = 0.0

            # High connections + LOW diversity = broad but shallow
            # This hub knows many things superficially
            if conn_count >= 10 and diversity < 0.3:
                word = self.lexicon.get_word(nid) if self.lexicon else nid
                priority = conn_count * (1.0 - diversity) * 2.0

                goals.append(AttentionGoal(
                    goal_type="curiosity",
                    target=nid,
                    priority=min(priority, 100.0),
                    rationale=(f"'{word}' has {conn_count} connections but "
                               f"low topic diversity ({diversity:.0%}). "
                               f"Needs deeper sub-topic knowledge."),
                    query=f"{word} detailed overview subtopics"
                ))

        return goals

    # ── Drive 2: ANTICIPATION ────────────────────────────────────

    def _generate_anticipation_goals(self) -> list:
        """
        Predict what the user will ask next based on query patterns.

        Strategy: If the user queried [A, B] and then [A, C],
        find other nodes connected to A that haven't been queried.
        Pre-load them.

        Biological analog: "They keep asking about Toronto.
        They'll probably want to know about nearby cities next."
        """
        goals = []

        if len(self.query_history) < 2:
            return goals

        # Find the most frequently queried topics
        top_topics = self.topic_frequency.most_common(5)

        for topic_id, freq in top_topics:
            if freq < 2 or topic_id not in self.kernel.nodes:
                continue

            word = self.lexicon.get_word(topic_id) if self.lexicon else topic_id

            # Find neighbors of this topic that have NEVER been queried
            queried_set = set()
            for _, seeds in self.query_history:
                queried_set.update(seeds)

            node = self.kernel.nodes[topic_id]
            unqueried_neighbors = []
            for neighbor_id in node.connections:
                if neighbor_id not in queried_set:
                    neighbor_word = (self.lexicon.get_word(neighbor_id)
                                    if self.lexicon else neighbor_id)
                    unqueried_neighbors.append((neighbor_id, neighbor_word))

            # If this hot topic has unexplored neighbors, anticipate
            if unqueried_neighbors and len(unqueried_neighbors) > 3:
                # Pick the most connected unqueried neighbor
                best = max(unqueried_neighbors,
                           key=lambda x: len(self.kernel.nodes.get(x[0],
                                            type('', (), {'connections': {}})
                                            ).connections))

                priority = freq * 15.0  # More queries = higher anticipation

                goals.append(AttentionGoal(
                    goal_type="anticipation",
                    target=best[0],
                    priority=min(priority, 80.0),
                    rationale=(f"User queried '{word}' {freq} times. "
                               f"'{best[1]}' is connected but unexplored. "
                               f"Pre-loading."),
                    query=f"{word} {best[1]}"
                ))

        return goals

    # ── Drive 3: STALENESS ───────────────────────────────────────

    def _generate_staleness_goals(self) -> list:
        """
        Find knowledge that might be outdated.

        Nodes wired many ticks ago that haven't been accessed
        recently may contain stale information.

        Biological analog: "I learned this in school 20 years ago.
        Is it still true?"
        """
        goals = []
        current_tick = self.kernel.current_tick

        if current_tick < 10:
            return goals  # Too early to judge staleness

        for nid, birth_tick in self.node_birth_tick.items():
            if nid not in self.kernel.nodes:
                continue

            age = current_tick - birth_tick
            last_access = self.node_last_access.get(nid, birth_tick)
            dormancy = current_tick - last_access

            # Staleness = old AND unused
            if age > 20 and dormancy > 15:
                word = self.lexicon.get_word(nid) if self.lexicon else nid
                conn_count = len(self.kernel.nodes[nid].connections)

                # Only flag important nodes (with connections)
                if conn_count < 3:
                    continue

                priority = (age * 0.5 + dormancy * 1.0) * (conn_count / 10.0)

                goals.append(AttentionGoal(
                    goal_type="staleness",
                    target=nid,
                    priority=min(priority, 60.0),
                    rationale=(f"'{word}' was wired at tick {birth_tick}, "
                               f"last accessed at tick {last_access}. "
                               f"Age={age}, dormancy={dormancy}. "
                               f"May contain outdated information."),
                    query=f"{word} latest information current"
                ))

        return goals

    # ── Main Entry Point ─────────────────────────────────────────

    def generate_goals(self, verbose: bool = True) -> list:
        """
        Run all three drives and return a prioritized goal list.

        This is what makes the system proactive — it doesn't wait
        for a user query. It looks at its own knowledge graph and
        decides what to learn next.
        """
        if verbose:
            print("\n[ATTENTION] Proactive Attention Controller scanning...")

        t0 = time.perf_counter()

        curiosity_goals = self._generate_curiosity_goals()
        anticipation_goals = self._generate_anticipation_goals()
        staleness_goals = self._generate_staleness_goals()

        all_goals = curiosity_goals + anticipation_goals + staleness_goals
        all_goals.sort(key=lambda g: g.priority, reverse=True)
        all_goals = all_goals[:self.max_goals]

        elapsed = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"[ATTENTION] Scan complete in {elapsed:.1f}ms")
            print(f"   [?] Curiosity goals:    {len(curiosity_goals)}")
            print(f"   [>] Anticipation goals:  {len(anticipation_goals)}")
            print(f"   [!] Staleness goals:     {len(staleness_goals)}")

            if all_goals:
                print(f"   Top goals:")
                for g in all_goals[:5]:
                    icon = {"curiosity": "?", "anticipation": ">",
                            "staleness": "!"}[g.goal_type]
                    print(f"     [{icon}] P={g.priority:.0f} | {g.rationale}")

        return all_goals

    def act_on_goals(self, forager, max_actions: int = 3,
                     verbose: bool = True) -> dict:
        """
        Execute the top goals using the WebForager.

        This is the autonomous action loop: the system decides
        what to learn, then actually learns it.
        """
        goals = self.generate_goals(verbose=verbose)

        if not goals:
            if verbose:
                print("[ATTENTION] No goals generated. Graph is healthy.")
            return {"goals_generated": 0, "actions_taken": 0,
                    "concepts_acquired": 0}

        actions_taken = 0
        concepts_acquired = 0

        for goal in goals[:max_actions]:
            if verbose:
                print(f"\n[ATTENTION] Acting on {goal.goal_type} goal: "
                      f"'{goal.query}'")

            new_nodes = forager.forage_query(goal.query, verbose=verbose)
            concepts_acquired += new_nodes
            actions_taken += 1

            if verbose:
                print(f"[ATTENTION] +{new_nodes} concepts from "
                      f"{goal.goal_type} goal")

        report = {
            "goals_generated": len(goals),
            "actions_taken": actions_taken,
            "concepts_acquired": concepts_acquired,
            "goals": goals,
        }

        if verbose:
            print(f"\n[ATTENTION] Session complete: "
                  f"{actions_taken} actions, "
                  f"+{concepts_acquired} concepts acquired")

        return report
