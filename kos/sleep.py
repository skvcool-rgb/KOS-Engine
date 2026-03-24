"""
KOS V6.1 — Sleep Consolidation Cycle.

Biological brains consolidate memories during sleep:
    - Strengthen frequently-used pathways (high myelin)
    - Weaken rarely-used pathways (low myelin)
    - Prune dead connections (weight < threshold)
    - Merge near-duplicate nodes (Jaccard similarity)
    - Infer new connections via triadic closure

The SleepCycle runs these operations on the KOS kernel graph,
producing a cleaner, faster, more accurate knowledge structure.

Can be scheduled to auto-run every N hours via a daemon thread,
mimicking the biological sleep-wake cycle.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Sleep Consolidation Cycle
# ---------------------------------------------------------------------------

class SleepCycle:
    """Runs graph consolidation operations analogous to biological sleep.

    During a consolidation cycle:
    1. High-myelin edges are strengthened (multiply weight by 1.05)
    2. Low-myelin edges are weakened (multiply weight by 0.95)
    3. Near-zero edges are pruned (weight < 0.05)
    4. Near-duplicate nodes are merged (Jaccard > 0.9)
    5. Triadic closure infers new edges for top-connected nodes

    Safety: nodes referenced in the SelfModel belief log are never
    pruned, preserving the system's self-knowledge integrity.

    Example::

        cycle = SleepCycle()
        stats = cycle.consolidate(kernel, lexicon)
        print(stats)  # {'strengthened': 42, 'weakened': 18, ...}
    """

    # Consolidation parameters
    STRENGTHEN_FACTOR = 1.05     # Multiply high-myelin weights
    WEAKEN_FACTOR = 0.95         # Multiply low-myelin weights
    PRUNE_THRESHOLD = 0.05       # Remove edges with weight below this
    MERGE_SIMILARITY = 0.9       # Jaccard threshold for node merging
    MYELIN_HIGH = 5              # Myelin above this = "high" (strengthen)
    MYELIN_LOW = 1               # Myelin at or below this = "low" (weaken)
    TRIADIC_TOP_N = 100          # Run triadic closure for top N nodes

    def __init__(self, protected_uids: Optional[Set[str]] = None) -> None:
        """Initialise the sleep cycle.

        Args:
            protected_uids: Set of node UIDs that must never be pruned.
                            Typically populated from SelfModel._belief_log.
        """
        self._protected_uids: Set[str] = protected_uids or set()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop = threading.Event()
        self._lock = threading.Lock()
        self._cycle_count = 0
        self._last_stats: Dict[str, int] = {}
        self._history: List[Dict] = []

    def set_protected_uids(self, uids: Set[str]) -> None:
        """Update the set of protected node UIDs.

        Protected nodes are never pruned or merged away during
        consolidation. Typically these are nodes in the SelfModel's
        belief log.

        Args:
            uids: Set of node UIDs to protect.
        """
        self._protected_uids = set(uids)

    def consolidate(self, kernel, lexicon=None) -> Dict[str, int]:
        """Run a full sleep consolidation cycle.

        Executes all five phases in order:
        1. Strengthen high-myelin edges
        2. Weaken low-myelin edges
        3. Prune near-zero edges
        4. Merge near-duplicate nodes
        5. Triadic closure inference

        Args:
            kernel: KOSKernel instance containing the graph.
            lexicon: Optional KASMLexicon for word lookups during merging.

        Returns:
            Dict with counts: {strengthened, weakened, pruned, merged, inferred}
        """
        start = time.perf_counter()

        stats = {
            'strengthened': 0,
            'weakened': 0,
            'pruned': 0,
            'merged': 0,
            'inferred': 0,
            'time_ms': 0.0,
        }

        with self._lock:
            # Phase 1 & 2: Strengthen / Weaken edges based on myelin
            stats['strengthened'], stats['weakened'] = self._modulate_edges(kernel)

            # Phase 3: Prune dead edges
            stats['pruned'] = self._prune_edges(kernel)

            # Phase 4: Merge near-duplicate nodes
            stats['merged'] = self._merge_duplicates(kernel, lexicon)

            # Phase 5: Triadic closure for top-connected nodes
            stats['inferred'] = self._triadic_closure(kernel)

        stats['time_ms'] = (time.perf_counter() - start) * 1000.0
        self._cycle_count += 1
        self._last_stats = stats
        self._history.append(dict(stats))

        return stats

    # ----- Phase 1 & 2: Edge Modulation -----------------------------------

    def _modulate_edges(self, kernel) -> Tuple[int, int]:
        """Strengthen high-myelin edges and weaken low-myelin edges.

        For dict-style connections (with 'w' and 'myelin' keys):
        - myelin > MYELIN_HIGH: weight *= STRENGTHEN_FACTOR
        - myelin <= MYELIN_LOW: weight *= WEAKEN_FACTOR

        For scalar connections, no modulation is applied (no myelin data).

        Returns:
            (strengthened_count, weakened_count)
        """
        strengthened = 0
        weakened = 0

        for node_id, node in kernel.nodes.items():
            for target_id, data in list(node.connections.items()):
                if not isinstance(data, dict):
                    continue
                myelin = data.get('myelin', 0)
                weight = data.get('w', 0.0)

                if myelin > self.MYELIN_HIGH:
                    data['w'] = weight * self.STRENGTHEN_FACTOR
                    strengthened += 1
                elif myelin <= self.MYELIN_LOW:
                    data['w'] = weight * self.WEAKEN_FACTOR
                    weakened += 1

        return strengthened, weakened

    # ----- Phase 3: Prune Dead Edges --------------------------------------

    def _prune_edges(self, kernel) -> int:
        """Remove edges with absolute weight below PRUNE_THRESHOLD.

        Never prunes edges connecting to protected nodes.

        Returns:
            Number of edges pruned.
        """
        pruned = 0
        for node_id, node in kernel.nodes.items():
            if node_id in self._protected_uids:
                continue

            to_remove = []
            for target_id, data in node.connections.items():
                if target_id in self._protected_uids:
                    continue

                if isinstance(data, dict):
                    weight = abs(data.get('w', 0.0))
                else:
                    weight = abs(data)

                if weight < self.PRUNE_THRESHOLD:
                    to_remove.append(target_id)

            for target_id in to_remove:
                del node.connections[target_id]
                pruned += 1

        return pruned

    # ----- Phase 4: Merge Near-Duplicate Nodes ----------------------------

    def _merge_duplicates(self, kernel, lexicon=None) -> int:
        """Merge nodes with Jaccard similarity > MERGE_SIMILARITY.

        Two nodes are considered near-duplicates if their connection
        neighbourhoods overlap by more than 90% (Jaccard index).

        The node with fewer connections is merged into the one with more.
        Protected nodes are never merged away (but can absorb merges).

        Returns:
            Number of nodes merged.
        """
        merged = 0
        node_ids = list(kernel.nodes.keys())
        merged_away: Set[str] = set()

        # Pre-compute neighbour sets for efficiency
        neighbour_sets: Dict[str, Set[str]] = {}
        for nid in node_ids:
            neighbour_sets[nid] = set(kernel.nodes[nid].connections.keys())

        # Only compare nodes with at least 2 connections
        candidates = [nid for nid in node_ids
                       if len(neighbour_sets[nid]) >= 2]

        for i in range(len(candidates)):
            nid_a = candidates[i]
            if nid_a in merged_away:
                continue

            for j in range(i + 1, len(candidates)):
                nid_b = candidates[j]
                if nid_b in merged_away:
                    continue

                set_a = neighbour_sets[nid_a]
                set_b = neighbour_sets[nid_b]

                # Jaccard similarity
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                if union == 0:
                    continue
                jaccard = intersection / union

                if jaccard >= self.MERGE_SIMILARITY:
                    # Merge smaller into larger
                    if len(set_a) >= len(set_b):
                        keeper, donor = nid_a, nid_b
                    else:
                        keeper, donor = nid_b, nid_a

                    # Never merge away a protected node
                    if donor in self._protected_uids:
                        if keeper in self._protected_uids:
                            continue  # Both protected, skip
                        keeper, donor = donor, keeper
                        if donor in self._protected_uids:
                            continue  # Both protected

                    self._merge_nodes(kernel, keeper, donor, lexicon)
                    merged_away.add(donor)
                    merged += 1

        return merged

    def _merge_nodes(self, kernel, keeper_id: str, donor_id: str,
                     lexicon=None) -> None:
        """Merge donor node into keeper node.

        All of donor's connections are transferred to keeper.
        Provenance sentences referencing donor are re-keyed to keeper.
        The donor node is removed from the graph.

        Args:
            kernel: KOSKernel instance.
            keeper_id: Node ID that absorbs the merge.
            donor_id: Node ID that is merged away.
            lexicon: Optional lexicon for updating word->uuid mappings.
        """
        keeper = kernel.nodes.get(keeper_id)
        donor = kernel.nodes.get(donor_id)
        if not keeper or not donor:
            return

        # Transfer connections from donor to keeper
        for target_id, data in donor.connections.items():
            if target_id == keeper_id:
                continue  # Skip self-loops
            if target_id not in keeper.connections:
                keeper.connections[target_id] = data
            # Also update reverse connections pointing to donor
        for nid, node in kernel.nodes.items():
            if donor_id in node.connections and nid != keeper_id:
                data = node.connections.pop(donor_id)
                if keeper_id not in node.connections:
                    node.connections[keeper_id] = data

        # Transfer provenance
        keys_to_update = []
        for pair_key in list(kernel.provenance.keys()):
            if donor_id in pair_key:
                keys_to_update.append(pair_key)
        for old_key in keys_to_update:
            new_key = tuple(sorted([
                keeper_id if x == donor_id else x for x in old_key
            ]))
            if new_key != old_key:
                kernel.provenance[new_key].update(kernel.provenance[old_key])
                del kernel.provenance[old_key]

        # Transfer properties
        for prop, val in donor.properties.items():
            if prop not in keeper.properties:
                keeper.properties[prop] = val

        # Remove donor from graph
        del kernel.nodes[donor_id]

        # Update lexicon mapping if available
        if lexicon and hasattr(lexicon, 'uuid_to_word'):
            donor_word = lexicon.uuid_to_word.get(donor_id)
            if donor_word and donor_word in lexicon.word_to_uuid:
                lexicon.word_to_uuid[donor_word] = keeper_id

    # ----- Phase 5: Triadic Closure ---------------------------------------

    def _triadic_closure(self, kernel) -> int:
        """Infer new edges via triadic closure for top-connected nodes.

        For the top N most-connected nodes, if A->B and B->C exist
        but A->C does not, create A->C with a weak weight (0.2).

        This mirrors the "friend of a friend" principle in social
        networks and closes structural holes in the knowledge graph.

        Returns:
            Number of new edges inferred.
        """
        # Find top N nodes by connection count
        sorted_nodes = sorted(
            kernel.nodes.items(),
            key=lambda x: len(x[1].connections),
            reverse=True
        )[:self.TRIADIC_TOP_N]

        inferred = 0
        for node_id, node in sorted_nodes:
            neighbours = list(node.connections.keys())
            for neighbour_id in neighbours:
                if neighbour_id not in kernel.nodes:
                    continue
                # Look at neighbour's connections (2-hop)
                for two_hop_id in list(kernel.nodes[neighbour_id].connections.keys()):
                    if (two_hop_id != node_id
                            and two_hop_id not in node.connections
                            and two_hop_id in kernel.nodes):
                        # Infer weak edge
                        kernel.add_connection(
                            node_id, two_hop_id, 0.2,
                            "[SLEEP] Triadic closure inference")
                        inferred += 1

                        # Cap inferred edges per cycle to avoid explosion
                        if inferred >= 500:
                            return inferred

        return inferred

    # ----- Scheduled Execution --------------------------------------------

    def schedule(self, kernel, lexicon=None,
                 interval_hours: float = 8.0) -> None:
        """Auto-run consolidation every N hours in a daemon thread.

        The thread runs in the background and does not prevent
        the Python process from exiting.

        Args:
            kernel: KOSKernel instance.
            lexicon: Optional KASMLexicon instance.
            interval_hours: Hours between consolidation cycles (default 8).
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return  # Already running

        self._scheduler_stop.clear()
        interval_seconds = interval_hours * 3600

        def _run_loop():
            while not self._scheduler_stop.is_set():
                self._scheduler_stop.wait(timeout=interval_seconds)
                if self._scheduler_stop.is_set():
                    break
                try:
                    self.consolidate(kernel, lexicon)
                except Exception:
                    pass  # Silently continue — daemon must not crash

        self._scheduler_thread = threading.Thread(
            target=_run_loop,
            name="KOS-SleepCycle",
            daemon=True,
        )
        self._scheduler_thread.start()

    def stop_schedule(self) -> None:
        """Stop the scheduled consolidation daemon."""
        self._scheduler_stop.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            self._scheduler_thread = None

    # ----- Stats / Monitoring ---------------------------------------------

    def get_history(self) -> List[Dict]:
        """Return past consolidation results.

        Each entry in the list corresponds to one consolidation cycle and
        contains the keys: strengthened, weakened, pruned, merged,
        inferred, and time_ms.

        Returns:
            List of dicts, one per completed cycle.
        """
        return list(self._history)

    def get_stats(self) -> Dict:
        """Return statistics from the last consolidation cycle.

        Returns:
            Dict with cycle count and last cycle stats.
        """
        return {
            'cycle_count': self._cycle_count,
            'last_stats': dict(self._last_stats),
            'protected_nodes': len(self._protected_uids),
            'scheduler_active': (
                self._scheduler_thread is not None
                and self._scheduler_thread.is_alive()
            ),
        }

    def __repr__(self) -> str:
        active = "active" if (self._scheduler_thread
                              and self._scheduler_thread.is_alive()) else "idle"
        return (f"SleepCycle(cycles={self._cycle_count}, "
                f"scheduler={active})")
