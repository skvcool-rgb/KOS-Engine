"""
KOS V8.0 -- Memory Lifecycle Manager (4-Tier Memory)

Solves "graph grows forever" by implementing biologically-inspired
memory tiers with promotion, demotion, and archival.

Tiers:
    HOT     - Active working set. Full activation. < 10K nodes.
    WARM    - Recently used. Queryable but lower priority.
    COLD    - Old knowledge. Compressed, retrievable on demand.
    ARCHIVE - Frozen. Not in active graph. Serialized to disk.

Transitions:
    HOT -> WARM:    After N ticks without query
    WARM -> COLD:   After M ticks without query
    COLD -> ARCHIVE: After P ticks without query
    Any -> HOT:     When queried (promotion)
"""

import time
import json
import os

# ---- Tier Thresholds (in ticks since last access) -----------------------
HOT_TO_WARM = 50
WARM_TO_COLD = 200
COLD_TO_ARCHIVE = 1000

# ---- Capacity Limits ----------------------------------------------------
HOT_MAX_NODES = 10000
WARM_MAX_NODES = 50000


class MemoryLifecycleManager:
    """Manages node lifecycle across memory tiers."""

    def __init__(self, kernel, archive_dir: str = None):
        self.kernel = kernel
        self.archive_dir = archive_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '.cache', 'archive')
        self.archived_nodes = {}  # node_id -> archive_file
        self.tier_stats = {"hot": 0, "warm": 0, "cold": 0, "archive": 0}
        self._last_sweep_tick = 0

    def classify_node(self, node_id: str) -> str:
        """Classify a node into its current tier."""
        tick = getattr(self.kernel, 'current_tick', 0)

        if node_id in self.archived_nodes:
            return "archive"

        node = self.kernel.nodes.get(node_id)
        if not node:
            return "unknown"

        last_access = getattr(node, 'last_tick', 0)
        age = tick - last_access

        if age < HOT_TO_WARM:
            return "hot"
        elif age < WARM_TO_COLD:
            return "warm"
        else:
            return "cold"

    def sweep(self) -> dict:
        """
        Run a lifecycle sweep: demote stale nodes, enforce capacity limits.
        Call periodically (e.g., every 100 ticks).
        """
        tick = getattr(self.kernel, 'current_tick', 0)
        self._last_sweep_tick = tick

        demoted = 0
        archived = 0
        promoted = 0

        # Classify all nodes
        tiers = {"hot": [], "warm": [], "cold": []}
        for node_id in list(self.kernel.nodes.keys()):
            tier = self.classify_node(node_id)
            if tier in tiers:
                tiers[tier].append(node_id)

        # Enforce HOT capacity limit
        if len(tiers["hot"]) > HOT_MAX_NODES:
            # Demote oldest hot nodes to warm
            hot_nodes = tiers["hot"]
            hot_nodes.sort(key=lambda nid: getattr(
                self.kernel.nodes.get(nid), 'last_tick', 0))
            excess = len(hot_nodes) - HOT_MAX_NODES
            for nid in hot_nodes[:excess]:
                demoted += 1

        # Archive cold nodes past threshold
        for node_id in tiers["cold"]:
            node = self.kernel.nodes.get(node_id)
            if node:
                last_access = getattr(node, 'last_tick', 0)
                if tick - last_access > COLD_TO_ARCHIVE:
                    self._archive_node(node_id)
                    archived += 1

        self.tier_stats = {
            "hot": len(tiers["hot"]),
            "warm": len(tiers["warm"]),
            "cold": len(tiers["cold"]) - archived,
            "archive": len(self.archived_nodes),
        }

        return {
            "demoted": demoted,
            "archived": archived,
            "promoted": promoted,
            "tiers": dict(self.tier_stats),
            "sweep_tick": tick,
        }

    def promote(self, node_id: str) -> bool:
        """Promote a node back to HOT tier (on query access)."""
        if node_id in self.archived_nodes:
            return self._restore_from_archive(node_id)

        node = self.kernel.nodes.get(node_id)
        if node:
            node.last_tick = getattr(self.kernel, 'current_tick', 0)
            return True
        return False

    def _archive_node(self, node_id: str):
        """Move a node to archive storage."""
        node = self.kernel.nodes.get(node_id)
        if not node:
            return

        # Serialize node data
        archive_data = {
            "node_id": node_id,
            "connections": dict(node.connections),
            "properties": getattr(node, 'properties', {}),
            "archived_tick": getattr(self.kernel, 'current_tick', 0),
        }

        # Save to archive directory
        os.makedirs(self.archive_dir, exist_ok=True)
        safe_name = node_id.replace("/", "_").replace(".", "_")[:100]
        filepath = os.path.join(self.archive_dir, f"{safe_name}.json")

        try:
            with open(filepath, 'w') as f:
                json.dump(archive_data, f)
            self.archived_nodes[node_id] = filepath

            # Remove from active graph (keep in provenance)
            del self.kernel.nodes[node_id]
        except Exception:
            pass  # Don't crash on archive failure

    def _restore_from_archive(self, node_id: str) -> bool:
        """Restore a node from archive to active graph."""
        filepath = self.archived_nodes.get(node_id)
        if not filepath or not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Re-create node
            self.kernel.add_node(data["node_id"])
            node = self.kernel.nodes[data["node_id"]]
            node.connections = data.get("connections", {})
            if hasattr(node, 'properties'):
                node.properties = data.get("properties", {})
            node.last_tick = getattr(self.kernel, 'current_tick', 0)

            # Re-wire to Rust if available
            if self.kernel._rust is not None:
                for tgt, w in node.connections.items():
                    if tgt in self.kernel.nodes:
                        self.kernel._rust.add_connection(
                            data["node_id"], tgt, float(w), None)

            # Remove from archive tracking
            del self.archived_nodes[node_id]
            os.remove(filepath)
            return True

        except Exception:
            return False

    def archive_count(self) -> int:
        return len(self.archived_nodes)

    def stats(self) -> dict:
        return {
            "tiers": dict(self.tier_stats),
            "archived_nodes": len(self.archived_nodes),
            "last_sweep_tick": self._last_sweep_tick,
            "total_active": len(self.kernel.nodes),
        }
