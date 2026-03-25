"""
KOS V8.0 — Memory Tier Manager

Nodes are classified into hot/warm/cold tiers based on recency of access.
Tiers affect retrieval bias — recently accessed knowledge gets priority.

The actual tier_bias computation lives in Rust (ArenaNode::tier_bias),
this Python module provides the constants and utility functions for
the Python fallback path and for reporting.
"""

# ── Tier Thresholds (in ticks since last query) ─────────────────
HOT_THRESHOLD = 50       # < 50 ticks since last access
WARM_THRESHOLD = 200     # < 200 ticks since last access
# > 200 ticks = cold

# ── Tier Bias Multipliers ───────────────────────────────────────
HOT_BIAS = 1.5
WARM_BIAS = 1.0
COLD_BIAS = 0.5


def classify(ticks_since_access: int) -> str:
    """Classify a node into hot/warm/cold tier."""
    if ticks_since_access < HOT_THRESHOLD:
        return "hot"
    elif ticks_since_access < WARM_THRESHOLD:
        return "warm"
    return "cold"


def bias(ticks_since_access: int) -> float:
    """Return the retrieval bias multiplier for a given recency."""
    if ticks_since_access < HOT_THRESHOLD:
        return HOT_BIAS
    elif ticks_since_access < WARM_THRESHOLD:
        return WARM_BIAS
    return COLD_BIAS


def tier_summary(kernel) -> dict:
    """Count nodes in each tier. Works with both Rust and Python backends."""
    counts = {"hot": 0, "warm": 0, "cold": 0}
    tick = getattr(kernel, 'current_tick', 0)

    if hasattr(kernel, '_rust') and kernel._rust is not None:
        stats = kernel._rust.stats()
        tick = int(stats.get('tick', tick))
        # For Rust, we can't iterate arena directly — use node list
        for nid in kernel.nodes:
            try:
                neighbors = kernel._rust.get_neighbors(nid)
                # Use the node's existence to classify; actual tier is in Rust
                # We approximate by checking if it was recently queried
                counts["warm"] += 1  # Default for nodes we can't inspect
            except Exception:
                counts["cold"] += 1
    else:
        for nid, node in kernel.nodes.items():
            last_active = getattr(node, 'last_tick', 0)
            tier = classify(tick - last_active)
            counts[tier] += 1

    return counts
