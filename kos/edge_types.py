"""
KOS V8.0 — Edge Type System

Every edge in the graph now carries a type that affects traversal,
scoring, and retrieval behavior. Types are stored as u8 in Rust
for arena compactness.
"""

import re

# ── Edge Type Constants (match Rust u8 values) ──────────────────
GENERIC = 0
IS_A = 1
CAUSES = 2
PART_OF = 3
OBSERVED_WITH = 4
CONTRADICTS = 5
SUPPORTS = 6
DERIVED_FROM = 7
PROCEDURE_STEP = 8
TEMPORAL_BEFORE = 9
TEMPORAL_AFTER = 10
LOCATED_IN = 11
HAS_PROPERTY = 12

NAMES = {
    GENERIC: "generic",
    IS_A: "is_a",
    CAUSES: "causes",
    PART_OF: "part_of",
    OBSERVED_WITH: "observed_with",
    CONTRADICTS: "contradicts",
    SUPPORTS: "supports",
    DERIVED_FROM: "derived_from",
    PROCEDURE_STEP: "procedure_step",
    TEMPORAL_BEFORE: "temporal_before",
    TEMPORAL_AFTER: "temporal_after",
    LOCATED_IN: "located_in",
    HAS_PROPERTY: "has_property",
}

# ── Traversal Configuration Per Type ────────────────────────────
# trust: how much to weight this edge in scoring (0.0-1.0)
# max_hops: how deep this edge type should propagate
# decay: per-hop energy multiplier for this type
EDGE_CONFIG = {
    GENERIC:        {"trust": 0.5, "max_hops": 3, "decay": 0.7},
    IS_A:           {"trust": 0.9, "max_hops": 5, "decay": 0.85},
    CAUSES:         {"trust": 0.85, "max_hops": 7, "decay": 0.8},
    PART_OF:        {"trust": 0.8, "max_hops": 4, "decay": 0.8},
    OBSERVED_WITH:  {"trust": 0.4, "max_hops": 2, "decay": 0.6},
    CONTRADICTS:    {"trust": 0.3, "max_hops": 2, "decay": 0.5},
    SUPPORTS:       {"trust": 0.8, "max_hops": 4, "decay": 0.8},
    DERIVED_FROM:   {"trust": 0.7, "max_hops": 3, "decay": 0.75},
    PROCEDURE_STEP: {"trust": 0.9, "max_hops": 10, "decay": 0.9},
    TEMPORAL_BEFORE: {"trust": 0.6, "max_hops": 5, "decay": 0.7},
    TEMPORAL_AFTER:  {"trust": 0.6, "max_hops": 5, "decay": 0.7},
    LOCATED_IN:     {"trust": 0.8, "max_hops": 3, "decay": 0.8},
    HAS_PROPERTY:   {"trust": 0.7, "max_hops": 2, "decay": 0.7},
}

# ── Inference Patterns ──────────────────────────────────────────
# Regex patterns to infer edge type from provenance text.
_PATTERNS = [
    (re.compile(r'\bis\s+a\b|\bare\s+a\b|\bis\s+an?\b|type\s+of', re.I), IS_A),
    (re.compile(r'\bcause[sd]?\b|\bleads?\s+to\b|\bresults?\s+in\b|\btrigger', re.I), CAUSES),
    (re.compile(r'\bpart\s+of\b|\bcomponent\b|\bcontain', re.I), PART_OF),
    (re.compile(r'\bobserved\b|\bseen\s+with\b|\bco-occur', re.I), OBSERVED_WITH),
    (re.compile(r'\bcontradict|\bnot\b.*\bbut\b|\bhowever\b|\bdespite', re.I), CONTRADICTS),
    (re.compile(r'\bsupport|\bconfirm|\bvalidat|\bcorroborat', re.I), SUPPORTS),
    (re.compile(r'\bderived\b|\bbased\s+on\b|\bfrom\b.*\bdata\b', re.I), DERIVED_FROM),
    (re.compile(r'\bstep\b|\bthen\b|\bprocedure\b|\bprocess\b', re.I), PROCEDURE_STEP),
    (re.compile(r'\bbefore\b|\bprior\s+to\b|\bprecede', re.I), TEMPORAL_BEFORE),
    (re.compile(r'\bafter\b|\bfollowing\b|\bsubsequent', re.I), TEMPORAL_AFTER),
    (re.compile(r'\bin\b.*\b(?:city|country|province|state|region)\b|\blocated', re.I), LOCATED_IN),
    (re.compile(r'\bhas\b|\bproperty\b|\bcharacteristic\b|\bfeature', re.I), HAS_PROPERTY),
]


def infer_type(provenance_text: str) -> int:
    """Infer edge type from provenance text using regex heuristics."""
    if not provenance_text:
        return GENERIC
    for pattern, edge_type in _PATTERNS:
        if pattern.search(provenance_text):
            return edge_type
    return GENERIC
