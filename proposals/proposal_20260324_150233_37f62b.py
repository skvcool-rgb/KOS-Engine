
# AUTO-GENERATED THRESHOLD CHANGE
# Parameter: edge_weight_for_verbs
# Old value: 0.9
# New value: 0.3
# Reason: DIAGNOSIS: Verbs get weight 0.9 same as nouns. But verbs are structural, not semantic. "Toronto PRODUCES energy" — the relationship is between Toronto and energy, not Toronto and produces. FIX: Set verb edge weight to 0.3 instead of 0.9. Nouns keep 0.9. This naturally suppresses verb propagation.

# Apply in the relevant module:
# edge_weight_for_verbs = 0.3

# To apply via config (preferred):
# Edit .cache/self_tuned_config.json:
#   "edge_weight_for_verbs": {"value": 0.3}
