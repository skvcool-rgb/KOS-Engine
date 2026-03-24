
# AUTO-GENERATED THRESHOLD CHANGE
# Parameter: edge_weight_normalization
# Old value: 0.0
# New value: 1.0
# Reason: Current edge weights range from -0.8 to 0.9 with no normalization. Over time, myelination can push effective weights above 1.0, distorting the physics. Proposal: normalize all edge weights to [-1.0, 1.0] after every daemon cycle. Prevents weight inflation and keeps the spreading activation physics numerically stable.

# Apply in the relevant module:
# edge_weight_normalization = 1.0

# To apply via config (preferred):
# Edit .cache/self_tuned_config.json:
#   "edge_weight_normalization": {"value": 1.0}
