
# AUTO-GENERATED THRESHOLD CHANGE
# Parameter: activation_threshold
# Old value: 0.1
# New value: 0.05
# Reason: Lower activation threshold improves recall for deep multi-hop chains without significant precision loss. Empirically validated in Contagion Audit test.

# Apply in the relevant module:
# activation_threshold = 0.05

# To apply via config (preferred):
# Edit .cache/self_tuned_config.json:
#   "activation_threshold": {"value": 0.05}
