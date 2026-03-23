"""
KOS V2.0 — ConceptNode (Perfected Physics Engine).

Each node represents a concept with dual state:
- activation: semantic score for result ranking (accumulates, never drains)
- fuel: propagation energy (drains to 0 after firing)

Hyperpolarization Gate: fuel only accumulates when incoming > 0 AND activation > 0.
Top-K Synaptic Routing: only fires down the top 15 strongest pathways.
"""


class ConceptNode:
    __slots__ = ['id', 'activation', 'fuel', 'connections',
                 'temporal_decay', 'max_energy', 'last_tick']

    def __init__(self, concept_id: str,
                 temporal_decay: float = 0.7,
                 max_energy: float = 3.0):
        self.id = concept_id
        self.temporal_decay = temporal_decay
        self.max_energy = max_energy
        self.activation = 0.0
        self.fuel = 0.0
        self.last_tick = 0
        self.connections = {}

    def _apply_lazy_decay(self, current_tick: int):
        if current_tick > self.last_tick:
            factor = (self.temporal_decay ** (current_tick - self.last_tick))
            self.activation *= factor
            self.fuel *= factor
            self.last_tick = current_tick

    def receive_signal(self, incoming_energy: float, current_tick: int):
        self._apply_lazy_decay(current_tick)

        # Update Epistemic Truth (Activation)
        self.activation += incoming_energy
        self.activation = max(-self.max_energy,
                              min(self.max_energy, self.activation))

        # Update Mechanical Spike (Fuel) via Hyperpolarization Gate
        if incoming_energy > 0 and self.activation > 0:
            self.fuel += incoming_energy
            self.fuel = max(0.0, min(self.max_energy, self.fuel))

    def propagate(self, current_tick: int = 0,
                  spatial_decay: float = 0.8,
                  base_threshold: float = 0.05):
        self._apply_lazy_decay(current_tick)

        if self.fuel < base_threshold:
            return []

        outbound = []

        # INCREASE FROM 15 TO 500!
        # Allows the node to access niche facts (Population, History) while the
        # Global Kernel queue (200 limit) prevents CPU crashing.
        edges = sorted(
            self.connections.items(),
            key=lambda x: abs(x[1]['w'] * (1 + x[1]['myelin'] * 0.01))
            if isinstance(x[1], dict)
            else abs(x[1]),
            reverse=True
        )[:500]

        for tgt, data in edges:
            if isinstance(data, dict):
                active_w = data['w'] * (1 + data['myelin'] * 0.01)
                passed = self.fuel * active_w * spatial_decay
                if abs(passed) >= base_threshold:
                    outbound.append((tgt, passed))
                    self.connections[tgt]['myelin'] += 1
            else:
                passed = self.fuel * data * spatial_decay
                if abs(passed) >= base_threshold:
                    outbound.append((tgt, passed))

        # Fire once rule
        self.fuel = 0.0
        return outbound
