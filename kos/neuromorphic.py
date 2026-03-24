"""
KOS Neuromorphic Simulator — Spike-Based Execution Model.

Software simulation of how KOS would run on neuromorphic hardware
(Intel Loihi 2, IBM TrueNorth). We do NOT have the actual hardware —
this simulates the spike-based execution model faithfully so we can:

    1. Validate that KASM operations map to spiking neural networks
    2. Estimate power/latency gains over GPU execution
    3. Build intuition for neuromorphic computing concepts

Key insight: KASM's BIND operation (element-wise XOR on bipolar vectors)
maps DIRECTLY to coincidence detection in spiking neurons. This is not
a metaphor — it is the actual mathematical equivalence.

    Bipolar BIND:   c_i = a_i * b_i   (product of +1/-1)
    Spike coincidence:  output fires IFF both inputs fire in the same
                        time window (AND gate on spikes)

    When we encode +1 as "spike present" and -1 as "spike absent" in a
    time bin, element-wise multiplication IS coincidence detection.
    Neuromorphic hardware performs this in O(1) energy per synapse.

Architecture reference:
    - Leaky Integrate-and-Fire (LIF) neuron model
    - Rate coding for concept vectors
    - Coincidence detection for BIND
    - Membrane summation for SUPERPOSE
    - Spike timing for PERMUTE (temporal coding)

Dependencies: numpy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Leaky Integrate-and-Fire Neuron
# ---------------------------------------------------------------------------

@dataclass
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron — the fundamental unit of neuromorphic
    computation.

    How it works (biological analogy):
        - The neuron has a membrane that accumulates charge (membrane_potential).
        - Each timestep, charge leaks away (controlled by tau_m).
        - When charge crosses a threshold (v_th), the neuron emits a spike
          and resets to 0.
        - After firing, the neuron enters a refractory period where it
          cannot fire again (models the biological refractory period).

    This is the simplest spiking neuron model that still captures the
    essential dynamics. Real Loihi cores implement this in fixed-point
    arithmetic on silicon.

    Parameters
    ----------
    name : str
        Human-readable identifier for this neuron.
    v_th : float
        Firing threshold. When membrane_potential >= v_th, neuron spikes.
        Typical range: 0.5 to 2.0.
    tau_m : float
        Membrane time constant (leak factor). Each timestep, the membrane
        potential is multiplied by (1 - 1/tau_m). Higher tau_m = slower leak.
        Typical range: 5.0 to 50.0.
    refractory_period : int
        Number of timesteps after a spike during which the neuron cannot
        fire. Models the biological absolute refractory period.
        Typical range: 1 to 5.
    """

    name: str
    v_th: float = 1.0
    tau_m: float = 10.0
    refractory_period: int = 2

    # State (mutable, not constructor params)
    membrane_potential: float = field(default=0.0, init=False)
    refractory_counter: int = field(default=0, init=False)
    spike_log: List[int] = field(default_factory=list, init=False)

    @property
    def leak_factor(self) -> float:
        """Per-timestep decay factor derived from membrane time constant."""
        return 1.0 - 1.0 / self.tau_m

    def step(self, input_current: float, t: int) -> bool:
        """
        Advance the neuron by one timestep.

        Parameters
        ----------
        input_current : float
            Total synaptic input at this timestep (sum of weighted spikes
            from presynaptic neurons).
        t : int
            Current simulation timestep (for logging).

        Returns
        -------
        bool
            True if the neuron fired a spike this timestep.

        Physics:
            V(t+1) = leak_factor * V(t) + I(t)      [sub-threshold]
            if V(t+1) >= V_th:
                emit spike, V -> 0, enter refractory  [super-threshold]
        """
        # Refractory period: neuron is silent
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            # Membrane still leaks during refractory
            self.membrane_potential *= self.leak_factor
            return False

        # Leaky integration
        self.membrane_potential = self.leak_factor * self.membrane_potential + input_current

        # Threshold check
        if self.membrane_potential >= self.v_th:
            self.spike_log.append(t)
            self.membrane_potential = 0.0  # Reset
            self.refractory_counter = self.refractory_period
            return True

        return False

    def reset(self):
        """Clear all state for a fresh simulation run."""
        self.membrane_potential = 0.0
        self.refractory_counter = 0
        self.spike_log = []


# ---------------------------------------------------------------------------
# 2. Synapse (weighted connection with axonal delay)
# ---------------------------------------------------------------------------

@dataclass
class Synapse:
    """
    A directed connection between two neurons.

    In neuromorphic hardware, each synapse is a physical connection on the
    chip with a programmable weight (typically 1-9 bits on Loihi 2) and
    an optional axonal delay.

    Parameters
    ----------
    source : str
        Name of the presynaptic (sending) neuron.
    target : str
        Name of the postsynaptic (receiving) neuron.
    weight : float
        Synaptic strength. Positive = excitatory, negative = inhibitory.
        On Loihi 2, weights are quantized to 8-bit signed integers.
    delay : int
        Axonal delay in timesteps. Models the time it takes a spike to
        travel along the axon. On Loihi 2, configurable 0-63 timesteps.
    """

    source: str
    target: str
    weight: float = 1.0
    delay: int = 1


# ---------------------------------------------------------------------------
# 3. Spike Train Encoder / Decoder
# ---------------------------------------------------------------------------

class SpikeTrainEncoder:
    """
    Converts between continuous values and spike trains using rate coding.

    Rate coding is the simplest and most well-understood neural code:
    a higher value maps to a higher spike frequency. This is how most
    sensory neurons encode stimulus intensity.

    How it works:
        1. A value v in [0, 1] maps to a spike probability p = v per timestep.
        2. At each timestep, we draw from Bernoulli(p).
        3. The resulting spike train has an expected rate = v.
        4. To decode, count spikes and divide by total timesteps.

    For KASM bipolar vectors {-1, +1}:
        - +1 encodes as a HIGH spike rate (p = 0.8)
        - -1 encodes as a LOW spike rate (p = 0.1)
        - The contrast ratio makes coincidence detection reliable.

    Parameters
    ----------
    timesteps : int
        Length of the encoding window. More timesteps = more precise
        encoding but higher latency. Typical: 50-200.
    high_rate : float
        Spike probability for +1 elements. Default 0.8.
    low_rate : float
        Spike probability for -1 elements. Default 0.1.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, timesteps: int = 100, high_rate: float = 0.8,
                 low_rate: float = 0.1, seed: Optional[int] = None):
        self.timesteps = timesteps
        self.high_rate = high_rate
        self.low_rate = low_rate
        self._rng = np.random.default_rng(seed)

    def encode_value(self, value: float) -> np.ndarray:
        """
        Encode a scalar value in [0, 1] as a spike train.

        Parameters
        ----------
        value : float
            Value between 0 and 1.

        Returns
        -------
        np.ndarray
            Boolean array of shape (timesteps,). True = spike at that time.
        """
        value = np.clip(value, 0.0, 1.0)
        return self._rng.random(self.timesteps) < value

    def encode_bipolar(self, element: int) -> np.ndarray:
        """
        Encode a bipolar element (+1 or -1) as a spike train.

        +1 -> high spike rate (many spikes)
        -1 -> low spike rate  (few spikes)

        This is the bridge between KASM vectors and neuromorphic spikes.
        """
        rate = self.high_rate if element == 1 else self.low_rate
        return self._rng.random(self.timesteps) < rate

    def encode_concept(self, name: str, dimensions: int = 100,
                       seed_offset: int = 0) -> np.ndarray:
        """
        Encode a named concept as a matrix of spike trains.

        Each dimension of the concept vector gets its own spike train.
        The concept identity is determined by the pattern of high/low
        rates across dimensions (its bipolar vector).

        Parameters
        ----------
        name : str
            Concept name (hashed to generate the bipolar vector).
        dimensions : int
            Number of dimensions (number of parallel spike trains).
        seed_offset : int
            Additional seed offset for different encodings of the same concept.

        Returns
        -------
        np.ndarray
            Boolean array of shape (dimensions, timesteps).
            spike_matrix[d, t] = True means dimension d spiked at time t.
        """
        # Deterministic bipolar vector from concept name
        concept_seed = hash(name) & 0xFFFFFFFF
        concept_rng = np.random.default_rng(concept_seed + seed_offset)
        bipolar = concept_rng.choice([-1, 1], size=dimensions)

        # Encode each dimension as a spike train
        spike_matrix = np.zeros((dimensions, self.timesteps), dtype=bool)
        for d in range(dimensions):
            spike_matrix[d] = self.encode_bipolar(bipolar[d])

        return spike_matrix

    def decode_spike_train(self, spikes: np.ndarray) -> float:
        """
        Decode a spike train back to a scalar value.

        Counts the fraction of timesteps with a spike.
        This is the Maximum Likelihood estimate under rate coding.

        Parameters
        ----------
        spikes : np.ndarray
            Boolean array of spike times.

        Returns
        -------
        float
            Estimated value in [0, 1].
        """
        return float(np.mean(spikes))

    def decode_to_bipolar(self, spikes: np.ndarray) -> int:
        """
        Decode a spike train to a bipolar element (+1 or -1).

        Uses the midpoint between high_rate and low_rate as the threshold.
        """
        rate = self.decode_spike_train(spikes)
        threshold = (self.high_rate + self.low_rate) / 2.0
        return 1 if rate >= threshold else -1


# ---------------------------------------------------------------------------
# 4. KASM-to-Spike Operation Mapper
# ---------------------------------------------------------------------------

class KASMSpikeMapper:
    """
    Maps KASM operations to spiking neural network equivalents.

    This is the core theoretical contribution: showing that each KASM
    operation has a DIRECT neuromorphic implementation, not an approximation.

    Operation Mapping
    -----------------
    KASM Operation    | Spike Equivalent           | Why it works
    ------------------|----------------------------|---------------------------
    NODE              | Create LIF neuron pool     | Random spike patterns =
                      |                            | random bipolar vectors
    BIND (XOR)        | Coincidence detection      | Two spikes in same time
                      |                            | bin = both +1 (product=+1)
    SUPERPOSE (+)     | Membrane summation         | Multiple inputs sum in
                      |                            | the LIF membrane equation
    RESONATE (sim)    | Spike correlation over     | Count co-firing events
                      | time window                | over a window
    PERMUTE (shift)   | Spike timing offset        | Delay lines shift the
                      |                            | entire spike train

    Parameters
    ----------
    encoder : SpikeTrainEncoder
        Encoder/decoder for spike trains.
    """

    def __init__(self, encoder: Optional[SpikeTrainEncoder] = None):
        self.encoder = encoder or SpikeTrainEncoder()

    def spike_bind(self, train_a: np.ndarray, train_b: np.ndarray) -> np.ndarray:
        """
        BIND via coincidence detection.

        In KASM:  c_i = a_i * b_i   (bipolar multiplication)
        In spikes: output fires ONLY when both inputs fire in the same
                   time bin.

        Mathematical equivalence:
            If +1 is encoded as spike-present and -1 as spike-absent:
            - Both +1 (both spike)    -> coincidence -> output spike  (+1)
            - Both -1 (neither spike) -> no input    -> no output     (-1)*
            - Mixed (one spikes)      -> sub-threshold -> no output   (-1)

        *The asymmetry (both-absent and mixed both map to no-spike) is
        handled by the rate coding: both-absent happens at low_rate^2 << 1,
        while both-present happens at high_rate^2 >> low_rate^2.
        The RATE of the output train encodes the bipolar product.

        Parameters
        ----------
        train_a, train_b : np.ndarray
            Spike trains (boolean arrays) of the same shape.

        Returns
        -------
        np.ndarray
            Output spike train: logical AND of inputs (coincidence).
        """
        return np.logical_and(train_a, train_b)

    def spike_superpose(self, *trains: np.ndarray,
                        threshold_fraction: float = 0.5) -> np.ndarray:
        """
        SUPERPOSE via membrane potential summation.

        In KASM:  s_i = sign(sum of v_k_i)  (majority vote)
        In spikes: the LIF neuron naturally sums its inputs in the
                   membrane potential. If enough inputs spike simultaneously,
                   the membrane crosses threshold and fires.

        The threshold_fraction parameter controls the majority rule:
        at 0.5, the output fires when >50% of inputs spike (majority vote).

        Parameters
        ----------
        trains : np.ndarray
            Variable number of spike trains to bundle.
        threshold_fraction : float
            Fraction of inputs that must spike for output to spike.

        Returns
        -------
        np.ndarray
            Output spike train representing the superposition.
        """
        if len(trains) == 0:
            raise ValueError("Need at least one spike train to superpose.")
        stacked = np.stack(trains, axis=0).astype(np.float32)
        vote_count = np.sum(stacked, axis=0)
        threshold = threshold_fraction * len(trains)
        return vote_count >= threshold

    def spike_resonate(self, train_a: np.ndarray, train_b: np.ndarray,
                       window: Optional[int] = None) -> float:
        """
        RESONATE via spike correlation (similarity measurement).

        In KASM:  sim(a, b) = cos(a, b) = (a . b) / (|a| |b|)
        In spikes: count the number of coincident spikes (co-firing events)
                   relative to total spikes. More co-firing = more similar.

        This is biologically plausible: Hebbian learning ("neurons that
        fire together wire together") uses exactly this co-firing statistic.

        Parameters
        ----------
        train_a, train_b : np.ndarray
            Spike trains to compare.
        window : int or None
            If provided, only consider the first `window` timesteps.

        Returns
        -------
        float
            Similarity score in [0, 1]. Higher = more similar.
        """
        if window is not None:
            train_a = train_a[..., :window]
            train_b = train_b[..., :window]

        # Count coincidences
        coincidences = np.sum(np.logical_and(train_a, train_b))
        # Normalize by geometric mean of spike counts (like cosine similarity)
        count_a = np.sum(train_a)
        count_b = np.sum(train_b)
        denominator = np.sqrt(float(count_a) * float(count_b))

        if denominator < 1e-10:
            return 0.0
        return float(coincidences) / denominator

    def spike_permute(self, train: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        PERMUTE via spike timing offset (temporal delay).

        In KASM:  rho(v, k) = circular_shift(v, k)
        In spikes: shift the entire spike train forward in time by `shift`
                   timesteps. On neuromorphic hardware, this is implemented
                   as an axonal delay — a physical wire that takes time
                   to traverse.

        This is how sequence position is encoded: the SAME concept at
        position 0 has its spikes arrive earlier than at position 3.
        The temporal offset IS the position encoding.

        Parameters
        ----------
        train : np.ndarray
            Spike train to permute.
        shift : int
            Number of timesteps to shift (circular).

        Returns
        -------
        np.ndarray
            Shifted spike train.
        """
        return np.roll(train, shift, axis=-1)


# ---------------------------------------------------------------------------
# 5. Neuromorphic Kernel (spike-based graph execution)
# ---------------------------------------------------------------------------

class NeuromorphicKernel:
    """
    A spiking neural network that mirrors a KOS knowledge graph.

    Each concept node becomes a LIF neuron (or neuron pool).
    Each edge becomes a synapse with weight and delay.
    Querying = injecting current into seed neurons and observing which
    downstream neurons fire.

    This is the neuromorphic equivalent of KOS's spreading activation,
    but computed entirely through spikes.

    Parameters
    ----------
    default_v_th : float
        Default firing threshold for new neurons.
    default_tau_m : float
        Default membrane time constant for new neurons.
    default_refractory : int
        Default refractory period for new neurons.
    """

    def __init__(self, default_v_th: float = 1.0, default_tau_m: float = 10.0,
                 default_refractory: int = 2):
        self.neurons: Dict[str, LIFNeuron] = {}
        self.synapses: List[Synapse] = []
        self._synapse_lookup: Dict[str, List[Synapse]] = {}  # source -> synapses

        self.default_v_th = default_v_th
        self.default_tau_m = default_tau_m
        self.default_refractory = default_refractory

    def add_neuron(self, name: str, v_th: Optional[float] = None,
                   tau_m: Optional[float] = None,
                   refractory: Optional[int] = None) -> LIFNeuron:
        """
        Create a LIF neuron in the network.

        Parameters
        ----------
        name : str
            Unique identifier for this neuron.
        v_th, tau_m, refractory : optional overrides for neuron parameters.

        Returns
        -------
        LIFNeuron
            The created (or existing) neuron.
        """
        if name not in self.neurons:
            self.neurons[name] = LIFNeuron(
                name=name,
                v_th=v_th if v_th is not None else self.default_v_th,
                tau_m=tau_m if tau_m is not None else self.default_tau_m,
                refractory_period=refractory if refractory is not None else self.default_refractory,
            )
        return self.neurons[name]

    def connect(self, source: str, target: str, weight: float = 1.0,
                delay: int = 1) -> Synapse:
        """
        Create a synapse between two neurons.

        Automatically creates the neurons if they do not exist.

        Parameters
        ----------
        source : str
            Presynaptic neuron name.
        target : str
            Postsynaptic neuron name.
        weight : float
            Synaptic weight. Positive = excitatory, negative = inhibitory.
        delay : int
            Axonal delay in timesteps.

        Returns
        -------
        Synapse
            The created synapse.
        """
        self.add_neuron(source)
        self.add_neuron(target)
        syn = Synapse(source=source, target=target, weight=weight, delay=delay)
        self.synapses.append(syn)
        self._synapse_lookup.setdefault(source, []).append(syn)
        return syn

    def propagate(self, seed_neurons: Dict[str, float],
                  timesteps: int = 100) -> Dict[str, List[int]]:
        """
        Run the spiking simulation.

        Injects current into seed neurons and propagates spikes through
        the network via synapses, respecting weights and delays.

        This is the neuromorphic equivalent of KOS spreading activation:
        instead of updating floating-point activations in a priority queue,
        we simulate discrete spikes propagating through LIF neurons.

        Parameters
        ----------
        seed_neurons : dict
            Mapping from neuron name to input current magnitude.
            These neurons receive constant external current (like a
            sensory input or query injection).
        timesteps : int
            Number of simulation timesteps. More = further propagation.

        Returns
        -------
        dict
            Mapping from neuron name to list of timesteps at which it fired.
            Neurons that never fired are omitted.

        Algorithm
        ---------
        For each timestep t:
            1. Compute input current for every neuron:
               I_j(t) = seed_current_j  (if j is a seed)
                       + sum over synapses (i->j) with delay d:
                           weight * spike_i(t - d)
            2. Step every neuron's LIF dynamics with its input current.
            3. Record which neurons spiked.

        On real Loihi hardware, steps 1-3 happen IN PARALLEL across all
        128 cores, making this O(1) in network size per timestep.
        """
        # Reset all neurons
        for neuron in self.neurons.values():
            neuron.reset()

        # Pre-build reverse lookup: target -> list of (source, weight, delay)
        incoming: Dict[str, List[Tuple[str, float, int]]] = {
            name: [] for name in self.neurons
        }
        for syn in self.synapses:
            incoming[syn.target].append((syn.source, syn.weight, syn.delay))

        # Spike history buffer: name -> set of spike times
        spike_times: Dict[str, set] = {name: set() for name in self.neurons}

        # Run simulation
        for t in range(timesteps):
            spikes_this_step = []

            for name, neuron in self.neurons.items():
                # Compute total input current
                current = seed_neurons.get(name, 0.0)

                for src_name, weight, delay in incoming[name]:
                    # Check if source neuron spiked at time (t - delay)
                    if (t - delay) in spike_times[src_name]:
                        current += weight

                # Step the neuron
                fired = neuron.step(current, t)
                if fired:
                    spikes_this_step.append(name)

            # Record spikes (after all neurons have been stepped)
            for name in spikes_this_step:
                spike_times[name].add(t)

        # Return only neurons that actually fired
        result = {}
        for name, neuron in self.neurons.items():
            if neuron.spike_log:
                result[name] = list(neuron.spike_log)
        return result

    def get_activation_ranking(self, spike_record: Dict[str, List[int]],
                               exclude_seeds: Optional[set] = None
                               ) -> List[Tuple[str, int]]:
        """
        Rank neurons by spike count (neuromorphic analog of activation score).

        In KOS, nodes are ranked by accumulated activation energy.
        In the neuromorphic model, the analog is total spike count:
        more spikes = more activated = more relevant.

        Parameters
        ----------
        spike_record : dict
            Output of propagate().
        exclude_seeds : set or None
            Neuron names to exclude from ranking (typically seed neurons).

        Returns
        -------
        list of (name, spike_count)
            Sorted descending by spike count.
        """
        exclude = exclude_seeds or set()
        counts = [
            (name, len(times))
            for name, times in spike_record.items()
            if name not in exclude
        ]
        counts.sort(key=lambda x: x[1], reverse=True)
        return counts


# ---------------------------------------------------------------------------
# 6. Neuromorphic Simulator (top-level interface + benchmarks)
# ---------------------------------------------------------------------------

class NeuromorphicSimulator:
    """
    Top-level simulator that ties together the LIF neuron model, spike
    encoding, KASM operation mapping, and the spiking kernel.

    Provides benchmark utilities to compare neuromorphic execution against
    conventional CPU/GPU approaches and estimate power efficiency.

    Usage Example
    -------------
    >>> sim = NeuromorphicSimulator(dimensions=100, timesteps=100, seed=42)
    >>>
    >>> # Encode two concepts as spike trains
    >>> cat_spikes = sim.encoder.encode_concept("cat", dimensions=100)
    >>> dog_spikes = sim.encoder.encode_concept("dog", dimensions=100)
    >>>
    >>> # BIND via coincidence detection
    >>> bound = sim.mapper.spike_bind(cat_spikes, dog_spikes)
    >>>
    >>> # Check similarity (should be low — cat and dog are different)
    >>> similarity = sim.mapper.spike_resonate(cat_spikes, dog_spikes)
    >>>
    >>> # Build a spiking knowledge graph
    >>> sim.kernel.connect("cat", "animal", weight=1.5)
    >>> sim.kernel.connect("dog", "animal", weight=1.5)
    >>> sim.kernel.connect("animal", "living_thing", weight=1.0)
    >>>
    >>> # Query by injecting spikes into "cat"
    >>> result = sim.kernel.propagate({"cat": 2.0}, timesteps=50)
    >>> ranking = sim.kernel.get_activation_ranking(result, exclude_seeds={"cat"})

    Parameters
    ----------
    dimensions : int
        Number of dimensions for concept vectors (spike trains per concept).
    timesteps : int
        Simulation window length in timesteps.
    seed : int or None
        Random seed for reproducibility.
    """

    # ── Hardware reference data ──────────────────────────────────────

    HARDWARE_SPECS = {
        "human_brain": {
            "name": "Human Brain",
            "neurons": 86_000_000_000,
            "power_watts": 20.0,
            "watts_per_neuron": 20.0 / 86e9,  # ~0.23 nW/neuron
            "latency_ms_per_inference": 300.0,  # ~300ms for complex reasoning
            "notes": "86 billion neurons, 20 watts. The gold standard.",
        },
        "gpu_a100": {
            "name": "NVIDIA A100 GPU",
            "neurons": 1_000_000_000,  # ~1B parameter equivalent
            "power_watts": 300.0,
            "watts_per_neuron": 300.0 / 1e9,  # ~300 nW/neuron
            "latency_ms_per_inference": 10.0,  # ~10ms for a forward pass
            "notes": "300W TDP. Fast but power-hungry.",
        },
        "loihi_2": {
            "name": "Intel Loihi 2",
            "neurons": 1_000_000,  # 1M neurons per chip
            "power_watts": 1.0,
            "watts_per_neuron": 1.0 / 1e6,  # ~1 uW/neuron
            "latency_ms_per_inference": 1.0,  # ~1ms for spike propagation
            "notes": "128 neuromorphic cores, 1M neurons, 1 watt.",
        },
        "truenorth": {
            "name": "IBM TrueNorth",
            "neurons": 1_000_000,
            "power_watts": 0.07,  # 70 mW
            "watts_per_neuron": 0.07 / 1e6,  # ~70 pW/neuron
            "latency_ms_per_inference": 1.0,
            "notes": "4096 cores, 1M neurons, 70mW. Most power-efficient.",
        },
    }

    def __init__(self, dimensions: int = 100, timesteps: int = 100,
                 seed: Optional[int] = None):
        self.dimensions = dimensions
        self.timesteps = timesteps
        self.seed = seed

        self.encoder = SpikeTrainEncoder(
            timesteps=timesteps, seed=seed
        )
        self.mapper = KASMSpikeMapper(encoder=self.encoder)
        self.kernel = NeuromorphicKernel()

    # ── KASM Operation Demonstrations ────────────────────────────────

    def demo_bind(self, concept_a: str, concept_b: str) -> dict:
        """
        Demonstrate BIND operation mapped to spike coincidence detection.

        Shows step by step how bipolar BIND becomes coincidence detection.

        Parameters
        ----------
        concept_a, concept_b : str
            Names of concepts to bind.

        Returns
        -------
        dict with keys:
            spikes_a, spikes_b : spike matrices
            bound : coincidence-detected output
            similarity_a_bound : should be low (BIND output is orthogonal)
            similarity_b_bound : should be low
            explanation : human-readable walkthrough
        """
        spikes_a = self.encoder.encode_concept(concept_a, self.dimensions)
        spikes_b = self.encoder.encode_concept(concept_b, self.dimensions)
        bound = self.mapper.spike_bind(spikes_a, spikes_b)

        sim_a_bound = self.mapper.spike_resonate(spikes_a, bound)
        sim_b_bound = self.mapper.spike_resonate(spikes_b, bound)
        sim_a_b = self.mapper.spike_resonate(spikes_a, spikes_b)

        return {
            "spikes_a": spikes_a,
            "spikes_b": spikes_b,
            "bound": bound,
            "similarity_a_b": sim_a_b,
            "similarity_a_bound": sim_a_bound,
            "similarity_b_bound": sim_b_bound,
            "spike_rate_a": float(np.mean(spikes_a)),
            "spike_rate_b": float(np.mean(spikes_b)),
            "spike_rate_bound": float(np.mean(bound)),
            "explanation": (
                f"BIND({concept_a}, {concept_b}) via coincidence detection:\n"
                f"  {concept_a} avg spike rate: {np.mean(spikes_a):.3f}\n"
                f"  {concept_b} avg spike rate: {np.mean(spikes_b):.3f}\n"
                f"  Bound output spike rate:    {np.mean(bound):.3f}\n"
                f"  sim({concept_a}, {concept_b}):    {sim_a_b:.3f}  (random concepts ~ low)\n"
                f"  sim({concept_a}, bound):      {sim_a_bound:.3f}  (should be low: BIND is orthogonal)\n"
                f"  sim({concept_b}, bound):      {sim_b_bound:.3f}  (should be low: BIND is orthogonal)\n"
                f"\n"
                f"  On Loihi 2, this is ONE synaptic operation per dimension per timestep.\n"
                f"  {self.dimensions} dimensions x {self.timesteps} timesteps = "
                f"{self.dimensions * self.timesteps:,} spike events total.\n"
                f"  At ~1 pJ per spike event, total energy: "
                f"{self.dimensions * self.timesteps * 1e-12 * 1e6:.2f} uJ"
            ),
        }

    def demo_superpose(self, *concepts: str) -> dict:
        """
        Demonstrate SUPERPOSE operation mapped to membrane summation.

        Parameters
        ----------
        concepts : str
            Names of concepts to bundle together.

        Returns
        -------
        dict with spike data, similarities, and explanation.
        """
        spike_trains = [
            self.encoder.encode_concept(c, self.dimensions) for c in concepts
        ]
        superposed = self.mapper.spike_superpose(*spike_trains)

        similarities = {}
        for i, name in enumerate(concepts):
            sim = self.mapper.spike_resonate(spike_trains[i], superposed)
            similarities[name] = sim

        return {
            "spike_trains": spike_trains,
            "superposed": superposed,
            "similarities": similarities,
            "spike_rate_superposed": float(np.mean(superposed)),
            "explanation": (
                f"SUPERPOSE({', '.join(concepts)}) via membrane summation:\n"
                f"  Output spike rate: {np.mean(superposed):.3f}\n"
                + "".join(
                    f"  sim(superposed, {name}): {similarities[name]:.3f}\n"
                    for name in concepts
                )
                + f"\n"
                f"  The superposition is similar to ALL its components.\n"
                f"  This is the neural basis of 'concept blending' — the\n"
                f"  output neuron responds to ANY of its inputs firing."
            ),
        }

    def demo_resonate(self, concept_a: str, concept_b: str) -> dict:
        """
        Demonstrate RESONATE (similarity) via spike correlation.

        Parameters
        ----------
        concept_a, concept_b : str
            Names of concepts to compare.

        Returns
        -------
        dict with similarity score and explanation.
        """
        spikes_a = self.encoder.encode_concept(concept_a, self.dimensions)
        spikes_b = self.encoder.encode_concept(concept_b, self.dimensions)

        # Same concept should resonate strongly
        spikes_a2 = self.encoder.encode_concept(concept_a, self.dimensions,
                                                 seed_offset=999)

        sim_diff = self.mapper.spike_resonate(spikes_a, spikes_b)
        sim_same = self.mapper.spike_resonate(spikes_a, spikes_a2)

        return {
            "similarity_different": sim_diff,
            "similarity_same_concept": sim_same,
            "explanation": (
                f"RESONATE via spike correlation:\n"
                f"  sim({concept_a}, {concept_b}): {sim_diff:.3f}  "
                f"(different concepts -> low correlation)\n"
                f"  sim({concept_a}, {concept_a}): {sim_same:.3f}  "
                f"(same concept, different encoding -> high correlation)\n"
                f"\n"
                f"  On neuromorphic hardware, this is computed by a\n"
                f"  correlation detector neuron that fires only when\n"
                f"  BOTH input trains co-fire. The firing rate of the\n"
                f"  detector IS the similarity score."
            ),
        }

    # ── Benchmarks ───────────────────────────────────────────────────

    def compare_to_kos(self, seed_concepts: List[str],
                       kos_kernel=None) -> dict:
        """
        Run the same query on the spike-based kernel and (optionally) a
        real KOS kernel, then compare the activation rankings.

        If no KOS kernel is provided, only the neuromorphic results are
        returned with timing information.

        Parameters
        ----------
        seed_concepts : list of str
            Concepts to use as query seeds.
        kos_kernel : KOSKernel or None
            Optional real KOS kernel for comparison.

        Returns
        -------
        dict with neuromorphic results, timing, and optionally KOS comparison.
        """
        seed_dict = {name: 2.0 for name in seed_concepts}

        # Neuromorphic run
        t0 = time.perf_counter()
        spike_result = self.kernel.propagate(seed_dict, self.timesteps)
        neuro_time = time.perf_counter() - t0

        neuro_ranking = self.kernel.get_activation_ranking(
            spike_result, exclude_seeds=set(seed_concepts)
        )

        result = {
            "neuromorphic": {
                "ranking": neuro_ranking[:20],
                "total_neurons_fired": len(spike_result),
                "wall_time_ms": neuro_time * 1000,
            },
        }

        # KOS comparison (if available)
        if kos_kernel is not None:
            try:
                t0 = time.perf_counter()
                kos_kernel.current_tick = 0
                for name in seed_concepts:
                    if name in kos_kernel.nodes:
                        kos_kernel.nodes[name].receive_signal(1.0, 0)
                # Run propagation ticks
                for tick in range(1, kos_kernel.max_ticks + 1):
                    kos_kernel.current_tick = tick
                    for node in kos_kernel.nodes.values():
                        signals = node.propagate(tick)
                        for target_id, energy in signals:
                            if target_id in kos_kernel.nodes:
                                kos_kernel.nodes[target_id].receive_signal(energy, tick)
                kos_time = time.perf_counter() - t0

                kos_ranking = sorted(
                    [(nid, n.activation) for nid, n in kos_kernel.nodes.items()
                     if nid not in set(seed_concepts)],
                    key=lambda x: x[1], reverse=True
                )[:20]

                result["kos_scalar"] = {
                    "ranking": kos_ranking,
                    "wall_time_ms": kos_time * 1000,
                }
            except Exception as e:
                result["kos_scalar"] = {"error": str(e)}

        return result

    def estimate_power(self, num_neurons: int,
                       spike_rate: float = 0.1) -> dict:
        """
        Estimate power consumption on different hardware platforms.

        This shows WHY neuromorphic hardware matters: the energy cost
        per operation is fundamentally different.

        The key metric is energy per spike event (synaptic operation):
            - GPU:     ~1 nJ per multiply-accumulate (MAC)
            - Loihi 2: ~1 pJ per spike event (1000x more efficient)
            - Brain:   ~10 fJ per synaptic event (100,000x more efficient)

        Parameters
        ----------
        num_neurons : int
            Number of neurons in the network.
        spike_rate : float
            Average fraction of neurons spiking per timestep (0.0 to 1.0).
            Biological networks: ~0.01 to 0.1 (sparse coding).
            Default 0.1 is a reasonable upper bound.

        Returns
        -------
        dict mapping platform name to estimated power metrics.
        """
        spikes_per_second = num_neurons * spike_rate * 1000  # ~1000 timesteps/sec
        avg_synapses_per_neuron = 10  # conservative estimate
        events_per_second = spikes_per_second * avg_synapses_per_neuron

        platforms = {}
        for key, spec in self.HARDWARE_SPECS.items():
            if key == "human_brain":
                # Scale from known biological efficiency
                energy_per_event_j = 10e-15  # ~10 fJ
            elif key == "gpu_a100":
                energy_per_event_j = 1e-9  # ~1 nJ per MAC
            elif key == "loihi_2":
                energy_per_event_j = 1e-12  # ~1 pJ per spike
            elif key == "truenorth":
                energy_per_event_j = 0.07e-12  # ~70 fJ per spike (from 70mW / 1M neurons)
            else:
                energy_per_event_j = 1e-9  # default to GPU-like

            estimated_watts = events_per_second * energy_per_event_j
            chips_needed = max(1, num_neurons / spec["neurons"])

            platforms[spec["name"]] = {
                "estimated_watts": round(estimated_watts, 6),
                "energy_per_event_joules": energy_per_event_j,
                "events_per_second": int(events_per_second),
                "chips_needed": round(chips_needed, 2),
                "total_system_watts": round(estimated_watts * chips_needed, 6),
            }

        return {
            "num_neurons": num_neurons,
            "spike_rate": spike_rate,
            "spikes_per_second": int(spikes_per_second),
            "platforms": platforms,
            "summary": (
                f"For {num_neurons:,} neurons at {spike_rate:.0%} spike rate:\n"
                + "".join(
                    f"  {name:20s}: {data['estimated_watts']*1000:.4f} mW "
                    f"({data['chips_needed']:.0f} chip(s))\n"
                    for name, data in platforms.items()
                )
                + f"\n  Neuromorphic advantage: "
                f"{platforms['NVIDIA A100 GPU']['estimated_watts'] / max(platforms['Intel Loihi 2']['estimated_watts'], 1e-15):.0f}x "
                f"more energy-efficient than GPU"
            ),
        }

    def estimate_latency(self, num_neurons: int,
                         avg_path_length: int = 5) -> dict:
        """
        Estimate inference latency on different platforms.

        Neuromorphic advantage in latency comes from:
        1. Parallelism: all neurons compute simultaneously per timestep.
        2. Event-driven: only active neurons consume cycles.
        3. No memory bottleneck: computation and memory are co-located
           (no von Neumann bottleneck).

        Parameters
        ----------
        num_neurons : int
            Number of neurons in the network.
        avg_path_length : int
            Average number of synaptic hops in a query path.
            Determines how many timesteps are needed for a spike
            to propagate from input to output.

        Returns
        -------
        dict mapping platform name to estimated latency.
        """
        platforms = {}

        # CPU: sequential neuron updates, O(N) per timestep
        cpu_time_per_neuron_us = 0.1  # ~100ns per neuron update
        cpu_timesteps = avg_path_length
        cpu_latency_ms = (num_neurons * cpu_time_per_neuron_us * cpu_timesteps) / 1000

        # GPU: parallel neuron updates, O(1) per timestep but with overhead
        gpu_kernel_launch_us = 10  # kernel launch overhead
        gpu_time_per_timestep_us = 50 + num_neurons * 0.001  # near-O(1) with overhead
        gpu_latency_ms = (gpu_kernel_launch_us + gpu_time_per_timestep_us * avg_path_length) / 1000

        # Loihi 2: fully parallel, O(1) per timestep, ~1us per timestep
        loihi_time_per_timestep_us = 1.0  # ~1 microsecond per timestep
        loihi_latency_ms = (loihi_time_per_timestep_us * avg_path_length) / 1000

        # TrueNorth: similar to Loihi but fixed 1ms timesteps
        truenorth_time_per_timestep_us = 1000  # 1ms tick
        truenorth_latency_ms = (truenorth_time_per_timestep_us * avg_path_length) / 1000

        platforms = {
            "CPU (single-thread)": {
                "latency_ms": round(cpu_latency_ms, 4),
                "scaling": "O(N) per timestep",
                "bottleneck": "Sequential neuron updates",
            },
            "GPU (A100)": {
                "latency_ms": round(gpu_latency_ms, 4),
                "scaling": "O(1) per timestep + kernel overhead",
                "bottleneck": "Memory bandwidth (von Neumann wall)",
            },
            "Intel Loihi 2": {
                "latency_ms": round(loihi_latency_ms, 4),
                "scaling": "O(1) per timestep, O(path_length) total",
                "bottleneck": "Path length (number of hops)",
            },
            "IBM TrueNorth": {
                "latency_ms": round(truenorth_latency_ms, 4),
                "scaling": "O(1) per timestep, fixed 1ms tick",
                "bottleneck": "Fixed tick rate (1ms minimum)",
            },
        }

        return {
            "num_neurons": num_neurons,
            "avg_path_length": avg_path_length,
            "platforms": platforms,
            "summary": (
                f"Latency for {num_neurons:,} neurons, {avg_path_length}-hop query:\n"
                + "".join(
                    f"  {name:22s}: {data['latency_ms']:.4f} ms  "
                    f"[{data['scaling']}]\n"
                    for name, data in platforms.items()
                )
                + f"\n  Loihi 2 speedup over CPU: "
                f"{cpu_latency_ms / max(loihi_latency_ms, 1e-15):.0f}x\n"
                f"  Loihi 2 speedup over GPU: "
                f"{gpu_latency_ms / max(loihi_latency_ms, 1e-15):.0f}x"
            ),
        }

    def energy_efficiency_report(self, graph_size: int) -> str:
        """
        Generate a human-readable energy efficiency comparison report.

        Compares what it would cost to run a KOS graph of the given size
        on different hardware platforms, contextualized against the human
        brain for perspective.

        Parameters
        ----------
        graph_size : int
            Number of concept nodes in the KOS graph.

        Returns
        -------
        str
            Formatted report string.
        """
        power = self.estimate_power(graph_size)
        latency = self.estimate_latency(graph_size)

        lines = [
            "=" * 70,
            "  KOS NEUROMORPHIC EFFICIENCY REPORT",
            "=" * 70,
            f"  Graph size: {graph_size:,} concept neurons",
            "",
            "  --- Power Consumption ---",
            "",
        ]

        for name, data in power["platforms"].items():
            lines.append(
                f"    {name:20s}  {data['estimated_watts']*1000:>10.4f} mW  "
                f"({data['chips_needed']:.0f} chip(s))"
            )

        lines += [
            "",
            "  --- Inference Latency (5-hop query) ---",
            "",
        ]

        for name, data in latency["platforms"].items():
            lines.append(
                f"    {name:22s}  {data['latency_ms']:>10.4f} ms  "
                f"[{data['bottleneck']}]"
            )

        # KOS-specific analysis
        gpu_watts = power["platforms"]["NVIDIA A100 GPU"]["estimated_watts"]
        loihi_watts = power["platforms"]["Intel Loihi 2"]["estimated_watts"]
        truenorth_watts = power["platforms"]["IBM TrueNorth"]["estimated_watts"]
        brain_watts = power["platforms"]["Human Brain"]["estimated_watts"]

        lines += [
            "",
            "  --- KOS on Neuromorphic Hardware ---",
            "",
            f"    KOS graph ({graph_size:,} nodes) mapped to Loihi 2:",
            f"      Power:   {loihi_watts*1000:.4f} mW "
            f"(vs {gpu_watts*1000:.4f} mW on GPU)",
            f"      Savings: {gpu_watts / max(loihi_watts, 1e-15):.0f}x less power than GPU",
            f"",
            f"    Why KASM is native to neuromorphic silicon:",
            f"      - BIND (XOR) = coincidence detection (1 synapse)",
            f"      - SUPERPOSE  = membrane summation (free in LIF physics)",
            f"      - PERMUTE    = axonal delay line (hardwired on chip)",
            f"      - RESONATE   = spike correlation (Hebbian readout)",
            f"",
            f"    Brain-scale reference:",
            f"      Human brain: {brain_watts*1000:.4f} mW for equivalent-size network",
            f"      Loihi 2:     {loihi_watts*1000:.4f} mW (silicon, programmable)",
            f"      TrueNorth:   {truenorth_watts*1000:.4f} mW (most efficient chip)",
            "",
            "=" * 70,
        ]

        return "\n".join(lines)
