"""
KOS V5.0 — Metacognition Engine (System 2 Thinking).

Layer 4: ShadowKernel — simulates multiple interpretations of a query
in parallel, measures logical tension (contradictions), and selects
the path with highest energy and lowest friction.

Layer 5: Active Inference — if System Entropy exceeds the safety
threshold, the OS autonomously forages for knowledge to resolve
its own uncertainty before answering.

References:
    - Kahneman (2011): System 1 (fast/intuitive) vs System 2 (slow/deliberate)
    - Friston (2010): Free Energy Principle — organisms minimize surprise
    - Browne et al. (2012): Monte Carlo Tree Search for decision evaluation
"""


class ShadowKernel:
    """
    The System 2 Thinker.

    Instead of answering immediately (System 1), the ShadowKernel
    evaluates multiple possible interpretations of a query by running
    them through the graph engine and measuring:

    1. Energy: How strongly does the graph respond? (signal strength)
    2. Tension: How many contradictions does the path hit? (friction)
    3. Score = Energy - Tension  (net confidence)

    SYSTEM_ENTROPY is the global "anxiety" metric:
        0.0  = perfect confidence, zero contradictions
        >15  = high uncertainty, triggers Active Inference
        100  = total confusion, no paths found at all
    """

    def __init__(self, kernel):
        self.k = kernel
        self.SYSTEM_ENTROPY = 0.0

    def _calculate_path_tension(self, seeds: list, results: list) -> float:
        """
        Measures the logical friction of a thought path.

        Tension accumulates when:
        - A seed has an inhibitory (negative weight) edge to a result
        - Multiple results contradict each other
        - The path crosses suppressed nodes

        High tension = the graph is saying "this doesn't make sense."
        """
        tension = 0.0
        result_dict = dict(results) if results else {}

        for ans_uuid, energy in result_dict.items():
            for seed in seeds:
                if seed not in self.k.nodes:
                    continue
                conn = self.k.nodes[seed].connections
                if ans_uuid in conn:
                    w = conn[ans_uuid]
                    # Extract weight whether dict-style or scalar
                    if isinstance(w, dict):
                        w = w.get('w', 0.0)
                    if w < 0.0:
                        # Hitting an inhibitory edge = contradiction
                        tension += abs(w) * 20.0

        # Additional tension: if multiple results have opposing signs
        # relative to the same seed, that's internal conflict
        energies = list(result_dict.values())
        if len(energies) >= 2:
            pos = sum(1 for e in energies if e > 0)
            neg = sum(1 for e in energies if e < 0)
            if pos > 0 and neg > 0:
                tension += min(pos, neg) * 10.0  # Internal conflict penalty

        return tension

    def _calculate_coverage(self, seeds: list, results: list) -> float:
        """
        How many of our seeds actually produced results?
        Low coverage = the graph doesn't have enough knowledge.
        """
        if not seeds:
            return 0.0
        result_nodes = {r[0] for r in results} if results else set()
        seeds_with_results = 0
        for s in seeds:
            if s in self.k.nodes and self.k.nodes[s].connections:
                # This seed has outbound edges — it's connected
                seeds_with_results += 1
        return seeds_with_results / len(seeds)

    def think_before_speaking(self, seed_branches: list,
                               verbose: bool = True) -> dict:
        """
        Evaluates 2-3 possible interpretations of the user's query.

        For each branch:
            1. Run spreading activation (the "simulation")
            2. Measure energy (signal strength)
            3. Measure tension (contradictions)
            4. Calculate net score = energy - tension
            5. Calculate coverage (how much of the query was answered)

        Returns the best branch, or None if all branches fail.
        Updates SYSTEM_ENTROPY to reflect the machine's confidence.
        """
        if verbose:
            print("\n[SYSTEM 2] Engaging Metacognitive Shadow Simulation...")

        best_branch = None
        best_score = -9999.0

        for i, branch_seeds in enumerate(seed_branches):
            if not branch_seeds:
                continue

            # Filter to only seeds that exist in the graph
            valid_seeds = [s for s in branch_seeds if s in self.k.nodes]
            if not valid_seeds:
                if verbose:
                    print(f"  -> Branch {i+1}: No valid seeds in graph. SKIP.")
                continue

            # Simulate the thought — run spreading activation
            results = self.k.query(valid_seeds, top_k=5)

            if not results:
                if verbose:
                    print(f"  -> Branch {i+1}: Zero activation. Dead path.")
                continue

            # Evaluate the thought
            top_energy = results[0][1]
            total_energy = sum(e for _, e in results)
            path_tension = self._calculate_path_tension(valid_seeds, results)
            coverage = self._calculate_coverage(valid_seeds, results)

            # The Metacognitive Formula:
            # High total energy (strong signal)
            # + High coverage (all seeds answered)
            # - High tension (contradictions)
            branch_score = total_energy + (coverage * 5.0) - path_tension

            if verbose:
                print(f"  -> Branch {i+1} [{len(valid_seeds)} seeds]: "
                      f"Energy={total_energy:.2f} | "
                      f"Coverage={coverage:.0%} | "
                      f"Tension={path_tension:.2f} | "
                      f"Score={branch_score:.2f}")

            if branch_score > best_score:
                best_score = branch_score
                best_branch = {
                    "seeds": valid_seeds,
                    "results": results,
                    "tension": path_tension,
                    "energy": total_energy,
                    "coverage": coverage,
                    "score": branch_score,
                }

        # Update System Entropy
        if best_branch:
            # Entropy = tension + inverse coverage penalty
            self.SYSTEM_ENTROPY = (
                best_branch["tension"]
                + (1.0 - best_branch["coverage"]) * 20.0
            )
            if verbose:
                print(f"  [*] Selected Branch: Score={best_branch['score']:.2f} | "
                      f"System Entropy={self.SYSTEM_ENTROPY:.2f}")
        else:
            self.SYSTEM_ENTROPY = 100.0
            if verbose:
                print(f"  [!] ALL BRANCHES FAILED. System Entropy = 100.0")

        return best_branch
