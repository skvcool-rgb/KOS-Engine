"""
KOS V6.0 — Experiment Engine

The autonomous scientific reasoning loop:
    HYPOTHESIZE → PREDICT → COMPUTE → COMPARE → HEAL → REPEAT

Combines all science drivers (Chemistry, Physics, Biology) with
the Predictive Coding Engine and Self-Healer to automatically
iterate toward viable solutions.

This is the scientific method, automated.
"""

import time
import copy
import re


class Hypothesis:
    """A testable scientific hypothesis with parameters."""

    def __init__(self, statement: str, parameters: dict = None,
                 expected_outcomes: dict = None):
        self.statement = statement
        self.parameters = parameters or {}
        self.expected_outcomes = expected_outcomes or {}
        self.iteration = 0
        self.history = []

    def modify(self, changes: dict, reason: str):
        """Create modified version tracking what changed and why."""
        self.history.append({
            "iteration": self.iteration,
            "parameters": copy.deepcopy(self.parameters),
            "reason": reason,
        })
        self.parameters.update(changes)
        self.iteration += 1

    def __repr__(self):
        return f"Hypothesis('{self.statement}', iter={self.iteration}, params={self.parameters})"


class ExperimentResult:
    """Result of a single experiment iteration."""

    def __init__(self):
        self.predictions = {}
        self.computations = {}
        self.errors = {}
        self.contradictions = []
        self.viable = False
        self.confidence = 0.0
        self.diagnosis = ""
        self.suggested_modification = {}

    def summary(self) -> str:
        lines = []
        lines.append(f"  Viable: {self.viable}")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        if self.predictions:
            lines.append(f"  Predictions: {self.predictions}")
        if self.computations:
            lines.append(f"  Computations: {self.computations}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        if self.contradictions:
            lines.append(f"  Contradictions: {self.contradictions}")
        if self.diagnosis:
            lines.append(f"  Diagnosis: {self.diagnosis}")
        return "\n".join(lines)


class ExperimentEngine:
    """
    Autonomous experimentation loop.

    Given a hypothesis, the engine:
    1. Predicts what should happen (from hypothesis parameters)
    2. Computes what actually happens (using science drivers)
    3. Compares prediction vs computation
    4. If error: diagnoses the failure, modifies hypothesis, loops
    5. If viable: returns confidence-scored result
    """

    def __init__(self, chemistry=None, physics=None, biology=None,
                 kernel=None, lexicon=None):
        self.chem = chemistry
        self.phys = physics
        self.bio = biology
        self.kernel = kernel
        self.lexicon = lexicon

    def run(self, hypothesis: Hypothesis, max_iterations: int = 10,
            energy_tolerance: float = 5.0, verbose: bool = True) -> dict:
        """
        Run the experiment loop until a viable solution is found
        or max iterations reached.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  EXPERIMENT ENGINE: Autonomous Hypothesis Testing")
            print(f"  Hypothesis: {hypothesis.statement}")
            print(f"  Max iterations: {max_iterations}")
            print(f"{'='*60}")

        results_history = []

        for i in range(max_iterations):
            if verbose:
                print(f"\n  --- Iteration {i+1} ---")
                print(f"  Parameters: {hypothesis.parameters}")

            result = self._run_single(hypothesis, verbose)
            results_history.append(result)

            if verbose:
                print(result.summary())

            if result.viable:
                if verbose:
                    print(f"\n  VIABLE SOLUTION FOUND at iteration {i+1}!")
                return {
                    "status": "VIABLE",
                    "iterations": i + 1,
                    "final_hypothesis": hypothesis,
                    "result": result,
                    "history": results_history,
                    "confidence": result.confidence,
                }

            # Self-heal: modify hypothesis based on diagnosis
            if result.suggested_modification:
                reason = result.diagnosis
                if verbose:
                    print(f"  Self-healer: {reason}")
                    print(f"  Modification: {result.suggested_modification}")
                hypothesis.modify(result.suggested_modification, reason)
            else:
                if verbose:
                    print(f"  No modification suggested. Stopping.")
                break

        return {
            "status": "INCONCLUSIVE",
            "iterations": len(results_history),
            "final_hypothesis": hypothesis,
            "history": results_history,
            "confidence": results_history[-1].confidence if results_history else 0,
        }

    def _run_single(self, hypothesis: Hypothesis, verbose: bool) -> ExperimentResult:
        """Run one iteration of predict-compute-compare."""
        result = ExperimentResult()
        params = hypothesis.parameters

        # ── PREDICT ──────────────────────────────────────
        # What should happen if the hypothesis is true?

        if "energy_required" in params and "energy_available" in params:
            result.predictions["energy_balance"] = (
                params["energy_available"] - params["energy_required"]
            )
            result.predictions["energy_sufficient"] = (
                params["energy_available"] >= params["energy_required"]
            )

        if "temperature" in params and "max_safe_temp" in params:
            result.predictions["temp_safe"] = (
                params["temperature"] <= params["max_safe_temp"]
            )

        if "wavelength_nm" in params and self.phys:
            photon = self.phys.photon_energy(params["wavelength_nm"])
            result.predictions["photon_energy_kJ_mol"] = photon["energy_kJ_mol"]
            result.predictions["photon_energy_eV"] = photon["energy_eV"]

        if "concentration" in params and "ec50" in params and "ld50" in params:
            conc = params["concentration"]
            result.predictions["in_therapeutic_window"] = (
                params["ec50"] <= conc <= params["ld50"]
            )

        # ── COMPUTE ──────────────────────────────────────
        # What actually happens according to science?

        checks_passed = 0
        checks_total = 0

        # Energy balance check
        if "energy_balance" in result.predictions:
            checks_total += 1
            balance = result.predictions["energy_balance"]
            result.computations["energy_balance_kJ"] = balance

            if balance >= -5.0:  # within tolerance
                checks_passed += 1
                result.computations["energy_verdict"] = "SUFFICIENT"
            else:
                result.computations["energy_verdict"] = "INSUFFICIENT"
                result.errors["energy"] = f"Deficit of {abs(balance):.1f} kJ/mol"

        # Temperature safety check
        if "temp_safe" in result.predictions:
            checks_total += 1
            if result.predictions["temp_safe"]:
                checks_passed += 1
                result.computations["temp_verdict"] = "SAFE"
            else:
                result.computations["temp_verdict"] = "EXCEEDS LIMIT"
                result.errors["temperature"] = (
                    f"{params['temperature']}K exceeds max {params['max_safe_temp']}K"
                )

        # Photon energy vs threshold
        if "photon_energy_kJ_mol" in result.predictions:
            pe = result.predictions["photon_energy_kJ_mol"]

            if "repair_threshold_kJ" in params:
                checks_total += 1
                if pe >= params["repair_threshold_kJ"]:
                    checks_passed += 1
                    result.computations["repair_energy_verdict"] = "SUFFICIENT"
                else:
                    result.computations["repair_energy_verdict"] = "INSUFFICIENT"
                    result.errors["repair_energy"] = (
                        f"Photon {pe:.0f} kJ/mol < threshold {params['repair_threshold_kJ']} kJ/mol"
                    )

            if "degradation_threshold_kJ" in params:
                checks_total += 1
                if pe < params["degradation_threshold_kJ"]:
                    checks_passed += 1
                    result.computations["degradation_safe"] = "SAFE"
                else:
                    result.computations["degradation_safe"] = "DANGEROUS"
                    result.errors["degradation"] = (
                        f"Photon {pe:.0f} kJ/mol >= degradation threshold {params['degradation_threshold_kJ']} kJ/mol"
                    )
                    result.contradictions.append(
                        "Light energy sufficient for repair but also causes degradation"
                    )

        # Therapeutic window check
        if "in_therapeutic_window" in result.predictions:
            checks_total += 1
            if result.predictions["in_therapeutic_window"]:
                checks_passed += 1
                result.computations["therapeutic_verdict"] = "IN WINDOW"
            else:
                conc = params["concentration"]
                if conc < params["ec50"]:
                    result.errors["dosage"] = "Sub-therapeutic (below EC50)"
                else:
                    result.errors["dosage"] = "TOXIC (above LD50)"

        # Bond feasibility check
        if "element_a" in params and "element_b" in params and self.chem:
            checks_total += 1
            bond_result = self.chem.check_reaction(params["element_a"], params["element_b"])
            if bond_result.get("feasible"):
                checks_passed += 1
                result.computations["bond_verdict"] = f"FEASIBLE ({bond_result['bond_type']})"
                result.computations["product"] = bond_result.get("product", "?")
            else:
                result.errors["bonding"] = bond_result.get("reason", "Cannot bond")

        # Biological feasibility
        if "enzyme_vmax" in params and "enzyme_km" in params and "substrate_conc" in params:
            checks_total += 1
            if self.bio:
                rate = self.bio.michaelis_menten(
                    params["enzyme_vmax"], params["enzyme_km"], params["substrate_conc"]
                )
                result.computations["reaction_rate"] = rate
                if rate.get("rate", 0) > 0:
                    checks_passed += 1
                    result.computations["enzyme_verdict"] = f"ACTIVE (rate={rate['rate']:.4f})"
                else:
                    result.errors["enzyme"] = "Zero reaction rate"

        # ── COMPARE ──────────────────────────────────────

        if checks_total == 0:
            result.confidence = 0.0
            result.diagnosis = "No testable parameters found"
            result.viable = False
        else:
            result.confidence = checks_passed / checks_total
            result.viable = (checks_passed == checks_total) and not result.contradictions

        # ── DIAGNOSE & SUGGEST ───────────────────────────

        if not result.viable:
            if result.contradictions:
                result.diagnosis = f"Contradiction: {result.contradictions[0]}"
                # Suggest resolving the contradiction
                if "wavelength_nm" in params:
                    # Try a different wavelength
                    current = params["wavelength_nm"]
                    if "degradation_threshold_kJ" in params and "repair_threshold_kJ" in params:
                        # Find wavelength between thresholds
                        # E = hc/lambda → lambda = hc/E
                        mid_energy = (params["repair_threshold_kJ"] + params["degradation_threshold_kJ"]) / 2
                        target_wl = 119627 / mid_energy  # hc in kJ*nm/mol
                        result.suggested_modification = {"wavelength_nm": round(target_wl)}
                        result.diagnosis += f". Try wavelength {round(target_wl)}nm (between thresholds)"

            elif "energy" in result.errors:
                deficit = abs(result.computations.get("energy_balance_kJ", 0))
                result.diagnosis = f"Energy deficit: {deficit:.1f} kJ/mol"

                # Suggest adding energy source
                if "cofactor_energy" not in params:
                    result.suggested_modification = {"cofactor_energy": 30.5}  # ATP
                    result.diagnosis += ". Adding ATP cofactor (30.5 kJ/mol)"
                elif "photon_assist" not in params:
                    result.suggested_modification = {
                        "photon_assist": True,
                        "wavelength_nm": 450,  # Blue light
                    }
                    result.diagnosis += ". Adding blue light photon assist"
                else:
                    # Increase existing energy sources
                    if "cofactor_count" in params:
                        result.suggested_modification = {
                            "cofactor_count": params["cofactor_count"] + 1
                        }
                    else:
                        result.suggested_modification = {"cofactor_count": 2}
                    result.diagnosis += ". Increasing cofactor count"

            elif "temperature" in result.errors:
                result.diagnosis = result.errors["temperature"]
                result.suggested_modification = {
                    "temperature": params.get("max_safe_temp", 300) - 10
                }

            elif "dosage" in result.errors:
                result.diagnosis = result.errors["dosage"]
                if "Sub-therapeutic" in result.errors["dosage"]:
                    result.suggested_modification = {
                        "concentration": params["ec50"] * 1.5
                    }
                else:
                    result.suggested_modification = {
                        "concentration": params["ld50"] * 0.5
                    }

            elif "bonding" in result.errors:
                result.diagnosis = result.errors["bonding"]
                # No auto-fix for bonding — elements either bond or don't

        return result

    # ── Convenience: pre-built experiment templates ──────

    def test_energy_hypothesis(self, source_energy: float,
                                required_energy: float,
                                source_name: str = "unknown") -> dict:
        """Quick test: does source provide enough energy?"""
        h = Hypothesis(
            f"Can {source_name} provide {required_energy} kJ/mol?",
            parameters={
                "energy_available": source_energy,
                "energy_required": required_energy,
            }
        )
        return self.run(h, max_iterations=5, verbose=False)

    def test_photon_repair(self, wavelength_nm: float,
                            repair_threshold_kJ: float,
                            degradation_threshold_kJ: float) -> dict:
        """Test if a light wavelength can repair without degrading."""
        h = Hypothesis(
            f"Can {wavelength_nm}nm light repair (>{repair_threshold_kJ} kJ/mol) "
            f"without degrading (<{degradation_threshold_kJ} kJ/mol)?",
            parameters={
                "wavelength_nm": wavelength_nm,
                "repair_threshold_kJ": repair_threshold_kJ,
                "degradation_threshold_kJ": degradation_threshold_kJ,
            }
        )
        return self.run(h, max_iterations=5)

    def test_drug_dosage(self, drug_name: str, concentration: float,
                          ec50: float, ld50: float) -> dict:
        """Test if a drug concentration is in the therapeutic window."""
        h = Hypothesis(
            f"Is {concentration}mg/L of {drug_name} therapeutically viable?",
            parameters={
                "concentration": concentration,
                "ec50": ec50,
                "ld50": ld50,
            }
        )
        return self.run(h, max_iterations=5)
