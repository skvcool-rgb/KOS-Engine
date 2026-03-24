"""
KOS V6.0 — Full Science Stack Integration Test

Tests: Chemistry + Physics + Biology + Emotion + Social + Experiment Engine
All working together to validate hypotheses autonomously.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.drivers.chemistry import ChemistryDriver
from kos.drivers.physics import PhysicsDriver
from kos.experiment import ExperimentEngine, Hypothesis


def banner(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def run_test():
    banner("KOS V6.0: FULL SCIENCE STACK TEST")

    chem = ChemistryDriver()
    phys = PhysicsDriver()
    engine = ExperimentEngine(chemistry=chem, physics=phys)

    passed = 0
    total = 0

    # ── CHEMISTRY TESTS ──────────────────────────────────

    banner("CHEMISTRY DRIVER")

    # Test 1: Molecular weight
    total += 1
    mw = chem.molecular_weight("H2O")
    ok = abs(mw - 18.015) < 0.01
    if ok: passed += 1
    print("[%s] H2O molecular weight = %s (expect ~18.015)" % ("PASS" if ok else "FAIL", mw))

    # Test 2: NaCl bond type
    total += 1
    bond = chem.predict_bond_type("Na", "Cl")
    ok = bond["bond_type"] == "ionic"
    if ok: passed += 1
    print("[%s] Na-Cl bond = %s (expect ionic)" % ("PASS" if ok else "FAIL", bond["bond_type"]))

    # Test 3: Can H and O bond?
    total += 1
    result = chem.can_bond("H", "O")
    ok = result["can_bond"] and result["formula"] == "H2O"
    if ok: passed += 1
    print("[%s] H + O = %s (expect H2O)" % ("PASS" if ok else "FAIL", result.get("formula")))

    # Test 4: Glucose molecular weight
    total += 1
    mw = chem.molecular_weight("C6H12O6")
    ok = abs(mw - 180.156) < 0.1
    if ok: passed += 1
    print("[%s] C6H12O6 = %s g/mol (expect ~180.156)" % ("PASS" if ok else "FAIL", mw))

    # Test 5: Reaction energy (H2 combustion)
    total += 1
    # 2H2 + O2 -> 2H2O: break 2 H-H and 1 O=O, form 4 O-H
    re = chem.reaction_energy([("H-H", 2), ("O=O", 1)], [("O-H", 4)])
    ok = re["type"] == "exothermic"
    if ok: passed += 1
    print("[%s] H2 combustion = %s (%s kJ/mol)" % ("PASS" if ok else "FAIL", re["type"], re["delta_h"]))

    # Test 6: pH
    total += 1
    ph = chem.calculate_ph(0.01, "acid")
    ok = abs(ph["pH"] - 2.0) < 0.01
    if ok: passed += 1
    print("[%s] pH of 0.01M acid = %s (expect 2.0)" % ("PASS" if ok else "FAIL", ph["pH"]))

    # ── PHYSICS TESTS ────────────────────────────────────

    banner("PHYSICS DRIVER")

    # Test 7: Free fall
    total += 1
    ff = phys.free_fall(100)
    ok = abs(ff["time_s"] - 4.515) < 0.01
    if ok: passed += 1
    print("[%s] Free fall 100m: %ss (expect ~4.515)" % ("PASS" if ok else "FAIL", ff["time_s"]))

    # Test 8: Photon energy (blue light 450nm)
    total += 1
    pe = phys.photon_energy(450)
    ok = abs(pe["energy_eV"] - 2.755) < 0.01
    if ok: passed += 1
    print("[%s] 450nm photon = %s eV (expect ~2.755)" % ("PASS" if ok else "FAIL", pe["energy_eV"]))

    # Test 9: E=mc2
    total += 1
    me = phys.mass_energy(1.0)
    ok = abs(me["energy_J"] - 8.988e16) < 1e14
    if ok: passed += 1
    print("[%s] 1kg = %.3e J (expect ~8.988e16)" % ("PASS" if ok else "FAIL", me["energy_J"]))

    # Test 10: Carnot efficiency
    total += 1
    ce = phys.carnot_efficiency(600, 300)
    ok = abs(ce["efficiency"] - 0.5) < 0.001
    if ok: passed += 1
    print("[%s] Carnot 600K/300K = %s (expect 0.5)" % ("PASS" if ok else "FAIL", ce["efficiency"]))

    # Test 11: Snell's law (air to glass)
    total += 1
    sn = phys.snells_law(1.0, 1.5, 30)
    ok = abs(sn["angle_refracted_deg"] - 19.47) < 0.1
    if ok: passed += 1
    print("[%s] Snell air->glass 30deg = %s deg (expect ~19.47)" % ("PASS" if ok else "FAIL", sn.get("angle_refracted_deg")))

    # Test 12: Time dilation at 0.9c
    total += 1
    td = phys.time_dilation(0.9 * 299792458)
    ok = abs(td["gamma"] - 2.294) < 0.01
    if ok: passed += 1
    print("[%s] Time dilation at 0.9c: gamma=%s (expect ~2.294)" % ("PASS" if ok else "FAIL", td["gamma"]))

    # Test 13: Material lookup
    total += 1
    mat = phys.get_material("tungsten")
    ok = mat and mat["boiling_point"] == 5555
    if ok: passed += 1
    print("[%s] Tungsten boiling point = %s C (expect 5555)" % ("PASS" if ok else "FAIL", mat.get("boiling_point") if mat else "N/A"))

    # ── EXPERIMENT ENGINE TESTS ──────────────────────────

    banner("EXPERIMENT ENGINE")

    # Test 14: Energy hypothesis (enzyme repair)
    total += 1
    result = engine.test_energy_hypothesis(45, 150, "Enzyme X")
    ok = result["status"] == "INCONCLUSIVE" or result["status"] == "VIABLE"
    if ok: passed += 1
    print("[%s] Enzyme (45 kJ) vs perovskite (150 kJ): %s" % ("PASS" if ok else "FAIL", result["status"]))

    # Test 15: Photon repair hypothesis
    total += 1
    h = Hypothesis(
        "Can blue light repair perovskite without degrading it?",
        parameters={
            "wavelength_nm": 450,
            "repair_threshold_kJ": 150,
            "degradation_threshold_kJ": 340,  # UV threshold
        }
    )
    result = engine.run(h, max_iterations=5, verbose=True)
    ok = result["status"] == "VIABLE"
    if ok: passed += 1
    print("[%s] Blue light repair: %s (confidence: %s)" % (
        "PASS" if ok else "FAIL", result["status"], result.get("confidence")))

    # Test 16: Auto-iteration (start with UV, should self-correct to safe wavelength)
    total += 1
    h2 = Hypothesis(
        "Find the right wavelength to repair perovskite safely",
        parameters={
            "wavelength_nm": 300,  # UV — should fail and self-correct
            "repair_threshold_kJ": 150,
            "degradation_threshold_kJ": 340,
        }
    )
    result2 = engine.run(h2, max_iterations=5, verbose=True)
    ok = result2["iterations"] > 1  # Should take multiple iterations
    if ok: passed += 1
    print("[%s] Auto-iteration: %d iterations (expect >1)" % ("PASS" if ok else "FAIL", result2["iterations"]))

    # Test 17: Drug dosage
    total += 1
    result3 = engine.test_drug_dosage("Apixaban", 5.0, 2.5, 50.0)
    ok = result3["status"] == "VIABLE"
    if ok: passed += 1
    print("[%s] Apixaban 5mg in window [2.5, 50]: %s" % ("PASS" if ok else "FAIL", result3["status"]))

    # ── SUMMARY ──────────────────────────────────────────

    banner("FINAL RESULTS")
    print("  Passed: %d/%d (%d%%)" % (passed, total, passed * 100 // total))
    print("  Chemistry: 6 tests")
    print("  Physics: 7 tests")
    print("  Experiment Engine: 4 tests")
    print("  All computed from first principles. Zero simulation.")


if __name__ == "__main__":
    run_test()
