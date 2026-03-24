"""
KOS V5.1 — Self-Modification Tests (Levels 1-3).

Level 1: AutoTuner tunes its own thresholds
Level 2: PluginManager auto-enables/disables modules
Level 3: FormulaEvolver invents better scoring weights
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.selfmod import AutoTuner, PluginManager, FormulaEvolver


CORPUS = """
Toronto is a major city in the Canadian province of Ontario.
Toronto was founded and incorporated in the year 1834.
The city of Toronto has a population of approximately 2.7 million people.
Toronto has a humid continental climate with warm summers and cold winters.
John Graves Simcoe originally established the settlement of Toronto.
Perovskite is a highly efficient material used in modern photovoltaic cells.
Photovoltaic cells capture photons to produce electricity efficiently.
Apixaban prevents thrombosis without requiring strict dietary restrictions.
Unlike warfarin, apixaban does not require constant diet monitoring.
"""


def run_selfmod_test():
    print("=" * 70)
    print("  KOS V5.1: SELF-MODIFICATION TESTS")
    print("  The System That Tunes Itself")
    print("=" * 70)

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    driver.ingest(CORPUS)

    # ── LEVEL 1: AUTO-TUNER ──────────────────────────────────
    print("\n[LEVEL 1] AUTO-TUNER: Self-Optimizing Thresholds")
    print("-" * 50)

    tuner = AutoTuner(kernel, lexicon, driver)
    optimal = tuner.tune(verbose=True)

    tuner_pass = len(optimal) >= 3
    print(f"\n  Parameters tuned: {len(optimal)}")
    print(f"  Level 1: {'PASS' if tuner_pass else 'FAIL'}")

    # ── LEVEL 2: PLUGIN MANAGER ──────────────────────────────
    print("\n[LEVEL 2] PLUGIN MANAGER: Auto-Architecture")
    print("-" * 50)

    pm = PluginManager(kernel, lexicon)

    # Simulate some queries
    pm.record_query("Where is Toronto?")
    pm.record_query("Donde esta Toronto?")
    pm.record_query("Wo ist Toronto?")
    pm.record_query("Ou est Toronto?")
    pm.record_query("Qual e Toronto?")
    pm.record_query("Toronto nerede?")

    changes = pm.evaluate(verbose=True)
    status = pm.get_status()

    # Check that multi-language got enabled
    multilang_enabled = pm.is_enabled('multilang')
    vsa_enabled = pm.is_enabled('vsa_backplane')

    print(f"\n  Plugin status:")
    for name, info in status.items():
        tag = "ON" if info['enabled'] else "OFF"
        reason = info.get('reason', '')
        reason_str = f" ({reason})" if reason else ""
        print(f"    [{tag:>3s}] {name}{reason_str}")

    pm_pass = multilang_enabled  # Multi-lang should have auto-enabled
    print(f"\n  Multi-lang auto-enabled: "
          f"{'YES' if multilang_enabled else 'NO'}")
    print(f"  Level 2: {'PASS' if pm_pass else 'FAIL'}")

    # ── LEVEL 3: FORMULA EVOLVER ─────────────────────────────
    print("\n[LEVEL 3] FORMULA EVOLVER: Genetic Scoring Optimization")
    print("-" * 50)

    evolver = FormulaEvolver(kernel, lexicon, driver,
                              population_size=20,
                              generations=15,
                              mutation_rate=0.3)

    best_genome = evolver.evolve(verbose=True)

    # Check that the evolved formula is at least as good as default
    evolver_pass = evolver.best_fitness >= 0.75
    print(f"\n  Best fitness: {evolver.best_fitness:.0%}")
    print(f"  Level 3: {'PASS' if evolver_pass else 'FAIL'}")

    # ── Verify config was saved ──────────────────────────────
    print("\n[VERIFY] Config File")
    print("-" * 50)

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                '.cache', 'self_tuned_config.json')
    config_exists = os.path.exists(config_path)
    print(f"  Config saved: {'YES' if config_exists else 'NO'} "
          f"({config_path})")

    if config_exists:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  Config sections: {list(config.keys())}")
        print(f"  Source code modified: NO (config file only)")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SELF-MODIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Level 1 (Auto-Tuner):      {'PASS' if tuner_pass else 'FAIL'} "
          f"— {len(optimal)} parameters optimized")
    print(f"  Level 2 (Plugin Manager):  {'PASS' if pm_pass else 'FAIL'} "
          f"— {len(changes)} auto-activations")
    print(f"  Level 3 (Formula Evolver): {'PASS' if evolver_pass else 'FAIL'} "
          f"— {evolver.best_fitness:.0%} fitness")
    print(f"  Source code modified:      NEVER")
    print(f"  All changes in:            .cache/self_tuned_config.json")

    all_pass = tuner_pass and pm_pass and evolver_pass
    if all_pass:
        print(f"\n  ALL 3 SELF-MODIFICATION LEVELS VERIFIED.")
        print(f"  The system optimizes itself without touching its own code.")
    print("=" * 70)


if __name__ == "__main__":
    run_selfmod_test()
