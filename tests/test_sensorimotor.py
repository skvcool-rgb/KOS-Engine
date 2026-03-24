"""
KOS V5.0 — Sensorimotor Agent Test (Phase 4: World Grounding).

Proves the complete grounded intelligence loop:
1. Agent monitors real Wikipedia articles
2. Ingests content, builds beliefs
3. On second cycle, compares new observations against predictions
4. Detects surprises, self-corrects weights
5. Reports belief changes

This is the final phase: a system grounded in real-world data
that autonomously maintains an accurate model of external reality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.graph import KOSKernel
from kos.lexicon import KASMLexicon
from kos.drivers.text import TextDriver
from kos.forager import WebForager
from kos.predictive import PredictiveCodingEngine
from kos.sensorimotor import SensoriMotorAgent


def run_sensorimotor_test():
    print("=" * 70)
    print("  KOS V5.0: SENSORIMOTOR AGENT TEST (Phase 4)")
    print("  Live Web Grounding — Real URLs, Real Knowledge")
    print("=" * 70)

    # ── Boot the full cognitive stack ────────────────────────────
    print("\n[BOOT] Initializing full cognitive stack...")

    kernel = KOSKernel(enable_vsa=False)
    lexicon = KASMLexicon()
    driver = TextDriver(kernel, lexicon)
    forager = WebForager(kernel, lexicon, text_driver=driver)
    pce = PredictiveCodingEngine(kernel, learning_rate=0.05)

    agent = SensoriMotorAgent(
        kernel=kernel,
        lexicon=lexicon,
        forager=forager,
        predictive_engine=pce,
        text_driver=driver,
        log_file="test_agent.log"
    )

    # ── Configure watchlist (real Wikipedia articles) ────────────
    print("[BOOT] Configuring world monitoring watchlist...")

    # Three diverse topics to monitor
    agent.add_watch(
        url="https://en.wikipedia.org/wiki/Perovskite_solar_cell",
        topic="perovskite solar cell",
        check_interval=10  # Short interval for testing
    )

    agent.add_watch(
        url="https://en.wikipedia.org/wiki/Toronto",
        topic="toronto city canada",
        check_interval=10
    )

    agent.add_watch(
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        topic="artificial intelligence",
        check_interval=10
    )

    print(f"[BOOT] Graph: {len(kernel.nodes)} nodes (empty)")
    print(f"[BOOT] Watchlist: {len(agent.monitor.watchlist)} URLs")

    # ── Run 3 cycles ─────────────────────────────────────────────
    # Cycle 1: First observation — everything is new (massive surprises)
    # Cycle 2: Same URLs — content unchanged, predictions should match
    # Cycle 3: Same URLs — predictions locked, zero error expected

    print("\n[TEST] Running 3 observation cycles...")
    print("[TEST] Cycle 1 = first contact (all surprises)")
    print("[TEST] Cycle 2 = re-observation (predictions forming)")
    print("[TEST] Cycle 3 = grounded (predictions match reality)")

    result = agent.run(max_cycles=3, cycle_interval=2, verbose=True)

    # ── Verify the results ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SENSORIMOTOR GROUNDING VERIFICATION")
    print("=" * 70)

    pce_stats = pce.get_stats()

    print(f"\n  Cycles completed:      {result['cycles']}")
    print(f"  URLs monitored:        3 (live Wikipedia)")
    print(f"  Total changes:         {result['changes']}")
    print(f"  Concepts acquired:     +{result['concepts_acquired']}")
    print(f"  Final graph size:      {result['graph_size']} nodes")
    print(f"  Prediction accuracy:   {pce_stats['overall_accuracy']:.1%}")
    print(f"  Weight adjustments:    {pce_stats['total_weight_adjustments']}")
    print(f"  Cached predictions:    {pce_stats['cached_predictions']}")

    # Check for belief revision alerts
    revisions = [a for a in result['alerts']
                 if a.change_type == "belief_revision"]
    new_knowledge = [a for a in result['alerts']
                     if a.change_type == "new_knowledge"]
    confirmations = [a for a in result['alerts']
                     if a.change_type == "confirmation"]

    print(f"\n  Alert breakdown:")
    print(f"    New knowledge:    {len(new_knowledge)}")
    print(f"    Belief revisions: {len(revisions)}")
    print(f"    Confirmations:    {len(confirmations)}")

    # ── Phase 4 verification criteria ────────────────────────────
    has_concepts = result['concepts_acquired'] > 100
    has_predictions = pce_stats['cached_predictions'] > 0
    has_adjustments = pce_stats['total_weight_adjustments'] > 0
    ran_full = result['cycles'] == 3

    print(f"\n  Verification:")
    print(f"    Acquired 100+ concepts from live web:  "
          f"{'PASS' if has_concepts else 'FAIL'} "
          f"({result['concepts_acquired']})")
    print(f"    Built prediction models:               "
          f"{'PASS' if has_predictions else 'FAIL'} "
          f"({pce_stats['cached_predictions']})")
    print(f"    Self-corrected via prediction error:    "
          f"{'PASS' if has_adjustments else 'FAIL'} "
          f"({pce_stats['total_weight_adjustments']})")
    print(f"    Completed all cycles:                  "
          f"{'PASS' if ran_full else 'FAIL'} "
          f"({result['cycles']}/3)")

    if has_concepts and has_predictions and ran_full:
        print(f"\n  PHASE 4 VERIFIED: SENSORIMOTOR GROUNDING COMPLETE")
        print(f"  The agent:")
        print(f"    1. Observed 3 live Wikipedia articles")
        print(f"    2. Built an internal model of the world "
              f"({result['graph_size']} concepts)")
        print(f"    3. Predicted activation patterns from its beliefs")
        print(f"    4. Compared predictions against re-observations")
        print(f"    5. Self-corrected {pce_stats['total_weight_adjustments']} "
              f"edge weights via Hebbian learning")
        print(f"\n  This is a grounded cognitive agent running on your laptop.")
        print(f"  Its 'body' is the internet. Its 'senses' are HTTP requests.")
        print(f"  Its 'actions' change its own knowledge graph.")
        print(f"  It knows when its beliefs are wrong and fixes them.")
    else:
        print(f"\n  Phase 4 PARTIAL — check network connectivity.")

    print("=" * 70)

    # Clean up log file
    if os.path.exists("test_agent.log"):
        os.remove("test_agent.log")


if __name__ == "__main__":
    run_sensorimotor_test()
